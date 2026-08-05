"""Real opt-in VILA, predicted-anchor decoder, and SAM2 video smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

import numpy as np
import torch
from PIL import Image

from llava.evo_seg.capability import SegmentationCapability
from llava.evo_seg.decoder import QueryConditionedSpatialDecoder
from llava.evo_seg.sam2_adapter import RGBFrameBatch, SAM2ImageFeatureProvider
from llava.evo_seg.sam2_video_adapter import (
    SAM2VideoMaskPropagator,
    build_sam2_video_image_predictor,
)
from llava.evo_seg.video_pipeline import VILAVideoSegmentationPipeline
from llava.evo_seg.vila_adapter import VILASegmentationAdapter
from scripts.evo_seg.smoke_image_segmentation import (
    _ForwardTimers,
    _build_query_token_mask,
    _git_commit,
    _load_config,
    _ordinary_last_logits,
    _repo_root,
    _timed_call,
)


def _git_worktree_dirty(root: Path) -> bool:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(status.strip())


def _synthetic_frames(count: int, height: int, width: int) -> list[Image.Image]:
    if count < 2:
        raise ValueError("synthetic video requires at least two frames")
    if min(height, width) <= 8:
        raise ValueError("synthetic frame dimensions must be greater than 8")
    frames = []
    box_height = max(height // 3, 2)
    box_width = max(width // 4, 2)
    for frame_index in range(count):
        array = np.full((height, width, 3), (36, 52, 72), dtype=np.uint8)
        fraction = frame_index / max(count - 1, 1)
        row_start = height // 3
        column_start = int((width - box_width) * fraction)
        array[row_start : row_start + box_height, column_start : column_start + box_width] = (
            220,
            50,
            38,
        )
        frames.append(Image.fromarray(array, mode="RGB"))
    return frames


def _load_frames(config: Dict[str, Any], video_path: Optional[Path]) -> tuple[list[Image.Image], str]:
    input_config = dict(config.get("input", {}))
    configured_path = input_config.get("video_path")
    selected_path = video_path or (Path(configured_path) if configured_path else None)
    frame_count = int(input_config.get("synthetic_frames", 3))
    if frame_count < 2:
        raise ValueError("S3 smoke requires at least two video frames")
    if selected_path is not None:
        selected_path = selected_path.expanduser().resolve()
        if not selected_path.exists():
            raise FileNotFoundError(f"input video does not exist: {selected_path}")
        from llava.utils.media import _load_video

        frames = _load_video(str(selected_path), num_frames=frame_count, fps=0.0)
        if len(frames) < 2:
            raise RuntimeError("input video did not yield at least two frames")
        return [frame.convert("RGB") for frame in frames], "external_video"
    return (
        _synthetic_frames(
            frame_count,
            int(input_config.get("synthetic_height", 96)),
            int(input_config.get("synthetic_width", 128)),
        ),
        "synthetic_video",
    )


def _write_frames(directory: Path, frames: Sequence[Image.Image], jpeg_quality: int) -> None:
    for frame_index, frame in enumerate(frames):
        frame.save(
            directory / f"{frame_index:05d}.jpg",
            format="JPEG",
            quality=jpeg_quality,
            subsampling=0,
        )


def _rgb_batch(frames: Sequence[Image.Image]) -> RGBFrameBatch:
    arrays = [np.asarray(frame.convert("RGB"), dtype=np.uint8).copy() for frame in frames]
    tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous().unsqueeze(0)
    return RGBFrameBatch(
        frames=tensor,
        frame_mask=torch.ones(1, len(frames), dtype=torch.bool),
        sample_ids=("s3-video-smoke",),
    )


def _prepare_vila_video_inputs(
    model: Any,
    frame_directory: Path,
    instruction: str,
    device: torch.device,
):
    from llava.media import Video
    from llava.mm_utils import process_images
    from llava.utils.media import extract_media
    from llava.utils.tokenizer import tokenize_conversation

    conversation = [{"from": "human", "value": [Video(str(frame_directory)), instruction]}]
    media = extract_media(conversation, config=model.config)
    processed = process_images(media["image"], model.vision_tower.image_processor, model.config)
    processed = processed.to(device=device, dtype=torch.float16)
    media["image"] = [item for item in processed]
    input_ids = tokenize_conversation(
        conversation,
        model.tokenizer,
        add_generation_prompt=True,
    ).unsqueeze(0).to(device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    return input_ids, media, defaultdict(dict), attention_mask


def _run(
    config: Dict[str, Any],
    config_path: Path,
    *,
    vila_model_path: Path,
    sam2_source_root: Path,
    sam2_checkpoint: Path,
    video_path: Optional[Path],
    device_text: Optional[str],
    anchor_index: Optional[int],
) -> Dict[str, Any]:
    if not vila_model_path.is_dir() or not (vila_model_path / "config.json").is_file():
        raise FileNotFoundError(f"VILA model directory is invalid: {vila_model_path}")
    if not sam2_source_root.is_dir():
        raise FileNotFoundError(f"SAM2 source root does not exist: {sam2_source_root}")
    if not sam2_checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {sam2_checkpoint}")
    sam2_config = dict(config.get("sam2", {}))
    video_config = dict(config.get("video", {}))
    device = torch.device(device_text or sam2_config.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real S3 end-to-end smoke requires an available CUDA device")
    torch.cuda.set_device(device)
    seed = int(config.get("seed", 123))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)

    frames, input_kind = _load_frames(config, video_path)
    rgb_frames = _rgb_batch(frames)
    prompt_config = dict(config.get("prompt", {}))
    query = str(prompt_config.get("query", "bright red rectangle"))
    template = str(
        prompt_config.get(
            "instruction_template",
            "Track and segment the object described by this referring expression: {query}.",
        )
    )
    instruction = template.format(query=query)

    import llava

    model, vila_initialization_ms = _timed_call(
        device,
        lambda: llava.load(
            str(vila_model_path),
            device=str(device),
            device_map={"": str(device)},
        ),
    )
    model.eval()
    model.config.num_video_frames = len(frames)
    jpeg_quality = int(video_config.get("jpeg_quality", 95))
    with tempfile.TemporaryDirectory(prefix="evovila-vila-video-") as directory:
        frame_directory = Path(directory)
        _write_frames(frame_directory, frames, jpeg_quality)
        input_ids, media, media_config, attention_mask = _prepare_vila_video_inputs(
            model,
            frame_directory,
            instruction,
            device,
        )
        query_token_mask = _build_query_token_mask(input_ids, model.tokenizer, query).to(device=device)

        baseline_before, ordinary_before_ms = _timed_call(
            device,
            lambda: _ordinary_last_logits(model, input_ids, media, media_config, attention_mask),
        )
        sam2_imported_before = any(name == "sam2" or name.startswith("sam2.") for name in sys.modules)
        if sam2_imported_before:
            raise AssertionError("ordinary VILA video path imported SAM2 before explicit initialization")

        provider = SAM2ImageFeatureProvider(
            {
                "source_root": str(sam2_source_root),
                "model_config": sam2_config["model_config"],
                "checkpoint_path": str(sam2_checkpoint),
                "device": str(device),
                "apply_postprocessing": bool(sam2_config.get("apply_postprocessing", True)),
            },
            predictor_factory=build_sam2_video_image_predictor,
        )
        propagator = SAM2VideoMaskPropagator(provider, video_config)
        _, sam2_initialization_ms = _timed_call(device, propagator.initialize)
        decoder_config = dict(config.get("decoder", {}))
        decoder = QueryConditionedSpatialDecoder(
            query_dim=int(model.llm.config.hidden_size),
            feature_dim=int(decoder_config.get("feature_dim", 256)),
            model_dim=int(decoder_config.get("model_dim", 128)),
            num_heads=int(decoder_config.get("num_heads", 8)),
            num_layers=int(decoder_config.get("num_layers", 2)),
            num_object_queries=int(decoder_config.get("num_object_queries", 1)),
            dropout=float(decoder_config.get("dropout", 0.0)),
        ).to(device=device)
        adapter = VILASegmentationAdapter(model, provider, SegmentationCapability(decoder))
        pipeline = VILAVideoSegmentationPipeline(adapter, propagator)
        request = dict(config.get("request", {"enabled": True, "task": "video"}))
        anchors = None
        if anchor_index is not None:
            anchors = torch.tensor([anchor_index], dtype=torch.long)

        with _ForwardTimers(
            device,
            {
                "vila_vision_encoder": model.get_vision_tower(),
                "vila_llm": model.get_llm(),
            },
        ) as forward_timers:
            output = pipeline.segment(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                query_token_mask=query_token_mask,
                rgb_frames=rgb_frames,
                request=request,
                attention_mask=attention_mask,
                anchor_indices=anchors,
            )

        pipeline.clear_request_state()
        if pipeline.adapter.capability.last_diagnostics:
            raise AssertionError("video pipeline retained capability diagnostics after explicit clear")
        baseline_after, ordinary_after_ms = _timed_call(
            device,
            lambda: _ordinary_last_logits(model, input_ids, media, media_config, attention_mask),
        )

    retention_max_abs_diff = float((baseline_before - baseline_after).abs().max().item())
    retention_exact = torch.equal(baseline_before, baseline_after)
    if not retention_exact:
        raise AssertionError(f"ordinary VILA logits changed after video extension: {retention_max_abs_diff}")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("VILA parameters must be frozen by the explicit adapter")
    predictor = provider._predictor
    if predictor is None or getattr(predictor, "_features", None) is not None:
        raise AssertionError("SAM2 image predictor retained request features")
    if any(parameter.requires_grad for parameter in predictor.model.parameters()):
        raise AssertionError("SAM2 parameters must remain frozen")
    sam2_imported_after = any(name == "sam2" or name.startswith("sam2.") for name in sys.modules)
    if not sam2_imported_after:
        raise AssertionError("explicit SAM2 video initialization did not import the local package")
    coarse = output.coarse_anchor_result
    propagated = output.propagation_result
    if not torch.isfinite(coarse.mask_logits).all() or not torch.isfinite(propagated.mask_logits).all():
        raise AssertionError("S3 video pipeline produced non-finite masks")
    if propagated.diagnostics["prompt_source"] != "predicted_anchor_mask":
        raise AssertionError("SAM2 video propagation did not record a predicted anchor prompt")
    if not propagated.diagnostics["state_cleared"]:
        raise AssertionError("SAM2 video propagation state was not cleared")

    coarse_timing = dict(coarse.diagnostics["component_timing_ms"])
    dense_diagnostics = coarse.diagnostics["dense_provider"]
    propagation_timing = dict(propagated.diagnostics["component_timing_ms"])
    component_timing_ms = {
        "vila_model_initialization": vila_initialization_ms,
        "sam2_initialization": sam2_initialization_ms,
        "ordinary_vila_before_extension": ordinary_before_ms,
        "ordinary_vila_after_extension": ordinary_after_ms,
        "vila_query_encoding_total": float(coarse_timing["vila_query_encoding"]),
        "vila_vision_encoder": forward_timers.elapsed_ms["vila_vision_encoder"],
        "vila_llm": forward_timers.elapsed_ms["vila_llm"],
        "sam2_anchor_image_encoder": float(
            dense_diagnostics["component_timing_ms"]["sam2_image_encoder"]
        ),
        "mask_decoder": float(coarse_timing["decoder"]),
        **{name: float(value) for name, value in propagation_timing.items()},
        "video_pipeline_total": float(output.diagnostics["component_timing_ms"]["video_pipeline_total"]),
    }
    query_positions = coarse.diagnostics["query_source_positions"]
    fused_positions = coarse.diagnostics["query_fused_positions"]
    return {
        "git_commit": _git_commit(_repo_root()),
        "git_worktree_dirty": _git_worktree_dirty(_repo_root()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "seed": seed,
        "execution_path": output.diagnostics["execution_path"],
        "input_kind": input_kind,
        "decoder_state": "randomly_initialized_plumbing_only",
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "offline_model_loading": True,
        },
        "assets": {
            "vila_model_name": vila_model_path.name,
            "sam2_checkpoint_name": sam2_checkpoint.name,
            "sam2_config": sam2_config["model_config"],
        },
        "query": {
            "text": query,
            "original_token_count": int(query_token_mask.sum().item()),
            "source_positions": query_positions.detach().cpu().tolist(),
            "fused_positions": fused_positions.detach().cpu().tolist(),
        },
        "anchor": {
            "policy": output.diagnostics["anchor_policy"],
            "source_indices": propagated.anchor_indices.detach().cpu().tolist(),
            "prompt_source": propagated.diagnostics["prompt_source"],
            "prompt_threshold": propagated.diagnostics["prompt_threshold"],
        },
        "tensor_shapes": {
            "raw_frames": list(rgb_frames.frames.shape),
            "vila_input_ids": list(input_ids.shape),
            "sam2_anchor_dense_features": list(dense_diagnostics["feature_shape"]),
            "coarse_anchor_mask_logits": list(coarse.mask_logits.shape),
            "propagated_mask_logits": list(propagated.mask_logits.shape),
        },
        "checks": {
            "finite": True,
            "vila_frozen": True,
            "sam2_frozen": True,
            "sam2_image_predictor_state_cleared": True,
            "sam2_video_state_cleared": True,
            "capability_diagnostics_cleared": True,
            "ordinary_vila_retention_exact": retention_exact,
            "ordinary_vila_retention_max_abs_diff": retention_max_abs_diff,
            "sam2_absent_before_explicit_initialize": not sam2_imported_before,
            "sam2_present_after_extension": sam2_imported_after,
        },
        "component_timing_ms": component_timing_ms,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_repo_root() / "configs/evo_seg/s3_video.yaml",
        help="S3 video-pipeline configuration path",
    )
    parser.add_argument("--vila-model", type=Path, required=True, help="Local VILA model directory")
    parser.add_argument("--sam2-source-root", type=Path, required=True, help="Local official SAM2 checkout")
    parser.add_argument("--sam2-checkpoint", type=Path, required=True, help="Local SAM2 checkpoint")
    parser.add_argument("--video", type=Path, default=None, help="Optional local MP4 or JPEG directory")
    parser.add_argument("--anchor-index", type=int, default=None, help="Optional explicit source-frame anchor")
    parser.add_argument("--device", default=None, help="CUDA device override, for example cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional smoke.json path; it must be outside the repository",
    )
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    payload = _run(
        _load_config(config_path),
        config_path,
        vila_model_path=args.vila_model.expanduser().resolve(),
        sam2_source_root=args.sam2_source_root.expanduser().resolve(),
        sam2_checkpoint=args.sam2_checkpoint.expanduser().resolve(),
        video_path=None if args.video is None else args.video.expanduser().resolve(),
        device_text=args.device,
        anchor_index=args.anchor_index,
    )
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        try:
            output_path.relative_to(_repo_root().resolve())
        except ValueError:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(serialized + "\n", encoding="utf-8")
        else:
            raise ValueError("smoke output must be outside the repository")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
