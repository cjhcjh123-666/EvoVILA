"""Real opt-in VILA, spatial-decoder, and SAM2 image smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
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
import yaml
from PIL import Image

from llava.evo_seg.capability import SegmentationCapability
from llava.evo_seg.decoder import QueryConditionedSpatialDecoder
from llava.evo_seg.image_pipeline import VILAImageSegmentationPipeline
from llava.evo_seg.sam2_adapter import RGBFrameBatch, SAM2ImageFeatureProvider
from llava.evo_seg.vila_adapter import VILASegmentationAdapter


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("S2 config must contain a mapping")
    return config


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_call(device: torch.device, function):
    _synchronize(device)
    started = time.perf_counter()
    value = function()
    _synchronize(device)
    return value, (time.perf_counter() - started) * 1000.0


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = getattr(encoded, "input_ids", encoded)
    if isinstance(values, torch.Tensor):
        values = values.flatten().tolist()
    if values and isinstance(values[0], (list, tuple)):
        values = values[0]
    return tuple(int(value) for value in values)


def _build_query_token_mask(input_ids: torch.Tensor, tokenizer: Any, query: str) -> torch.Tensor:
    """Locate one explicit multi-token query span in a batch-size-one prompt."""

    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("query span smoke helper requires input_ids with shape [1,L]")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    row = input_ids[0].tolist()
    spans = set()
    for candidate_text in (query, f" {query}"):
        candidate = _token_ids(tokenizer, candidate_text)
        if not candidate:
            continue
        for start in range(len(row) - len(candidate) + 1):
            if tuple(row[start : start + len(candidate)]) == candidate:
                spans.add((start, start + len(candidate)))
    if len(spans) != 1:
        raise ValueError(f"query must map to exactly one token span, found {sorted(spans)}")
    start, end = next(iter(spans))
    if end - start < 2:
        raise ValueError("S2 smoke requires a multi-token referring expression")
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mask[0, start:end] = True
    return mask


def _synthetic_image(height: int, width: int) -> Image.Image:
    if min(height, width) <= 8:
        raise ValueError("synthetic image dimensions must be greater than 8")
    array = np.full((height, width, 3), (36, 52, 72), dtype=np.uint8)
    row_start, row_end = height // 5, 4 * height // 5
    column_start, column_end = width // 5, 4 * width // 5
    array[row_start:row_end, column_start:column_end] = (220, 50, 38)
    return Image.fromarray(array, mode="RGB")


def _load_image(config: Dict[str, Any], image_path: Optional[Path]) -> tuple[Image.Image, str]:
    input_config = dict(config.get("input", {}))
    configured_path = input_config.get("image_path")
    selected_path = image_path or (Path(configured_path) if configured_path else None)
    if selected_path is not None:
        selected_path = selected_path.expanduser().resolve()
        if not selected_path.is_file():
            raise FileNotFoundError(f"input image does not exist: {selected_path}")
        with Image.open(selected_path) as handle:
            return handle.convert("RGB"), "external_image"
    return (
        _synthetic_image(
            int(input_config.get("synthetic_height", 96)),
            int(input_config.get("synthetic_width", 128)),
        ),
        "synthetic_image",
    )


def _rgb_batch(image: Image.Image) -> RGBFrameBatch:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    frames = torch.from_numpy(array).permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(0)
    return RGBFrameBatch(
        frames=frames,
        frame_mask=torch.ones(1, 1, dtype=torch.bool),
        sample_ids=("s2-image-smoke",),
    )


def _prepare_vila_inputs(model: Any, image: Image.Image, instruction: str, device: torch.device):
    from llava.mm_utils import process_images
    from llava.utils.media import extract_media
    from llava.utils.tokenizer import tokenize_conversation

    conversation = [{"from": "human", "value": [image.copy(), instruction]}]
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


def _ordinary_last_logits(
    model: Any,
    input_ids: torch.Tensor,
    media: Dict[str, Any],
    media_config: Dict[str, Any],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            media=media,
            media_config=media_config,
            attention_mask=attention_mask,
            packing=False,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
    return outputs.logits[:, -1].detach().to(device="cpu", dtype=torch.float32)


class _ForwardTimers:
    def __init__(self, device: torch.device, modules: Dict[str, torch.nn.Module]) -> None:
        self.device = device
        self.modules = modules
        self.elapsed_ms = {name: 0.0 for name in modules}
        self._started = {}
        self._handles = []

    def __enter__(self):
        for name, module in self.modules.items():
            self._handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
            self._handles.append(module.register_forward_hook(self._post_hook(name)))
        return self

    def _pre_hook(self, name: str):
        def hook(_module, _inputs):
            _synchronize(self.device)
            self._started[name] = time.perf_counter()

        return hook

    def _post_hook(self, name: str):
        def hook(_module, _inputs, _output):
            _synchronize(self.device)
            self.elapsed_ms[name] += (time.perf_counter() - self._started.pop(name)) * 1000.0

        return hook

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _run(
    config: Dict[str, Any],
    config_path: Path,
    *,
    vila_model_path: Path,
    sam2_source_root: Path,
    sam2_checkpoint: Path,
    image_path: Optional[Path],
    device_text: Optional[str],
) -> Dict[str, Any]:
    if not vila_model_path.is_dir() or not (vila_model_path / "config.json").is_file():
        raise FileNotFoundError(f"VILA model directory is invalid: {vila_model_path}")
    if not sam2_source_root.is_dir():
        raise FileNotFoundError(f"SAM2 source root does not exist: {sam2_source_root}")
    if not sam2_checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {sam2_checkpoint}")
    sam2_config = dict(config.get("sam2", {}))
    device = torch.device(device_text or sam2_config.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real S2 end-to-end smoke requires an available CUDA device")
    torch.cuda.set_device(device)
    seed = int(config.get("seed", 123))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)

    image, input_kind = _load_image(config, image_path)
    rgb_batch = _rgb_batch(image)
    prompt_config = dict(config.get("prompt", {}))
    query = str(prompt_config.get("query", "bright red rectangle"))
    template = str(
        prompt_config.get(
            "instruction_template",
            "Segment the object described by this referring expression: {query}.",
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
    input_ids, media, media_config, attention_mask = _prepare_vila_inputs(
        model,
        image,
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
        raise AssertionError("ordinary VILA path imported SAM2 before explicit initialization")

    provider = SAM2ImageFeatureProvider(
        {
            "source_root": str(sam2_source_root),
            "model_config": sam2_config["model_config"],
            "checkpoint_path": str(sam2_checkpoint),
            "device": str(device),
            "apply_postprocessing": bool(sam2_config.get("apply_postprocessing", True)),
        }
    )
    _, sam2_initialization_ms = _timed_call(device, provider.initialize)
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
    capability = SegmentationCapability(decoder)
    adapter = VILASegmentationAdapter(model, provider, capability)
    pipeline = VILAImageSegmentationPipeline(adapter, provider)
    request = dict(config.get("request", {"enabled": True, "task": "image"}))

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
            rgb_frames=rgb_batch,
            request=request,
            attention_mask=attention_mask,
        )

    pipeline.clear_request_state()
    if pipeline.adapter.capability.last_diagnostics:
        raise AssertionError("image pipeline retained capability diagnostics after explicit clear")

    baseline_after, ordinary_after_ms = _timed_call(
        device,
        lambda: _ordinary_last_logits(model, input_ids, media, media_config, attention_mask),
    )
    retention_max_abs_diff = float((baseline_before - baseline_after).abs().max().item())
    retention_exact = torch.equal(baseline_before, baseline_after)
    if not retention_exact:
        raise AssertionError(f"ordinary VILA logits changed after extension execution: {retention_max_abs_diff}")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("VILA parameters must be frozen by the explicit adapter")
    predictor = provider._predictor
    if predictor is None or getattr(predictor, "_features", None) is not None:
        raise AssertionError("SAM2 predictor retained request image features")
    if any(parameter.requires_grad for parameter in predictor.model.parameters()):
        raise AssertionError("SAM2 parameters must remain frozen")
    sam2_imported_after = any(name == "sam2" or name.startswith("sam2.") for name in sys.modules)
    if not sam2_imported_after:
        raise AssertionError("explicit SAM2 initialization did not import the local package")
    coarse = output.coarse_result
    refined = output.refined_mask_logits
    if not torch.isfinite(coarse.mask_logits).all() or not torch.isfinite(refined).all():
        raise AssertionError("S2 image pipeline produced non-finite masks")

    coarse_timing = dict(coarse.diagnostics["component_timing_ms"])
    dense_diagnostics = coarse.diagnostics["dense_provider"]
    component_timing_ms = {
        "vila_model_initialization": vila_initialization_ms,
        "sam2_initialization": sam2_initialization_ms,
        "ordinary_vila_before_extension": ordinary_before_ms,
        "ordinary_vila_after_extension": ordinary_after_ms,
        "vila_query_encoding_total": float(coarse_timing["vila_query_encoding"]),
        "vila_vision_encoder": forward_timers.elapsed_ms["vila_vision_encoder"],
        "vila_llm": forward_timers.elapsed_ms["vila_llm"],
        "sam2_image_encoder": float(dense_diagnostics["component_timing_ms"]["sam2_image_encoder"]),
        "mask_decoder": float(coarse_timing["decoder"]),
        "sam2_mask_refinement_with_reencode": float(
            output.diagnostics["component_timing_ms"]["sam2_mask_refinement_with_reencode"]
        ),
        "sam2_video_propagation": None,
        "image_pipeline_total": float(output.diagnostics["component_timing_ms"]["image_pipeline_total"]),
    }
    query_positions = coarse.diagnostics["query_source_positions"]
    fused_positions = coarse.diagnostics["query_fused_positions"]
    return {
        "git_commit": _git_commit(_repo_root()),
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
        "tensor_shapes": {
            "raw_frames": list(rgb_batch.frames.shape),
            "vila_input_ids": list(input_ids.shape),
            "sam2_dense_features": list(dense_diagnostics["feature_shape"]),
            "coarse_mask_logits": list(coarse.mask_logits.shape),
            "refined_mask_logits": list(refined.shape),
        },
        "checks": {
            "finite": True,
            "vila_frozen": True,
            "sam2_frozen": True,
            "sam2_predictor_state_cleared": True,
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
        default=_repo_root() / "configs/evo_seg/s2_image.yaml",
        help="S2 image-pipeline configuration path",
    )
    parser.add_argument("--vila-model", type=Path, required=True, help="Local VILA model directory")
    parser.add_argument("--sam2-source-root", type=Path, required=True, help="Local official SAM2 checkout")
    parser.add_argument("--sam2-checkpoint", type=Path, required=True, help="Local SAM2 checkpoint")
    parser.add_argument("--image", type=Path, default=None, help="Optional local image; synthetic RGB is default")
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
        image_path=None if args.image is None else args.image.expanduser().resolve(),
        device_text=args.device,
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
