"""Real SAM2 image-provider smoke using local source and checkpoint paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

import torch
import yaml

from llava.evo_seg.sam2_adapter import RGBFrameBatch, SAM2ImageFeatureProvider


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


def _elapsed_ms(started: float, device: torch.device) -> float:
    _synchronize(device)
    return (time.perf_counter() - started) * 1000.0


def _run(
    config: Dict[str, Any],
    config_path: Path,
    *,
    source_root: Path,
    checkpoint: Path,
    device_text: Optional[str],
) -> Dict[str, Any]:
    sam2_config = dict(config.get("sam2", {}))
    device = torch.device(device_text or sam2_config.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real S2 smoke requires an available CUDA device")
    torch.cuda.set_device(device)
    if not source_root.is_dir():
        raise FileNotFoundError(f"SAM2 source root does not exist: {source_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint}")

    seed = int(config.get("seed", 123))
    torch.manual_seed(seed)
    input_config = dict(config.get("input", {}))
    height = int(input_config.get("synthetic_height", 96))
    width = int(input_config.get("synthetic_width", 128))
    frames = torch.zeros(1, 1, 3, height, width, dtype=torch.uint8)
    row_start, row_end = height // 5, 4 * height // 5
    column_start, column_end = width // 5, 4 * width // 5
    frames[:, :, 0, row_start:row_end, column_start:column_end] = 220
    frames[:, :, 1, row_start:row_end, column_start:column_end] = 80
    frames[:, :, 2, row_start:row_end, column_start:column_end] = 40
    batch = RGBFrameBatch(
        frames=frames,
        frame_mask=torch.ones(1, 1, dtype=torch.bool),
        sample_ids=("synthetic",),
    )
    provider = SAM2ImageFeatureProvider(
        {
            "source_root": str(source_root),
            "model_config": sam2_config["model_config"],
            "checkpoint_path": str(checkpoint),
            "device": str(device),
            "apply_postprocessing": bool(sam2_config.get("apply_postprocessing", True)),
        }
    )
    sam2_imported_before = "sam2" in sys.modules
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    provider.initialize()
    initialization_ms = _elapsed_ms(started, device)

    started = time.perf_counter()
    dense = provider.encode_frames(batch)
    encoder_ms = _elapsed_ms(started, device)
    coarse = torch.full(
        (1, 1, 1, dense.features.shape[-2], dense.features.shape[-1]),
        -4.0,
        device=device,
    )
    coarse[..., dense.features.shape[-2] // 4 : 3 * dense.features.shape[-2] // 4,
           dense.features.shape[-1] // 4 : 3 * dense.features.shape[-1] // 4] = 4.0

    started = time.perf_counter()
    refined = provider.refine_masks(batch, coarse)
    refinement_ms = _elapsed_ms(started, device)
    predictor = provider._predictor
    model = predictor.model
    state_cleared = getattr(predictor, "_features", None) is None
    if not state_cleared:
        raise AssertionError("SAM2 predictor retained request image features")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("SAM2 parameters must remain frozen")
    if not torch.isfinite(dense.features).all() or not torch.isfinite(refined).all():
        raise AssertionError("SAM2 smoke produced non-finite tensors")

    return {
        "git_commit": _git_commit(_repo_root()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "seed": seed,
        "execution_path": "sam2_image_provider",
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
        },
        "tensor_shapes": {
            "raw_frames": list(batch.frames.shape),
            "dense_features": list(dense.features.shape),
            "coarse_mask_logits": list(coarse.shape),
            "refined_mask_logits": list(refined.shape),
        },
        "finite": True,
        "sam2_frozen": True,
        "predictor_state_cleared": True,
        "imported_optional_modules": {
            "sam2_before_explicit_initialize": sam2_imported_before,
            "sam2_after_explicit_initialize": "sam2" in sys.modules,
        },
        "component_timing_ms": {
            "sam2_initialization": initialization_ms,
            "sam2_image_encoder": encoder_ms,
            "sam2_mask_refinement_with_reencode": refinement_ms,
        },
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_repo_root() / "configs/evo_seg/s2_image.yaml",
        help="S2 image-provider configuration path",
    )
    parser.add_argument("--source-root", type=Path, required=True, help="Local official SAM2 checkout")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local SAM2 checkpoint")
    parser.add_argument("--device", default=None, help="CUDA device override, for example cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional smoke.json path; it must be outside the repository",
    )
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    payload = _run(
        _load_config(config_path),
        config_path,
        source_root=args.source_root.expanduser().resolve(),
        checkpoint=args.checkpoint.expanduser().resolve(),
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
