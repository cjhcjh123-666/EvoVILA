"""No-weight S0 smoke test for the public EvoVILA segmentation entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

import torch
import yaml

from llava.evo_seg import (
    GroundingBatch,
    QueryConditionedSpatialDecoder,
    SegmentationCapability,
)


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
        raise ValueError("S0 config must contain a mapping")
    return config


def _run(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    device = torch.device(config.get("device", "cpu"))
    if device.type != "cpu":
        raise ValueError("S0 smoke is intentionally CPU-only")

    batch_size = int(config["batch_size"])
    frames = int(config["frames"])
    query_length = int(config["query_length"])
    query_dim = int(config["query_dim"])
    feature_dim = int(config["feature_dim"])
    height = int(config["height"])
    width = int(config["width"])
    query_mask = torch.ones(batch_size, query_length, dtype=torch.bool, device=device)
    if query_length > 2:
        query_mask[0, -1] = False
    frame_mask = torch.ones(batch_size, frames, dtype=torch.bool, device=device)
    if frames > 1:
        frame_mask[0, -1] = False
    batch = GroundingBatch(
        query_states=torch.randn(batch_size, query_length, query_dim, device=device),
        query_mask=query_mask,
        dense_features=torch.randn(batch_size, frames, feature_dim, height, width, device=device),
        frame_mask=frame_mask,
    )
    decoder = QueryConditionedSpatialDecoder(
        query_dim=query_dim,
        feature_dim=feature_dim,
        model_dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        num_object_queries=int(config["num_object_queries"]),
        dropout=float(config.get("dropout", 0.0)),
    ).to(device)
    capability = SegmentationCapability(decoder)
    before_sam2 = "sam2" in sys.modules
    request_task = "image" if frames == 1 else "video"
    result = capability(batch, {"enabled": True, "task": request_task})
    loss = result.mask_logits.square().mean() + result.object_logits.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in decoder.parameters() if parameter.requires_grad]
    finite = all(
        torch.isfinite(tensor).all().item()
        for tensor in (
            result.mask_logits,
            result.object_logits,
            result.object_embeddings,
            result.frame_embeddings,
        )
    )
    gradient_ok = bool(gradients) and all(
        gradient is not None and torch.isfinite(gradient).all().item() for gradient in gradients
    )
    if "sam2" in sys.modules and not before_sam2:
        raise AssertionError("S0 imported SAM2 unexpectedly")
    return {
        "git_commit": _git_commit(_repo_root()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "seed": seed,
        "execution_path": result.diagnostics["execution_path"],
        "tensor_shapes": {
            "query_states": list(batch.query_states.shape),
            "dense_features": list(batch.dense_features.shape),
            "mask_logits": list(result.mask_logits.shape),
            "frame_embeddings": list(result.frame_embeddings.shape),
        },
        "finite": finite,
        "gradient_check": gradient_ok,
        "imported_optional_modules": {"sam2": "sam2" in sys.modules},
        "component_timing_ms": dict(result.diagnostics["component_timing_ms"]),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_repo_root() / "configs/evo_seg/s0_decoder.yaml",
        help="S0 synthetic configuration path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional smoke.json path; it must be outside the repository",
    )
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    payload = _run(_load_config(config_path), config_path)
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        output_path = args.output.resolve()
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
