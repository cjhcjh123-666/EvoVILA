#!/usr/bin/env python
"""Run one external-asset SAM2.1/DAVIS video segmentation smoke test."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from llava.capabilities import build_capability_pipeline
from llava.model.llava_arch import LlavaMetaModel


class SmokeModel(LlavaMetaModel, nn.Module):
    """Use the same public model entry point without loading VILA weights."""

    def __init__(self, capability_options):
        nn.Module.__init__(self)
        self.capabilities = build_capability_pipeline(
            {
                "names": ["video_segmentation"],
                "options": {"video_segmentation": capability_options},
            }
        )
        self.capability_modules = nn.ModuleList(
            [capability for capability in self.capabilities.capabilities if isinstance(capability, nn.Module)]
        )
        self._active_capability_context = None


def _prompt_from_annotation(annotation_path: Path):
    annotation = np.asarray(Image.open(annotation_path))
    foreground = np.argwhere(annotation > 0)
    if len(foreground) == 0:
        raise RuntimeError(f"DAVIS annotation has no foreground pixels: {annotation_path}")
    y, x = foreground[len(foreground) // 2]
    return [[float(x), float(y)]], [1]


def _mean_iou(predicted: torch.Tensor, annotation_dir: Path) -> float:
    scores = []
    for frame_idx, predicted_mask in enumerate(predicted):
        annotation_path = annotation_dir / f"{frame_idx:05d}.png"
        target = torch.from_numpy(np.asarray(Image.open(annotation_path)) > 0)
        predicted_mask = predicted_mask.to(dtype=torch.bool)
        intersection = torch.logical_and(predicted_mask, target).sum().item()
        union = torch.logical_or(predicted_mask, target).sum().item()
        scores.append(1.0 if union == 0 else intersection / union)
    return float(sum(scores) / len(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-root", default="/9950backfile/zhangyafei/sam2")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--davis-root", default="/9950backfile/zhangyafei/DAVIS-2017")
    parser.add_argument("--sequence", default="bear")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anchor-frame", type=int, default=0)
    parser.add_argument("--vos-optimized", action="store_true")
    args = parser.parse_args()

    davis_root = Path(args.davis_root).expanduser()
    video_dir = davis_root / "JPEGImages" / "480p" / args.sequence
    annotation_dir = davis_root / "Annotations" / "480p" / args.sequence
    if not video_dir.is_dir() or not annotation_dir.is_dir():
        raise SystemExit(f"DAVIS sequence not found: {args.sequence} under {davis_root}")
    points, labels = _prompt_from_annotation(annotation_dir / f"{args.anchor_frame:05d}.png")

    capability_options = {
        "sam2_root": str(Path(args.sam2_root).expanduser()),
        "device": args.device,
        "vos_optimized": args.vos_optimized,
    }
    if args.checkpoint is not None:
        capability_options["checkpoint"] = args.checkpoint

    model = SmokeModel(capability_options).eval()
    if args.device.startswith("cuda"):
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    with autocast:
        result = model.segment_videos(
            video_dir,
            anchor_frame=args.anchor_frame,
            points=points,
            labels=labels,
            return_result=True,
        )

    print(
        json.dumps(
            {
                "sequence": args.sequence,
                "mask_shape": list(result.masks.shape),
                "mask_dtype": str(result.masks.dtype),
                "object_ids": list(result.object_ids),
                "video_size": list(result.video_size),
                "mean_iou": _mean_iou(result.masks, annotation_dir),
                "instrumentation": dict(result.instrumentation),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
