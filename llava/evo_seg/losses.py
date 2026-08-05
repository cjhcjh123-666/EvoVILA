"""Losses and negative controls for the opt-in segmentation decoder."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import GroundingBatch, SegmentationResult


def _validate_valid(valid: Tensor, shape: tuple[int, int, int]) -> Tensor:
    if not isinstance(valid, Tensor) or valid.dtype != torch.bool or tuple(valid.shape) != shape:
        raise ValueError(f"valid must be a bool tensor with shape {shape}")
    return valid


def _align_targets(logits: Tensor, targets: Tensor) -> Tensor:
    if not isinstance(targets, Tensor) or targets.ndim != 5:
        raise ValueError("targets must have shape [B,N,T,H,W]")
    if logits.shape[:3] != targets.shape[:3]:
        raise ValueError("logits and targets must align in batch, object, and time dimensions")
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    if targets.shape[-2:] != logits.shape[-2:]:
        batch_size, objects, frames, target_height, target_width = targets.shape
        targets = F.interpolate(
            targets.reshape(batch_size * objects * frames, 1, target_height, target_width),
            size=logits.shape[-2:],
            mode="nearest",
        ).reshape(batch_size, objects, frames, *logits.shape[-2:])
    return targets.clamp(0, 1)


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    weights = valid.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1)


def binary_mask_loss(logits: Tensor, targets: Tensor, valid: Tensor) -> Tensor:
    """Compute per-object/frame BCE, excluding invalid entries."""

    if logits.ndim != 5 or not logits.is_floating_point():
        raise ValueError("logits must be a floating tensor with shape [B,N,T,H,W]")
    valid = _validate_valid(valid, tuple(logits.shape[:3]))
    targets = _align_targets(logits, targets)
    per_entry = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=(-1, -2))
    return _masked_mean(per_entry, valid)


def dice_loss(logits: Tensor, targets: Tensor, valid: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute soft Dice loss over valid object/frame entries."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    if logits.ndim != 5 or not logits.is_floating_point():
        raise ValueError("logits must be a floating tensor with shape [B,N,T,H,W]")
    valid = _validate_valid(valid, tuple(logits.shape[:3]))
    targets = _align_targets(logits, targets)
    probabilities = logits.sigmoid().flatten(start_dim=-2)
    targets = targets.flatten(start_dim=-2)
    intersection = (probabilities * targets).sum(dim=-1)
    denominator = probabilities.sum(dim=-1) + targets.sum(dim=-1)
    per_entry = 1 - (2 * intersection + eps) / (denominator + eps)
    return _masked_mean(per_entry, valid)


def objectness_loss(logits: Tensor, presence: Tensor, frame_mask: Tensor) -> Tensor:
    """Supervise whether each referred object exists in any valid frame."""

    if logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("objectness logits must have shape [B,N]")
    if presence.ndim != 3 or presence.dtype != torch.bool:
        raise ValueError("presence must be a bool tensor with shape [B,N,T]")
    if frame_mask.ndim != 2 or frame_mask.dtype != torch.bool:
        raise ValueError("frame_mask must be a bool tensor with shape [B,T]")
    if tuple(presence.shape[:2]) != tuple(logits.shape) or presence.shape[2] != frame_mask.shape[1]:
        raise ValueError("presence, frame_mask, and objectness logits are misaligned")
    if frame_mask.shape[0] != logits.shape[0]:
        raise ValueError("frame_mask batch size must match objectness logits")
    valid_presence = presence & frame_mask[:, None, :]
    labels = valid_presence.any(dim=-1).to(device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, labels)


def _region_score(logits: Tensor, targets: Tensor) -> Tensor:
    targets = _align_targets(logits, targets)
    positive_count = targets.flatten(start_dim=-2).sum(dim=-1).clamp_min(1)
    negative = 1 - targets
    negative_count = negative.flatten(start_dim=-2).sum(dim=-1).clamp_min(1)
    positive_score = (logits * targets).flatten(start_dim=-2).sum(dim=-1) / positive_count
    negative_score = (logits * negative).flatten(start_dim=-2).sum(dim=-1) / negative_count
    return positive_score - negative_score


def query_swap_margin_loss(
    correct_logits: Tensor,
    swapped_logits: Tensor,
    targets: Tensor,
    valid: Tensor,
    margin: float = 0.1,
) -> Tensor:
    """Require the correct query to score its target above a same-media swap."""

    if correct_logits.shape != swapped_logits.shape or correct_logits.ndim != 5:
        raise ValueError("correct_logits and swapped_logits must share shape [B,N,T,H,W]")
    if not correct_logits.is_floating_point() or not swapped_logits.is_floating_point():
        raise ValueError("correct_logits and swapped_logits must be floating point")
    if not torch.isfinite(correct_logits).all() or not torch.isfinite(swapped_logits).all():
        raise ValueError("correct_logits and swapped_logits must be finite")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    valid = _validate_valid(valid, tuple(correct_logits.shape[:3]))
    correct_score = _region_score(correct_logits, targets)
    swapped_score = _region_score(swapped_logits, targets)
    return _masked_mean(F.relu(margin - correct_score + swapped_score), valid)


def temporal_consistency_loss(frame_embeddings: Tensor, presence: Tensor, frame_mask: Tensor) -> Tensor:
    """Penalize adjacent-frame representation drift for present objects."""

    if frame_embeddings.ndim != 4 or not frame_embeddings.is_floating_point():
        raise ValueError("frame_embeddings must have shape [B,N,T,D]")
    batch_size, objects, frames, _ = frame_embeddings.shape
    if presence.shape != (batch_size, objects, frames) or presence.dtype != torch.bool:
        raise ValueError("presence must be bool with shape [B,N,T]")
    if frame_mask.shape != (batch_size, frames) or frame_mask.dtype != torch.bool:
        raise ValueError("frame_mask must be bool with shape [B,T]")
    if frames <= 1:
        return frame_embeddings.sum() * 0
    adjacent_valid = (
        presence[:, :, 1:]
        & presence[:, :, :-1]
        & frame_mask[:, None, 1:]
        & frame_mask[:, None, :-1]
    )
    drift = 1 - F.cosine_similarity(frame_embeddings[:, :, 1:], frame_embeddings[:, :, :-1], dim=-1)
    return _masked_mean(drift, adjacent_valid)


def compute_segmentation_loss(
    result: SegmentationResult,
    batch: GroundingBatch,
    weights: Mapping[str, float],
    controls: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Tensor]:
    """Compute all enabled segmentation losses and their weighted total."""

    controls = {} if controls is None else controls
    valid = batch.frame_mask[:, None, :]
    if batch.target_presence is not None:
        valid = batch.target_presence & valid
    if result.mask_logits.shape[:3] != tuple(valid.shape):
        raise ValueError("result and batch object/time dimensions are misaligned")
    if batch.target_masks is None:
        raise ValueError("target_masks are required for segmentation losses")
    if batch.target_presence is None:
        raise ValueError("target_presence is required for segmentation losses")

    zero = result.mask_logits.sum() * 0
    bce = binary_mask_loss(result.mask_logits, batch.target_masks, valid)
    dice = dice_loss(result.mask_logits, batch.target_masks, valid)
    objectness = objectness_loss(result.object_logits, batch.target_presence, batch.frame_mask)
    query_swap = zero
    if float(weights.get("query_swap", 0.0)) != 0:
        swapped_logits = controls.get("swapped_logits")
        if swapped_logits is None:
            raise ValueError("swapped_logits are required when query_swap weight is nonzero")
        query_swap = query_swap_margin_loss(
            result.mask_logits,
            swapped_logits,
            batch.target_masks,
            valid,
            margin=float(controls.get("margin", 0.1)),
        )
    temporal = temporal_consistency_loss(result.frame_embeddings, batch.target_presence, batch.frame_mask)
    terms = {"bce": bce, "dice": dice, "objectness": objectness, "query_swap": query_swap, "temporal": temporal}
    total = zero
    for name, term in terms.items():
        weight = float(weights.get(name, 0.0 if name in {"query_swap", "temporal"} else 1.0))
        if weight < 0 or not math.isfinite(weight):
            raise ValueError(f"loss weight for {name} must be finite and non-negative")
        total = total + term * weight
    terms["total"] = total
    return terms
