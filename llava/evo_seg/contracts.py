"""Request and tensor contracts for the opt-in EvoVILA segmentation path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Optional, Tuple

import torch
from torch import Tensor


SegmentationTask = Literal["image", "video"]


def freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively freeze capability options without copying tensor values."""

    if not isinstance(value, Mapping):
        raise TypeError("options must be a mapping")

    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, (list, tuple)):
            return tuple(freeze(child) for child in item)
        if isinstance(item, (set, frozenset)):
            return frozenset(freeze(child) for child in item)
        return item

    return MappingProxyType({key: freeze(item) for key, item in value.items()})


def _check_tensor(name: str, value: Any, rank: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got {value.ndim}")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _check_bool_mask(name: str, value: Any, shape: Tuple[int, ...]) -> Tensor:
    tensor = _check_tensor(name, value, len(shape))
    if tensor.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}")
    return tensor


@dataclass(frozen=True)
class SegmentationRequest:
    """Request-local opt-in switch for dense segmentation computation."""

    enabled: bool = False
    task: Optional[SegmentationTask] = None
    options: Mapping[str, Any] = MappingProxyType({})
    request_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        if self.task not in (None, "image", "video"):
            raise ValueError("task must be 'image', 'video', or None")
        if self.enabled and self.task is None:
            raise ValueError("an enabled segmentation request requires task")
        if self.request_id is not None and not isinstance(self.request_id, str):
            raise TypeError("request_id must be a string or None")
        object.__setattr__(self, "options", freeze_mapping(self.options))

    @classmethod
    def disabled(cls) -> "SegmentationRequest":
        return cls()

    @classmethod
    def from_value(cls, value: Any) -> "SegmentationRequest":
        if value is None:
            return cls.disabled()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("segmentation request must be None, a request, or a mapping")

        allowed = {"enabled", "task", "options", "request_id"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown segmentation request fields: {sorted(unknown)}")
        options = value.get("options", {})
        if options is None:
            options = {}
        return cls(
            enabled=value.get("enabled", False),
            task=value.get("task"),
            options=options,
            request_id=value.get("request_id"),
        )


@dataclass(frozen=True)
class GroundingBatch:
    """Validated language and dense-feature batch shared by image and video."""

    query_states: Tensor
    query_mask: Tensor
    dense_features: Tensor
    frame_mask: Tensor
    target_masks: Optional[Tensor] = None
    target_presence: Optional[Tensor] = None
    sample_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.validate()
        if not isinstance(self.sample_ids, tuple):
            raise TypeError("sample_ids must be a tuple")
        if not all(isinstance(item, str) for item in self.sample_ids):
            raise TypeError("sample_ids must contain strings")
        if self.sample_ids and len(self.sample_ids) != self.query_states.shape[0]:
            raise ValueError("sample_ids length must match batch size")

    def validate(self) -> None:
        query_states = _check_tensor("query_states", self.query_states, 3)
        if not query_states.is_floating_point():
            raise TypeError("query_states must be floating point")
        batch_size, query_length, query_dim = query_states.shape
        if query_length <= 0 or query_dim <= 0:
            raise ValueError("query_states must have positive sequence and feature dimensions")
        query_mask = _check_bool_mask("query_mask", self.query_mask, (batch_size, query_length))
        if not query_mask.any(dim=1).all():
            raise ValueError("every sample must contain at least one valid query token")

        dense_features = _check_tensor("dense_features", self.dense_features, 5)
        if not dense_features.is_floating_point():
            raise TypeError("dense_features must be floating point")
        dense_batch, frames, channels, height, width = dense_features.shape
        if dense_batch != batch_size:
            raise ValueError("dense_features batch size must match query_states")
        if min(frames, channels, height, width) <= 0:
            raise ValueError("dense_features dimensions must be positive")
        frame_mask = _check_bool_mask("frame_mask", self.frame_mask, (batch_size, frames))
        if not frame_mask.any(dim=1).all():
            raise ValueError("every sample must contain at least one valid frame")

        tensors = [query_states, query_mask, dense_features, frame_mask]

        if self.target_masks is not None:
            target_masks = _check_tensor("target_masks", self.target_masks, 5)
            target_batch, objects, target_frames, target_height, target_width = target_masks.shape
            if target_batch != batch_size or target_frames != frames:
                raise ValueError("target_masks must align with batch and frame dimensions")
            if min(objects, target_height, target_width) <= 0:
                raise ValueError("target_masks object and spatial dimensions must be positive")
            if not (target_masks.dtype == torch.bool or target_masks.is_floating_point()):
                raise TypeError("target_masks must be bool or floating point")
            tensors.append(target_masks)
            if self.target_presence is not None and tuple(self.target_presence.shape) != (
                batch_size,
                objects,
                frames,
            ):
                raise ValueError("target_presence must align with target_masks")

        if self.target_presence is not None:
            target_presence_tensor = _check_tensor("target_presence", self.target_presence, 3)
            target_presence = _check_bool_mask(
                "target_presence", target_presence_tensor, (batch_size, target_presence_tensor.shape[1], frames)
            )
            if target_presence.shape[1] <= 0:
                raise ValueError("target_presence must contain at least one object")
            tensors.append(target_presence)
        if any(tensor.device != query_states.device for tensor in tensors[1:]):
            raise ValueError("all GroundingBatch tensors must be on the same device")


@dataclass(frozen=True)
class SegmentationResult:
    """Validated decoder output consumed by losses and optional refiners."""

    mask_logits: Tensor
    object_logits: Tensor
    object_embeddings: Tensor
    frame_embeddings: Tensor
    frame_mask: Tensor
    diagnostics: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        self.validate()
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))

    def validate(self) -> None:
        mask_logits = _check_tensor("mask_logits", self.mask_logits, 5)
        object_logits = _check_tensor("object_logits", self.object_logits, 2)
        object_embeddings = _check_tensor("object_embeddings", self.object_embeddings, 3)
        frame_embeddings = _check_tensor("frame_embeddings", self.frame_embeddings, 4)
        for name, tensor in (
            ("mask_logits", mask_logits),
            ("object_logits", object_logits),
            ("object_embeddings", object_embeddings),
            ("frame_embeddings", frame_embeddings),
        ):
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must be floating point")
        batch_size, objects, frames, height, width = mask_logits.shape
        if min(objects, frames, height, width) <= 0:
            raise ValueError("mask_logits object, time, and spatial dimensions must be positive")
        if tuple(object_logits.shape) != (batch_size, objects):
            raise ValueError("object_logits must align with mask_logits")
        embedding_dim = object_embeddings.shape[-1]
        if embedding_dim <= 0 or object_embeddings.shape[:2] != (batch_size, objects):
            raise ValueError("object_embeddings shape is invalid")
        if tuple(frame_embeddings.shape) != (batch_size, objects, frames, embedding_dim):
            raise ValueError("frame_embeddings must align with object embeddings and mask logits")
        frame_mask = _check_bool_mask("frame_mask", self.frame_mask, (batch_size, frames))
        if any(
            tensor.device != mask_logits.device
            for tensor in (object_logits, object_embeddings, frame_embeddings, frame_mask)
        ):
            raise ValueError("all SegmentationResult tensors must be on the same device")
        invalid = ~frame_mask[:, None, :, None, None]
        if invalid.any() and mask_logits.masked_select(invalid).abs().max().item() != 0:
            raise ValueError("mask_logits must be zero on invalid frames")
        invalid_embeddings = ~frame_mask[:, None, :, None]
        if invalid_embeddings.any() and frame_embeddings.masked_select(invalid_embeddings).abs().max().item() != 0:
            raise ValueError("frame_embeddings must be zero on invalid frames")
