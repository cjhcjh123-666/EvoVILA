"""Explicit predicted-anchor VILA-to-SAM2 video segmentation composition."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional

import torch
from torch import Tensor

from .contracts import SegmentationRequest, SegmentationResult, freeze_mapping
from .sam2_adapter import RGBFrameBatch
from .sam2_video_adapter import SAM2VideoMaskPropagator, VideoPropagationResult
from .vila_adapter import VILASegmentationAdapter


def _synchronize(tensor: Tensor) -> None:
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def _anchor_indices(batch: RGBFrameBatch, value: Optional[Tensor]) -> Tensor:
    if value is None:
        return batch.frame_mask.to(dtype=torch.int64).argmax(dim=1)
    if not isinstance(value, Tensor) or value.dtype != torch.long:
        raise TypeError("anchor_indices must be a torch.long tensor or None")
    if tuple(value.shape) != (batch.frames.shape[0],):
        raise ValueError("anchor_indices must have shape [B]")
    anchors = value.detach().to(device="cpu")
    frames = batch.frames.shape[1]
    if ((anchors < 0) | (anchors >= frames)).any():
        raise ValueError("anchor_indices must lie inside the video frame range")
    if not batch.frame_mask.gather(1, anchors[:, None]).all():
        raise ValueError("every anchor index must select a valid frame")
    return anchors


def _select_anchor_batch(batch: RGBFrameBatch, anchors: Tensor) -> RGBFrameBatch:
    batch_indices = torch.arange(batch.frames.shape[0], dtype=torch.long)
    frames = batch.frames[batch_indices, anchors].unsqueeze(1).contiguous()
    return RGBFrameBatch(
        frames=frames,
        frame_mask=torch.ones(batch.frames.shape[0], 1, dtype=torch.bool),
        sample_ids=batch.sample_ids,
    )


@dataclass(frozen=True)
class VideoSegmentationOutput:
    """Coarse predicted anchor plus SAM2-propagated video mask logits."""

    coarse_anchor_result: SegmentationResult
    propagation_result: VideoPropagationResult
    diagnostics: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        coarse = self.coarse_anchor_result
        propagated = self.propagation_result
        if not isinstance(coarse, SegmentationResult):
            raise TypeError("coarse_anchor_result must be a SegmentationResult")
        if not isinstance(propagated, VideoPropagationResult):
            raise TypeError("propagation_result must be a VideoPropagationResult")
        if coarse.mask_logits.shape[2] != 1:
            raise ValueError("coarse anchor result must contain exactly one frame")
        if tuple(coarse.mask_logits.shape[:2]) != tuple(propagated.mask_logits.shape[:2]):
            raise ValueError("coarse and propagated masks must align in batch and object axes")
        if coarse.mask_logits.device != propagated.mask_logits.device:
            raise ValueError("coarse and propagated masks must be on the same device")
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))


class VILAVideoSegmentationPipeline:
    """Compose frozen VILA anchor prediction with official SAM2 propagation."""

    def __init__(
        self,
        adapter: VILASegmentationAdapter,
        propagator: SAM2VideoMaskPropagator,
    ) -> None:
        if not isinstance(adapter, VILASegmentationAdapter):
            raise TypeError("adapter must be a VILASegmentationAdapter")
        if not isinstance(propagator, SAM2VideoMaskPropagator):
            raise TypeError("propagator must be a SAM2VideoMaskPropagator")
        if adapter.dense_provider is not propagator.provider:
            raise ValueError("the VILA adapter dense provider and SAM2 propagator must share one provider")
        self.adapter = adapter
        self.propagator = propagator

    def segment(
        self,
        input_ids: Tensor,
        media: Optional[Mapping[str, Any]],
        media_config: Optional[Mapping[str, Any]],
        query_token_mask: Tensor,
        rgb_frames: RGBFrameBatch,
        request: Any,
        attention_mask: Optional[Tensor] = None,
        hidden_layer: int = -1,
        anchor_indices: Optional[Tensor] = None,
    ) -> VideoSegmentationOutput:
        """Predict one fixed anchor mask and propagate it over the raw video."""

        normalized = SegmentationRequest.from_value(request)
        if not normalized.enabled:
            raise ValueError("video segmentation pipeline is disabled for this request")
        if normalized.task != "video":
            raise ValueError("video segmentation pipeline requires task='video'")
        if not isinstance(rgb_frames, RGBFrameBatch):
            raise TypeError("rgb_frames must be an RGBFrameBatch")
        if rgb_frames.frames.shape[1] <= 1 or (rgb_frames.frame_mask.sum(dim=1) <= 1).any():
            raise ValueError("video segmentation pipeline requires at least two valid frames per sample")
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,L]")
        if input_ids.shape[0] != rgb_frames.frames.shape[0]:
            raise ValueError("VILA input batch must match raw RGB batch")
        anchors = _anchor_indices(rgb_frames, anchor_indices)
        anchor_batch = _select_anchor_batch(rgb_frames, anchors)

        _synchronize(input_ids)
        pipeline_started = time.perf_counter()
        coarse = self.adapter.segment(
            input_ids=input_ids,
            media=media,
            media_config=media_config,
            query_token_mask=query_token_mask,
            request=normalized,
            attention_mask=attention_mask,
            hidden_layer=hidden_layer,
            dense_input=anchor_batch,
        )
        _synchronize(coarse.mask_logits)
        propagation = self.propagator.propagate(
            rgb_frames,
            anchors,
            coarse.mask_logits[:, :, 0],
        )
        _synchronize(propagation.mask_logits)
        total_ms = (time.perf_counter() - pipeline_started) * 1000.0

        component_timing = dict(coarse.diagnostics.get("component_timing_ms", {}))
        component_timing.update(propagation.diagnostics.get("component_timing_ms", {}))
        component_timing["video_pipeline_total"] = total_ms
        diagnostics = {
            "execution_path": "vila_sam2_predicted_anchor_video_pipeline",
            "task": "video",
            "anchor_policy": "first_valid" if anchor_indices is None else "explicit",
            "anchor_indices": anchors.tolist(),
            "coarse_execution_path": coarse.diagnostics.get("execution_path"),
            "component_timing_ms": component_timing,
            "output_shapes": {
                "coarse_anchor_mask_logits": tuple(coarse.mask_logits.shape),
                "propagated_mask_logits": tuple(propagation.mask_logits.shape),
            },
        }
        return VideoSegmentationOutput(
            coarse_anchor_result=coarse,
            propagation_result=propagation,
            diagnostics=diagnostics,
        )

    def clear_request_state(self) -> None:
        """Clear extension diagnostics and predictor-side request state."""

        self.adapter.capability.clear_request_state()
        self.propagator.clear_request_state()
