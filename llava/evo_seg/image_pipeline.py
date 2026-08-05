"""Explicit VILA-to-SAM2 image segmentation composition."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional

import torch
from torch import Tensor

from .contracts import SegmentationRequest, SegmentationResult, freeze_mapping
from .sam2_adapter import RGBFrameBatch, SAM2ImageFeatureProvider
from .vila_adapter import VILASegmentationAdapter


def _synchronize(tensor: Tensor) -> None:
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


@dataclass(frozen=True)
class ImageSegmentationOutput:
    """Coarse decoder result plus SAM2-refined image mask logits."""

    coarse_result: SegmentationResult
    refined_mask_logits: Tensor
    diagnostics: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.coarse_result, SegmentationResult):
            raise TypeError("coarse_result must be a SegmentationResult")
        refined = self.refined_mask_logits
        if not isinstance(refined, Tensor) or refined.ndim != 5:
            raise ValueError("refined_mask_logits must have shape [B,N,1,H,W]")
        if not refined.is_floating_point() or not torch.isfinite(refined).all():
            raise ValueError("refined_mask_logits must be finite and floating point")
        expected_prefix = tuple(self.coarse_result.mask_logits.shape[:3])
        if tuple(refined.shape[:3]) != expected_prefix or refined.shape[2] != 1:
            raise ValueError("refined masks must align with the coarse image batch, objects, and T=1")
        if min(refined.shape[-2:]) <= 0:
            raise ValueError("refined masks must have positive spatial dimensions")
        if refined.device != self.coarse_result.mask_logits.device:
            raise ValueError("coarse and refined masks must be on the same device")
        invalid = ~self.coarse_result.frame_mask[:, None, :, None, None]
        if invalid.any() and refined.masked_select(invalid).abs().max().item() != 0:
            raise ValueError("refined masks must be zero on invalid frames")
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))


class VILAImageSegmentationPipeline:
    """Compose the explicit VILA adapter and its SAM2 image provider."""

    def __init__(
        self,
        adapter: VILASegmentationAdapter,
        refiner: SAM2ImageFeatureProvider,
    ) -> None:
        if not isinstance(adapter, VILASegmentationAdapter):
            raise TypeError("adapter must be a VILASegmentationAdapter")
        if not isinstance(refiner, SAM2ImageFeatureProvider):
            raise TypeError("refiner must be a SAM2ImageFeatureProvider")
        if adapter.dense_provider is not refiner:
            raise ValueError("the VILA adapter dense provider and SAM2 refiner must be the same object")
        self.adapter = adapter
        self.refiner = refiner

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
    ) -> ImageSegmentationOutput:
        """Run the complete opt-in VILA, spatial decoder, and SAM2 image path."""

        normalized = SegmentationRequest.from_value(request)
        if not normalized.enabled:
            raise ValueError("image segmentation pipeline is disabled for this request")
        if normalized.task != "image":
            raise ValueError("image segmentation pipeline requires task='image'")
        if not isinstance(rgb_frames, RGBFrameBatch):
            raise TypeError("rgb_frames must be an RGBFrameBatch")
        if rgb_frames.frames.shape[1] != 1:
            raise ValueError("image segmentation pipeline requires T=1 raw RGB input")
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,L]")
        if input_ids.shape[0] != rgb_frames.frames.shape[0]:
            raise ValueError("VILA input batch must match raw RGB batch")

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
            dense_input=rgb_frames,
        )
        _synchronize(coarse.mask_logits)
        refinement_started = time.perf_counter()
        refined = self.refiner.refine_masks(rgb_frames, coarse.mask_logits)
        _synchronize(refined)
        refinement_ms = (time.perf_counter() - refinement_started) * 1000.0
        total_ms = (time.perf_counter() - pipeline_started) * 1000.0

        component_timing = dict(coarse.diagnostics.get("component_timing_ms", {}))
        component_timing.update(
            {
                "sam2_mask_refinement_with_reencode": refinement_ms,
                "image_pipeline_total": total_ms,
            }
        )
        diagnostics = {
            "execution_path": "vila_sam2_image_pipeline",
            "task": "image",
            "coarse_execution_path": coarse.diagnostics.get("execution_path"),
            "component_timing_ms": component_timing,
            "output_shapes": {
                "coarse_mask_logits": tuple(coarse.mask_logits.shape),
                "refined_mask_logits": tuple(refined.shape),
            },
        }
        return ImageSegmentationOutput(
            coarse_result=coarse,
            refined_mask_logits=refined,
            diagnostics=diagnostics,
        )

    def clear_request_state(self) -> None:
        """Clear extension diagnostics and predictor-side request images."""

        self.adapter.capability.clear_request_state()
        self.refiner.clear_request_state()
