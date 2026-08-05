"""Lazy shared-weight SAM2 video propagation adapter."""

from __future__ import annotations

import importlib
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn

from .contracts import freeze_mapping
from .sam2_adapter import (
    RGBFrameBatch,
    SAM2BuildOptions,
    SAM2ImageFeatureProvider,
    _freeze_predictor_model,
    _resolve_sam2_package,
    _synchronize_device,
)


@dataclass(frozen=True)
class SAM2VideoPropagationOptions:
    """Request-independent controls for official SAM2 video inference."""

    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = False
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        if not isinstance(self.offload_video_to_cpu, bool):
            raise TypeError("offload_video_to_cpu must be a bool")
        if not isinstance(self.offload_state_to_cpu, bool):
            raise TypeError("offload_state_to_cpu must be a bool")
        if not isinstance(self.jpeg_quality, int) or isinstance(self.jpeg_quality, bool):
            raise TypeError("jpeg_quality must be an integer")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1,100]")

    @classmethod
    def from_value(cls, value: Any) -> "SAM2VideoPropagationOptions":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("SAM2 video options must be an options object or mapping")
        allowed = {"offload_video_to_cpu", "offload_state_to_cpu", "jpeg_quality"}
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(f"unknown SAM2 video option fields: {sorted(unknown)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class VideoPropagationResult:
    """Validated padded SAM2 video logits reconstructed in source-frame order."""

    mask_logits: Tensor
    frame_mask: Tensor
    anchor_indices: Tensor
    diagnostics: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        logits = self.mask_logits
        if not isinstance(logits, Tensor) or logits.ndim != 5:
            raise ValueError("mask_logits must have shape [B,N,T,H,W]")
        if not logits.is_floating_point() or not torch.isfinite(logits).all():
            raise ValueError("mask_logits must be finite and floating point")
        batch_size, objects, frames, height, width = logits.shape
        if min(batch_size, objects, frames, height, width) <= 0:
            raise ValueError("mask_logits dimensions must be positive")
        frame_mask = self.frame_mask
        if not isinstance(frame_mask, Tensor) or frame_mask.dtype != torch.bool:
            raise TypeError("frame_mask must be a torch.bool tensor")
        if tuple(frame_mask.shape) != (batch_size, frames):
            raise ValueError("frame_mask must align with mask_logits")
        if frame_mask.device != logits.device:
            raise ValueError("frame_mask and mask_logits must be on the same device")
        anchors = self.anchor_indices
        if not isinstance(anchors, Tensor) or anchors.dtype != torch.long:
            raise TypeError("anchor_indices must be a torch.long tensor")
        if tuple(anchors.shape) != (batch_size,):
            raise ValueError("anchor_indices must have shape [B]")
        if anchors.device != logits.device:
            raise ValueError("anchor_indices and mask_logits must be on the same device")
        if ((anchors < 0) | (anchors >= frames)).any():
            raise ValueError("anchor_indices must lie inside the video frame range")
        if not frame_mask.gather(1, anchors[:, None]).all():
            raise ValueError("every anchor index must select a valid frame")
        invalid = ~frame_mask[:, None, :, None, None]
        if invalid.any() and logits.masked_select(invalid).abs().max().item() != 0:
            raise ValueError("mask_logits must be zero on invalid frames")
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))


def build_sam2_video_image_predictor(options: Any) -> Any:
    """Build one frozen video-capable SAM2 model behind the image predictor API."""

    normalized = SAM2BuildOptions.from_value(options)
    _resolve_sam2_package(normalized)
    checkpoint_path = Path(normalized.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint_path}")

    build_module = importlib.import_module("sam2.build_sam")
    predictor_module = importlib.import_module("sam2.sam2_image_predictor")
    build_video = getattr(build_module, "build_sam2_video_predictor", None)
    predictor_type = getattr(predictor_module, "SAM2ImagePredictor", None)
    if not callable(build_video) or not callable(predictor_type):
        raise ImportError("installed SAM2 package does not expose the official video predictor API")
    model = build_video(
        config_file=normalized.model_config,
        ckpt_path=str(checkpoint_path),
        device=normalized.device,
        mode="eval",
        apply_postprocessing=normalized.apply_postprocessing,
    )
    predictor = predictor_type(model)
    _freeze_predictor_model(predictor)
    return predictor


class SAM2VideoMaskPropagator:
    """Propagate predicted anchor masks with a shared frozen SAM2 video model."""

    def __init__(self, provider: SAM2ImageFeatureProvider, options: Any = None) -> None:
        if not isinstance(provider, SAM2ImageFeatureProvider):
            raise TypeError("provider must be a SAM2ImageFeatureProvider")
        self.provider = provider
        self.options = SAM2VideoPropagationOptions.from_value(options)

    def _video_model(self) -> nn.Module:
        with self.provider._lock, torch.no_grad():
            predictor = self.provider._get_predictor_locked()
            model = getattr(predictor, "model", None)
            if not isinstance(model, nn.Module):
                raise TypeError("SAM2 image predictor must expose a torch model")
            for method in ("init_state", "add_new_mask", "propagate_in_video", "reset_state"):
                if not callable(getattr(model, method, None)):
                    raise TypeError(f"shared SAM2 model must implement {method}")
            model.eval()
            model.requires_grad_(False)
            return model

    def initialize(self) -> None:
        """Explicitly initialize and validate the shared video-capable model."""

        self._video_model()

    @staticmethod
    def _normalize_anchor_indices(batch: RGBFrameBatch, anchor_indices: Tensor) -> Tensor:
        if not isinstance(anchor_indices, Tensor) or anchor_indices.dtype != torch.long:
            raise TypeError("anchor_indices must be a torch.long tensor")
        if tuple(anchor_indices.shape) != (batch.frames.shape[0],):
            raise ValueError("anchor_indices must have shape [B]")
        anchors = anchor_indices.detach().to(device="cpu")
        frames = batch.frames.shape[1]
        if ((anchors < 0) | (anchors >= frames)).any():
            raise ValueError("anchor_indices must lie inside the video frame range")
        if not batch.frame_mask.gather(1, anchors[:, None]).all():
            raise ValueError("every anchor index must select a valid frame")
        return anchors

    @staticmethod
    def _store_prediction(
        output: Tensor,
        batch_index: int,
        source_frame_index: int,
        object_ids: Any,
        mask_logits: Any,
    ) -> None:
        expected_ids = tuple(range(1, output.shape[1] + 1))
        try:
            received_ids = tuple(int(value) for value in object_ids)
        except TypeError as error:
            raise RuntimeError("SAM2 video predictor returned invalid object ids") from error
        if len(received_ids) != len(set(received_ids)) or set(received_ids) != set(expected_ids):
            raise RuntimeError(
                f"SAM2 video predictor returned object ids {received_ids}, expected {expected_ids}"
            )
        masks = torch.as_tensor(mask_logits)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        expected_shape = (len(received_ids), output.shape[-2], output.shape[-1])
        if tuple(masks.shape) != expected_shape:
            raise RuntimeError(
                f"SAM2 propagated masks must have shape {expected_shape}, got {tuple(masks.shape)}"
            )
        source_by_id = {object_id: index for index, object_id in enumerate(received_ids)}
        ordered = torch.stack([masks[source_by_id[object_id]] for object_id in expected_ids])
        output[batch_index, :, source_frame_index] = ordered.to(
            device=output.device,
            dtype=output.dtype,
        )

    def propagate(
        self,
        batch: RGBFrameBatch,
        anchor_indices: Tensor,
        anchor_mask_logits: Tensor,
    ) -> VideoPropagationResult:
        """Threshold predicted anchors and propagate them over each valid video."""

        if not isinstance(batch, RGBFrameBatch):
            raise TypeError("batch must be an RGBFrameBatch")
        if batch.frames.shape[1] <= 1 or (batch.frame_mask.sum(dim=1) <= 1).any():
            raise ValueError("video propagation requires at least two valid frames per sample")
        anchors = self._normalize_anchor_indices(batch, anchor_indices)
        if not isinstance(anchor_mask_logits, Tensor) or anchor_mask_logits.ndim != 4:
            raise ValueError("anchor_mask_logits must have shape [B,N,H,W]")
        if not anchor_mask_logits.is_floating_point() or not torch.isfinite(anchor_mask_logits).all():
            raise ValueError("anchor_mask_logits must be finite and floating point")
        batch_size, objects, mask_height, mask_width = anchor_mask_logits.shape
        if batch_size != batch.frames.shape[0] or min(objects, mask_height, mask_width) <= 0:
            raise ValueError("anchor masks must have positive dimensions and match the RGB batch")

        model = self._video_model()
        output = anchor_mask_logits.new_zeros(
            (batch_size, objects, batch.frames.shape[1], batch.frames.shape[-2], batch.frames.shape[-1])
        )
        prompt_masks = (anchor_mask_logits > 0).detach().to(device="cpu")
        timing = {
            "sam2_video_frame_io": 0.0,
            "sam2_video_state_initialization": 0.0,
            "sam2_video_anchor_prompt": 0.0,
            "sam2_video_forward_propagation": 0.0,
            "sam2_video_reverse_propagation": 0.0,
        }
        compact_anchor_indices = []
        _synchronize_device(self.provider.options.device)
        total_started = time.perf_counter()
        with self.provider._lock, torch.no_grad():
            for batch_index in range(batch_size):
                valid_indices = batch.frame_mask[batch_index].nonzero(as_tuple=False).flatten().tolist()
                anchor_source = int(anchors[batch_index].item())
                compact_anchor = valid_indices.index(anchor_source)
                compact_anchor_indices.append(compact_anchor)
                state = None
                with tempfile.TemporaryDirectory(prefix="evovila-sam2-video-") as directory:
                    io_started = time.perf_counter()
                    for compact_index, source_index in enumerate(valid_indices):
                        array = (
                            batch.frames[batch_index, source_index]
                            .permute(1, 2, 0)
                            .contiguous()
                            .numpy()
                        )
                        Image.fromarray(array, mode="RGB").save(
                            Path(directory) / f"{compact_index:05d}.jpg",
                            format="JPEG",
                            quality=self.options.jpeg_quality,
                            subsampling=0,
                        )
                    timing["sam2_video_frame_io"] += (time.perf_counter() - io_started) * 1000.0

                    _synchronize_device(self.provider.options.device)
                    initialized_at = time.perf_counter()
                    state = model.init_state(
                        video_path=directory,
                        offload_video_to_cpu=self.options.offload_video_to_cpu,
                        offload_state_to_cpu=self.options.offload_state_to_cpu,
                        async_loading_frames=False,
                    )
                    _synchronize_device(self.provider.options.device)
                    timing["sam2_video_state_initialization"] += (
                        time.perf_counter() - initialized_at
                    ) * 1000.0
                    if not isinstance(state, dict):
                        raise RuntimeError("SAM2 init_state must return a mutable state dictionary")
                    try:
                        _synchronize_device(self.provider.options.device)
                        prompted_at = time.perf_counter()
                        for object_index in range(objects):
                            model.add_new_mask(
                                inference_state=state,
                                frame_idx=compact_anchor,
                                obj_id=object_index + 1,
                                mask=prompt_masks[batch_index, object_index],
                            )
                        _synchronize_device(self.provider.options.device)
                        timing["sam2_video_anchor_prompt"] += (
                            time.perf_counter() - prompted_at
                        ) * 1000.0

                        seen_frames = set()
                        _synchronize_device(self.provider.options.device)
                        forward_at = time.perf_counter()
                        for compact_index, object_ids, masks in model.propagate_in_video(
                            inference_state=state,
                            start_frame_idx=compact_anchor,
                            reverse=False,
                        ):
                            compact_index = int(compact_index)
                            if compact_index < 0 or compact_index >= len(valid_indices):
                                raise RuntimeError("SAM2 returned a frame outside the valid video range")
                            self._store_prediction(
                                output,
                                batch_index,
                                valid_indices[compact_index],
                                object_ids,
                                masks,
                            )
                            seen_frames.add(compact_index)
                        _synchronize_device(self.provider.options.device)
                        timing["sam2_video_forward_propagation"] += (
                            time.perf_counter() - forward_at
                        ) * 1000.0

                        if compact_anchor > 0:
                            _synchronize_device(self.provider.options.device)
                            reverse_at = time.perf_counter()
                            for compact_index, object_ids, masks in model.propagate_in_video(
                                inference_state=state,
                                start_frame_idx=compact_anchor,
                                reverse=True,
                            ):
                                compact_index = int(compact_index)
                                if compact_index < 0 or compact_index >= len(valid_indices):
                                    raise RuntimeError("SAM2 returned a frame outside the valid video range")
                                self._store_prediction(
                                    output,
                                    batch_index,
                                    valid_indices[compact_index],
                                    object_ids,
                                    masks,
                                )
                                seen_frames.add(compact_index)
                            _synchronize_device(self.provider.options.device)
                            timing["sam2_video_reverse_propagation"] += (
                                time.perf_counter() - reverse_at
                            ) * 1000.0
                        if seen_frames != set(range(len(valid_indices))):
                            raise RuntimeError(
                                "SAM2 propagation did not return every valid frame: "
                                f"received {sorted(seen_frames)}, expected {list(range(len(valid_indices)))}"
                            )
                    finally:
                        if state is not None:
                            try:
                                model.reset_state(state)
                            finally:
                                state.clear()

        _synchronize_device(self.provider.options.device)
        timing["sam2_video_total"] = (time.perf_counter() - total_started) * 1000.0
        frame_mask = batch.frame_mask.to(device=output.device)
        anchors_on_output = anchors.to(device=output.device)
        return VideoPropagationResult(
            mask_logits=output,
            frame_mask=frame_mask,
            anchor_indices=anchors_on_output,
            diagnostics={
                "provider": "sam2_video_predictor",
                "prompt_source": "predicted_anchor_mask",
                "prompt_threshold": 0.0,
                "temporary_frame_bridge": "request_local_jpeg_directory",
                "valid_frames_per_sample": batch.frame_mask.sum(dim=1).tolist(),
                "compact_anchor_indices": compact_anchor_indices,
                "state_cleared": True,
                "component_timing_ms": timing,
            },
        )

    def clear_request_state(self) -> None:
        """Clear image-predictor request state without unloading shared weights."""

        self.provider.clear_request_state()
