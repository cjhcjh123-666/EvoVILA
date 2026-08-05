"""Lazy SAM2 image feature and mask-refinement adapter.

This module contains no module-level SAM2 import.  The official package and a
local checkpoint are resolved only when an explicitly enabled provider call
first needs a predictor.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .contracts import SegmentationRequest, freeze_mapping
from .vila_adapter import DenseFeatureBatch


@dataclass(frozen=True)
class RGBFrameBatch:
    """Raw RGB pixels for SAM2, represented as ``[B,T,3,H,W]`` CPU uint8."""

    frames: Tensor
    frame_mask: Tensor
    sample_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.frames, Tensor) or self.frames.ndim != 5:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        if self.frames.dtype != torch.uint8:
            raise TypeError("frames must have dtype torch.uint8 with values in [0,255]")
        if self.frames.device.type != "cpu":
            raise ValueError("raw RGB frames must remain on CPU until SAM2 preprocessing")
        batch_size, frames, channels, height, width = self.frames.shape
        if channels != 3 or min(batch_size, frames, height, width) <= 0:
            raise ValueError("frames must have positive [B,T,3,H,W] dimensions")
        if not isinstance(self.frame_mask, Tensor) or self.frame_mask.dtype != torch.bool:
            raise TypeError("frame_mask must be a torch.bool tensor")
        if tuple(self.frame_mask.shape) != (batch_size, frames):
            raise ValueError("frame_mask must align with frames batch and time dimensions")
        if self.frame_mask.device.type != "cpu":
            raise ValueError("frame_mask must be on CPU with raw RGB frames")
        if not self.frame_mask.any(dim=1).all():
            raise ValueError("every sample must contain at least one valid frame")
        if not isinstance(self.sample_ids, tuple) or not all(isinstance(item, str) for item in self.sample_ids):
            raise TypeError("sample_ids must be a tuple of strings")
        if self.sample_ids and len(self.sample_ids) != batch_size:
            raise ValueError("sample_ids length must match batch size")


@dataclass(frozen=True)
class SAM2BuildOptions:
    """Local-only SAM2 construction options; no remote model IDs are accepted."""

    model_config: str
    checkpoint_path: str
    source_root: Optional[str] = None
    device: str = "cuda"
    apply_postprocessing: bool = True

    def __post_init__(self) -> None:
        for name in ("model_config", "checkpoint_path", "device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if Path(self.model_config).is_absolute():
            raise ValueError("model_config must be a SAM2 package-relative config name")
        if self.source_root is not None:
            if not isinstance(self.source_root, str):
                raise TypeError("source_root must be a string or None")
            if not self.source_root.strip():
                raise ValueError("source_root must be non-empty when provided")
        if not isinstance(self.apply_postprocessing, bool):
            raise TypeError("apply_postprocessing must be a bool")

    @classmethod
    def from_value(cls, value: Any) -> "SAM2BuildOptions":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("SAM2 options must be a SAM2BuildOptions or mapping")
        allowed = {"model_config", "checkpoint_path", "source_root", "device", "apply_postprocessing"}
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(f"unknown SAM2 option fields: {sorted(unknown)}")
        return cls(**dict(value))


def _resolve_sam2_package(options: SAM2BuildOptions) -> Path:
    if options.source_root is not None:
        source_root = Path(options.source_root).expanduser().resolve()
        if not source_root.is_dir() or not (source_root / "sam2" / "__init__.py").is_file():
            raise FileNotFoundError(f"SAM2 source_root does not contain a sam2 package: {source_root}")
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
            importlib.invalidate_caches()

    spec = importlib.util.find_spec("sam2")
    if spec is None or spec.origin is None:
        raise ImportError(
            "SAM2 is not installed; provide source_root pointing to a local official SAM2 checkout"
        )
    package_root = Path(spec.origin).resolve().parent
    if options.source_root is not None:
        expected_root = (Path(options.source_root).expanduser().resolve() / "sam2")
        if package_root != expected_root:
            raise ImportError(f"loaded SAM2 package {package_root} does not match source_root {expected_root}")
    config_path = (package_root / options.model_config).resolve()
    if not config_path.is_relative_to(package_root):
        raise ValueError("SAM2 model_config cannot escape the package directory")
    if not config_path.is_file():
        raise FileNotFoundError(f"SAM2 model config does not exist inside the package: {config_path}")
    return package_root


def _freeze_predictor_model(predictor: Any) -> None:
    model = getattr(predictor, "model", None)
    if not isinstance(model, nn.Module):
        raise TypeError("SAM2 predictor must expose its torch model as predictor.model")
    model.eval()
    model.requires_grad_(False)


def build_sam2_image_predictor(options: Any) -> Any:
    """Build the official SAM2 image predictor from local source and weights."""

    normalized = SAM2BuildOptions.from_value(options)
    _resolve_sam2_package(normalized)
    checkpoint_path = Path(normalized.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint_path}")

    build_module = importlib.import_module("sam2.build_sam")
    predictor_module = importlib.import_module("sam2.sam2_image_predictor")
    build_sam2 = getattr(build_module, "build_sam2", None)
    predictor_type = getattr(predictor_module, "SAM2ImagePredictor", None)
    if not callable(build_sam2) or not callable(predictor_type):
        raise ImportError("installed SAM2 package does not expose the official image predictor API")
    model = build_sam2(
        config_file=normalized.model_config,
        ckpt_path=str(checkpoint_path),
        device=normalized.device,
        mode="eval",
        apply_postprocessing=normalized.apply_postprocessing,
    )
    predictor = predictor_type(model)
    _freeze_predictor_model(predictor)
    return predictor


class SAM2ImageFeatureProvider:
    """Lazily encode raw frames and optionally refine decoder anchor masks."""

    def __init__(
        self,
        options: Any,
        predictor_factory: Optional[Callable[[SAM2BuildOptions], Any]] = None,
    ) -> None:
        self.options = SAM2BuildOptions.from_value(options)
        if predictor_factory is not None and not callable(predictor_factory):
            raise TypeError("predictor_factory must be callable or None")
        self._predictor_factory = predictor_factory
        self._predictor: Optional[Any] = None
        self._lock = threading.RLock()

    @property
    def initialized(self) -> bool:
        return self._predictor is not None

    def initialize(self) -> None:
        """Explicitly build frozen SAM2 weights without setting request images."""

        with self._lock, torch.no_grad():
            self._get_predictor_locked()

    def _get_predictor_locked(self) -> Any:
        if self._predictor is None:
            factory = self._predictor_factory or build_sam2_image_predictor
            predictor = factory(self.options)
            for method in ("set_image_batch", "predict_batch"):
                if not callable(getattr(predictor, method, None)):
                    raise TypeError(f"SAM2 predictor must implement {method}")
            _freeze_predictor_model(predictor)
            self._predictor = predictor
        return self._predictor

    @staticmethod
    def _valid_images(batch: RGBFrameBatch) -> tuple[list[Any], list[tuple[int, int]]]:
        images, indices = [], []
        for batch_index, frame_index in batch.frame_mask.nonzero(as_tuple=False).tolist():
            image = batch.frames[batch_index, frame_index].permute(1, 2, 0).contiguous().numpy()
            images.append(image)
            indices.append((batch_index, frame_index))
        return images, indices

    @staticmethod
    def _extract_image_embedding(predictor: Any) -> Tensor:
        getter = getattr(predictor, "get_image_embedding", None)
        if callable(getter):
            embedding = getter()
        else:
            features = getattr(predictor, "_features", None)
            embedding = features.get("image_embed") if isinstance(features, Mapping) else None
        if not isinstance(embedding, Tensor) or embedding.ndim != 4:
            raise RuntimeError("SAM2 predictor did not expose image_embed as [N,C,H,W]")
        if not embedding.is_floating_point() or not torch.isfinite(embedding).all():
            raise RuntimeError("SAM2 image embeddings must be finite floating tensors")
        return embedding.detach().clone()

    @staticmethod
    def _reset_predictor(predictor: Any) -> None:
        reset = getattr(predictor, "reset_predictor", None)
        if callable(reset):
            reset()

    def encode_frames(self, batch: RGBFrameBatch) -> DenseFeatureBatch:
        """Run SAM2's own RGB preprocessing and return padded dense features."""

        if not isinstance(batch, RGBFrameBatch):
            raise TypeError("batch must be an RGBFrameBatch")
        images, indices = self._valid_images(batch)
        started = time.perf_counter()
        with self._lock, torch.no_grad():
            predictor = self._get_predictor_locked()
            try:
                predictor.set_image_batch(images)
                valid_features = self._extract_image_embedding(predictor)
            finally:
                self._reset_predictor(predictor)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if valid_features.shape[0] != len(indices):
            raise RuntimeError("SAM2 image embedding batch does not match the number of valid frames")

        batch_size, frames = batch.frame_mask.shape
        features = valid_features.new_zeros((batch_size, frames, *valid_features.shape[1:]))
        for feature_index, (batch_index, frame_index) in enumerate(indices):
            features[batch_index, frame_index] = valid_features[feature_index]
        frame_mask = batch.frame_mask.to(device=features.device)
        return DenseFeatureBatch(
            features=features,
            frame_mask=frame_mask,
            diagnostics={
                "provider": "sam2_image_encoder",
                "preprocessing": "SAM2ImagePredictor.set_image_batch",
                "raw_frame_shape": tuple(batch.frames.shape),
                "valid_frames": len(indices),
                "feature_shape": tuple(features.shape),
                "component_timing_ms": {"sam2_image_encoder": elapsed_ms},
            },
        )

    def encode(self, media: Any, media_config: Any, request: Any) -> DenseFeatureBatch:
        """Dense-provider protocol used by :class:`VILASegmentationAdapter`."""

        del media_config
        normalized = SegmentationRequest.from_value(request)
        if not normalized.enabled:
            raise ValueError("SAM2 image provider is disabled for this request")
        if not isinstance(media, RGBFrameBatch):
            raise TypeError("SAM2 dense input must be an RGBFrameBatch")
        if normalized.task == "image" and media.frames.shape[1] != 1:
            raise ValueError("image segmentation requests require T=1 raw RGB input")
        return self.encode_frames(media)

    def refine_masks(self, batch: RGBFrameBatch, coarse_mask_logits: Tensor) -> Tensor:
        """Use decoder mask logits as SAM2 mask prompts and return RGB-size logits."""

        if not isinstance(batch, RGBFrameBatch):
            raise TypeError("batch must be an RGBFrameBatch")
        if not isinstance(coarse_mask_logits, Tensor) or coarse_mask_logits.ndim != 5:
            raise ValueError("coarse_mask_logits must have shape [B,N,T,H,W]")
        if not coarse_mask_logits.is_floating_point() or not torch.isfinite(coarse_mask_logits).all():
            raise ValueError("coarse_mask_logits must be finite and floating point")
        batch_size, objects, frames, _, _ = coarse_mask_logits.shape
        if (batch_size, frames) != tuple(batch.frame_mask.shape):
            raise ValueError("coarse masks must align with RGB batch and time dimensions")
        images, indices = self._valid_images(batch)
        frame_logits = coarse_mask_logits.permute(0, 2, 1, 3, 4)

        with self._lock, torch.no_grad():
            predictor = self._get_predictor_locked()
            prompt_encoder = getattr(getattr(predictor, "model", None), "sam_prompt_encoder", None)
            mask_input_size = getattr(prompt_encoder, "mask_input_size", (256, 256))
            if not isinstance(mask_input_size, (tuple, list)) or len(mask_input_size) != 2:
                raise RuntimeError("SAM2 mask_input_size must contain height and width")
            valid_logits = torch.stack([frame_logits[b, t] for b, t in indices], dim=0)
            resized_logits = F.interpolate(
                valid_logits,
                size=(int(mask_input_size[0]), int(mask_input_size[1])),
                mode="bilinear",
                align_corners=False,
            )
            mask_inputs = [item.detach().to(dtype=torch.float32, device="cpu").numpy() for item in resized_logits]
            try:
                predictor.set_image_batch(images)
                prediction = predictor.predict_batch(
                    mask_input_batch=mask_inputs,
                    multimask_output=False,
                    return_logits=True,
                )
            finally:
                self._reset_predictor(predictor)

        if not isinstance(prediction, (tuple, list)) or not prediction:
            raise RuntimeError("SAM2 predict_batch did not return mask logits")
        masks = prediction[0]
        if not isinstance(masks, (tuple, list)) or len(masks) != len(indices):
            raise RuntimeError("SAM2 refined mask batch does not match valid frames")
        output = coarse_mask_logits.new_zeros(
            (batch_size, objects, frames, batch.frames.shape[-2], batch.frames.shape[-1])
        )
        for refined, (batch_index, frame_index) in zip(masks, indices):
            tensor = torch.as_tensor(refined)
            if tensor.ndim == 2 and objects == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim == 4 and tensor.shape[1] == 1:
                tensor = tensor.squeeze(1)
            if tuple(tensor.shape) != (objects, batch.frames.shape[-2], batch.frames.shape[-1]):
                raise RuntimeError(
                    "SAM2 refined masks must have shape [N,H,W], got " f"{tuple(tensor.shape)}"
                )
            output[batch_index, :, frame_index] = tensor.to(
                device=output.device,
                dtype=output.dtype,
            )
        return output

    def clear_request_state(self) -> None:
        """Drop any predictor-side image state without unloading frozen weights."""

        with self._lock:
            if self._predictor is not None:
                self._reset_predictor(self._predictor)

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        return freeze_mapping({"initialized": self.initialized, "device": self.options.device})
