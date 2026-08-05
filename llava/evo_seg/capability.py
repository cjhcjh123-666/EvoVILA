"""Request-scoped ownership for the optional segmentation decoder."""

from __future__ import annotations

import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Optional

import torch
from torch import Tensor, nn

from .contracts import GroundingBatch, SegmentationRequest, SegmentationResult, freeze_mapping


def _synchronize(tensor: Tensor) -> None:
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


class SegmentationCapability(nn.Module):
    """Expose dense prediction without replacing VILA's ordinary forward path."""

    def __init__(self, decoder: nn.Module, providers: Optional[Mapping[str, nn.Module]] = None) -> None:
        super().__init__()
        if not isinstance(decoder, nn.Module):
            raise TypeError("decoder must be a torch.nn.Module")
        self.decoder = decoder
        provider_modules = dict(providers or {})
        if not all(isinstance(provider, nn.Module) for provider in provider_modules.values()):
            raise TypeError("providers must contain only torch.nn.Module values")
        self.providers = nn.ModuleDict(provider_modules)
        self._last_diagnostics: Mapping[str, Any] = MappingProxyType({})

    @staticmethod
    def is_enabled(request: Any) -> bool:
        """Parse a request without constructing a decoder or optional provider."""

        normalized = SegmentationRequest.from_value(request)
        return normalized.enabled

    @property
    def last_diagnostics(self) -> Mapping[str, Any]:
        return self._last_diagnostics

    def clear_request_state(self) -> None:
        """Clear diagnostics only; no media or training tensor is retained."""

        self._last_diagnostics = MappingProxyType({})

    def forward(self, batch: GroundingBatch, request: Any) -> SegmentationResult:
        """Run the decoder only for an explicitly enabled request."""

        normalized = SegmentationRequest.from_value(request)
        if not normalized.enabled:
            raise ValueError("segmentation capability is disabled for this request")
        if normalized.task not in ("image", "video"):
            raise ValueError("enabled segmentation request requires image or video task")
        if not isinstance(batch, GroundingBatch):
            raise TypeError("batch must be a GroundingBatch")

        _synchronize(batch.dense_features)
        started = time.perf_counter()
        result = self.decoder(batch)
        _synchronize(result.mask_logits)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        diagnostics = dict(result.diagnostics)
        component_timing = dict(diagnostics.get("component_timing_ms", {}))
        component_timing["decoder"] = elapsed_ms
        component_timing["total"] = elapsed_ms
        diagnostics.update(
            {
                "execution_path": "evo_seg_decoder",
                "task": normalized.task,
                "component_timing_ms": component_timing,
            }
        )
        self._last_diagnostics = freeze_mapping(diagnostics)
        return SegmentationResult(
            mask_logits=result.mask_logits,
            object_logits=result.object_logits,
            object_embeddings=result.object_embeddings,
            frame_embeddings=result.frame_embeddings,
            frame_mask=result.frame_mask,
            diagnostics=diagnostics,
        )
