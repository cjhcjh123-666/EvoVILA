"""Request-scoped bridge from repository-native VILA to EvoVILA-Seg.

The adapter deliberately keeps dense feature extraction injectable.  S1 can
therefore exercise the complete language/provenance/decoder contract without
importing SAM2 or downloading a vision model.  A production provider can be
added at a later milestone without changing the VILA call site.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional

import torch
from torch import Tensor, nn

from .capability import SegmentationCapability
from .contracts import GroundingBatch, SegmentationRequest, SegmentationResult, freeze_mapping
from .provenance import FusionCapture, QueryStateBatch


def _synchronize(tensor: Tensor) -> None:
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


@dataclass(frozen=True)
class DenseFeatureBatch:
    """Validated dense provider output shared by image and video requests."""

    features: Tensor
    frame_mask: Tensor
    diagnostics: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.features, Tensor) or self.features.ndim != 5:
            raise ValueError("features must have shape [B,T,C,H,W]")
        if not self.features.is_floating_point():
            raise TypeError("features must be floating point")
        if not torch.isfinite(self.features).all():
            raise ValueError("features must contain only finite values")
        if not isinstance(self.frame_mask, Tensor) or self.frame_mask.ndim != 2:
            raise ValueError("frame_mask must have shape [B,T]")
        if self.frame_mask.dtype != torch.bool:
            raise TypeError("frame_mask must have dtype torch.bool")
        if tuple(self.frame_mask.shape) != tuple(self.features.shape[:2]):
            raise ValueError("frame_mask must align with features batch and time dimensions")
        if not self.frame_mask.any(dim=1).all():
            raise ValueError("every sample must contain at least one valid frame")
        if self.features.shape[1] <= 0 or self.features.shape[2] <= 0:
            raise ValueError("features must contain at least one frame and channel")
        if self.features.shape[3] <= 0 or self.features.shape[4] <= 0:
            raise ValueError("features must have positive spatial dimensions")
        if self.frame_mask.device != self.features.device:
            raise ValueError("features and frame_mask must be on the same device")
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))


class VILASegmentationAdapter:
    """Extract explicit VILA text states and invoke an injected dense provider.

    ``dense_provider`` must either implement ``encode(media, media_config,
    request)`` or be callable with those three positional arguments, returning
    a :class:`DenseFeatureBatch`.  The provider is intentionally outside the
    ordinary VILA model path and is only called by :meth:`segment`.
    """

    def __init__(
        self,
        model: Any,
        dense_provider: Any,
        capability: SegmentationCapability,
    ) -> None:
        if model is None or not callable(model):
            raise TypeError("model must be callable")
        if not (callable(dense_provider) or callable(getattr(dense_provider, "encode", None))):
            raise TypeError("dense_provider must be callable or implement encode")
        if not isinstance(capability, SegmentationCapability):
            raise TypeError("capability must be a SegmentationCapability")
        if isinstance(model, nn.Module):
            model.eval()
            model.requires_grad_(False)
        self.model = model
        self.dense_provider = dense_provider
        self.capability = capability

    @staticmethod
    def _validate_inputs(
        input_ids: Tensor,
        query_token_mask: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Optional[Tensor]:
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,L]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must have dtype torch.long")
        if not isinstance(query_token_mask, Tensor) or query_token_mask.ndim != 2:
            raise ValueError("query_token_mask must have shape [B,L]")
        if query_token_mask.dtype != torch.bool:
            raise TypeError("query_token_mask must have dtype torch.bool")
        if tuple(query_token_mask.shape) != tuple(input_ids.shape):
            raise ValueError("query_token_mask must align with input_ids")
        if query_token_mask.device != input_ids.device:
            raise ValueError("query_token_mask and input_ids must be on the same device")
        if not query_token_mask.any(dim=1).all():
            raise ValueError("every sample must select at least one query token")
        if attention_mask is not None:
            if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 2:
                raise ValueError("attention_mask must have shape [B,L]")
            if tuple(attention_mask.shape) != tuple(input_ids.shape):
                raise ValueError("attention_mask must align with input_ids")
            if attention_mask.device != input_ids.device:
                raise ValueError("attention_mask and input_ids must be on the same device")
            if attention_mask.dtype == torch.bool:
                normalized_attention = attention_mask
            elif attention_mask.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                normalized_attention = attention_mask.to(dtype=torch.bool)
            else:
                raise TypeError("attention_mask must be boolean or integer")
            if (query_token_mask & ~normalized_attention).any():
                raise ValueError("query_token_mask cannot select padded input tokens")
            return normalized_attention
        return None

    @staticmethod
    def _hidden_states(outputs: Any) -> Any:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None and isinstance(outputs, Mapping):
            hidden_states = outputs.get("hidden_states")
        if hidden_states is None and isinstance(outputs, (tuple, list)):
            # A few lightweight test models return a tuple rather than a
            # ModelOutput.  Select the first tuple/list made entirely of tensors.
            for value in outputs:
                if isinstance(value, (tuple, list)) and value and all(isinstance(item, Tensor) for item in value):
                    hidden_states = value
                    break
        if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
            raise RuntimeError("VILA output_hidden_states=True did not return hidden states")
        if not all(isinstance(item, Tensor) and item.ndim == 3 for item in hidden_states):
            raise RuntimeError("VILA hidden states must be a sequence of [B,F,D] tensors")
        return hidden_states

    def _extract_query_states_with_timing(
        self,
        input_ids: Tensor,
        media: Optional[Mapping[str, Any]],
        media_config: Optional[Mapping[str, Any]],
        query_token_mask: Tensor,
        attention_mask: Optional[Tensor],
        hidden_layer: int,
        enable_grad: bool = False,
    ) -> tuple[QueryStateBatch, float]:
        # Keep package import free of VILA model initialization.  The model
        # package has heavyweight registrations, so resolve this hook only
        # when the explicitly requested adapter path is executed.
        from llava.model.fusion_observer import fusion_observation

        normalized_attention = self._validate_inputs(input_ids, query_token_mask, attention_mask)
        if not isinstance(hidden_layer, int):
            raise TypeError("hidden_layer must be an integer")
        capture = FusionCapture()
        _synchronize(input_ids)
        started = time.perf_counter()
        grad_context = torch.enable_grad() if enable_grad else torch.no_grad()
        with grad_context, fusion_observation(capture):
            outputs = self.model(
                input_ids=input_ids,
                media=media,
                media_config={} if media_config is None else media_config,
                attention_mask=normalized_attention,
                packing=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        hidden_states = self._hidden_states(outputs)
        try:
            selected_hidden = hidden_states[hidden_layer]
        except IndexError as error:
            raise ValueError(
                f"hidden_layer {hidden_layer} is outside {len(hidden_states)} returned hidden-state layers"
            ) from error
        _synchronize(selected_hidden)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if selected_hidden.shape[:2] != capture.provenance.source_position.shape:
            raise RuntimeError(
                "VILA hidden-state sequence does not align with fused provenance: "
                f"hidden={tuple(selected_hidden.shape)}, provenance={capture.provenance.source_position.shape}"
            )
        # Query masks refer to original padded token positions.  The provenance
        # object performs the media expansion and left/right padding mapping.
        return capture.gather_query_states(selected_hidden, query_token_mask), elapsed_ms

    def extract_query_states_training(
        self,
        input_ids: Tensor,
        media: Optional[Mapping[str, Any]],
        media_config: Optional[Mapping[str, Any]],
        query_token_mask: Tensor,
        attention_mask: Optional[Tensor] = None,
        hidden_layer: int = -1,
    ) -> QueryStateBatch:
        """Gradient-enabled query extraction for S4b training only.

        Ordinary requests continue to use :meth:`extract_query_states`, which
        runs under ``torch.no_grad()``.  Trainable LoRA/[SEG]/projector
        parameters need gradients through the frozen VILA forward, so this
        training entry keeps autograd enabled.
        """

        query_states, _ = self._extract_query_states_with_timing(
            input_ids,
            media,
            media_config,
            query_token_mask,
            attention_mask,
            hidden_layer,
            enable_grad=True,
        )
        return query_states

    def extract_query_states(
        self,
        input_ids: Tensor,
        media: Optional[Mapping[str, Any]],
        media_config: Optional[Mapping[str, Any]],
        query_token_mask: Tensor,
        attention_mask: Optional[Tensor] = None,
        hidden_layer: int = -1,
    ) -> QueryStateBatch:
        """Run frozen VILA teacher forcing and gather all selected query tokens."""

        query_states, _ = self._extract_query_states_with_timing(
            input_ids,
            media,
            media_config,
            query_token_mask,
            attention_mask,
            hidden_layer,
        )
        return query_states

    def _run_dense_provider(
        self,
        media: Any,
        media_config: Optional[Mapping[str, Any]],
        request: SegmentationRequest,
    ) -> tuple[DenseFeatureBatch, float]:
        started = time.perf_counter()
        encode = getattr(self.dense_provider, "encode", None)
        if callable(encode):
            dense = encode(media, media_config, request)
        else:
            dense = self.dense_provider(media, media_config, request)
        if not isinstance(dense, DenseFeatureBatch):
            raise TypeError("dense_provider must return a DenseFeatureBatch")
        _synchronize(dense.features)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return dense, elapsed_ms

    def segment(
        self,
        input_ids: Tensor,
        media: Optional[Mapping[str, Any]],
        media_config: Optional[Mapping[str, Any]],
        query_token_mask: Tensor,
        request: Any,
        attention_mask: Optional[Tensor] = None,
        hidden_layer: int = -1,
        dense_input: Any = None,
    ) -> SegmentationResult:
        """Run the complete opt-in language-conditioned segmentation path."""

        normalized = SegmentationRequest.from_value(request)
        if not normalized.enabled:
            raise ValueError("segmentation capability is disabled for this request")
        if normalized.task not in ("image", "video"):
            raise ValueError("enabled segmentation request requires image or video task")
        _synchronize(input_ids)
        started = time.perf_counter()
        query_states, vila_ms = self._extract_query_states_with_timing(
            input_ids,
            media,
            media_config,
            query_token_mask,
            attention_mask,
            hidden_layer,
        )
        provider_input = media if dense_input is None else dense_input
        dense, provider_ms = self._run_dense_provider(provider_input, media_config, normalized)
        if dense.features.shape[0] != query_states.states.shape[0]:
            raise ValueError("dense provider batch size must match VILA query batch")
        if normalized.task == "image" and dense.features.shape[1] != 1:
            raise ValueError("image segmentation requests require exactly one dense frame")
        batch = GroundingBatch(
            query_states=query_states.states,
            query_mask=query_states.mask,
            dense_features=dense.features,
            frame_mask=dense.frame_mask,
        )
        result = self.capability(batch, normalized)
        _synchronize(result.mask_logits)
        total_ms = (time.perf_counter() - started) * 1000.0
        diagnostics = dict(result.diagnostics)
        component_timing = dict(diagnostics.get("component_timing_ms", {}))
        component_timing.update(
            {
                "vila_query_encoding": vila_ms,
                "dense_provider": provider_ms,
                "total": total_ms,
            }
        )
        diagnostics.update(
            {
                "execution_path": "vila_evo_seg_adapter",
                "task": normalized.task,
                "query_source_positions": query_states.source_positions,
                "query_fused_positions": query_states.fused_positions,
                "dense_provider": dict(dense.diagnostics),
                "component_timing_ms": component_timing,
            }
        )
        return SegmentationResult(
            mask_logits=result.mask_logits,
            object_logits=result.object_logits,
            object_embeddings=result.object_embeddings,
            frame_embeddings=result.frame_embeddings,
            frame_mask=result.frame_mask,
            diagnostics=diagnostics,
        )
