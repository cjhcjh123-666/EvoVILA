"""S4b training policy: LoRA, [SEG] embedding, grounding projector, SAM2 mask-decoder tuning.

All trainable components are inactive outside an explicit training scope, so a
normal VILA request executes the original frozen path.  No SAM2 import happens
at module level.
"""

from __future__ import annotations

import contextlib
import contextvars
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


_TRAINING_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "evo_seg_training_active", default=False
)


def is_seg_training_active() -> bool:
    """Return whether the trainable segmentation scope is active."""

    return _TRAINING_ACTIVE.get()


@contextlib.contextmanager
def seg_training_active(active: bool = True) -> Iterator[None]:
    """Enable LoRA/[SEG] injection only inside the explicit training scope."""

    token = _TRAINING_ACTIVE.set(active)
    try:
        yield
    finally:
        _TRAINING_ACTIVE.reset(token)


class LoRAAdapter(nn.Module):
    """Low-rank additive adapter applied as ``scaling * x @ A^T @ B^T``."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, hidden: Tensor) -> Tensor:
        return (hidden @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


def _layer_projection(layer: nn.Module, name: str) -> nn.Linear:
    module = layer
    for part in name.split("."):
        module = getattr(module, part)
    if not isinstance(module, nn.Linear):
        raise TypeError(f"LoRA target must be nn.Linear, got {type(module).__name__}")
    return module


def apply_lora(
    layers: Sequence[nn.Module],
    projection_names: Sequence[str],
    rank: int,
    alpha: float,
) -> Tuple[Dict[str, List[LoRAAdapter]], List[Any]]:
    """Attach additive LoRA adapters to named projections of every layer.

    The base weights stay frozen and unmodified; adapters only contribute when
    :func:`seg_training_active` is enabled.  Returns adapters by projection
    name and the forward-hook handles used to remove them later.
    """

    if not layers or not projection_names:
        raise ValueError("LoRA requires at least one layer and projection")
    adapters: Dict[str, List[LoRAAdapter]] = {name: [] for name in projection_names}
    handles: List[Any] = []
    for layer in layers:
        for name in projection_names:
            linear = _layer_projection(layer, name)
            adapter = LoRAAdapter(
                linear.in_features, linear.out_features, rank, alpha
            ).to(device=linear.weight.device, dtype=torch.float32)
            adapters[name].append(adapter)

            def make_hook(adapter: LoRAAdapter):
                def hook(_module: nn.Module, args: Tuple[Any, ...], output: Tensor) -> Tensor:
                    if not is_seg_training_active():
                        return output
                    hidden = args[0]
                    return output + adapter(hidden)

                return hook

            handles.append(linear.register_forward_hook(make_hook(adapter)))
    return adapters, handles


def remove_hooks(handles: Sequence[Any]) -> None:
    for handle in handles:
        handle.remove()


class SegEmbeddingInjector:
    """Trainable additive [SEG] embedding without mutating the frozen table."""

    def __init__(self, embed_module: nn.Module, seg_id: int, hidden_size: int) -> None:
        if seg_id < 0:
            raise ValueError("seg_id must be a non-negative token id")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.seg_id = seg_id
        self.seg_embedding = nn.Parameter(torch.empty(1, hidden_size))
        nn.init.normal_(self.seg_embedding, std=0.02)
        self._embed_module = embed_module
        self._handle = embed_module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module: nn.Module, args: Tuple[Any, ...], output: Tensor) -> Tensor:
        if not is_seg_training_active():
            return output
        input_ids = args[0]
        if not isinstance(input_ids, Tensor) or input_ids.dtype != torch.long:
            return output
        mask = input_ids == self.seg_id
        if not mask.any():
            return output
        output = output.clone()
        output[mask] = output[mask] + self.seg_embedding.to(dtype=output.dtype, device=output.device)
        return output

    def detach(self) -> None:
        self._handle.remove()


class GroundingProjector(nn.Module):
    """Keep full multi-token query states and append an explicit [SEG] token."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.seg_norm = nn.LayerNorm(hidden_size)
        self.seg_projection = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        query_states: Tensor,
        query_mask: Tensor,
        seg_positions: Tensor,
    ) -> Tensor:
        """Return ``[B,L+1,D]``: full query states plus the [SEG] token state."""

        if query_states.ndim != 3 or query_states.shape[0] != query_mask.shape[0]:
            raise ValueError("query_states must be [B,L,D] aligned with query_mask")
        if seg_positions.shape != (query_states.shape[0],):
            raise ValueError("seg_positions must be [B]")
        batch = torch.arange(query_states.shape[0], device=query_states.device)
        seg_state = query_states[batch, seg_positions]
        seg_token = self.seg_projection(self.seg_norm(seg_state)).unsqueeze(1)
        return torch.cat([query_states, seg_token], dim=1)


@dataclass
class SegTrainingPolicy:
    """Trainable S4b components and their optimizer parameter groups."""

    lora_adapters: Dict[str, List[LoRAAdapter]]
    lora_handles: List[Any]
    seg_injector: SegEmbeddingInjector
    projector: nn.Module
    decoder: nn.Module
    sam2_mask_decoder: List[nn.Parameter]
    sam2_mask_decoder_trainable: bool

    def optimizer_param_groups(self, lora_lr: float, base_lr: float) -> List[Dict[str, Any]]:
        groups = [
            {
                "name": "decoder_projector_seg",
                "params": list(self.projector.parameters())
                + list(self.decoder.parameters())
                + [self.seg_injector.seg_embedding],
                "lr": base_lr,
            }
        ]
        lora_params = [
            parameter
            for adapters in self.lora_adapters.values()
            for adapter in adapters
            for parameter in adapter.parameters()
        ]
        if lora_params:
            groups.append({"name": "lora", "params": lora_params, "lr": lora_lr})
        if self.sam2_mask_decoder_trainable and self.sam2_mask_decoder:
            groups.append(
                {
                    "name": "sam2_mask_decoder",
                    "params": self.sam2_mask_decoder,
                    "lr": base_lr,
                }
            )
        return groups

    def freeze_state_summary(self, vila_model: nn.Module) -> Dict[str, Any]:
        vila_trainable = sum(
            1 for parameter in vila_model.parameters() if parameter.requires_grad
        )
        trainable_counts = {
            "vila_base_trainable_parameters": vila_trainable,
            "lora_adapters": sum(len(adapters) for adapters in self.lora_adapters.values()),
            "projector_parameters": sum(parameter.numel() for parameter in self.projector.parameters()),
            "decoder_parameters": sum(parameter.numel() for parameter in self.decoder.parameters()),
            "seg_embedding_parameters": self.seg_injector.seg_embedding.numel(),
            "sam2_mask_decoder_trainable": self.sam2_mask_decoder_trainable,
            "sam2_mask_decoder_parameters": sum(
                parameter.numel() for parameter in self.sam2_mask_decoder
            ),
        }
        return trainable_counts
