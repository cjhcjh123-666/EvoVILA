"""Validated fused-token provenance and multi-token query gathering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor


FUSED_SOURCE_TYPES = ("text", "image", "video", "padding")


def _validate_tensor(name: str, value: Any, dtype: torch.dtype, rank: int) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype != dtype or value.ndim != rank:
        raise TypeError(f"{name} must be a {dtype} tensor with rank {rank}")
    return value


@dataclass(frozen=True)
class FusedSequenceProvenance:
    """Mapping from the fused sequence to original token positions."""

    source_type: Tuple[Tuple[str, ...], ...]
    source_position: Tensor
    fused_position: Tensor
    attention_mask: Tensor
    valid_length: Tensor

    def __post_init__(self) -> None:
        source_type = tuple(tuple(str(item) for item in row) for row in self.source_type)
        object.__setattr__(self, "source_type", source_type)
        source_position = _validate_tensor("source_position", self.source_position, torch.long, 2)
        fused_position = _validate_tensor("fused_position", self.fused_position, torch.long, 2)
        attention_mask = _validate_tensor("attention_mask", self.attention_mask, torch.bool, 2)
        valid_length = _validate_tensor("valid_length", self.valid_length, torch.long, 1)
        if source_position.shape != fused_position.shape or source_position.shape != attention_mask.shape:
            raise ValueError("provenance position and attention tensors must share [B,F] shape")
        batch_size, sequence_length = source_position.shape
        if len(source_type) != batch_size or any(len(row) != sequence_length for row in source_type):
            raise ValueError("source_type must align with [B,F] tensors")
        if valid_length.shape != (batch_size,):
            raise ValueError("valid_length must have shape [B]")
        if (valid_length < 0).any() or (valid_length > sequence_length).any():
            raise ValueError("valid_length must be within the fused sequence length")
        if not torch.equal(attention_mask.sum(dim=1).to(dtype=torch.long), valid_length):
            raise ValueError("valid_length must equal attention_mask sums")
        expected_fused = torch.arange(sequence_length, device=fused_position.device).expand(batch_size, -1)
        if not torch.equal(fused_position, expected_fused):
            raise ValueError("fused_position must enumerate each padded fused row")
        for row_index, row in enumerate(source_type):
            invalid = set(row).difference(FUSED_SOURCE_TYPES)
            if invalid:
                raise ValueError(f"unknown fused source type(s): {sorted(invalid)}")
            for position, source in enumerate(row):
                if source == "padding":
                    if bool(attention_mask[row_index, position]) or source_position[row_index, position] != -1:
                        raise ValueError("padding provenance must be masked and use source position -1")
                elif not bool(attention_mask[row_index, position]):
                    raise ValueError("non-padding provenance cannot be outside attention_mask")
                elif source == "text" and source_position[row_index, position] < 0:
                    raise ValueError("text provenance must retain an original position")
                elif source in {"image", "video"} and source_position[row_index, position] != -1:
                    raise ValueError("media expansion provenance must use source position -1")

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        padding_side: str,
        fused_attention_mask: Tensor,
        device: Optional[torch.device] = None,
    ) -> "FusedSequenceProvenance":
        if not rows:
            raise ValueError("cannot build provenance for an empty batch")
        if padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'")
        fused_attention_mask = _validate_tensor("fused_attention_mask", fused_attention_mask, torch.bool, 2)
        max_length = max(len(row["source_type"]) for row in rows)
        if tuple(fused_attention_mask.shape) != (len(rows), max_length):
            raise ValueError("fused_attention_mask must align with the longest fused row")
        source_types, source_positions = [], []
        for row_index, row in enumerate(rows):
            types = list(row["source_type"])
            positions = list(row["source_position"])
            if len(types) != len(positions):
                raise ValueError("source_type and source_position rows must have equal lengths")
            pad_length = max_length - len(types)
            padding_types = ["padding"] * pad_length
            padding_positions = [-1] * pad_length
            if padding_side == "right":
                final_types = types + padding_types
                final_positions = positions + padding_positions
            else:
                final_types = padding_types + types
                final_positions = padding_positions + positions
            source_types.append(tuple(final_types))
            source_positions.append(final_positions)
            expected_valid = len(types)
            if int(fused_attention_mask[row_index].sum()) != expected_valid:
                raise ValueError("fused_attention_mask does not match the observed row length")
        target_device = device or fused_attention_mask.device
        source_position = torch.tensor(source_positions, dtype=torch.long, device=target_device)
        fused_position = torch.arange(max_length, dtype=torch.long, device=target_device).expand(len(rows), -1).clone()
        attention_mask = fused_attention_mask.to(device=target_device)
        valid_length = attention_mask.sum(dim=1, dtype=torch.long)
        return cls(tuple(source_types), source_position, fused_position, attention_mask, valid_length)

    @property
    def batch_size(self) -> int:
        return int(self.source_position.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.source_position.shape[1])

    def gather_query_states(self, hidden_states: Tensor, query_token_mask: Tensor) -> "QueryStateBatch":
        """Gather all explicitly selected original text tokens from fused states."""

        if not isinstance(hidden_states, Tensor) or hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B,F,D]")
        if tuple(hidden_states.shape[:2]) != tuple(self.source_position.shape):
            raise ValueError("hidden_states must align with fused provenance")
        query_token_mask = _validate_tensor("query_token_mask", query_token_mask, torch.bool, 2)
        if query_token_mask.shape[0] != self.batch_size:
            raise ValueError("query_token_mask batch size must align with provenance")
        selected_positions = [row.nonzero(as_tuple=False).flatten().tolist() for row in query_token_mask]
        if any(not positions for positions in selected_positions):
            raise ValueError("every sample must select at least one query token")

        fused_positions = []
        for batch_index, positions in enumerate(selected_positions):
            row_fused = []
            for source_position in positions:
                matches = [
                    fused_index
                    for fused_index, (source_type, original_position) in enumerate(
                        zip(self.source_type[batch_index], self.source_position[batch_index].tolist())
                    )
                    if source_type == "text" and original_position == source_position
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "query token did not map to exactly one fused text position: "
                        f"batch={batch_index}, source_position={source_position}, matches={matches}"
                    )
                row_fused.append(matches[0])
            fused_positions.append(row_fused)

        max_query_length = max(len(row) for row in fused_positions)
        states = hidden_states.new_zeros((self.batch_size, max_query_length, hidden_states.shape[-1]))
        mask = torch.zeros((self.batch_size, max_query_length), dtype=torch.bool, device=hidden_states.device)
        source_positions_tensor = torch.full(
            (self.batch_size, max_query_length), -1, dtype=torch.long, device=hidden_states.device
        )
        fused_positions_tensor = torch.full_like(source_positions_tensor, -1)
        for batch_index, row_fused in enumerate(fused_positions):
            length = len(row_fused)
            row_fused_tensor = torch.tensor(row_fused, dtype=torch.long, device=hidden_states.device)
            row_source_tensor = torch.tensor(
                selected_positions[batch_index], dtype=torch.long, device=hidden_states.device
            )
            states[batch_index, :length] = hidden_states[batch_index, row_fused_tensor]
            mask[batch_index, :length] = True
            source_positions_tensor[batch_index, :length] = row_source_tensor
            fused_positions_tensor[batch_index, :length] = row_fused_tensor
        return QueryStateBatch(
            states=states,
            mask=mask,
            source_positions=source_positions_tensor,
            fused_positions=fused_positions_tensor,
            provenance=self,
        )


@dataclass(frozen=True)
class QueryStateBatch:
    """Padded multi-token query states aligned to one VILA batch."""

    states: Tensor
    mask: Tensor
    source_positions: Tensor
    fused_positions: Tensor
    provenance: FusedSequenceProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, FusedSequenceProvenance):
            raise TypeError("provenance must be FusedSequenceProvenance")
        if self.states.ndim != 3 or not self.states.is_floating_point():
            raise ValueError("states must be floating point with shape [B,L,D]")
        batch_size, query_length, _ = self.states.shape
        for name, value in (("mask", self.mask), ("source_positions", self.source_positions), ("fused_positions", self.fused_positions)):
            if value.ndim != 2 or value.shape != (batch_size, query_length):
                raise ValueError(f"{name} must align with states [B,L]")
        if self.mask.dtype != torch.bool or self.source_positions.dtype != torch.long or self.fused_positions.dtype != torch.long:
            raise TypeError("query state metadata has invalid dtype")
        if batch_size != self.provenance.batch_size or not self.mask.any(dim=1).all():
            raise ValueError("query states must align with provenance and contain one token per sample")
        if (~self.mask).any() and self.states.masked_select((~self.mask).unsqueeze(-1)).abs().max() != 0:
            raise ValueError("padded query states must be zero")


class FusionCapture:
    """One-shot observer populated by VILA's request-local fusion hook."""

    def __init__(self) -> None:
        self._provenance: Optional[FusedSequenceProvenance] = None

    def observe_fusion(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        padding_side: str,
        fused_attention_mask: Tensor,
    ) -> None:
        if self._provenance is not None:
            raise RuntimeError("FusionCapture cannot observe more than one fused sequence")
        self._provenance = FusedSequenceProvenance.from_rows(
            rows,
            padding_side=padding_side,
            fused_attention_mask=fused_attention_mask,
        )

    @property
    def provenance(self) -> FusedSequenceProvenance:
        if self._provenance is None:
            raise RuntimeError("FusionCapture has not observed a fused sequence")
        return self._provenance

    def gather_query_states(self, hidden_states: Tensor, query_token_mask: Tensor) -> QueryStateBatch:
        return self.provenance.gather_query_states(hidden_states, query_token_mask)
