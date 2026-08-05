"""Language-conditioned spatial decoder used by the opt-in S0 capability."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import GroundingBatch, SegmentationResult


def _axis_encoding(length: int, dimension: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build a deterministic sinusoidal encoding for one coordinate axis."""

    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
    frequencies = torch.exp(-frequencies * (torch.log(torch.tensor(10000.0, device=device)) / dimension))
    encoding = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    if dimension > 1:
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


def _spatiotemporal_encoding(
    frames: int, height: int, width: int, dimension: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Return `[T*H*W,D]` coordinates without assuming a fixed image size."""

    time = _axis_encoding(frames, dimension, device, dtype)[:, None, None, :]
    row = _axis_encoding(height, dimension, device, dtype)[None, :, None, :]
    column = _axis_encoding(width, dimension, device, dtype)[None, None, :, :]
    return (time + row + column).reshape(frames * height * width, dimension)


class _QueryDecoderLayer(nn.Module):
    """One object-to-language then object-to-spatial attention block."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.language_norm = nn.LayerNorm(model_dim)
        self.language_attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(model_dim)
        self.spatial_attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        object_states: Tensor,
        language_states: Tensor,
        spatial_states: Tensor,
        language_padding: Tensor,
        spatial_padding: Tensor,
    ) -> Tensor:
        language_query = self.language_norm(object_states)
        language_update, _ = self.language_attention(
            language_query,
            language_states,
            language_states,
            key_padding_mask=language_padding,
            need_weights=False,
        )
        object_states = object_states + language_update

        spatial_query = self.spatial_norm(object_states)
        spatial_update, _ = self.spatial_attention(
            spatial_query,
            spatial_states,
            spatial_states,
            key_padding_mask=spatial_padding,
            need_weights=False,
        )
        object_states = object_states + spatial_update
        return object_states + self.ffn(self.ffn_norm(object_states))


class QueryConditionedSpatialDecoder(nn.Module):
    """Cross-attend multi-token language states to dense spatiotemporal features."""

    def __init__(
        self,
        query_dim: int,
        feature_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        num_object_queries: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("query_dim", query_dim),
            ("feature_dim", feature_dim),
            ("model_dim", model_dim),
            ("num_heads", num_heads),
            ("num_layers", num_layers),
            ("num_object_queries", num_object_queries),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.query_norm = nn.LayerNorm(query_dim)
        self.query_projection = nn.Linear(query_dim, model_dim)
        self.feature_projection = nn.Conv2d(feature_dim, model_dim, kernel_size=1)
        self.object_queries = nn.Parameter(torch.randn(num_object_queries, model_dim) * 0.02)
        self.layers = nn.ModuleList(
            [_QueryDecoderLayer(model_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.mask_norm = nn.LayerNorm(model_dim)
        self.mask_projection = nn.Linear(model_dim, model_dim)
        self.object_norm = nn.LayerNorm(model_dim)
        self.object_head = nn.Linear(model_dim, 1)
        self.model_dim = model_dim
        self.num_object_queries = num_object_queries

    def forward(self, batch: GroundingBatch) -> SegmentationResult:
        """Predict object masks and representations for a validated batch."""

        query_input = batch.query_states.to(dtype=self.query_projection.weight.dtype)
        query_states = self.query_projection(self.query_norm(query_input))
        query_mask = batch.query_mask
        language_padding = ~query_mask
        batch_size, frames, _, height, width = batch.dense_features.shape

        feature_input = batch.dense_features.to(dtype=self.feature_projection.weight.dtype)
        projected_frames = self.feature_projection(
            feature_input.reshape(batch_size * frames, -1, height, width)
        ).reshape(batch_size, frames, self.model_dim, height, width)
        spatial_tokens = projected_frames.permute(0, 1, 3, 4, 2).reshape(
            batch_size, frames * height * width, self.model_dim
        )
        spatial_tokens = spatial_tokens + _spatiotemporal_encoding(
            frames,
            height,
            width,
            self.model_dim,
            spatial_tokens.device,
            spatial_tokens.dtype,
        ).unsqueeze(0)
        spatial_padding = (~batch.frame_mask)[:, :, None].expand(batch_size, frames, height * width).reshape(
            batch_size, frames * height * width
        )

        valid_query_count = query_mask.sum(dim=1, keepdim=True).to(dtype=query_states.dtype).clamp_min(1)
        language_context = (query_states * query_mask.unsqueeze(-1)).sum(dim=1) / valid_query_count
        object_states = self.object_queries.unsqueeze(0) + language_context.unsqueeze(1)
        for layer in self.layers:
            object_states = layer(
                object_states,
                query_states,
                spatial_tokens,
                language_padding,
                spatial_padding,
            )

        mask_embeddings = self.mask_projection(self.mask_norm(object_states))
        normalized_masks = F.normalize(mask_embeddings, dim=-1)
        normalized_pixels = F.normalize(projected_frames, dim=2)
        mask_logits = torch.einsum("bnd,btdhw->bnthw", normalized_masks, normalized_pixels)
        frame_mask = batch.frame_mask[:, None, :, None, None]
        mask_logits = mask_logits * frame_mask.to(dtype=mask_logits.dtype)

        # Soft spatial pooling keeps the representation differentiable before S2/S3 refinement.
        weights = torch.softmax(mask_logits.reshape(batch_size, self.num_object_queries, frames, -1), dim=-1)
        pixel_values = projected_frames.permute(0, 1, 3, 4, 2).reshape(
            batch_size, frames, height * width, self.model_dim
        )
        frame_embeddings = torch.einsum("bnth,bthd->bntd", weights, pixel_values)
        frame_embeddings = frame_embeddings * batch.frame_mask[:, None, :, None].to(dtype=frame_embeddings.dtype)
        valid_frame_count = batch.frame_mask.sum(dim=1, keepdim=True).to(dtype=frame_embeddings.dtype).clamp_min(1)
        object_embeddings = frame_embeddings.sum(dim=2) / valid_frame_count[:, None, :]
        object_logits = self.object_head(self.object_norm(object_embeddings)).squeeze(-1)
        return SegmentationResult(
            mask_logits=mask_logits,
            object_logits=object_logits,
            object_embeddings=object_embeddings,
            frame_embeddings=frame_embeddings,
            frame_mask=batch.frame_mask,
            diagnostics={
                "language_source_length": int(query_states.shape[1]),
                "spatial_source_length": int(frames * height * width),
                "output_shapes": {
                    "mask_logits": tuple(mask_logits.shape),
                    "frame_embeddings": tuple(frame_embeddings.shape),
                },
            },
        )
