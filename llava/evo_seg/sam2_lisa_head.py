"""LISA-style segmentation head: [SEG] token state -> SAM2 mask decoder.

The referring expression is summarized by the frozen VILA into the hidden
state of the trainable [SEG] token.  That state is projected to SAM2's prompt
embedding space and fed to the (trainable) SAM2 mask decoder together with the
frozen SAM2 image embeddings, producing a single object mask.  This replaces
the earlier query-conditioned cross-attention decoder, whose query states the
audit showed to be almost query-insensitive (same-image different-query cosine
~0.87-0.98 at every layer).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .contracts import SegmentationResult


class Sam2LisaHead(nn.Module):
    """Project the [SEG] state and run the SAM2 mask decoder as the head.

    The SAM2 mask decoder parameters are exposed as part of this module so
    they become trainable with the rest of the segmentation stack.  The SAM2
    image encoder and prompt encoder remain frozen.
    """

    def __init__(
        self,
        hidden_size: int,
        sam2_model: Any,
        prompt_dim: int = 256,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if prompt_dim <= 0:
            raise ValueError("prompt_dim must be positive")
        if sam2_model is None or not hasattr(sam2_model, "sam_mask_decoder"):
            raise TypeError("sam2_model must expose sam_mask_decoder")

        self.prompt_dim = prompt_dim
        self.seg_norm = nn.LayerNorm(hidden_size)
        self.seg_projection = nn.Linear(hidden_size, prompt_dim)
        self.object_norm = nn.LayerNorm(hidden_size)
        self.object_head = nn.Linear(hidden_size, 1)
        # Reference the SAM2 mask decoder so its parameters become part of this
        # module's trainable parameters (LISA-style joint fine-tuning).
        self.sam_mask_decoder = sam2_model.sam_mask_decoder
        self.register_buffer(
            "image_pe",
            sam2_model.sam_prompt_encoder.get_dense_pe().detach().clone(),
        )

    def forward(
        self,
        seg_state: Tensor,
        image_features: Tensor,
        frame_mask: Tensor,
        high_res_features: tuple = (),
    ) -> SegmentationResult:
        """Predict a mask for a single frame (image) or per-frame batch.

        Args:
            seg_state: [B, D] fp32 hidden state of the [SEG] token.
            image_features: [B, T, C, H, W] frozen SAM2 image embeddings.
            frame_mask: [B, T] valid-frame mask.
        """

        batch_size = seg_state.shape[0]
        prompt = self.seg_projection(self.seg_norm(seg_state))  # [B, prompt_dim]
        object_logits = self.object_head(self.object_norm(seg_state))  # [B, 1]

        features = image_features.to(dtype=torch.float32)
        if features.ndim != 5:
            raise ValueError("image_features must have shape [B,T,C,H,W]")
        batch_size, frames, _, height, width = features.shape
        dense = torch.zeros(
            batch_size,
            1,
            height,
            width,
            device=features.device,
            dtype=torch.float32,
        )
        valid = frame_mask.to(device=features.device)
        per_frame_masks = []
        for frame_index in range(frames):
            if bool(valid[:, frame_index].all()):
                frame_masks, _, _, _ = self.sam_mask_decoder(
                    image_embeddings=features[:, frame_index],
                    image_pe=self.image_pe.to(dtype=torch.float32),
                    sparse_prompt_embeddings=prompt.unsqueeze(1),  # [B, 1, 256]
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=(
                        [level[:, frame_index].to(dtype=torch.float32) for level in high_res_features]
                        if high_res_features
                        else None
                    ),
                )
                per_frame_masks.append(frame_masks.unsqueeze(2))  # [B, 1, 1, H, W]
            else:
                height_out, width_out = dense.shape[-2] * 4, dense.shape[-1] * 4
                per_frame_masks.append(
                    features.new_zeros(batch_size, 1, 1, height_out, width_out)
                )
        mask_logits = torch.cat(per_frame_masks, dim=2)  # [B, 1, T, H, W]
        mask_logits = mask_logits * valid[:, None, :, None, None].to(dtype=mask_logits.dtype)
        frame_embeddings = seg_state.unsqueeze(1).unsqueeze(1).expand(
            batch_size, 1, frames, -1
        )
        return SegmentationResult(
            mask_logits=mask_logits,
            object_logits=object_logits,
            object_embeddings=seg_state.unsqueeze(1),
            frame_embeddings=frame_embeddings,
            frame_mask=frame_mask.to(device=seg_state.device),
            diagnostics={
                "head": "sam2_lisa",
                "mask_output_size": tuple(mask_logits.shape),
                "prompt_dim": self.prompt_dim,
            },
        )
