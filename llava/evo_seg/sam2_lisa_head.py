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

    def forward_video_memory(
        self,
        seg_state: Tensor,
        image_features: Tensor,
        frame_mask: Tensor,
        high_res_features: tuple = (),
        memory_encoder: Any = None,
        memory_attention: Any = None,
        maskmem_tpos_enc: Any = None,
        max_memory: int = 6,
    ) -> SegmentationResult:
        """Per-frame mask decoding conditioned on SAM2 memory (differentiable).

        Runs the mask decoder over the video frames in temporal order; frames
        after the first are decoded from memory-conditioned features produced by
        the (optionally trainable) SAM2 memory encoder + memory attention.  This
        teaches temporal consistency and propagation directly in the training
        loop (GLUS/Sa2VA-style) instead of relying on post-hoc eval propagation.

        NOTE: SAM2's MemoryAttention expects seq-first [S, B, C] tensors; we
        follow the same layout as SAM2Base._prepare_memory_conditioned_features.
        """

        batch_size, frames, channels, height, width = image_features.shape
        prompt = self.seg_projection(self.seg_norm(seg_state))  # [B, prompt_dim]
        object_logits = self.object_head(self.object_norm(seg_state))  # [B, 1]
        dense = torch.zeros(
            batch_size, 1, height, width, device=image_features.device, dtype=torch.float32
        )
        valid = frame_mask.to(device=image_features.device)
        # per-frame spatial pos enc (seq-first) for the memory attention
        curr_pos = self.image_pe.to(dtype=torch.float32).flatten(2).permute(2, 0, 1)  # [HW, 1, C]
        memory: list = []
        per_frame_masks = []
        for frame_index in range(frames):
            if not bool(valid[:, frame_index].all()):
                height_out, width_out = dense.shape[-2] * 4, dense.shape[-1] * 4
                per_frame_masks.append(
                    image_features.new_zeros(batch_size, 1, 1, height_out, width_out)
                )
                continue
            pix_feat = image_features[:, frame_index].to(dtype=torch.float32)  # [B, C, H, W]
            if memory and memory_attention is not None:
                # memories: M x {"vision_features": [B,C,H,W], "vision_pos_enc": [[B,C,H,W]]}
                m_count = len(memory)
                mem_feats = torch.stack([m["vision_features"] for m in memory])  # [M,B,C,H,W]
                mem_feats = mem_feats.permute(1, 0, 2, 3, 4).reshape(batch_size, m_count, channels, -1)
                mem_feats = mem_feats.permute(0, 3, 1, 2).reshape(batch_size, m_count * height * width, channels)
                mem_feats = mem_feats.permute(1, 0, 2)  # [M*HW, B, C]
                mem_pos = torch.stack([m["vision_pos_enc"][-1] for m in memory])  # [M,B,C,H,W]
                mem_pos = mem_pos.permute(1, 0, 2, 3, 4).reshape(batch_size, m_count, channels, -1)
                mem_pos = mem_pos.permute(0, 3, 1, 2).reshape(batch_size, m_count * height * width, channels)
                if maskmem_tpos_enc is not None:
                    tpos = maskmem_tpos_enc.to(dtype=torch.float32)  # [num_maskmem, C]
                    # memory[0] is the oldest; give older frames a higher tpos index
                    offsets = torch.arange(m_count, device=mem_pos.device)
                    tpos_idx = ((m_count - 1 - offsets) % tpos.shape[0]).long()
                    tpos_sel = tpos[tpos_idx]  # [M, C]
                    tpos_b = tpos_sel.view(m_count, 1, channels).repeat(1, height * width, 1)
                    tpos_b = tpos_b.view(m_count * height * width, 1, channels).expand(-1, batch_size, -1)
                    mem_pos = mem_pos + tpos_b
                mem_pos = mem_pos.permute(1, 0, 2)  # [M*HW, B, C]
                curr = pix_feat.flatten(2).permute(2, 0, 1)  # [HW, B, C]
                fused = memory_attention(
                    curr=curr,
                    memory=mem_feats,
                    curr_pos=curr_pos.expand(-1, batch_size, -1),
                    memory_pos=mem_pos,
                    num_obj_ptr_tokens=0,
                )
                pix_feat = fused.permute(1, 2, 0).view(batch_size, channels, height, width)
            frame_masks, _, _, _ = self.sam_mask_decoder(
                image_embeddings=pix_feat,
                image_pe=self.image_pe.to(dtype=torch.float32),
                sparse_prompt_embeddings=prompt.unsqueeze(1),
                dense_prompt_embeddings=dense,
                multimask_output=False,
                repeat_image=False,
                high_res_features=(
                    [level[:, frame_index].to(dtype=torch.float32) for level in high_res_features]
                    if high_res_features
                    else None
                ),
            )
            per_frame_masks.append(frame_masks.unsqueeze(2))  # [B, 1, 1, H', W']
            if memory_encoder is not None:
                mem_out = memory_encoder(pix_feat, frame_masks.float().sigmoid())
                memory.append(mem_out)
                if max_memory > 0 and len(memory) > max_memory:
                    memory = memory[-max_memory:]
        mask_logits = torch.cat(per_frame_masks, dim=2)
        mask_logits = mask_logits * valid[:, None, :, None, None].to(dtype=mask_logits.dtype)
        frame_embeddings = seg_state.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, frames, -1)
        return SegmentationResult(
            mask_logits=mask_logits,
            object_logits=object_logits,
            object_embeddings=seg_state.unsqueeze(1),
            frame_embeddings=frame_embeddings,
            frame_mask=frame_mask.to(device=seg_state.device),
            diagnostics={
                "head": "sam2_lisa_memory",
                "mask_output_size": tuple(mask_logits.shape),
                "prompt_dim": self.prompt_dim,
                "memory_frames": len(memory),
            },
        )
