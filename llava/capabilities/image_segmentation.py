"""Optional image segmentation warm-up capability.

This module deliberately consumes projected image features from the existing
VILA image path. It is not a SAM2 implementation and is never imported unless
the ``image_segmentation`` capability is explicitly requested.
"""

import math
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .base import Capability, CapabilityError, MediaContext


class ImageSegmentationCapability(Capability, nn.Module):
    """Decode a dense image mask from static square visual tokens."""

    name = "image_segmentation"

    def __init__(self, config: Any = None, options: Optional[Mapping[str, Any]] = None) -> None:
        nn.Module.__init__(self)
        Capability.__init__(self, config=config, options=options)
        hidden_channels = int(self.options.get("hidden_channels", 128))
        self.mask_decoder = nn.Sequential(
            nn.LazyConv2d(hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.mask_loss_weight = float(self.options.get("mask_loss_weight", 1.0))
        self.dice_loss_weight = float(self.options.get("dice_loss_weight", 1.0))
        self._context: Optional[MediaContext] = None
        self._mask_logits: Optional[torch.Tensor] = None
        self._mask_loss: Optional[torch.Tensor] = None

    def on_media_context(self, context: MediaContext) -> None:
        self._context = context
        self._mask_logits = None
        self._mask_loss = None

    @staticmethod
    def _as_feature_batch(features: Any) -> torch.Tensor:
        if isinstance(features, (list, tuple)):
            if not features:
                raise CapabilityError("image_segmentation received no image features")
            if not all(isinstance(feature, torch.Tensor) for feature in features):
                raise CapabilityError("image_segmentation expects tensor image features")
            if len({tuple(feature.shape) for feature in features}) != 1:
                raise CapabilityError(
                    "image_segmentation requires equal-size image feature grids; "
                    "disable dynamic tiling for the M2 warm-up"
                )
            features = torch.stack(list(features), dim=0)
        if not isinstance(features, torch.Tensor) or features.dim() != 3:
            raise CapabilityError("image_segmentation expects features shaped [batch, tokens, channels]")
        return features

    @staticmethod
    def _as_target_batch(targets: Any, batch_size: int) -> torch.Tensor:
        if not isinstance(targets, torch.Tensor):
            raise CapabilityError("segmentation_masks must be a torch.Tensor")
        if targets.dim() == 2:
            targets = targets.unsqueeze(0)
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if targets.dim() != 3:
            raise CapabilityError("segmentation_masks must be shaped [batch, height, width]")
        if targets.shape[0] != batch_size:
            raise CapabilityError(
                f"segmentation_masks batch ({targets.shape[0]}) does not match image features ({batch_size})"
            )
        return targets

    @staticmethod
    def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = logits.sigmoid().flatten(1)
        targets = targets.flatten(1)
        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        return (1 - (2 * intersection + 1e-6) / (denominator + 1e-6)).mean()

    def on_vision_features(self, context: MediaContext, features: Any, media_name: str = "image") -> None:
        if media_name != "image" or context.media_count("image") == 0:
            return None

        features = self._as_feature_batch(features)
        token_count = int(features.shape[1])
        side = math.isqrt(token_count)
        if side * side != token_count:
            raise CapabilityError(
                f"image_segmentation requires square visual tokens, received {token_count}; "
                "use the static resize image path for M2"
            )

        feature_map = features.transpose(1, 2).reshape(features.shape[0], features.shape[2], side, side)
        mask_logits = self.mask_decoder(feature_map)
        targets = context.metadata.get("segmentation_masks")
        if targets is not None:
            targets = self._as_target_batch(targets, mask_logits.shape[0])
            output_size = tuple(targets.shape[-2:])
        else:
            configured_size = self.options.get("output_size")
            output_size = tuple(configured_size) if configured_size is not None else None

        if output_size is not None:
            mask_logits = F.interpolate(mask_logits, size=output_size, mode="bilinear", align_corners=False)
        self._mask_logits = mask_logits

        if targets is not None:
            targets = targets.to(device=mask_logits.device, dtype=mask_logits.dtype).unsqueeze(1)
            self._mask_loss = self.mask_loss_weight * F.binary_cross_entropy_with_logits(mask_logits, targets)
            self._mask_loss = self._mask_loss + self.dice_loss_weight * self._dice_loss(mask_logits, targets)
        return None

    def get_outputs(self):
        if self._mask_logits is None:
            return {}
        return {"mask_logits": self._mask_logits}

    def compute_loss(self):
        return self._mask_loss

    def clear(self) -> None:
        self._context = None
        self._mask_logits = None
        self._mask_loss = None


__all__ = ["ImageSegmentationCapability"]
