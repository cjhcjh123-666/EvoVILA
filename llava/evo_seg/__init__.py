"""Opt-in EvoVILA segmentation extensions.

The package-level import intentionally remains free of SAM2 and VILA model
imports.
"""

from .contracts import GroundingBatch, SegmentationRequest, SegmentationResult
from .decoder import QueryConditionedSpatialDecoder
from .capability import SegmentationCapability
from .provenance import FusedSequenceProvenance, FusionCapture, QueryStateBatch
from .vila_adapter import DenseFeatureBatch, VILASegmentationAdapter

__all__ = [
    "DenseFeatureBatch",
    "FusedSequenceProvenance",
    "FusionCapture",
    "GroundingBatch",
    "QueryStateBatch",
    "SegmentationRequest",
    "SegmentationResult",
    "QueryConditionedSpatialDecoder",
    "SegmentationCapability",
    "VILASegmentationAdapter",
]
