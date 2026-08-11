"""Optional capability-extension interfaces for EvoVILA."""

from .base import Capability, CapabilityContext, CapabilityError, CapabilityRequest, MediaContext
from .registry import (
    DEFAULT_CAPABILITY_REGISTRY,
    CapabilityFactory,
    CapabilityPipeline,
    CapabilityRegistry,
    build_capability_pipeline,
)

__all__ = [
    "Capability",
    "CapabilityContext",
    "CapabilityError",
    "CapabilityFactory",
    "CapabilityPipeline",
    "CapabilityRegistry",
    "CapabilityRequest",
    "DEFAULT_CAPABILITY_REGISTRY",
    "MediaContext",
    "build_capability_pipeline",
]
