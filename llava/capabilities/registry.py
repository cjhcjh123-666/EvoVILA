"""Registry and dispatch pipeline for optional EvoVILA capabilities."""

from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from .base import Capability, CapabilityError, CapabilityRequest, MediaContext


CapabilityFactory = Callable[..., Capability]


class CapabilityPipeline:
    """Dispatches context observations to explicitly selected capabilities."""

    def __init__(self, capabilities: Iterable[Capability] = ()) -> None:
        self._capabilities = tuple(capabilities)

    @property
    def capabilities(self) -> Tuple[Capability, ...]:
        return self._capabilities

    @property
    def enabled_names(self) -> Tuple[str, ...]:
        return tuple(capability.name for capability in self._capabilities)

    def on_media_context(self, context: MediaContext) -> Tuple[Any, ...]:
        # Results are observations for future consumers. The model never uses
        # them to alter media embeddings or token alignment in M1.
        return tuple(capability.on_media_context(context) for capability in self._capabilities)

    def on_vision_features(
        self, context: MediaContext, features: Any, media_name: str = "image"
    ) -> Tuple[Any, ...]:
        return tuple(
            capability.on_vision_features(context, features, media_name=media_name)
            for capability in self._capabilities
        )

    def get_outputs(self) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        for capability in self._capabilities:
            outputs.update(capability.get_outputs())
        return outputs

    def compute_loss(self) -> Any:
        losses = [loss for loss in (capability.compute_loss() for capability in self._capabilities) if loss is not None]
        if not losses:
            return None
        return sum(losses)

    def clear(self) -> None:
        for capability in self._capabilities:
            capability.clear()


class CapabilityRegistry:
    """Name-to-factory registry with explicit, deterministic construction."""

    def __init__(self) -> None:
        self._factories: Dict[str, CapabilityFactory] = {}

    def register(
        self,
        name: str,
        factory: Optional[CapabilityFactory] = None,
        *,
        overwrite: bool = False,
    ):
        """Register a factory, or return a decorator when ``factory`` is omitted."""
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("Capability name cannot be empty")

        def add(candidate: CapabilityFactory) -> CapabilityFactory:
            if normalized_name in self._factories and not overwrite:
                raise ValueError(f"Capability '{normalized_name}' is already registered")
            self._factories[normalized_name] = candidate
            return candidate

        return add(factory) if factory is not None else add

    def unregister(self, name: str) -> None:
        self._factories.pop(name, None)

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._factories))

    def build(self, request: CapabilityRequest, *, config: Any = None) -> CapabilityPipeline:
        capabilities = []
        for name in request.names:
            factory = self._factories.get(name)
            if factory is None:
                available = ", ".join(self.names()) or "none"
                if request.strict:
                    raise CapabilityError(
                        f"Unknown capability '{name}'. Registered capabilities: {available}"
                    )
                continue
            try:
                capability = factory(config=config, options=request.options.get(name, {}))
            except TypeError as exc:
                raise CapabilityError(
                    f"Capability factory '{name}' must accept config= and options="
                ) from exc
            if not isinstance(capability, Capability):
                raise CapabilityError(f"Capability factory '{name}' returned {type(capability)!r}, expected Capability")
            capabilities.append(capability)
        return CapabilityPipeline(capabilities)


DEFAULT_CAPABILITY_REGISTRY = CapabilityRegistry()


def _build_image_segmentation_capability(**kwargs) -> Capability:
    # Keep the optional decoder and torch modules out of ordinary VILA imports.
    from .image_segmentation import ImageSegmentationCapability

    return ImageSegmentationCapability(**kwargs)


DEFAULT_CAPABILITY_REGISTRY.register("image_segmentation", _build_image_segmentation_capability)


def _build_video_segmentation_capability(**kwargs) -> Capability:
    # Keep SAM2 and its Hydra configuration out of ordinary VILA imports.
    from .video_segmentation import VideoSegmentationCapability

    return VideoSegmentationCapability(**kwargs)


DEFAULT_CAPABILITY_REGISTRY.register("video_segmentation", _build_video_segmentation_capability)


def build_capability_pipeline(
    request: Any = None,
    *,
    config: Any = None,
    registry: CapabilityRegistry = DEFAULT_CAPABILITY_REGISTRY,
) -> CapabilityPipeline:
    """Build the no-op pipeline by default, or an explicitly requested one."""
    if request is None and config is not None:
        request = getattr(config, "capabilities", None)
    return registry.build(CapabilityRequest.from_config(request), config=config)


__all__ = [
    "CapabilityFactory",
    "CapabilityPipeline",
    "CapabilityRegistry",
    "DEFAULT_CAPABILITY_REGISTRY",
    "build_capability_pipeline",
]
