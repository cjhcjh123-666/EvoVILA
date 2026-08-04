"""Capability bridge for exported VILA remote-code models.

When remote code is loaded from an EvoVILA checkout, the shared registry is
used. When a converted model is copied as a standalone remote-code bundle,
the default remains a no-op without adding a dependency to that bundle.
"""

try:
    from llava.capabilities import CapabilityContext, CapabilityError, MediaContext, build_capability_pipeline
except ImportError:  # pragma: no cover - exercised only by standalone bundles
    from types import MappingProxyType

    class MediaContext:
        def __init__(self, input_ids, media, media_config, stage="embed", training=False, metadata=None):
            self.input_ids = input_ids
            self.media = MappingProxyType(dict(media or {}))
            self.media_config = MappingProxyType(dict(media_config or {}))
            self.stage = stage
            self.training = training
            self.metadata = MappingProxyType(dict(metadata or {}))

        @classmethod
        def from_inputs(
            cls, input_ids, media, media_config, *, stage="embed", training=False, metadata=None
        ):
            return cls(input_ids, media, media_config, stage, training, metadata)

    CapabilityContext = MediaContext

    class CapabilityError(RuntimeError):
        pass

    class _NoOpPipeline:
        enabled_names = ()
        capabilities = ()

        def on_media_context(self, context):
            return ()

        def on_vision_features(self, context, features, media_name="image"):
            return ()

        def get_outputs(self):
            return {}

        def compute_loss(self):
            return None

        def clear(self):
            return None

    def build_capability_pipeline(request=None, *, config=None, registry=None):
        if request is None and config is not None:
            request = getattr(config, "capabilities", None)
        if isinstance(request, str):
            requested_names = (request,)
        elif isinstance(request, dict):
            requested_names = request.get("names", request.get("enabled", ()))
        else:
            requested_names = request or ()
        if requested_names:
            raise RuntimeError("Optional capabilities require the EvoVILA package when using standalone remote code")
        return _NoOpPipeline()


__all__ = ["CapabilityContext", "CapabilityError", "MediaContext", "build_capability_pipeline"]
