"""Core contracts for optional EvoVILA capabilities.

Capability callbacks observe the inputs to the multimodal embedding path. They
do not own or replace VILA's media-token fusion, which keeps the baseline path
compatible with existing checkpoints and callers.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Optional, Sequence, Tuple


def _freeze(value: Any) -> Any:
    """Freeze container structure while leaving tensors and other payloads intact."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class CapabilityRequest:
    """Explicit selection of optional capabilities.

    ``names`` is intentionally empty by default. A capability is never
    activated merely because its implementation is importable.
    """

    names: Tuple[str, ...] = ()
    options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    strict: bool = True

    def __post_init__(self) -> None:
        names = tuple(dict.fromkeys(str(name) for name in self.names if str(name)))
        options = _freeze(dict(self.options))
        if not isinstance(options, Mapping):
            raise TypeError("CapabilityRequest.options must be a mapping")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "options", options)

    @classmethod
    def from_config(cls, value: Any) -> "CapabilityRequest":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(names=(value,))
        if isinstance(value, (list, tuple)):
            if not all(isinstance(name, str) for name in value):
                raise TypeError("Capability names must be strings")
            return cls(names=tuple(value))
        if not isinstance(value, Mapping):
            raise TypeError("capabilities must be a string, sequence, mapping, or None")

        names_value = value.get("names", value.get("enabled"))
        if names_value is None:
            names_value = tuple(
                key for key in value if key not in {"options", "strict", "names", "enabled"}
            )
        if isinstance(names_value, str):
            names_value = (names_value,)
        if not isinstance(names_value, (list, tuple)) or not all(
            isinstance(name, str) for name in names_value
        ):
            raise TypeError("Capability names must be strings")

        options = value.get("options", {})
        if not isinstance(options, Mapping):
            raise TypeError("Capability options must be a mapping")
        return cls(names=tuple(names_value), options=options, strict=bool(value.get("strict", True)))


@dataclass(frozen=True)
class MediaContext:
    """Read-only view of a VILA multimodal embedding request.

    The media payloads are the already-prepared values used by VILA. The
    outer mappings and sequences are immutable, and a capability callback's
    return value is never fed back into token fusion. Implementations must
    treat payload objects such as tensors as read-only as well.
    """

    input_ids: Any
    media: Mapping[str, Sequence[Any]]
    media_config: Mapping[str, Mapping[str, Any]]
    stage: str = "embed"
    training: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "media", _freeze(dict(self.media or {})))
        object.__setattr__(self, "media_config", _freeze(dict(self.media_config or {})))
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata or {})))

    @classmethod
    def from_inputs(
        cls,
        input_ids: Any,
        media: Optional[Mapping[str, Sequence[Any]]],
        media_config: Optional[Mapping[str, Mapping[str, Any]]],
        *,
        stage: str = "embed",
        training: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "MediaContext":
        return cls(
            input_ids=input_ids,
            media=media or {},
            media_config=media_config or {},
            stage=stage,
            training=training,
            metadata=metadata or {},
        )

    def media_count(self, name: Optional[str] = None) -> int:
        if name is not None:
            return len(self.media.get(name, ()))
        return sum(len(values) for values in self.media.values())


class Capability:
    """Base class for an optional, observation-only capability."""

    name: ClassVar[str] = "capability"

    def __init__(self, config: Any = None, options: Optional[Mapping[str, Any]] = None) -> None:
        self.config = config
        self.options = _freeze(dict(options or {}))

    def on_media_context(self, context: MediaContext) -> Any:
        """Observe one embedding request without modifying VILA inputs."""
        return None

    def on_vision_features(self, context: MediaContext, features: Any, media_name: str = "image") -> Any:
        """Observe projected vision features produced during the VILA path."""
        return None

    def get_outputs(self) -> Mapping[str, Any]:
        """Return optional outputs produced for the current request."""
        return {}

    def compute_loss(self) -> Any:
        """Return an optional loss for the current request."""
        return None

    def clear(self) -> None:
        """Release request-local outputs retained by the capability."""
        return None


class CapabilityError(RuntimeError):
    """Raised when an explicitly requested capability cannot be built."""


CapabilityContext = MediaContext


__all__ = [
    "Capability",
    "CapabilityContext",
    "CapabilityError",
    "CapabilityRequest",
    "MediaContext",
]
