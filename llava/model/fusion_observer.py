"""Request-local fusion observation hook with no segmentation imports."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_CURRENT_FUSION_OBSERVER: ContextVar[Any] = ContextVar("evovila_fusion_observer", default=None)


def current_fusion_observer() -> Any:
    """Return the observer for the current request, or ``None`` by default."""

    return _CURRENT_FUSION_OBSERVER.get()


@contextmanager
def fusion_observation(observer: Any) -> Iterator[None]:
    """Install one request-local observer and always restore the prior scope."""

    if observer is None or not hasattr(observer, "observe_fusion"):
        raise TypeError("observer must provide observe_fusion")
    token = _CURRENT_FUSION_OBSERVER.set(observer)
    try:
        yield
    finally:
        _CURRENT_FUSION_OBSERVER.reset(token)
