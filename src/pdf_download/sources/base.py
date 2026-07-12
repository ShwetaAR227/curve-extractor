"""Stage-2 PDF source interface + registry.

A *source* answers one question: "given a device record, what URL might
serve its datasheet PDF?" Downloading/verifying is the downloader's job.

This is the pluggable interface the legacy README promised but never built
(legacy `sources/` actually held stage-1 CSV configs). Registration is data:
adding a vendor source is one module + one `register()` call, never a new
code path in the downloader.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PdfSource(ABC):
    """One strategy for finding a datasheet URL for a device."""

    #: Registry key, unique.
    name: str = ""
    #: Lower tries first.
    priority: int = 100

    @abstractmethod
    def find_url(self, device: dict) -> Optional[str]:
        """Return a candidate PDF URL for ``device``, or None if this source
        has nothing to offer. Must not raise for missing device fields."""


_REGISTRY: Dict[str, PdfSource] = {}


def register(source: PdfSource) -> None:
    """Add a source instance to the registry (idempotent by name)."""
    _REGISTRY[source.name] = source


def get_source(name: str) -> PdfSource:
    """Look up a registered source by name.

    Raises:
        KeyError: Unknown name; message lists every registered source.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown PDF source '{name}'. Registered sources: {list_sources()}"
        ) from None


def list_sources() -> List[str]:
    """Return every registered source name, sorted."""
    return sorted(_REGISTRY)


def iter_sources() -> List[PdfSource]:
    """Return all registered sources in priority order (lowest first)."""
    return sorted(_REGISTRY.values(), key=lambda s: s.priority)
