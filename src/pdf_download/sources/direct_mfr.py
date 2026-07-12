"""Source: well-known manufacturer URL patterns (no scraping, no API).

Patterns are DATA: ``MFR_URL_PATTERNS`` is an ordered list of
``(manufacturer_matcher, url_template)`` pairs; the template receives
``lower``/``upper`` part-number variants. Adding a manufacturer is a
one-entry edit.

Pattern provenance (legacy 2026-05-02 probe): ST reliable, onsemi ~50%
hit rate, Microchip/Nexperia/Diodes plausible-but-spottier. Infineon is
deliberately absent — its URLs embed an unpredictable version slug.
"""
import logging
from typing import Callable, List, Optional, Tuple

from .base import PdfSource

logger = logging.getLogger(__name__)

#: (matcher over lowercased manufacturer string, URL template).
#: First matching entry wins.
MFR_URL_PATTERNS: List[Tuple[Callable[[str], bool], str]] = [
    (
        lambda m: "st" in m and "micro" in m,
        "https://www.st.com/resource/en/datasheet/{lower}.pdf",
    ),
    (
        lambda m: "onsemi" in m or "on semiconductor" in m,
        "https://www.onsemi.com/download/data-sheet/pdf/{lower}-d.pdf",
    ),
    (
        lambda m: "microchip" in m,
        "https://ww1.microchip.com/downloads/aemDocuments/documents/SCBU/"
        "ProductDocuments/DataSheets/{upper}.pdf",
    ),
    (
        lambda m: "nexperia" in m,
        "https://assets.nexperia.com/documents/data-sheet/{upper}.pdf",
    ),
    (
        lambda m: "diodes" in m,
        "https://www.diodes.com/assets/Datasheets/{upper}.pdf",
    ),
]


class DirectMfrSource(PdfSource):
    """Predictable manufacturer datasheet URL patterns."""

    name = "direct_mfr"
    priority = 10

    def find_url(self, device: dict) -> Optional[str]:
        part = (device.get("ModelName") or "").strip()
        mfr = (device.get("manufacturer") or "").strip().lower()
        if not part or not mfr:
            return None
        for matcher, template in MFR_URL_PATTERNS:
            if matcher(mfr):
                return template.format(lower=part.lower(), upper=part.upper())
        return None
