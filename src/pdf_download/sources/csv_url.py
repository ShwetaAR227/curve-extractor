"""Source: the datasheet URL already present in the stage-1 CSV record.

Highest priority — the vendor-provided URL is right most of the time and
costs nothing to discover.
"""
from typing import Optional

from .base import PdfSource


class CsvUrlSource(PdfSource):
    """Return the device record's own ``pdf_url`` field."""

    name = "csv_url"
    priority = 0

    def find_url(self, device: dict) -> Optional[str]:
        url = (device.get("pdf_url") or "").strip()
        if not url or url == "-":
            return None
        if url.startswith("//"):
            url = "https:" + url
        return url
