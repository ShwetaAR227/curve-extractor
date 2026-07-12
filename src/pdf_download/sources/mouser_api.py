"""Source: Mouser Search API (free tier — register at mouser.com/api-hub/).

Fires only when the ``MOUSER_API_KEY`` environment variable is set
(CLAUDE.md §3: secrets via env, never in the repo). Without a key this
source is a silent no-op — verified by test to make zero network calls.
"""
import json
import logging
import os
import ssl
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import PdfSource

logger = logging.getLogger(__name__)

_API_URL = "https://api.mouser.com/api/v1/search/partnumber?apiKey={key}"
_TIMEOUT_S = 20


class MouserApiSource(PdfSource):
    """Look up the DataSheetUrl for a part via the Mouser Search API."""

    name = "mouser_api"
    priority = 20

    def find_url(self, device: dict) -> Optional[str]:
        api_key = os.environ.get("MOUSER_API_KEY", "").strip()
        if not api_key:
            return None
        part = (device.get("ModelName") or "").strip()
        if not part:
            return None

        payload = json.dumps({
            "SearchByPartRequest": {
                "mouserPartNumber": part,
                "partSearchOptions": "1",
            }
        }).encode("utf-8")
        req = Request(
            _API_URL.format(key=api_key),
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urlopen(req, timeout=_TIMEOUT_S, context=ssl.create_default_context()) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        except (URLError, HTTPError, TimeoutError, OSError, json.JSONDecodeError) as e:
            logger.warning("mouser_api lookup failed for %s: %s", part, e)
            return None

        parts = (data.get("SearchResults") or {}).get("Parts") or []
        # Prefer the exact part match, else the first hit with a datasheet.
        for p in parts:
            if (p.get("ManufacturerPartNumber") or "").lower() == part.lower():
                url = (p.get("DataSheetUrl") or "").strip()
                if url:
                    return url
        for p in parts:
            url = (p.get("DataSheetUrl") or "").strip()
            if url:
                return url
        return None
