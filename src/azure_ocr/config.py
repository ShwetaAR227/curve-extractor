"""Stage-3 configuration — all constants in one place (CLAUDE.md §3).

NOTE: the legacy snapshot's config module was never copied into
``D:\\Extractor`` (every legacy stage-3 module imports a ``..config`` that
does not exist there), so these values were reconstructed: DEFAULT_DPI was
verified against real rendered pages in the legacy data (A4 page PNG of
1654x2339 px = exactly 200 DPI); the rest are the legacy call-site defaults.

Azure credentials are NEVER stored here — the client reads
``AZURE_DOC_INTEL_ENDPOINT`` / ``AZURE_DOC_INTEL_KEY`` /
``AZURE_OCR_ENDPOINT`` / ``AZURE_OCR_KEY`` from the environment.
"""

#: Page-render resolution. Doc Intel polygons are in inches; multiplying by
#: this DPI maps them onto the rendered page PNGs.
DEFAULT_DPI = 200

#: Padding added around figure bounding boxes before cropping (pixels).
FIGURE_PADDING_PX = 10

#: Seconds between Azure Read API calls (free-tier friendly).
OCR_RATE_LIMIT_S = 0.5

#: Default budget of OCR calls per document run.
DEFAULT_MAX_OCR_CALLS = 200
