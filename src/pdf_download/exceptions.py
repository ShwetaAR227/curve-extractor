"""Stage-2 exception types.

Distinct types per failure mode so callers can react appropriately:
retry/queue on network trouble, move on for a true 404, and treat an
invalid payload (HTML error page with a .pdf name) as a hard data problem.
DELIBERATE CHANGE from legacy, which collapsed every failure into ``False``.
"""


class PdfDownloadError(Exception):
    """Base class for all stage-2 download failures."""


class PdfNetworkError(PdfDownloadError):
    """Transient network failure: timeout, connection error, 5xx. Retryable."""


class PdfNotFoundError(PdfDownloadError):
    """The server says the document does not exist (404/410). Not retryable."""


class InvalidPdfError(PdfDownloadError):
    """Payload is not a real PDF (bad magic bytes / truncated) — the
    'HTML error page saved as .pdf' failure mode. Never written to disk."""
