"""Stage-2 downloader: fetch, verify, and store datasheet PDFs.

Orchestration: for each device, walk the source registry in priority order,
download the first candidate URL that yields a *verified* PDF (magic bytes +
minimum size, checked before the file reaches its final name), with retry +
exponential backoff on transient network failures only.

DELIBERATE CHANGES from legacy ``pdf_downloader.py``:
- Typed exceptions (see :mod:`exceptions`) instead of True/False returns.
- 404/410 is not retried (legacy hammered dead URLs three times).
- TLS verification ON by default; ``PIPELINE_SSL_NO_VERIFY=1`` opts out for
  the known broken-cert vendor sites (legacy disabled verification always).
- Per-host throttle is an in-process monotonic-clock interval (legacy used
  fcntl.flock — silently a no-op on Windows, where this pipeline runs).
- Payload is written to a temp file and renamed only after verification.
- The batch never mutates input device dicts (legacy set
  ``_pdf_download_failed`` on them); results live in the returned report.
- Legacy checkpoint JSON dropped: skip-if-exists gives the same resume
  behavior without a second state file that can go stale or corrupt.
"""
import gzip
import logging
import os
import ssl
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from src.ingest.model_name_utils import sanitize_for_filesystem

from .exceptions import InvalidPdfError, PdfDownloadError, PdfNetworkError, PdfNotFoundError
from .sources import iter_sources

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF"
MIN_PDF_BYTES = 100
DEFAULT_TIMEOUT_S = 30
DEFAULT_MAX_RETRIES = 3

#: Rotated across retry attempts — some vendor CDNs 403 a repeated UA.
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

#: Minimum seconds between requests to the same host (legacy-verified 1.0s
#: default keeps vendor CDNs happy). Overridable via env, config-in-one-place.
HOST_MIN_INTERVAL_S = float(os.environ.get("PIPELINE_HOST_MIN_INTERVAL", "1.0"))

_last_request_at: Dict[str, float] = {}


def _browser_headers(ua: str) -> dict:
    """Full browser-style header set — empirically required (legacy probe
    2026-05-02): a bare User-Agent gets 403'd by Cloudflare-fronted vendor
    CDNs (onsemi, ST, Toshiba, Infineon, Mouser)."""
    return {
        "User-Agent": ua,
        "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.digikey.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Fetch-User": "?1",
    }


def _ssl_context() -> ssl.SSLContext:
    """Verifying context by default; PIPELINE_SSL_NO_VERIFY=1 opts out for
    vendor sites with broken certs (logged so it's never silent)."""
    ctx = ssl.create_default_context()
    if os.environ.get("PIPELINE_SSL_NO_VERIFY", "").strip() == "1":
        logger.warning("TLS verification DISABLED via PIPELINE_SSL_NO_VERIFY=1")
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _acquire_host_slot(host: Optional[str]) -> None:
    """Sleep until at least HOST_MIN_INTERVAL_S has passed since the last
    request to ``host`` from this process."""
    if not host:
        return
    now = time.monotonic()
    wait = _last_request_at.get(host, 0.0) + HOST_MIN_INTERVAL_S - now
    if wait > 0:
        time.sleep(wait)
    _last_request_at[host] = time.monotonic()


def _normalize_url(url: str) -> str:
    url = url.strip()
    if url.startswith("//"):
        url = "https:" + url
    # Quote path/query so unescaped spaces don't crash http.client.
    parts = urlsplit(url)
    return urlunsplit((
        parts.scheme, parts.netloc,
        quote(parts.path, safe="/%"),
        quote(parts.query, safe="=&%"),
        parts.fragment,
    ))


def _decode_body(data: bytes, content_encoding: str) -> bytes:
    enc = (content_encoding or "").lower()
    if enc == "gzip":
        return gzip.decompress(data)
    if enc == "deflate":
        try:
            return zlib.decompress(data)
        except zlib.error:
            return zlib.decompress(data, -zlib.MAX_WBITS)
    return data


def _verify_pdf(data: bytes, url: str) -> None:
    """Raise :class:`InvalidPdfError` unless ``data`` looks like a real PDF."""
    if len(data) < MIN_PDF_BYTES:
        raise InvalidPdfError(f"Payload too small ({len(data)} bytes) from {url}")
    if not data.startswith(PDF_MAGIC):
        raise InvalidPdfError(f"Not a PDF (HTML error page?) from {url}")


def download_pdf(
    url: str,
    dest_path: Union[str, Path],
    timeout: int = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
    force: bool = False,
) -> str:
    """Download one URL to ``dest_path`` as a verified PDF.

    Args:
        url: Source URL (protocol-relative ``//`` accepted).
        dest_path: Final PDF path; parent directories are created.
        timeout: Per-request timeout in seconds.
        max_retries: Attempts for *transient* failures (backoff 1s, 2s, ...).
        force: Re-download even if ``dest_path`` exists.

    Returns:
        ``"downloaded"`` or ``"skipped"`` (already present and not ``force``).

    Raises:
        PdfDownloadError: Blank/invalid URL.
        PdfNotFoundError: Server returned 404/410 (not retried).
        PdfNetworkError: Transient failures exhausted ``max_retries``.
        InvalidPdfError: Payload failed PDF verification (nothing written).
    """
    dest_path = Path(dest_path)
    if not url or not url.strip() or url.strip() == "-":
        raise PdfDownloadError(f"Blank or invalid URL for {dest_path.name}")

    if dest_path.exists() and not force:
        logger.debug("Exists, skipping: %s", dest_path)
        return "skipped"

    url = _normalize_url(url)
    host = (urlparse(url).hostname or "").lower() or None
    ctx = _ssl_context()
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        _acquire_host_slot(host)
        req = Request(url, headers=_browser_headers(USER_AGENTS[attempt % len(USER_AGENTS)]))
        try:
            with urlopen(req, timeout=timeout, context=ctx) as resp:
                data = _decode_body(resp.read(), resp.headers.get("Content-Encoding"))
            break
        except HTTPError as e:
            if e.code in (404, 410):
                raise PdfNotFoundError(f"HTTP {e.code} for {url}") from e
            last_error = e  # 5xx/403 etc.: transient-ish, retry
        except (URLError, TimeoutError, OSError) as e:
            last_error = e
        logger.debug("Attempt %d/%d failed for %s: %s", attempt + 1, max_retries, url, last_error)
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    else:
        raise PdfNetworkError(
            f"Download failed after {max_retries} attempts: {url} ({last_error})"
        ) from last_error

    _verify_pdf(data, url)  # raises InvalidPdfError before anything is written

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        tmp_path.write_bytes(data)
        os.replace(tmp_path, dest_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    logger.info("Downloaded %s (%d bytes)", dest_path.name, len(data))
    return "downloaded"


@dataclass(frozen=True)
class DeviceDownloadResult:
    """Outcome for one device: status is one of
    downloaded / skipped / no_url / failed."""

    device: str
    status: str
    source: Optional[str] = None
    path: Optional[Path] = None
    reason: Optional[str] = None


@dataclass
class BatchReport:
    """Outcome of a batch run; input device dicts are never mutated."""

    results: List[DeviceDownloadResult] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        """Return ``{status: count}`` over all results."""
        out: Dict[str, int] = {}
        for r in self.results:
            out[r.status] = out.get(r.status, 0) + 1
        return out


def download_for_device(
    device: dict,
    output_dir: Union[str, Path],
    force: bool = False,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> DeviceDownloadResult:
    """Download the datasheet for one device, walking sources by priority.

    Layout matches legacy: ``<output_dir>/<safe_name>/<safe_name>.pdf``.
    Every source's URL is tried through :func:`download_pdf`; the first
    verified PDF wins. All failures are collected into the result's reason.
    """
    model_name = (device.get("ModelName") or "").strip()
    if not model_name:
        return DeviceDownloadResult(
            device="<unknown>", status="failed",
            reason="device record has no ModelName",
        )

    safe_name = sanitize_for_filesystem(model_name)
    dest_path = Path(output_dir) / safe_name / f"{safe_name}.pdf"

    if dest_path.exists() and not force:
        return DeviceDownloadResult(
            device=model_name, status="skipped", path=dest_path,
        )

    failures: List[str] = []
    tried_any_url = False
    for source in iter_sources():
        url = source.find_url(device)
        if not url:
            continue
        tried_any_url = True
        try:
            download_pdf(url, dest_path, timeout=timeout, force=force)
            return DeviceDownloadResult(
                device=model_name, status="downloaded",
                source=source.name, path=dest_path,
            )
        except PdfDownloadError as e:
            failures.append(f"{source.name}: {type(e).__name__}: {e}")
            logger.info("Source %s failed for %s: %s", source.name, model_name, e)

    if not tried_any_url:
        return DeviceDownloadResult(
            device=model_name, status="no_url",
            reason="no source produced a candidate URL",
        )
    return DeviceDownloadResult(
        device=model_name, status="failed", reason="; ".join(failures),
    )


def download_batch(
    devices: List[dict],
    output_dir: Union[str, Path],
    force: bool = False,
) -> BatchReport:
    """Download datasheets for a batch. One device's failure never stops the
    batch; per-device outcomes and counts live in the returned report."""
    report = BatchReport()
    for device in devices:
        result = download_for_device(device, output_dir, force=force)
        report.results.append(result)

    counts = report.counts()
    logger.info(
        "PDF batch done: %s",
        ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "empty batch",
    )
    for r in report.results:
        if r.status in ("failed", "no_url"):
            logger.warning("  %s: %s (%s)", r.device, r.status, r.reason)
    return report
