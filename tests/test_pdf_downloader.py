"""Tests for src/pdf_download/downloader.py — all network mocked, no real calls.

Covers: success, 404 (typed, no retry), network timeout (typed, retried with
backoff, gives up cleanly), invalid-PDF content (HTML error page saved as
.pdf — typed error, nothing written), skip-if-exists vs force, and the
source-fallback walk in download_for_device.
"""
import io
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from src.pdf_download.downloader import (
    download_pdf,
    download_for_device,
    download_batch,
)
from src.pdf_download.exceptions import (
    PdfDownloadError,
    PdfNetworkError,
    PdfNotFoundError,
    InvalidPdfError,
)

PDF_BYTES = b"%PDF-1.4\n" + b"x" * 200
HTML_BYTES = b"<!DOCTYPE html><html><body>404 Not Found</body></html>" + b" " * 100


class FakeResponse:
    """Minimal stand-in for urlopen's response context manager."""

    def __init__(self, body: bytes, headers: dict = None):
        self._body = body
        self.headers = headers or {}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def no_sleep(monkeypatch):
    """Capture backoff sleeps instead of actually sleeping."""
    sleeps = []
    monkeypatch.setattr("src.pdf_download.downloader.time.sleep", sleeps.append)
    monkeypatch.setattr("src.pdf_download.downloader._acquire_host_slot", lambda host: None)
    return sleeps


def mock_urlopen(monkeypatch, side_effect):
    """Install a urlopen replacement; side_effect is a list of responses or
    exceptions consumed one call at a time."""
    calls = []

    def fake(req, timeout=None, context=None):
        calls.append(req)
        effect = side_effect[min(len(calls) - 1, len(side_effect) - 1)]
        if isinstance(effect, Exception):
            raise effect
        return effect

    monkeypatch.setattr("src.pdf_download.downloader.urlopen", fake)
    return calls


def http_404(url="https://x.test/a.pdf"):
    return HTTPError(url, 404, "Not Found", hdrs=None, fp=io.BytesIO(b""))


class TestDownloadSuccess:
    def test_writes_pdf_and_reports_downloaded(self, tmp_path, no_sleep, monkeypatch):
        calls = mock_urlopen(monkeypatch, [FakeResponse(PDF_BYTES)])
        dest = tmp_path / "dev" / "dev.pdf"
        outcome = download_pdf("https://x.test/a.pdf", dest)
        assert outcome == "downloaded"
        assert dest.read_bytes() == PDF_BYTES
        assert len(calls) == 1

    def test_gzip_content_decoded(self, tmp_path, no_sleep, monkeypatch):
        import gzip
        resp = FakeResponse(gzip.compress(PDF_BYTES), {"Content-Encoding": "gzip"})
        mock_urlopen(monkeypatch, [resp])
        dest = tmp_path / "a.pdf"
        assert download_pdf("https://x.test/a.pdf", dest) == "downloaded"
        assert dest.read_bytes() == PDF_BYTES

    def test_protocol_relative_url_normalized(self, tmp_path, no_sleep, monkeypatch):
        calls = mock_urlopen(monkeypatch, [FakeResponse(PDF_BYTES)])
        download_pdf("//x.test/a.pdf", tmp_path / "a.pdf")
        assert calls[0].full_url.startswith("https://")

    def test_url_with_spaces_quoted(self, tmp_path, no_sleep, monkeypatch):
        calls = mock_urlopen(monkeypatch, [FakeResponse(PDF_BYTES)])
        download_pdf("https://x.test/data sheet.pdf", tmp_path / "a.pdf")
        assert " " not in calls[0].full_url


class TestNotFound:
    def test_404_typed_and_not_retried(self, tmp_path, no_sleep, monkeypatch):
        calls = mock_urlopen(monkeypatch, [http_404()])
        with pytest.raises(PdfNotFoundError):
            download_pdf("https://x.test/a.pdf", tmp_path / "a.pdf")
        assert len(calls) == 1          # no pointless retry of a 404
        assert not (tmp_path / "a.pdf").exists()

    def test_is_subclass_of_base(self):
        assert issubclass(PdfNotFoundError, PdfDownloadError)


class TestNetworkFailure:
    def test_timeout_retried_then_typed_error(self, tmp_path, no_sleep, monkeypatch):
        calls = mock_urlopen(monkeypatch, [TimeoutError("timed out")])
        with pytest.raises(PdfNetworkError):
            download_pdf("https://x.test/a.pdf", tmp_path / "a.pdf", max_retries=3)
        assert len(calls) == 3          # retried up to max_retries
        assert no_sleep == [1, 2]       # exponential backoff between attempts

    def test_recovers_on_second_attempt(self, tmp_path, no_sleep, monkeypatch):
        calls = mock_urlopen(
            monkeypatch, [URLError("reset"), FakeResponse(PDF_BYTES)]
        )
        dest = tmp_path / "a.pdf"
        assert download_pdf("https://x.test/a.pdf", dest) == "downloaded"
        assert len(calls) == 2
        assert dest.read_bytes() == PDF_BYTES

    def test_5xx_retried(self, tmp_path, no_sleep, monkeypatch):
        err = HTTPError("https://x.test/a.pdf", 503, "Unavailable", None, io.BytesIO(b""))
        calls = mock_urlopen(monkeypatch, [err, FakeResponse(PDF_BYTES)])
        assert download_pdf("https://x.test/a.pdf", tmp_path / "a.pdf") == "downloaded"
        assert len(calls) == 2


class TestInvalidPdf:
    def test_html_error_page_never_saved(self, tmp_path, no_sleep, monkeypatch):
        mock_urlopen(monkeypatch, [FakeResponse(HTML_BYTES)])
        dest = tmp_path / "a.pdf"
        with pytest.raises(InvalidPdfError):
            download_pdf("https://x.test/a.pdf", dest)
        assert not dest.exists()        # garbage never reaches dest
        assert not list(tmp_path.iterdir())  # no temp file left behind either

    def test_too_small_rejected(self, tmp_path, no_sleep, monkeypatch):
        mock_urlopen(monkeypatch, [FakeResponse(b"%PDF")])
        with pytest.raises(InvalidPdfError):
            download_pdf("https://x.test/a.pdf", tmp_path / "a.pdf")

    def test_empty_url_typed_error(self, tmp_path, no_sleep):
        with pytest.raises(PdfDownloadError):
            download_pdf("  ", tmp_path / "a.pdf")


class TestExistsPolicy:
    def test_skip_if_exists(self, tmp_path, no_sleep, monkeypatch):
        dest = tmp_path / "a.pdf"
        dest.write_bytes(PDF_BYTES)
        calls = mock_urlopen(monkeypatch, [FakeResponse(PDF_BYTES)])
        assert download_pdf("https://x.test/a.pdf", dest) == "skipped"
        assert calls == []              # no network touched

    def test_force_redownloads(self, tmp_path, no_sleep, monkeypatch):
        dest = tmp_path / "a.pdf"
        dest.write_bytes(b"%PDF old" + b"x" * 100)
        calls = mock_urlopen(monkeypatch, [FakeResponse(PDF_BYTES)])
        assert download_pdf("https://x.test/a.pdf", dest, force=True) == "downloaded"
        assert len(calls) == 1
        assert dest.read_bytes() == PDF_BYTES


class TestDownloadForDevice:
    DEVICE = {
        "ModelName": "STP55NF06",
        "manufacturer": "STMicroelectronics",
        "pdf_url": "https://csv.test/stp55nf06.pdf",
    }

    def test_csv_url_tried_first(self, tmp_path, no_sleep, monkeypatch):
        calls = mock_urlopen(monkeypatch, [FakeResponse(PDF_BYTES)])
        result = download_for_device(self.DEVICE, tmp_path)
        assert result.status == "downloaded"
        assert result.source == "csv_url"
        assert calls[0].full_url == "https://csv.test/stp55nf06.pdf"
        assert result.path.read_bytes() == PDF_BYTES

    def test_falls_back_to_next_source(self, tmp_path, no_sleep, monkeypatch):
        # csv_url 404s -> direct_mfr (ST pattern) succeeds
        calls = mock_urlopen(monkeypatch, [http_404(), FakeResponse(PDF_BYTES)])
        result = download_for_device(self.DEVICE, tmp_path)
        assert result.status == "downloaded"
        assert result.source == "direct_mfr"
        assert "st.com" in calls[1].full_url

    def test_all_sources_fail_reports_failed_with_reason(self, tmp_path, no_sleep, monkeypatch):
        mock_urlopen(monkeypatch, [http_404()])
        result = download_for_device(self.DEVICE, tmp_path)
        assert result.status == "failed"
        assert result.reason          # human-readable why
        assert result.path is None or not result.path.exists()

    def test_no_url_from_any_source(self, tmp_path, no_sleep, monkeypatch):
        calls = mock_urlopen(monkeypatch, [FakeResponse(PDF_BYTES)])
        device = {"ModelName": "X999", "manufacturer": "ACME"}
        result = download_for_device(device, tmp_path)
        assert result.status == "no_url"
        assert calls == []

    def test_missing_model_name_fails_cleanly(self, tmp_path, no_sleep):
        result = download_for_device({"pdf_url": "https://x.test/a.pdf"}, tmp_path)
        assert result.status == "failed"
        assert "ModelName" in result.reason


class TestDownloadBatch:
    def test_batch_isolates_failures_and_reports(self, tmp_path, no_sleep, monkeypatch):
        devices = [
            {"ModelName": "GOOD1", "manufacturer": "m", "pdf_url": "https://x.test/1.pdf"},
            {"ModelName": "BAD404", "manufacturer": "m", "pdf_url": "https://x.test/2.pdf"},
            {"ModelName": "GOOD2", "manufacturer": "m", "pdf_url": "https://x.test/3.pdf"},
        ]

        def fake(req, timeout=None, context=None):
            if "2.pdf" in req.full_url:
                raise http_404(req.full_url)
            return FakeResponse(PDF_BYTES)

        monkeypatch.setattr("src.pdf_download.downloader.urlopen", fake)
        report = download_batch(devices, tmp_path)
        assert report.counts()["downloaded"] == 2
        assert report.counts()["failed"] == 1
        failed = [r for r in report.results if r.status == "failed"]
        assert failed[0].device == "BAD404"

    def test_batch_input_devices_not_mutated(self, tmp_path, no_sleep, monkeypatch):
        # LEGACY BUG NOT PORTED: legacy wrote _pdf_download_failed into inputs.
        mock_urlopen(monkeypatch, [http_404()])
        device = {"ModelName": "BAD", "manufacturer": "m", "pdf_url": "https://x.test/a.pdf"}
        before = dict(device)
        download_batch([device], tmp_path)
        assert device == before
