"""Tests for src/azure_ocr/figure_ocr.py — OCR text attachment, fully mocked."""
import cv2
import numpy as np
import pytest

from src.azure_ocr.figure_ocr import ocr_figures, parse_ocr_result


def write_png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((50, 50, 3), 255, np.uint8))


def read_api_result(lines):
    """Build an Azure Read API analyzeResult with the given (text, poly) lines."""
    return {
        "readResults": [
            {"lines": [{"text": t, "boundingBox": poly} for t, poly in lines]}
        ]
    }


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr("src.azure_ocr.figure_ocr.time.sleep", lambda s: None)


class TestOcrFigures:
    def test_populates_text_and_lines(self, tmp_path, no_sleep):
        write_png(tmp_path / "figures" / "f0.png")
        figs = [{"index": 0, "image_path": "figures/f0.png"}]
        fake = lambda img_bytes: read_api_result(
            [("VDS (V)", [0, 0, 10, 0, 10, 5, 0, 5])]
        )
        out = ocr_figures(figs, tmp_path, ocr_fn=fake)
        assert out[0]["ocr_text"] == ["VDS (V)"]
        assert out[0]["ocr_lines"][0]["bounding_box"] == {"x1": 0, "y1": 0, "x2": 10, "y2": 5}

    def test_idempotent_skips_already_ocrd(self, tmp_path, no_sleep):
        write_png(tmp_path / "figures" / "f0.png")
        figs = [{"index": 0, "image_path": "figures/f0.png", "ocr_text": ["cached"]}]
        calls = []
        ocr_figures(figs, tmp_path, ocr_fn=lambda b: calls.append(1))
        assert calls == []
        assert figs[0]["ocr_text"] == ["cached"]

    def test_budget_cap_respected(self, tmp_path, no_sleep):
        figs = []
        for i in range(3):
            write_png(tmp_path / "figures" / f"f{i}.png")
            figs.append({"index": i, "image_path": f"figures/f{i}.png"})
        calls = []

        def fake(b):
            calls.append(1)
            return read_api_result([("x", [0, 0, 1, 0, 1, 1, 0, 1])])

        ocr_figures(figs, tmp_path, ocr_fn=fake, max_calls=2)
        assert len(calls) == 2
        assert figs[2].get("ocr_text") is None  # left un-OCR'd, not faked

    def test_missing_image_gets_empty_ocr_not_crash(self, tmp_path, no_sleep):
        figs = [{"index": 0, "image_path": "figures/nope.png"}]
        out = ocr_figures(figs, tmp_path, ocr_fn=lambda b: None)
        assert out[0]["ocr_text"] == []
        assert out[0]["ocr_lines"] == []

    def test_ocr_exception_isolated(self, tmp_path, no_sleep):
        write_png(tmp_path / "figures" / "f0.png")
        write_png(tmp_path / "figures" / "f1.png")
        figs = [
            {"index": 0, "image_path": "figures/f0.png"},
            {"index": 1, "image_path": "figures/f1.png"},
        ]

        def flaky(b):
            if not hasattr(flaky, "called"):
                flaky.called = True
                raise RuntimeError("boom")
            return read_api_result([("ok", [0, 0, 1, 0, 1, 1, 0, 1])])

        out = ocr_figures(figs, tmp_path, ocr_fn=flaky)
        assert out[0]["ocr_text"] == []      # failure -> empty, logged
        assert out[1]["ocr_text"] == ["ok"]  # batch continued

    def test_outline_flag_reevaluated_after_ocr(self, tmp_path, no_sleep):
        write_png(tmp_path / "figures" / "f0.png")
        figs = [{"index": 0, "image_path": "figures/f0.png"}]
        fake = lambda b: read_api_result(
            [("Package Outline Dimensions", [0, 0, 1, 0, 1, 1, 0, 1])]
        )
        out = ocr_figures(figs, tmp_path, ocr_fn=fake)
        assert out[0]["is_package_outline"] is True


class TestParseOcrResult:
    def test_empty_result(self):
        assert parse_ocr_result({}) == ([], [])

    def test_blank_lines_dropped(self):
        result = read_api_result([("  ", [0, 0, 1, 0, 1, 1, 0, 1])])
        assert parse_ocr_result(result) == ([], [])

    def test_malformed_polygon_zero_bbox(self):
        result = read_api_result([("text", [1, 2])])
        text, lines = parse_ocr_result(result)
        assert text == ["text"]
        assert lines[0]["bounding_box"] == {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
