"""Tests for src.extraction.classical — written FIRST (CLAUDE.md §2, red phase).

Classical (non-AI) extraction front-end for rdson_vs_tj: a single solid-
colored curve line isolated with plain image processing — no LineFormer, no
GPU. The module under test does not exist yet; this file defines its
contract:

- ``detect_curve_classical(image)`` -> ``List[Detection]`` — the SAME
  ``src.extraction.inference.Detection`` objects the AI path produces
  (score + boolean HxW mask), so everything downstream (dedup,
  skeletonize, naming, calibration, schema) is the existing frozen
  Stage-5 code, reused not reimplemented.
- ``run_classical_pipeline(device, curve_type, source_image, image,
  ocr_lines)`` -> the exact Stage-5 result dict Stage 6 already consumes
  (validated by ``src.extraction.schema.validate_result``), with
  ``expected_curve_count=1`` semantics: anything other than exactly one
  credible curve is ``needs_review`` (quarantine), never guessed.

Owner-specified scenarios covered (2026-07-13):
  1. clean solid curve line
  2. curve crossing a gridline
  3. curve near an axis edge
  4. curve with small gaps/breaks
  5. legend swatches / other colors that aren't the curve
  6. faint/low-contrast curve color
  7. curve touching the image border
  8. blank image / no curve found -> quarantine, not crash
  9. missing/invalid calibration data -> clear error/needs_review
 10. one x-value must never produce two y-values
 11. output values within the chart's axis range

All images are synthetic numpy arrays (no fixtures on disk, no GPU, no
network — CLAUDE.md §2).
"""
import numpy as np
import pytest

from src.extraction.classical import detect_curve_classical, run_classical_pipeline
from src.extraction.schema import validate_result

# ---------------------------------------------------------------- fixtures

IMG_W, IMG_H = 400, 300

# Chart geometry shared by every synthetic figure. Axis calibration
# (via the existing parse_numeric_ticks zoning) needs: x tick labels in the
# bottom 30% band (cy/img_h > 0.70), y tick labels in the left 30% band.
X_AXIS_ROWS = (240, 242)   # black x-axis line
Y_AXIS_COLS = (55, 57)     # black y-axis line
CURVE_BGR = (40, 40, 220)  # solid red — a typical datasheet curve color
FAINT_BGR = (170, 170, 240)  # washed-out low-contrast red
GRID_GRAY = 180

# Tick pixel positions (exact linear fits):
#   x: 25 degC @ col 60 ... 150 degC @ col 350 (2.32 px per degC)
#   y: 0 mOhm @ row 240 ... 80 mOhm @ row 40 (-2.5 px per mOhm)
X_TICKS = [(25, 60), (50, 118), (75, 176), (100, 234), (125, 292), (150, 350)]
Y_TICKS = [(80, 40), (60, 90), (40, 140), (20, 190), (0, 240)]


def ocr_line(text, x1, y1, x2, y2):
    return {"text": text, "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def good_ocr_lines():
    """OCR tick labels matching the drawn chart, plus the y-axis unit label."""
    lines = [
        ocr_line(str(val), col - 15, 265, col + 15, 285) for val, col in X_TICKS
    ]
    lines += [
        ocr_line(str(val), 15, row - 8, 50, row + 8) for val, row in Y_TICKS
    ]
    # Rotated y-axis caption as Azure OCR typically returns it — the unit
    # source for detection ("mΩ" canonicalized to ASCII "mOhm" in output).
    lines.append(ocr_line("RDS(on) [mΩ]", 2, 100, 26, 190))
    return lines


def blank_chart():
    return np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8)


def draw_axes(img):
    img[X_AXIS_ROWS[0]:X_AXIS_ROWS[1], Y_AXIS_COLS[0]:355] = 0
    img[35:X_AXIS_ROWS[1], Y_AXIS_COLS[0]:Y_AXIS_COLS[1]] = 0


def draw_gridlines(img):
    for _, row in Y_TICKS[:-1]:          # horizontal gridlines (skip the axis row)
        img[row, Y_AXIS_COLS[1]:355] = GRID_GRAY
    for _, col in X_TICKS[1:]:           # vertical gridlines (skip the axis col)
        img[40:X_AXIS_ROWS[0], col] = GRID_GRAY


def curve_row(col):
    """The drawn curve: rdson rising with temperature, rows 200 -> 70."""
    frac = (col - 60) / (350 - 60)
    return int(round(200 - 130 * frac))


def draw_curve(img, col_start=60, col_end=350, gaps=(), color=CURVE_BGR,
               thickness=3):
    for col in range(col_start, col_end):
        if any(g0 <= col < g1 for g0, g1 in gaps):
            continue
        row = curve_row(col)
        img[row:row + thickness, col] = color


def standard_chart(**draw_curve_kwargs):
    """Axes + gridlines + one solid curve — the canonical happy-path figure."""
    img = blank_chart()
    draw_axes(img)
    draw_gridlines(img)
    draw_curve(img, **draw_curve_kwargs)
    return img


def run_standard(img, ocr_lines=None):
    return run_classical_pipeline(
        device="DEV1",
        curve_type="rdson_vs_tj",
        source_image="fig.png",
        image=img,
        ocr_lines=good_ocr_lines() if ocr_lines is None else ocr_lines,
    )


# ------------------------------------------------- detect_curve_classical

class TestDetectCurveClassical:
    def test_clean_curve_yields_single_detection_with_valid_score_and_mask(self):
        # Scenario 1: clean solid curve line.
        img = standard_chart()
        detections = detect_curve_classical(img)
        assert len(detections) == 1
        det = detections[0]
        assert det.mask.dtype == np.bool_
        assert det.mask.shape == (IMG_H, IMG_W)
        assert 0.0 < det.score <= 1.0
        # The mask must actually cover the drawn curve, end to end.
        for col in (60, 150, 250, 349):
            assert det.mask[:, col].any(), f"curve pixel at col {col} not in mask"

    def test_gridline_crossing_keeps_one_detection_and_excludes_gridline_pixels(self):
        # Scenario 2: the curve crosses gray gridlines; the gridlines are
        # background, not curve — they must be neither absorbed nor allowed
        # to split the detection in two.
        img = standard_chart()
        detections = detect_curve_classical(img)
        assert len(detections) == 1
        det = detections[0]
        # Gridline-only pixels far from the curve stay out of the mask:
        # top horizontal gridline (row 40) where the curve is near row 182,
        assert not det.mask[40, 100:110].any()
        # and a vertical gridline (col 118) well below the curve (row ~174).
        assert not det.mask[225:239, 118].any()

    def test_curve_near_axis_edge_does_not_absorb_axis_pixels(self):
        # Scenario 3: a low, flat curve hugging the x-axis (2 px above it).
        img = blank_chart()
        draw_axes(img)
        img[236:238, 60:350] = CURVE_BGR
        detections = detect_curve_classical(img)
        assert len(detections) == 1
        det = detections[0]
        # The black axis lines must not leak into the curve mask.
        black = img.sum(axis=2) < 100
        assert not (det.mask & black).any()
        assert not det.mask[X_AXIS_ROWS[0]:X_AXIS_ROWS[1], :].any()

    def test_small_gaps_still_one_detection_spanning_full_extent(self):
        # Scenario 4: small breaks (compression/anti-aliasing dropouts) in a
        # solid curve are bridged — one curve, not three fragments.
        img = standard_chart(gaps=((150, 153), (220, 223), (280, 283)))
        detections = detect_curve_classical(img)
        assert len(detections) == 1
        det = detections[0]
        assert det.mask[:, 65].any()    # before the first gap
        assert det.mask[:, 345].any()   # after the last gap

    def test_legend_swatch_and_text_blob_are_not_detected_as_curves(self):
        # Scenario 5: a short legend swatch (same color as the curve) and a
        # dark text blob must not become detections or join the curve mask.
        img = standard_chart()
        img[50:53, 280:292] = CURVE_BGR   # legend swatch: 12 px, way too short
        img[48:58, 296:330] = 60          # dark "Tj = 25degC"-style text blob
        detections = detect_curve_classical(img)
        assert len(detections) == 1
        assert not detections[0].mask[45:62, 275:335].any()

    def test_faint_low_contrast_curve_is_still_detected(self):
        # Scenario 6: washed-out scan/print colors still count as a curve.
        img = blank_chart()
        draw_axes(img)
        draw_gridlines(img)
        draw_curve(img, color=FAINT_BGR)
        detections = detect_curve_classical(img)
        assert len(detections) == 1
        assert detections[0].mask[:, 200].any()

    def test_curve_touching_image_border_is_detected_without_crash(self):
        # Scenario 7: the crop sometimes clips the plot; a curve running to
        # the image edge must not crash border-sensitive morphology.
        img = blank_chart()
        for col in range(0, IMG_W):
            row = 5 + int(round((250 - 5) * (1 - col / (IMG_W - 1))))
            img[row:min(row + 3, IMG_H), col] = CURVE_BGR
        detections = detect_curve_classical(img)
        assert len(detections) == 1
        assert detections[0].mask[:, 0].any()
        assert detections[0].mask[:, IMG_W - 1].any()

    def test_blank_image_returns_no_detections_without_crash(self):
        # Scenario 8 (detector half): nothing to find -> empty list, no throw.
        assert detect_curve_classical(blank_chart()) == []

    def test_two_distinct_curves_are_both_returned_never_silently_picked(self):
        # Never-guess rule: if the chart actually has two long curves (e.g.
        # two Vgs conditions), BOTH come back — the pipeline's count gate
        # quarantines the figure; the detector must not pick a winner.
        img = standard_chart()
        for col in range(60, 350):  # second, blue curve well below the first
            row = curve_row(col) + 60
            img[row:row + 3, col] = (220, 40, 40)
        detections = detect_curve_classical(img)
        assert len(detections) == 2


# ---------------------------------------------- run_classical_pipeline

class TestRunClassicalPipeline:
    def test_ok_result_matches_stage6_schema_and_names_curve_rdson(self):
        # The output contract IS the AI path's contract: Stage 6's gallery
        # and Stage 7's orchestrator consume this dict unchanged.
        result = run_standard(standard_chart())
        validate_result(result)  # raises if anything is off-schema
        assert result["status"] == "ok"
        assert result["review_reason"] is None
        assert result["device"] == "DEV1"
        assert result["curve_type"] == "rdson_vs_tj"
        assert result["source_image"] == "fig.png"
        assert result["calibration"] is not None
        assert len(result["curves"]) == 1
        curve = result["curves"][0]
        assert curve["curve_name"] == "rdson"
        assert 0.0 < curve["confidence"] <= 1.0
        assert curve["points"], "ok result must carry traced points"

    def test_ok_result_units_detected_as_mohm(self):
        # Unit comes from the y-axis label zone only, canonicalized to
        # ASCII "mOhm" (Windows-console-safe; Ω never round-trips reliably).
        result = run_standard(standard_chart())
        assert result["units"] == "mOhm"

    def test_one_x_value_never_maps_to_two_y_values(self):
        # Scenario 10: even with a vertical jog in the drawn line (thick
        # marker, retrace artifact), each x appears exactly once — the
        # legacy single-valued-x tracing bug must stay dead.
        img = standard_chart()
        jog_top = curve_row(200)
        img[jog_top:jog_top + 40, 200:203] = CURVE_BGR  # vertical segment
        result = run_standard(img)
        assert result["status"] == "ok"
        xs = [p["x"] for p in result["curves"][0]["points"]]
        assert all(b > a for a, b in zip(xs, xs[1:])), \
            "x values must be strictly increasing (one y per x)"

    def test_extracted_values_stay_within_axis_tick_ranges(self):
        # Scenario 11: engineering values must land inside the calibrated
        # axis span (25..150 degC, 0..80 mOhm), small tolerance for the
        # 3-px stroke thickness.
        result = run_standard(standard_chart())
        assert result["status"] == "ok"
        points = result["curves"][0]["points"]
        for p in points:
            assert 25 - 2 <= p["x"] <= 150 + 2, f"x out of axis range: {p['x']}"
            assert 0 - 2 <= p["y"] <= 80 + 2, f"y out of axis range: {p['y']}"
        # And the curve genuinely spans the chart, not just a fragment of it.
        xs = [p["x"] for p in points]
        assert min(xs) < 35 and max(xs) > 140

    def test_no_curve_found_flags_needs_review_quarantine_not_crash(self):
        # Scenario 8 (pipeline half): axes and gridlines but no curve ->
        # schema-valid quarantine, never an exception, never a guessed curve.
        img = blank_chart()
        draw_axes(img)
        draw_gridlines(img)
        result = run_standard(img)
        validate_result(result)
        assert result["status"] == "needs_review"
        assert result["review_reason"]
        assert all(not c["points"] for c in result["curves"])

    def test_missing_calibration_ocr_flags_needs_review_with_clear_reason(self):
        # Scenario 9a: a perfectly good curve but no usable tick text ->
        # needs_review that SAYS calibration failed; no silent zeros.
        result = run_standard(standard_chart(), ocr_lines=[])
        validate_result(result)
        assert result["status"] == "needs_review"
        assert "calibration" in result["review_reason"]
        assert result["calibration"] is None

    def test_malformed_ocr_line_errors_loudly_not_silently(self):
        # Scenario 9b: structurally invalid calibration input (an OCR line
        # with no bounding_box) is a caller bug — raise, don't swallow
        # (CLAUDE.md §7: errors are never silently swallowed). Stage 7's
        # orchestrator maps the raise to failed_extraction.
        bad_lines = good_ocr_lines() + [{"text": "100"}]  # no bounding_box
        with pytest.raises((KeyError, ValueError)):
            run_standard(standard_chart(), ocr_lines=bad_lines)
