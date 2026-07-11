"""Tests for src.review.gallery — written FIRST (CLAUDE.md §2).

Stage 6 is a pure VIEWER of Stage 5's already-saved results: it never
recalculates calibration or re-derives values (a real legacy bug — Stage 6
had a drifted copy of the calibration math). Overlay projection uses the
shared data_to_pixel from src.calibration.ticks with Stage 5's STORED
calibration dict.
"""
import json

import cv2
import numpy as np
import pytest

from src.review.gallery import (
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    bucket_result,
    build_gallery,
    sample_confident,
)
from src.review.review_state import save_state, set_decision


def make_result(device="DEV1", status="ok", review_reason=None,
                confidences=(0.95, 0.9, 0.85), points=True, units="pF"):
    cal = {"x_slope": 10.0, "x_intercept": 50.0, "y_slope": -50.0,
           "y_intercept": 250.0, "x_log": False, "y_log": True}
    curves = []
    for i, conf in enumerate(confidences):
        curve_points = [{"x": 1.0, "y": 10.0}, {"x": 5.0, "y": 100.0}] if points else []
        curves.append({"curve_name": f"C{i}", "confidence": conf, "points": curve_points})
    return {
        "device": device,
        "curve_type": "capacitance_vs_vds",
        "source_image": f"{device}__fig.png",
        "status": status,
        "review_reason": review_reason,
        "duplicates_removed": 0,
        "calibration": None if status == "needs_review" else cal,
        "curves": curves,
        "units": units,
    }


# ------------------------------------------------------------------- bucketing

def test_needs_review_bucketed_flagged_even_with_high_confidence():
    result = make_result(status="needs_review", review_reason="units_undetected",
                         confidences=(0.99, 0.99, 0.99))
    assert bucket_result(result) == "needs_review"


def test_ok_with_low_min_confidence_bucketed_low_confidence():
    result = make_result(confidences=(0.95, 0.9, DEFAULT_LOW_CONFIDENCE_THRESHOLD - 0.01))
    assert bucket_result(result) == "low_confidence"


def test_ok_at_exact_threshold_is_confident():
    result = make_result(confidences=(DEFAULT_LOW_CONFIDENCE_THRESHOLD,) * 3)
    assert bucket_result(result) == "confident"


def test_ok_above_threshold_is_confident():
    assert bucket_result(make_result(confidences=(0.95, 0.9, 0.85))) == "confident"


def test_custom_threshold_respected():
    result = make_result(confidences=(0.95, 0.9, 0.85))
    assert bucket_result(result, threshold=0.96) == "low_confidence"


def test_ok_with_no_curves_treated_as_low_confidence():
    result = make_result(confidences=())
    assert bucket_result(result) == "low_confidence"


# -------------------------------------------------------------------- sampling

def test_sample_confident_none_returns_everything():
    items = [make_result(device=f"D{i}") for i in range(10)]
    assert sample_confident(items, None) == items


def test_sample_confident_caps_count():
    items = [make_result(device=f"D{i}") for i in range(10)]
    sampled = sample_confident(items, 4)
    assert len(sampled) == 4


def test_sample_confident_is_deterministic():
    items = [make_result(device=f"D{i}") for i in range(10)]
    a = sample_confident(items, 4)
    b = sample_confident(items, 4)
    assert [r["device"] for r in a] == [r["device"] for r in b]


def test_sample_confident_larger_than_population_returns_all():
    items = [make_result(device=f"D{i}") for i in range(3)]
    assert len(sample_confident(items, 10)) == 3


# ------------------------------------------------------------- build_gallery

def write_stage5_batch(stage5_dir, results):
    stage5_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        (stage5_dir / f"{result['device']}.json").write_text(
            json.dumps(result), encoding="utf-8")


def write_image_for(images_dir, result, size=(300, 400)):
    images_dir.mkdir(parents=True, exist_ok=True)
    img = np.full((size[0], size[1], 3), 255, dtype=np.uint8)
    cv2.imwrite(str(images_dir / result["source_image"]), img)


def test_build_gallery_groups_and_counts(tmp_path):
    results = [
        make_result(device="A_CONF"),
        make_result(device="B_LOWC", confidences=(0.5, 0.9, 0.9)),
        make_result(device="C_FLAG", status="needs_review", review_reason="units_undetected"),
    ]
    stage5_dir, images_dir, out_dir = tmp_path / "s5", tmp_path / "img", tmp_path / "out"
    write_stage5_batch(stage5_dir, results)
    for r in results:
        write_image_for(images_dir, r)

    summary = build_gallery(stage5_dir, [images_dir], out_dir)
    assert summary["counts"] == {"needs_review": 1, "low_confidence": 1, "confident": 1}
    html_text = (out_dir / "gallery.html").read_text(encoding="utf-8")
    assert "A_CONF" in html_text and "B_LOWC" in html_text and "C_FLAG" in html_text
    assert "units_undetected" in html_text


def test_build_gallery_sample_size_caps_only_confident(tmp_path):
    results = (
        [make_result(device=f"CONF{i}") for i in range(6)]
        + [make_result(device=f"LOW{i}", confidences=(0.4, 0.9, 0.9)) for i in range(3)]
        + [make_result(device=f"FLAG{i}", status="needs_review",
                       review_reason="x") for i in range(3)]
    )
    stage5_dir, images_dir, out_dir = tmp_path / "s5", tmp_path / "img", tmp_path / "out"
    write_stage5_batch(stage5_dir, results)
    for r in results:
        write_image_for(images_dir, r)

    summary = build_gallery(stage5_dir, [images_dir], out_dir, sample_size=2)
    assert summary["shown"]["confident"] == 2
    # needs_review and low_confidence are ALWAYS shown in full.
    assert summary["shown"]["needs_review"] == 3
    assert summary["shown"]["low_confidence"] == 3


def test_build_gallery_missing_image_flagged_not_crash(tmp_path):
    results = [make_result(device="HASIMG"), make_result(device="NOIMG")]
    stage5_dir, images_dir, out_dir = tmp_path / "s5", tmp_path / "img", tmp_path / "out"
    write_stage5_batch(stage5_dir, results)
    write_image_for(images_dir, results[0])  # only the first has an image

    summary = build_gallery(stage5_dir, [images_dir], out_dir)
    assert "NOIMG__fig.png" in summary["missing_images"]
    html_text = (out_dir / "gallery.html").read_text(encoding="utf-8")
    assert "NOIMG" in html_text  # still listed, flagged as missing


def test_build_gallery_writes_overlay_pngs(tmp_path):
    results = [make_result(device="DEV1")]
    stage5_dir, images_dir, out_dir = tmp_path / "s5", tmp_path / "img", tmp_path / "out"
    write_stage5_batch(stage5_dir, results)
    write_image_for(images_dir, results[0])

    build_gallery(stage5_dir, [images_dir], out_dir)
    overlays = list((out_dir / "overlays").glob("*.png"))
    assert len(overlays) == 1


def test_build_gallery_prefills_existing_decisions(tmp_path):
    results = [make_result(device="DEV1")]
    stage5_dir, images_dir, out_dir = tmp_path / "s5", tmp_path / "img", tmp_path / "out"
    write_stage5_batch(stage5_dir, results)
    write_image_for(images_dir, results[0])
    out_dir.mkdir(parents=True)
    save_state(set_decision({}, "DEV1", "capacitance_vs_vds", "approve"),
               out_dir / "review_state.json")

    build_gallery(stage5_dir, [images_dir], out_dir)
    html_text = (out_dir / "gallery.html").read_text(encoding="utf-8")
    # The approve radio for DEV1 is pre-checked from the loaded state.
    assert "checked" in html_text


def test_build_gallery_empty_input_dir_produces_empty_gallery(tmp_path):
    stage5_dir, images_dir, out_dir = tmp_path / "s5", tmp_path / "img", tmp_path / "out"
    stage5_dir.mkdir()
    images_dir.mkdir()
    summary = build_gallery(stage5_dir, [images_dir], out_dir)
    assert summary["counts"] == {"needs_review": 0, "low_confidence": 0, "confident": 0}
    assert (out_dir / "gallery.html").exists()


def test_cli_run_emits_bucket_counts_to_console(tmp_path):
    # Regression: `python -m src.review.gallery` must not lose its INFO logs
    # to the __main__-logger-name trap (the known Session-1 cvat_to_coco CLI
    # defect, PROGRESS.md) — bucket counts are required output (CLAUDE.md §7).
    import subprocess
    import sys

    results = [make_result(device="DEV1")]
    stage5_dir, images_dir, out_dir = tmp_path / "s5", tmp_path / "img", tmp_path / "out"
    write_stage5_batch(stage5_dir, results)
    write_image_for(images_dir, results[0])

    proc = subprocess.run(
        [sys.executable, "-m", "src.review.gallery", str(stage5_dir),
         str(images_dir), "--out", str(out_dir)],
        capture_output=True, text=True, check=True,
    )
    assert "buckets" in proc.stderr


def test_build_gallery_never_recomputes_calibration(tmp_path, monkeypatch):
    # THE stage-6 contract: viewer must not re-derive calibration. If
    # gallery code ever calls derive_calibration, this fails loudly.
    import src.review.gallery as gallery_mod

    def boom(*args, **kwargs):
        raise AssertionError("Stage 6 must never call derive_calibration")

    import src.calibration.ticks as ticks_mod
    monkeypatch.setattr(ticks_mod, "derive_calibration", boom)

    results = [make_result(device="DEV1")]
    stage5_dir, images_dir, out_dir = tmp_path / "s5", tmp_path / "img", tmp_path / "out"
    write_stage5_batch(stage5_dir, results)
    write_image_for(images_dir, results[0])
    build_gallery(stage5_dir, [images_dir], out_dir)  # must not raise
