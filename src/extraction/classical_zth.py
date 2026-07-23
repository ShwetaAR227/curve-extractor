"""Classical (non-AI) Stage-5 extraction front-end for zth_vs_time.

Ported from the legacy ``D:\\Extractor\\5_opencv_extract\\zth_extract.py``
(CLAUDE.md §6 — reference material only, read and re-verified, never
trusted blindly). Owner-approved design (2026-07-23):

Unlike rdson_vs_tj/vgsth_vs_tj (thin wrappers around the shared
:mod:`src.extraction.curve_detection` + :func:`src.extraction.pipeline.
process_detections` core), zth_vs_time genuinely needs its own pipeline:
time-unit-aware tick parsing (ns/us/ms/s), auto linear-vs-log axis fitting,
gridline removal tuned for a multi-duty-cycle-curve family, a single-pulse
picker among several detected curves, a printed-Foster-table OCR reader
that skips curve-tracing entirely when it succeeds, and a real physics fit
(2-element Foster RC network) rather than plain point interpolation. Only
the FINAL step is shared: :func:`src.extraction.schema.build_result`.

The one piece of device-level data this needs beyond the standard 5-arg
``run_classical_pipeline(device, curve_type, source_image, image,
ocr_lines)`` call every classical extractor gets from
:mod:`src.orchestrator.live_stages` — the device's ``full_extraction.json``
``tables`` key, for the printed Rth_JC lookup — is read by THIS module
itself via an optional ``stage3_root`` kwarg (unused by live_stages.py's
fixed call site), falling back to the ``LINEFORMER_STAGE3_ROOT`` env var
(the same convention :mod:`src.classification.stage3_loader` established).
Best-effort: if the root/file/table is unavailable for ANY reason, the
pipeline proceeds with no Rth constraint (an unconstrained Foster fit is
the legacy code's own normal, common path), logged, never a crash.

Curve name is always ``"single_pulse"`` — matches the CVAT-annotated
training dataset already collected for this curve type (owner decision).
Units are always ``"K/W"`` (owner decision — no real ambiguity for this
curve type, never detected). Confidence is the Foster fit's r-squared
clamped to ``[0, 1]`` (never negative — the schema requires confidence in
that range); a printed-table read (no fit performed, ground truth) is
always ``1.0``; a total fit failure is ``0.0`` (no information).

No classification-registry entry exists for zth_vs_time yet (the legacy
``score_zth``/``auto_pick_zth_figure`` figure-picker duplicates
:mod:`src.classification.curve_registry`'s job and is deliberately NOT
migrated here — a separate, smaller follow-up task, owner-approved). By
the time this module runs, the figure has already been chosen upstream.

Legacy status strings are remapped onto our two-status system
(``"ok"``/``"needs_review"``), keeping the ORIGINAL descriptive message as
``review_reason`` text. One deliberate remap (not a literal 1:1 port):
legacy's "skip_fit" case — a printed Rth value that disagrees with the
observed late-time curve value — sets their own status to "clean" despite
saying in the same breath "curve trace unreliable, tau values not
extracted". This module maps that to ``needs_review`` instead (the message
itself describes something a reviewer should see), keeping their exact
warning text as the review_reason.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import curve_fit

from src.common.log import get_logger
from src.extraction.schema import build_result

logger = get_logger(__name__)

STAGE3_ROOT_ENV_VAR = "LINEFORMER_STAGE3_ROOT"
CURVE_NAME = "single_pulse"
UNITS = "K/W"

# ---------------------------------------------------------------------------
# Tick parsing (unit-aware) — ported from zth_extract.py, unchanged logic.
# ---------------------------------------------------------------------------

_PLAIN_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_SI_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(ns|us|\u00b5s|ms|s)$", re.IGNORECASE)
_EXP_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*[eE]\s*(-?\d+)$|^10\s*\^\s*(-?\d+)$")
_SUPER_DIGIT_MAP = {"\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3",
                    "\u2074": "4", "\u2075": "5", "\u2076": "6", "\u2077": "7",
                    "\u2078": "8", "\u2079": "9", "\u207b": "-"}
_SI_FACTORS = {"ns": 1e-9, "us": 1e-6, "\u00b5s": 1e-6, "ms": 1e-3, "s": 1.0}


def _normalize_super(text: str) -> str:
    return "".join(_SUPER_DIGIT_MAP.get(ch, ch) for ch in text)


def parse_tick(text: str, axis: str = "x") -> Optional[float]:
    """Parse OCR text as a numeric tick value (time-unit-aware on X). See
    the module docstring — ported from zth_extract.py's own ``parse_tick``.
    """
    if not text:
        return None
    t = text.strip().replace("\u2212", "-").replace(",", ".")
    t = _normalize_super(t)
    m = re.match(r"^10([0-9])$", t)
    if m:
        try:
            return 10.0 ** int(m.group(1))
        except ValueError:
            pass
    if _PLAIN_RE.match(t):
        try:
            return float(t)
        except ValueError:
            return None
    if axis == "x":
        m = _SI_RE.match(t)
        if m:
            return float(m.group(1)) * _SI_FACTORS[m.group(2).lower().replace("\u00b5", "u").replace("us", "us")]
    t2 = t.replace("\u00b5", "u")
    if axis == "x":
        m = _SI_RE.match(t2)
        if m:
            return float(m.group(1)) * _SI_FACTORS[m.group(2).lower()]
    m = _EXP_RE.match(t)
    if m:
        if m.group(1) and m.group(2):
            try:
                return float(m.group(1)) * (10 ** int(m.group(2)))
            except ValueError:
                return None
        if m.group(3):
            try:
                return 10 ** int(m.group(3))
            except ValueError:
                return None
    m = re.match(r"^10\s*-\s*(\d+)$", t)
    if m:
        try:
            return 10 ** (-int(m.group(1)))
        except ValueError:
            return None
    return None


def parse_axis_ticks(
    ocr_lines: Sequence[Dict[str, Any]], img_w: float, img_h: float
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Pull axis tick candidates from OCR lines: x-ticks (time, bottom 20%
    of the figure), y-ticks (left 20%). Ported unchanged."""
    x_ticks, y_ticks = [], []
    for line in ocr_lines:
        bb = line.get("bounding_box") or {}
        if not bb:
            continue
        cx = (bb["x1"] + bb["x2"]) / 2.0
        cy = (bb["y1"] + bb["y2"]) / 2.0
        norm_cx = cx / max(img_w, 1)
        norm_cy = cy / max(img_h, 1)
        text = line.get("text", "")
        if norm_cy > 0.80:
            v = parse_tick(text, axis="x")
            if v is not None:
                x_ticks.append((v, cx))
        elif norm_cx < 0.20:
            v = parse_tick(text, axis="y")
            if v is not None:
                y_ticks.append((v, cy))
    return x_ticks, y_ticks


# ---------------------------------------------------------------------------
# Auto-scale axis fit — ported unchanged.
# ---------------------------------------------------------------------------

def _fit_linear(ticks: Sequence[Tuple[float, float]], min_n: int = 3) -> Optional[Dict[str, Any]]:
    if len(ticks) < min_n:
        return None
    pts = list(ticks)
    for _ in range(len(ticks)):
        if len(pts) < min_n:
            break
        vals = np.array([t[0] for t in pts])
        pos = np.array([t[1] for t in pts])
        if len(set(vals.tolist())) < 2:
            return None
        A = np.vstack([vals, np.ones_like(vals)]).T
        slope, intercept = np.linalg.lstsq(A, pos, rcond=None)[0]
        pred = slope * vals + intercept
        resid = np.abs(pos - pred)
        med = float(np.median(resid))
        worst_i = int(np.argmax(resid))
        if med > 0 and resid[worst_i] > 4 * med and resid[worst_i] > 8:
            pts.pop(worst_i)
            continue
        return {
            "slope": float(slope), "intercept": float(intercept),
            "median_resid": med,
            "used": [(float(p[0]), float(p[1])) for p in pts],
        }
    return None


def _fit_log(ticks: Sequence[Tuple[float, float]], min_n: int = 3) -> Optional[Dict[str, Any]]:
    pos_ticks = [(v, px) for v, px in ticks if v is not None and v > 0]
    if len(pos_ticks) < min_n:
        return None
    log_ticks = [(float(np.log10(v)), px) for v, px in pos_ticks]
    fit = _fit_linear(log_ticks, min_n)
    if fit is None:
        return None
    used_log_vals = sorted(v for v, _ in fit["used"])
    decade_span = used_log_vals[-1] - used_log_vals[0]
    fit["decade_span"] = float(decade_span)
    fit["used_data"] = [(10 ** v, px) for v, px in fit["used"]]
    return fit


def fit_axis_auto(ticks: Sequence[Tuple[float, float]], axis_label: str = "?") -> Optional[Dict[str, Any]]:
    """Try linear and log fits; pick whichever fits better, preferring log
    only when positive ticks span >= 2 decades. Ported unchanged."""
    lin = _fit_linear(ticks)
    log = _fit_log(ticks)
    if lin is None and log is None:
        return None

    def _norm_resid(fit, used):
        if not used:
            return 1e9
        px_span = max(p for _, p in used) - min(p for _, p in used)
        return fit["median_resid"] / max(px_span, 1.0)

    if log is not None and log.get("decade_span", 0) >= 2.0:
        lin_score = _norm_resid(lin, lin["used"]) if lin else 1e9
        log_score = _norm_resid(log, log["used"])
        if log_score <= lin_score * 1.2:
            return {
                "scale": "log", "slope": log["slope"], "intercept": log["intercept"],
                "median_resid": log["median_resid"], "decade_span": log["decade_span"],
                "used": log["used_data"],
            }
    if lin is not None:
        return {
            "scale": "linear", "slope": lin["slope"], "intercept": lin["intercept"],
            "median_resid": lin["median_resid"], "used": lin["used"],
        }
    if log is not None:
        return {
            "scale": "log", "slope": log["slope"], "intercept": log["intercept"],
            "median_resid": log["median_resid"], "decade_span": log["decade_span"],
            "used": log["used_data"],
        }
    return None


def derive_calibration_zth(fig_meta: Dict[str, Any], img_w: float, img_h: float) -> Optional[Dict[str, Any]]:
    """Derive zth's own calibration dict (x_scale/y_scale as 'log'|'linear'
    strings, plus fit diagnostics) — ported unchanged."""
    ocr_lines = fig_meta.get("ocr_lines", [])
    x_ticks, y_ticks = parse_axis_ticks(ocr_lines, img_w, img_h)
    x_fit = fit_axis_auto(x_ticks, "x")
    y_fit = fit_axis_auto(y_ticks, "y")
    if x_fit is None or y_fit is None:
        return None
    used_x_px = sorted(p for _, p in x_fit["used"])
    used_y_px = sorted(p for _, p in y_fit["used"])
    return {
        "plot_bbox": {
            "left": used_x_px[0], "right": used_x_px[-1],
            "top": used_y_px[0], "bottom": used_y_px[-1],
        },
        "x_scale": x_fit["scale"], "x_slope": x_fit["slope"], "x_intercept": x_fit["intercept"],
        "x_median_resid": x_fit["median_resid"], "x_decade_span": x_fit.get("decade_span"),
        "y_scale": y_fit["scale"], "y_slope": y_fit["slope"], "y_intercept": y_fit["intercept"],
        "y_median_resid": y_fit["median_resid"], "y_decade_span": y_fit.get("decade_span"),
        "x_ticks_used": x_fit["used"], "y_ticks_used": y_fit["used"],
    }


def pixel_to_data(px: float, py: float, cal: Dict[str, Any]) -> Tuple[float, float]:
    """zth's own pixel->data inverse (NOT src.calibration.ticks.pixel_to_data
    — a different, module-private calibration dict shape; kept as a plain
    module-level function here purely to mirror the legacy port, never
    imported by any other module, so there is no actual name collision at
    runtime). Ported unchanged, including the 12-decade clamp."""
    if cal["x_scale"] == "log":
        log_x = (px - cal["x_intercept"]) / cal["x_slope"]
        x = 10 ** max(min(log_x, 12.0), -12.0)
    else:
        x = (px - cal["x_intercept"]) / cal["x_slope"]
    if cal["y_scale"] == "log":
        log_y = (py - cal["y_intercept"]) / cal["y_slope"]
        y = 10 ** max(min(log_y, 12.0), -12.0)
    else:
        y = (py - cal["y_intercept"]) / cal["y_slope"]
    return float(x), float(y)


# ---------------------------------------------------------------------------
# CV pipeline (multi-curve detection) — ported unchanged, except the
# threshold/gridline-removal/close steps (inline in legacy process()) are
# split into their own named helper for testability. Same math.
# ---------------------------------------------------------------------------

def _clean_for_clustering(crop_bgr: np.ndarray) -> np.ndarray:
    """Threshold + gridline removal + gap-close on a plot-bbox crop. Same
    math as the inline block in legacy zth_extract.py's ``process()``."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    h, w = binary.shape[:2]
    grid_len_h = max(int(w * 0.30), 30)
    grid_len_v = max(int(h * 0.30), 30)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (grid_len_h, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, grid_len_v))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    grid_mask = cv2.bitwise_or(horiz_lines, vert_lines)
    cleaned = cv2.subtract(binary, grid_mask)
    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    return cleaned


def cluster_into_curves_zth(
    cleaned: np.ndarray, min_x_span_ratio: float = 0.30,
    max_density: float = 0.20, min_area: int = 30,
) -> List[Dict[str, Any]]:
    """Connected-component clustering, NO top-N truncation (every credible
    curve-shaped component is returned). Ported unchanged."""
    crop_w = cleaned.shape[1]
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    items = []
    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        if w / crop_w < min_x_span_ratio:
            continue
        density = area / max(w * h, 1)
        if density > max_density:
            continue
        ys, xs = np.where(labels == lbl)
        items.append({
            "id": int(lbl), "area": int(area), "xs": xs, "ys": ys,
            "mean_y": float(np.mean(ys)), "bbox_x": int(x), "bbox_y": int(y),
            "width": int(w), "height": int(h), "density": float(density),
        })
    items.sort(key=lambda c: -c["area"])
    return items


def trace_curve(cluster: Dict[str, Any], x_step: int = 1) -> List[Tuple[float, float]]:
    """For each x column, the BOTTOM-most (95th percentile) pixel of the
    cluster — pins the trace to the single-pulse line specifically even
    when duty-cycle curves merge into one component. Ported unchanged."""
    xs, ys = cluster["xs"], cluster["ys"]
    if len(xs) == 0:
        return []
    df: Dict[int, List[int]] = {}
    for x, y in zip(xs, ys):
        df.setdefault(int(x), []).append(int(y))
    points = []
    for x in sorted(df.keys()):
        if x % x_step != 0:
            continue
        col_ys = df[x]
        if len(col_ys) >= 5:
            y_bot = float(np.percentile(col_ys, 95))
        else:
            y_bot = float(np.max(col_ys))
        points.append((float(x), y_bot))
    return points


# ---------------------------------------------------------------------------
# Single-pulse picker — ported unchanged.
# ---------------------------------------------------------------------------

_SP_LABEL_RE = re.compile(
    r"single\s*pulse|single-?pulse|^\s*sp\s*$|d\s*=\s*0(?!\.\d)|d\s*=\s*0\.0+",
    re.IGNORECASE,
)
PICKER_V2_WEIGHTS = {"edge": 0.45, "flat": 0.25, "ocr": 0.30}


def _cluster_left_edge_y(cluster, plot_left_local, plot_right_local, frac=0.15):
    L = 0
    R = plot_right_local - plot_left_local
    cutoff = L + frac * (R - L)
    xs = cluster["xs"]
    ys = cluster["ys"]
    mask = xs <= cutoff
    if not np.any(mask):
        return None
    return float(np.mean(ys[mask]))


def _cluster_flat_run_right(cluster, plot_w, half_frac=0.5, tol_y_frac=0.10, min_tol_px=3.0):
    xs = cluster["xs"]
    ys = cluster["ys"]
    if len(xs) == 0 or plot_w <= 0:
        return 0.0
    right_start = plot_w * (1.0 - half_frac)
    df: Dict[int, List[int]] = {}
    for x, y in zip(xs.tolist(), ys.tolist()):
        if x >= right_start:
            df.setdefault(int(x), []).append(int(y))
    sorted_xs = sorted(df.keys())
    if len(sorted_xs) < 4:
        return 0.0
    trace_y = [float(np.mean(df[x])) for x in sorted_xs]
    y_range = max(max(trace_y) - min(trace_y), 1.0)
    tol_px = max(tol_y_frac * y_range, min_tol_px)
    best_run = 0
    run_start_i = 0
    for i in range(1, len(trace_y)):
        if abs(trace_y[i] - trace_y[run_start_i]) > tol_px:
            run_len = sorted_xs[i - 1] - sorted_xs[run_start_i]
            if run_len > best_run:
                best_run = run_len
            run_start_i = i
    final_run = sorted_xs[-1] - sorted_xs[run_start_i]
    if final_run > best_run:
        best_run = final_run
    half_w = plot_w * half_frac
    return float(best_run) / max(half_w, 1.0)


def _ocr_label_for_cluster(cluster, fig_meta, plot_left, plot_top, max_dist_px=80):
    nearby_texts = []
    matched = None
    if len(cluster["xs"]) == 0:
        return None, []
    n = len(cluster["xs"])
    step = max(1, n // 200)
    cx_img = (cluster["xs"][::step] + plot_left).astype(np.int32)
    cy_img = (cluster["ys"][::step] + plot_top).astype(np.int32)
    for line in fig_meta.get("ocr_lines", []):
        text = (line.get("text") or "").strip()
        if not text or len(text) > 30:
            continue
        bb = line.get("bounding_box") or {}
        if not bb:
            continue
        bx = (bb["x1"] + bb["x2"]) / 2.0
        by = (bb["y1"] + bb["y2"]) / 2.0
        dx = cx_img - bx
        dy = cy_img - by
        d = np.sqrt(dx * dx + dy * dy).min()
        if d <= max_dist_px:
            nearby_texts.append(text)
            if _SP_LABEL_RE.search(text):
                matched = text
    return matched, nearby_texts


def pick_single_pulse(clusters, fig_meta, cal, plot_left, plot_top, plot_right):
    """Picker v2: weighted score over edge-y / flat-run-right / OCR
    proximity. Ported unchanged."""
    if not clusters:
        return None
    plot_w = plot_right - plot_left

    if len(clusters) == 1:
        c0 = clusters[0]
        matched, _ = _ocr_label_for_cluster(c0, fig_meta, plot_left, plot_top, max_dist_px=80)
        return {
            "cluster": c0, "selection_method": "single_cluster_v2", "label_text": matched,
            "geom_idx": 0, "ocr_idx": 0 if matched else None, "picker_version": 2,
            "scores": {"edge": [1.0], "flat": [1.0], "ocr": [1.0 if matched else 0.0], "total": [1.0]},
        }

    edge_ys = []
    for c in clusters:
        ey = _cluster_left_edge_y(c, 0, plot_w)
        edge_ys.append(ey if ey is not None else c["mean_y"])
    min_ey, max_ey = min(edge_ys), max(edge_ys)
    span_ey = max_ey - min_ey
    edge_score = [(ey - min_ey) / span_ey for ey in edge_ys] if span_ey > 1e-6 else [1.0] * len(clusters)

    flat_runs = [_cluster_flat_run_right(c, plot_w) for c in clusters]
    max_fr = max(flat_runs) if flat_runs else 0.0
    flat_score = [fr / max_fr for fr in flat_runs] if max_fr > 1e-6 else [0.0] * len(clusters)

    cluster_labels = []
    ocr_score = []
    for c in clusters:
        matched, nearby = _ocr_label_for_cluster(c, fig_meta, plot_left, plot_top, max_dist_px=80)
        cluster_labels.append((matched, nearby))
        ocr_score.append(1.0 if matched else 0.0)

    w = PICKER_V2_WEIGHTS
    totals = [
        w["edge"] * edge_score[i] + w["flat"] * flat_score[i] + w["ocr"] * ocr_score[i]
        for i in range(len(clusters))
    ]
    best_idx = int(np.argmax(totals))
    geom_idx = int(np.argmax(edge_ys))

    parts = []
    if edge_score[best_idx] >= 0.99:
        parts.append("edge")
    if flat_score[best_idx] >= 0.50:
        parts.append("flat")
    if ocr_score[best_idx] > 0:
        parts.append("ocr")
    if not parts:
        parts = ["weighted"]
    method = "+".join(parts) + "_v2"
    if best_idx != geom_idx:
        method += "_override"

    out = {
        "cluster": clusters[best_idx], "selection_method": method,
        "label_text": cluster_labels[best_idx][0], "geom_idx": geom_idx,
        "ocr_idx": best_idx if ocr_score[best_idx] > 0 else None,
        "edge_ys": edge_ys, "picker_version": 2,
        "scores": {"edge": edge_score, "flat": flat_score, "ocr": ocr_score, "total": totals},
    }
    if best_idx != geom_idx:
        out["warning"] = (
            f"v2 picked cluster {best_idx} (total={totals[best_idx]:.3f}) "
            f"over geometric pick {geom_idx} (total={totals[geom_idx]:.3f})"
        )
    return out


# ---------------------------------------------------------------------------
# Foster RC table OCR parser (PDF-wins path) — ported unchanged.
# ---------------------------------------------------------------------------

_DEG_C = r"(?:[\u00b0\u00bao]?\s*c|\u2103)"
_RI_HEADER_RE = re.compile(
    rf"\bri\s*\(\s*{_DEG_C}\s*/\s*w\s*\)|"
    r"\bri\s*\(\s*k\s*/\s*w\s*\)|"
    r"\br\s*i\s*\[.*?w\s*\]|"
    r"^r\s*i\s*$",
    re.IGNORECASE,
)
_TAU_HEADER_RE = re.compile(
    r"[t\u03c4]\s*[li\u03c4]?\s*\(\s*sec\s*\)|"
    r"\btau\s*\(\s*s(?:ec)?\s*\)|"
    r"[t\u03c4]\s*[li\u03c4]?\s*\(\s*s\s*\)|"
    r"^\s*[t\u03c4]\s*[li\u03c4]?\s*$|"
    r"^\s*tau\s*$",
    re.IGNORECASE,
)
_NUMERIC_VALUE_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$")


def _ocr_line_centers(fig_meta):
    out = []
    for ln in fig_meta.get("ocr_lines", []):
        text = (ln.get("text") or "").strip()
        if not text:
            continue
        bb = ln.get("bounding_box") or {}
        if not bb:
            continue
        cx = (bb["x1"] + bb["x2"]) / 2.0
        cy = (bb["y1"] + bb["y2"]) / 2.0
        out.append((cx, cy, text))
    return out


def _try_parse_value(text):
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^[-_.\s]+|[-_.\s]+$", "", t)
    t = t.replace("O", "0").replace("o", "0") if re.search(r"[Oo]", t) and re.search(r"\d", t) else t
    if _NUMERIC_VALUE_RE.match(t):
        try:
            return float(t)
        except ValueError:
            return None
    return None


def parse_foster_table_from_ocr(fig_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find an embedded Foster RC table in figure OCR. Ported unchanged."""
    lines = _ocr_line_centers(fig_meta)
    if not lines:
        return None

    ri_headers = [(cx, cy, t) for cx, cy, t in lines if _RI_HEADER_RE.search(t)]
    tau_headers = [(cx, cy, t) for cx, cy, t in lines if _TAU_HEADER_RE.search(t)]
    if not ri_headers or not tau_headers:
        return None

    best = None
    for rcx, rcy, _rt in ri_headers:
        for tcx, tcy, _tt in tau_headers:
            if tcx <= rcx + 5:
                continue
            dy = abs(rcy - tcy)
            if dy > 30:
                continue
            score = dy + abs(tcx - rcx) * 0.01
            if best is None or score < best[0]:
                best = (score, rcx, rcy, tcx, tcy)
    if best is None:
        return None
    _, ri_x, ri_y, tau_x, tau_y = best
    header_y = (ri_y + tau_y) / 2.0
    col_tol = max((tau_x - ri_x) * 0.45, 30.0)

    ri_col, tau_col = [], []
    for cx, cy, t in lines:
        if cy <= header_y + 8:
            continue
        v = _try_parse_value(t)
        if v is None or v <= 0:
            continue
        if abs(cx - ri_x) <= col_tol and abs(cx - ri_x) < abs(cx - tau_x):
            ri_col.append((cy, v, t))
        elif abs(cx - tau_x) <= col_tol:
            tau_col.append((cy, v, t))

    if len(ri_col) < 2 or len(tau_col) < 2:
        return None

    ri_col.sort(key=lambda r: r[0])
    tau_col.sort(key=lambda r: r[0])

    pairs = []
    used_tau = set()
    for cy_r, val_r, _rt in ri_col:
        best_j, best_dy = None, 9999
        for j, (cy_t, _vt, _tt) in enumerate(tau_col):
            if j in used_tau:
                continue
            dy = abs(cy_t - cy_r)
            if dy < best_dy:
                best_dy, best_j = dy, j
        if best_j is None or best_dy > 35:
            continue
        used_tau.add(best_j)
        pairs.append({"R": val_r, "tau": tau_col[best_j][1], "_cy": cy_r})

    if len(pairs) < 2:
        return None

    pairs.sort(key=lambda p: p["_cy"])

    for p in pairs:
        if not (1e-6 <= p["R"] <= 1e3):
            return None
        if not (1e-9 <= p["tau"] <= 1e3):
            return None

    rc_pairs = []
    for i, p in enumerate(pairs, 1):
        rc_pairs.append({"R": p["R"], "tau": p["tau"], "row": i})

    rth_steady = sum(p["R"] for p in rc_pairs)

    return {
        "rc_pairs": rc_pairs, "n_pairs": len(rc_pairs),
        "ri_col_x": float(ri_x), "tau_col_x": float(tau_x), "header_y": float(header_y),
        "rth_steady": float(rth_steady),
    }


# ---------------------------------------------------------------------------
# Foster RC fit (1-element) — ported unchanged.
# ---------------------------------------------------------------------------

def fit_foster(
    xs: Sequence[float], ys: Sequence[float], rth_constraint: Optional[float] = None,
) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    """1st-order Foster fit: Z(t) = R * (1 - exp(-t/tau)). Ported unchanged."""
    if len(xs) < 6:
        return None, None
    xs_a = np.asarray(xs, dtype=float)
    ys_a = np.asarray(ys, dtype=float)
    mask = (xs_a > 0) & np.isfinite(ys_a) & (ys_a >= 0)
    xs_a, ys_a = xs_a[mask], ys_a[mask]
    if len(xs_a) < 6:
        return None, None
    y_max = float(np.max(ys_a))
    x_max = float(np.max(xs_a))

    if rth_constraint is not None and rth_constraint > 0:
        rth = float(rth_constraint)

        def _foster_fixed_r(t, tau):
            return rth * (1.0 - np.exp(-t / tau))

        p0 = [max(x_max * 0.3, 1e-3)]
        bounds = ([1e-9], [1e3])
        try:
            popt, _ = curve_fit(_foster_fixed_r, xs_a, ys_a, p0=p0, bounds=bounds, maxfev=5000)
        except Exception:
            return None, None
        tau = float(popt[0])
        r = rth
        pred = _foster_fixed_r(xs_a, *popt)
    else:
        def _foster_1(t, r, tau):
            return r * (1.0 - np.exp(-t / tau))

        p0 = [max(y_max, 1e-4), max(x_max * 0.3, 1e-3)]
        bounds = ([1e-6, 1e-9], [1e3, 1e3])
        try:
            popt, _ = curve_fit(_foster_1, xs_a, ys_a, p0=p0, bounds=bounds, maxfev=5000)
        except Exception:
            return None, None
        r, tau = float(popt[0]), float(popt[1])
        pred = _foster_1(xs_a, *popt)

    ss_res = float(np.sum((ys_a - pred) ** 2))
    ss_tot = float(np.sum((ys_a - np.mean(ys_a)) ** 2))
    r2_score = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return {"r1": r, "tau1": tau}, float(r2_score)


# ---------------------------------------------------------------------------
# Manufacturer Rth_JC table parser — ported unchanged.
# ---------------------------------------------------------------------------

_RTH_SYM_RE = re.compile(
    r"^\s*r\s*[_(]?\s*th\s*[_]?\s*(?:\(?\s*j\s*[-_]?\s*c\s*\)?|jc)\s*\)?\s*$", re.IGNORECASE,
)
_RTH_DESC_RE = re.compile(
    r"thermal\s+resistance.*(?:junction.*case|j[\s\-]+c\b)|junction[\s\-]+case", re.IGNORECASE,
)
_RTH_JA_RE = re.compile(r"junction.*ambient|j[\s\-]?a\b|rthja", re.IGNORECASE)
_KW_RE = re.compile(r"(?:k|[\u00b0\u00ba\u2103]\s*c)\s*[/]\s*w", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _parse_num(s):
    if not s:
        return None
    s = s.replace(",", ".").strip()
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def find_rth_jc_in_full_extraction(full_extraction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Scan tables for junction-to-case thermal-resistance rows. Ported
    unchanged."""
    out = []
    for ti, t in enumerate(full_extraction.get("tables") or []):
        col_typ = col_max = col_unit = col_symbol = col_param = None
        for h in (t.get("headers") or []):
            if not isinstance(h, dict):
                continue
            text = (h.get("text") or "").strip().lower()
            col = h.get("col")
            if col is None:
                continue
            if "typ" in text and col_typ is None:
                col_typ = col
            elif "max" in text and col_max is None:
                col_max = col
            elif text in ("unit", "units") and col_unit is None:
                col_unit = col
            elif "symbol" in text and col_symbol is None:
                col_symbol = col
            elif "parameter" in text and col_param is None:
                col_param = col
        if col_unit is None:
            col_unit = t.get("unit_column_index")

        for row in (t.get("rows") or []):
            cells = row.get("cells") if isinstance(row, dict) else row
            if not cells:
                continue
            by_col = {}
            for c in cells:
                if isinstance(c, dict):
                    col = c.get("col")
                    if col is not None:
                        by_col[col] = (c.get("text") or "").strip()

            sym = by_col.get(col_symbol, "") if col_symbol is not None else ""
            param = by_col.get(col_param, "") if col_param is not None else ""
            unit = by_col.get(col_unit, "") if col_unit is not None else ""
            full_row = " | ".join(by_col.get(i, "") for i in sorted(by_col))

            sym_compact = sym.lower().replace(" ", "")
            sym_match = bool(_RTH_SYM_RE.search(sym)) or "rthjc" in sym_compact
            desc_match = bool(_RTH_DESC_RE.search(param)) or bool(_RTH_DESC_RE.search(full_row))
            if not (sym_match or desc_match):
                continue
            if "rthjc" not in sym_compact:
                if _RTH_JA_RE.search(sym) or _RTH_JA_RE.search(param):
                    continue
            if not (_KW_RE.search(unit) or _KW_RE.search(full_row)):
                continue
            typ = _parse_num(by_col.get(col_typ, "")) if col_typ is not None else None
            mx = _parse_num(by_col.get(col_max, "")) if col_max is not None else None
            if typ is None and mx is None:
                continue
            out.append({"typ": typ, "max": mx, "source_table": ti, "raw_row": full_row[:200]})
    return out[0] if out else None


def pick_rth_constraint(full_extraction: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    """Prefer Rth_max over Rth_typ. Ported unchanged."""
    found = find_rth_jc_in_full_extraction(full_extraction)
    if not found:
        return None, None
    if found["max"] is not None and found["max"] > 0:
        return found["max"], "table_max"
    if found["typ"] is not None and found["typ"] > 0:
        return found["typ"], "table_typ"
    return None, None


# ---------------------------------------------------------------------------
# NEW: self-contained Rth-table file read (owner-approved design, 2026-07-23)
# ---------------------------------------------------------------------------

def _read_full_extraction_for_rth(
    device: str, stage3_root: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Best-effort read of ``<stage3_root>/<device>/full_extraction.json``
    for the printed Rth_JC table lookup ONLY. Falls back to the
    ``LINEFORMER_STAGE3_ROOT`` env var when ``stage3_root`` is omitted
    (CLAUDE.md §3 — never a hardcoded path). Any failure (root unset,
    device folder missing, malformed JSON) degrades to ``None`` — this
    piece of data is optional (an unconstrained Foster fit is a normal,
    common outcome), never a crash. Logged either way.
    """
    root = stage3_root or os.environ.get(STAGE3_ROOT_ENV_VAR)
    if not root:
        logger.info(
            "classical_zth(%s): no stage3_root and %s unset — "
            "proceeding without an Rth_JC table constraint", device, STAGE3_ROOT_ENV_VAR,
        )
        return None
    json_path = os.path.join(str(root), device, "full_extraction.json")
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.info(
            "classical_zth(%s): could not read %s (%s) — "
            "proceeding without an Rth_JC table constraint", device, json_path, exc,
        )
        return None


# ---------------------------------------------------------------------------
# Result construction helpers
# ---------------------------------------------------------------------------

def _clamp_confidence(r_squared: Optional[float]) -> float:
    if r_squared is None:
        return 0.0
    return max(0.0, min(1.0, r_squared))


def _needs_review(
    device: str, curve_type: str, source_image: str, reason: str,
    calibration: Optional[Dict[str, Any]] = None,
    points: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, Any]:
    logger.warning("classical_zth(%s, %s): needs_review - %s", device, curve_type, reason)
    curves = [{"curve_name": CURVE_NAME, "confidence": 0.0, "points": points or []}]
    return build_result(
        device=device, curve_type=curve_type, source_image=source_image,
        status="needs_review", review_reason=reason, duplicates_removed=0,
        calibration=calibration, curves=curves, units=None if calibration is None else UNITS,
    )


def _calibration_with_bonus_fields(cal_zth: Dict[str, Any]) -> Dict[str, Any]:
    """Map zth's own calibration dict onto our 6 required fields, keeping
    every extra legacy field as bonus detail alongside them."""
    calibration = dict(cal_zth)
    calibration["x_log"] = cal_zth["x_scale"] == "log"
    calibration["y_log"] = cal_zth["y_scale"] == "log"
    return calibration


def run_classical_pipeline(
    device: str,
    curve_type: str,
    source_image: str,
    image: np.ndarray,
    ocr_lines: Sequence[Dict[str, Any]],
    stage3_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the zth_vs_time classical pipeline (GPU-free): printed-table
    read first, otherwise digitize + physics-fit. See the module docstring
    for the full design.

    Args:
        device: Device identifier.
        curve_type: Registry key (``"zth_vs_time"``).
        source_image: Figure image path/identifier, recorded in the result.
        image: HxWx3 uint8 BGR figure crop.
        ocr_lines: The figure's OCR lines (dict-shaped:
            ``{"text": str, "bounding_box": {"x1","y1","x2","y2"}}``).
        stage3_root: Optional Stage-3 output root for the Rth_JC table
            lookup (falls back to ``LINEFORMER_STAGE3_ROOT``). live_stages.py's
            standard call site never passes this — it's read here, not
            threaded through the shared adapter protocol.

    Returns:
        A schema-validated Stage-5 result dict.
    """
    fig_meta = {"ocr_lines": list(ocr_lines)}

    # ---- PDF-wins path: read the printed Foster RC table first ----
    table = parse_foster_table_from_ocr(fig_meta)
    if table is not None:
        logger.info(
            "classical_zth(%s, %s): printed Foster table found (%d pairs) — "
            "skipping the CV pipeline entirely", device, curve_type, table["n_pairs"],
        )
        curve = {
            "curve_name": CURVE_NAME, "confidence": 1.0, "points": [],
            "extraction_source": "pdf_table",
            "rc_pairs": table["rc_pairs"],
            "rth_jc_steady_state": table["rth_steady"],
        }
        return build_result(
            device=device, curve_type=curve_type, source_image=source_image,
            status="ok", review_reason=None, duplicates_removed=0,
            calibration=None, curves=[curve], units=UNITS,
        )

    # ---- Fallback: CV pipeline (curve digitisation + Foster fit) ----
    img_h, img_w = image.shape[:2]
    cal_zth = derive_calibration_zth(fig_meta, img_w, img_h)
    if cal_zth is None:
        return _needs_review(
            device, curve_type, source_image,
            "calibration_failed: axis calibration failed (insufficient/degenerate tick marks)",
        )
    calibration = _calibration_with_bonus_fields(cal_zth)
    bb = cal_zth["plot_bbox"]
    L, R = int(bb["left"]) + 2, int(bb["right"]) - 2
    T, B = int(bb["top"]) + 2, int(bb["bottom"]) - 2
    if R - L < 50 or B - T < 50:
        return _needs_review(
            device, curve_type, source_image,
            f"plot_bbox_too_small: plot area {R - L}x{B - T}px is too small to trace",
            calibration=calibration,
        )

    axis_x_min, axis_y_top = pixel_to_data(bb["left"], bb["top"], cal_zth)
    axis_x_max, axis_y_bot = pixel_to_data(bb["right"], bb["bottom"], cal_zth)
    axis_x_min, axis_x_max = sorted([axis_x_min, axis_x_max])
    axis_y_min, axis_y_max = sorted([axis_y_bot, axis_y_top])
    calibration["axis_data_range"] = {
        "x_min": axis_x_min, "x_max": axis_x_max, "y_min": axis_y_min, "y_max": axis_y_max,
    }

    crop = image[T:B, L:R].copy()
    cleaned = _clean_for_clustering(crop)
    clusters = cluster_into_curves_zth(cleaned)
    if not clusters:
        return _needs_review(
            device, curve_type, source_image, "no_curves_found: no curve-shaped components detected",
            calibration=calibration,
        )

    pick = pick_single_pulse(clusters, fig_meta, cal_zth, L, T, R)
    if pick is None:
        return _needs_review(
            device, curve_type, source_image,
            "single_pulse_pick_failed: could not select the single-pulse curve",
            calibration=calibration,
        )

    chosen = pick["cluster"]
    local_pts = trace_curve(chosen, x_step=2)
    eng_pts = []
    for x_local, y_local in local_pts:
        x_img = x_local + L
        y_img = y_local + T
        x_data, y_data = pixel_to_data(x_img, y_img, cal_zth)
        eng_pts.append((x_data, y_data))

    if len(eng_pts) < 6:
        return _needs_review(
            device, curve_type, source_image,
            f"too_few_points: fewer than 6 digitized points ({len(eng_pts)} found)",
            calibration=calibration,
        )

    xs = [p[0] for p in eng_pts]
    ys = [p[1] for p in eng_pts]
    sorted_pairs = sorted(zip(xs, ys))
    sxs = np.array([p[0] for p in sorted_pairs])
    sys_ = np.array([p[1] for p in sorted_pairs])
    pos_mask = sys_ > 0
    pos_y = sys_[pos_mask] if pos_mask.any() else sys_
    left_q = float(np.median(pos_y[: max(len(pos_y) // 5, 1)]))
    right_q = float(np.median(pos_y[-max(len(pos_y) // 5, 1):]))
    rise_ratio = right_q / max(left_q, 1e-9)

    full_extraction = _read_full_extraction_for_rth(device, stage3_root)
    rth_constraint, rth_source = pick_rth_constraint(full_extraction or {})

    if rth_constraint is None and (rise_ratio > 1e6 or rise_ratio < 1.0 or right_q > 100.0):
        return _needs_review(
            device, curve_type, source_image,
            f"calibration_disaster: rise_ratio={rise_ratio:.3g} out of [1, 1e6] or "
            f"right_q={right_q:.3g} > 100 K/W — calibration is not physically plausible",
            calibration=calibration,
        )

    fit_constraint = None
    constraint_warning = None
    skip_fit = False
    if rth_constraint is not None:
        scale_ratio = right_q / float(rth_constraint)
        if 0.05 <= scale_ratio <= 3.0:
            fit_constraint = rth_constraint
        else:
            skip_fit = True
            constraint_warning = (
                f"calibration_broken: y_obs_late={right_q:.3g} vs rth_table="
                f"{rth_constraint:.3g} (ratio {scale_ratio:.2g}); "
                f"curve trace unreliable, tau values not extracted"
            )

    if skip_fit:
        # Deliberate remap (see module docstring): legacy calls this
        # "clean", we call it needs_review — the message itself describes
        # something a reviewer should see.
        return _needs_review(
            device, curve_type, source_image, constraint_warning, calibration=calibration,
        )

    fitted_params, r2 = fit_foster(sxs.tolist(), sys_.tolist(), rth_constraint=fit_constraint)
    if fitted_params is None or r2 is None or r2 < 0.5:
        r2_text = f"{r2:.3g}" if r2 is not None else "N/A"
        return _needs_review(
            device, curve_type, source_image,
            f"foster_fit_failed: r_squared={r2_text} (< 0.5 or fit did not converge)",
            calibration=calibration,
            points=[{"x": x, "y": y} for x, y in eng_pts],
        )

    if rth_constraint is not None:
        rth_steady = float(rth_constraint)
        rth_steady_source = rth_source
    else:
        rth_steady = fitted_params["r1"]
        rth_steady_source = "foster_unconstrained"

    curve = {
        "curve_name": CURVE_NAME,
        "confidence": _clamp_confidence(r2),
        "points": [{"x": x, "y": y} for x, y in eng_pts],
        "extraction_source": "curve_fit_v3" if fit_constraint is None else "curve_fit_v3_constrained",
        "fitted_params": fitted_params,
        "r_squared": r2,
        "r_fixed_at_rth_jc": fit_constraint is not None,
        "rth_jc_steady_state": rth_steady,
        "rth_jc_steady_state_source": rth_steady_source,
        "rise_ratio": rise_ratio,
    }
    logger.info(
        "classical_zth(%s, %s): ok, r_squared=%.4f, rth_jc=%.4g", device, curve_type, r2, rth_steady,
    )
    return build_result(
        device=device, curve_type=curve_type, source_image=source_image,
        status="ok", review_reason=None, duplicates_removed=0,
        calibration=calibration, curves=[curve], units=UNITS,
    )
