"""Axis-tick calibration, ported from legacy `cv_curve_extract.py` (CLAUDE.md §6).

Source: ``D:\\Extractor\\5_opencv_extract\\cv_curve_extract.py``, lines
**116-197** (``parse_numeric_ticks``), **200-293** (``fit_axis``), and
**391-402** (``pixel_to_data``) — the canonical version per
``LEGACY_REVIEW.md`` §3, confirmed to be the only implementation in that
family with log-axis support, compound-token parsing, and RANSAC outlier
rejection. Deliberately lifted, not reinvented (CLAUDE.md §1, §6) — do NOT
port from any of the five documented diverged forks (render_cv_gallery.py,
overlay_on_figure.py, level_1 extract.py, igbt_extract.py,
classify_curves/extract/calibrate.py), and do NOT copy the broken 3-tuple
consumer pattern from ``D:\\LineFormerModel\\extract_curves_auto.py``.

``fit_axis`` keeps the documented **4-tuple** return contract:
``(slope, intercept, used, is_log)`` or ``None``.

Behavioral caveats carried over unchanged from the legacy review (not bugs,
documented limitations of the ported algorithm):
1. Tick zoning (bottom 30% / left 30% / tight-corner 15%) assumes a
   bottom x-axis and left y-axis — right-hand or dual axes will mis-bucket.
2. ``inlier_threshold=15.0`` px is absolute, not resolution-scaled.
3. RANSAC is O(n^3) over tick count (fine for realistic tick counts).
4. Log auto-detection wants >=2 positive-valued decades to be reliable.
5. ``plot_bbox`` spans only the outermost *used* ticks, not the true plot
   rectangle.
6. Compound-token even-spacing assumes uniformly spaced values in one OCR
   line — wrong for log-decade tick rows OCR'd as a single token.
"""
import re
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.common.log import get_logger

logger = get_logger(__name__)

NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
MAX_TICK_MAGNITUDE = 9_999_999  # rejects OCR-merged runs like "100120140160180"

# Log-axis exponent repair (T16 — owner-approved deviation from the verbatim
# legacy port, after T15 showed 5/8 real capacitance charts silently
# calibrating as linear): Azure OCR renders superscript decade labels like
# "10^3" as "10 3" (one token with a space), "10³"/"10º" (unicode
# superscript / ordinal chars), or "103" (digits concatenated). The legacy
# parser split "10 3" into two bogus ticks (10 and 3) via the compound-token
# path and read "103" as the value 103 — either way fit_axis then found a
# plausible-looking LINEAR fit through the exponent digits.
_EXPONENT_SPACE_RE = re.compile(r"^10\s+(\d)$")
_SUPERSCRIPT_DIGITS = {
    "⁰": 0, "¹": 1, "²": 2, "³": 3, "⁴": 4,
    "⁵": 5, "⁶": 6, "⁷": 7, "⁸": 8, "⁹": 9,
    "º": 0, "°": 0,  # ordinal/degree chars OCR'd in place of superscript 0
}
_EXPONENT_SUPERSCRIPT_RE = re.compile(
    "^10([" + "".join(_SUPERSCRIPT_DIGITS) + "])$"
)
# Negative-exponent form: "10-1", "10-2" (no space) -> 10^-1, 10^-2. Seen on
# BSS127H6327XTSA2/BSS127IXTSA1/BTS132E3129NKSA1's y-axes (T17/T18). Distinct
# from the "-\s+\d"-style negative-tick collapse above: that only fires with
# a space around the dash; this is a literal "10-d" token, no space.
_EXPONENT_NEGATIVE_RE = re.compile(r"^10-(\d)$")
# Bare concatenated form: "10" + single exponent digit read as one integer
# (100–109, i.e. 10^0..10^9). Ambiguous with a genuine value (e.g. a real
# 104 or a real 100V tick), so it is only reinterpreted when at least
# MIN_BARE_EXPONENT_TICKS ticks on the SAME axis match the pattern (or the
# axis already has unambiguous exponent labels) — a lone 104 or 100 among
# normal ticks stays 104/100. Lower bound is 100, not 101 (T18): bare "100"
# is used as the 10^0 decade label on some real charts (BSS127H6327XTSA2),
# alongside 102/101 on the same axis, so the same corroboration gate now
# covers it too.
_BARE_EXPONENT_MIN, _BARE_EXPONENT_MAX = 100, 109
MIN_BARE_EXPONENT_TICKS = 2
# Mirrors fit_axis's own >=10x value-range-ratio log-preference threshold
# (see prefer_log below) — reused, not reinvented, so the zero-tick-drop
# heuristic stays consistent with fit_axis's own log/linear judgment.
LOG_ZERO_DROP_RATIO = 10.0

Tick = Tuple[float, float]  # (value, pixel_position)
OcrLine = Dict[str, Any]  # {"text": str, "bounding_box": {"x1","y1","x2","y2"}}


def _exponent_value(text: str) -> Optional[float]:
    """Return 10**d if ``text`` is an OCR'd log-decade label, else None."""
    match = _EXPONENT_SPACE_RE.match(text)
    if match:
        return 10.0 ** int(match.group(1))
    match = _EXPONENT_SUPERSCRIPT_RE.match(text)
    if match:
        return 10.0 ** _SUPERSCRIPT_DIGITS[match.group(1)]
    match = _EXPONENT_NEGATIVE_RE.match(text)
    if match:
        return 10.0 ** (-int(match.group(1)))
    return None


def _repair_exponent_ticks(
    ticks: List[Tick], has_exponent_labels: bool, already_resolved: Optional[List[bool]] = None
) -> List[Tick]:
    """Per-axis repair for concatenated exponent labels + log-axis junk ticks.

    Reinterprets bare "10d" values (100–109) as 10**d when the axis shows
    enough evidence of being log-decade labeled (>= MIN_BARE_EXPONENT_TICKS
    such ticks, or >= 1 alongside an unambiguous exponent label). On any
    axis identified as log-decade labeled, non-positive ticks are dropped —
    a "0" cannot be a real tick there, and it blocks fit_axis's log
    detection (which requires all-positive values).

    ``already_resolved[i]`` (parallel to ``ticks``) marks entries already
    produced by explicit exponent-notation parsing (e.g. "10 2" -> 100) —
    excluded from bare-digit reinterpretation so a correctly-resolved 100
    is never re-reinterpreted as 10**0 just because its value happens to
    land in the bare-digit range.
    """
    if already_resolved is None:
        already_resolved = [False] * len(ticks)
    # "100" alone is too likely to be a genuine engineering value (100V,
    # 100pF, ...) to reinterpret on its own -- unlike 101-109, which are
    # essentially never real tick values in this domain. Only fold a bare
    # "100" into the reinterpretation set when an unambiguous 101-109
    # sibling (or explicit exponent-notation evidence) already establishes
    # this axis as log-decade labeled. Without that corroboration, a
    # duplicate/outlier "100" is exactly the kind of point fit_axis's own
    # RANSAC is already designed to reject -- don't pre-empt it (T18
    # regression found on AUIRLU3114Z: reinterpreting two literal "100"
    # duplicates destroyed a tick set RANSAC was correctly handling as-is).
    strong_indices = [
        i for i, (value, _) in enumerate(ticks)
        if not already_resolved[i]
        and value == int(value) and 101 <= value <= _BARE_EXPONENT_MAX
    ]
    if strong_indices or has_exponent_labels:
        weak_indices = [
            i for i, (value, _) in enumerate(ticks)
            if not already_resolved[i] and value == _BARE_EXPONENT_MIN
        ]
        bare_indices = strong_indices + weak_indices
    else:
        bare_indices = strong_indices
    if len(bare_indices) >= MIN_BARE_EXPONENT_TICKS or (
        has_exponent_labels and len(bare_indices) >= 1
    ):
        for i in bare_indices:
            value, pixel = ticks[i]
            ticks[i] = (10.0 ** (value - 100), pixel)
        logger.info(
            "exponent repair: reinterpreted %d bare '10d' tick(s) as powers of ten",
            len(bare_indices),
        )
        has_exponent_labels = True

    if has_exponent_labels:
        dropped = [t for t in ticks if t[0] <= 0]
        if dropped:
            logger.info(
                "exponent repair: dropped %d non-positive tick(s) from a "
                "log-decade-labeled axis: %s", len(dropped), dropped,
            )
        ticks = [t for t in ticks if t[0] > 0]
    return ticks


def _drop_stray_zero_on_log_axis(ticks: List[Tick]) -> List[Tick]:
    """Drop a lone ``0`` tick when the remaining ticks span a wide enough
    positive range to indicate a log-labeled axis a real 0 cannot belong to
    (log10(0) is undefined). Unlike :func:`_repair_exponent_ticks`, this
    fires without needing an explicit exponent-notation label — e.g. plain
    "0", "1", "10", "100" tick text, no superscript/space-exponent form.

    Only triggers when every OTHER tick on the axis is positive (a mixed
    +/- range is never a log axis, so 0 is left alone) and their
    max/min ratio is >= LOG_ZERO_DROP_RATIO — the same threshold
    :func:`fit_axis` itself uses to prefer a log fit, so this heuristic
    can't disagree with fit_axis's own judgment once applied.
    """
    zero_ticks = [t for t in ticks if t[0] == 0]
    other = [t for t in ticks if t[0] != 0]
    if not zero_ticks or len(other) < 2:
        return ticks
    if not all(value > 0 for value, _ in other):
        return ticks
    ratio = max(v for v, _ in other) / min(v for v, _ in other)
    if ratio >= LOG_ZERO_DROP_RATIO:
        logger.info(
            "dropped %d stray zero tick(s) from a likely log-labeled axis "
            "(remaining value range ratio %.1f >= %.1f)",
            len(zero_ticks), ratio, LOG_ZERO_DROP_RATIO,
        )
        return other
    return ticks


def parse_numeric_ticks(
    ocr_lines: Sequence[OcrLine], img_w: float, img_h: float
) -> Tuple[List[Tick], List[Tick]]:
    """Split a figure's OCR lines into x-axis and y-axis numeric tick candidates.

    Args:
        ocr_lines: Azure-style OCR lines for the figure crop, each
            ``{"text": str, "bounding_box": {"x1","y1","x2","y2"}}``.
        img_w: Figure crop width in pixels.
        img_h: Figure crop height in pixels.

    Returns:
        ``(x_ticks, y_ticks)`` — each a list of ``(value, pixel_center)``.
    """
    x_ticks: List[Tick] = []
    y_ticks: List[Tick] = []
    x_has_exponent = False
    y_has_exponent = False
    # Parallel to x_ticks/y_ticks: True where the entry already came from
    # explicit exponent-notation parsing (_exponent_value), so the bare-digit
    # reinterpretation pass in _repair_exponent_ticks never touches it again.
    x_exp_origin: List[bool] = []
    y_exp_origin: List[bool] = []

    for line in ocr_lines:
        text = line.get("text", "").strip().replace("−", "-")
        text = re.sub(r"-\s+(\d)", r"-\1", text)  # collapse "- 40" -> "-40"
        text = text.replace(",", "")
        bbox = line["bounding_box"]
        cx = (bbox["x1"] + bbox["x2"]) / 2
        cy = (bbox["y1"] + bbox["y2"]) / 2
        in_x_zone = cy / img_h > 0.70
        in_y_zone = cx / img_w < 0.30
        in_tight_y = cx / img_w < 0.15

        def _place(val: float, px_x: float, px_y: float) -> None:
            if in_x_zone and in_tight_y:
                if val >= 0:
                    y_ticks.append((val, px_y))
                    y_exp_origin.append(False)
                else:
                    x_ticks.append((val, px_x))
                    x_exp_origin.append(False)
            elif in_x_zone:
                x_ticks.append((val, px_x))
                x_exp_origin.append(False)
            elif in_y_zone:
                y_ticks.append((val, px_y))
                y_exp_origin.append(False)

        # Log-decade exponent label ("10 3", "10³", "10º") — must be checked
        # BEFORE the compound-token path, which would otherwise split "10 3"
        # into two bogus ticks (the exact T15 silent-linear-fit failure).
        exponent = _exponent_value(text)
        if exponent is not None:
            # y-zone wins for exponent labels: the bottom-most y-axis decade
            # label (e.g. "10 0" at the plot's lower-left) falls inside the
            # x-zone band too, but its left position marks it as a y label.
            if in_y_zone:
                y_ticks.append((exponent, cy))
                y_exp_origin.append(True)
                y_has_exponent = True
            elif in_x_zone:
                x_ticks.append((exponent, cx))
                x_exp_origin.append(True)
                x_has_exponent = True
            continue

        text_clean = text.rstrip(" -")
        if NUMERIC_RE.match(text_clean):
            try:
                val = float(text_clean)
            except ValueError:
                continue
            if abs(val) <= MAX_TICK_MAGNITUDE or (
                val == int(val) and len(str(int(abs(val)))) <= 7
            ):
                _place(val, cx, cy)
            continue

        # Compound token: "-60 -40 -20 0 20 40 60 80" (every part numeric).
        parts = text.split()
        if len(parts) >= 2:
            nums: List[float] = []
            for part in parts:
                part_clean = part.rstrip("-").strip()
                if not NUMERIC_RE.match(part_clean):
                    nums = []
                    break
                value = float(part_clean)
                if abs(value) > MAX_TICK_MAGNITUDE:
                    nums = []
                    break
                nums.append(value)
            if len(nums) >= 2:
                n = len(nums)
                x1, x2 = bbox["x1"], bbox["x2"]
                y1, y2 = bbox["y1"], bbox["y2"]
                for i, val in enumerate(nums):
                    frac = i / (n - 1)
                    if in_x_zone:
                        x_ticks.append((val, x1 + frac * (x2 - x1)))
                        x_exp_origin.append(False)
                    elif in_y_zone:
                        y_ticks.append((val, y1 + frac * (y2 - y1)))
                        y_exp_origin.append(False)

    x_ticks = _repair_exponent_ticks(x_ticks, x_has_exponent, x_exp_origin)
    y_ticks = _repair_exponent_ticks(y_ticks, y_has_exponent, y_exp_origin)
    x_ticks = _drop_stray_zero_on_log_axis(x_ticks)
    y_ticks = _drop_stray_zero_on_log_axis(y_ticks)
    return x_ticks, y_ticks


def _log_roundness(value: float) -> float:
    if value > 0:
        frac = abs(np.log10(value)) % 1
        return min(frac, 1.0 - frac)
    return abs(value - round(value))


def _ransac(vals: np.ndarray, pos: np.ndarray, inlier_threshold: float) -> List[int]:
    """Exhaustive 3-combination RANSAC; returns the largest inlier index set."""
    best_inliers: List[int] = []
    for idx in combinations(range(len(vals)), min(3, len(vals))):
        sample_vals = vals[list(idx)]
        sample_pos = pos[list(idx)]
        if len(set(sample_vals)) < 2:
            continue
        design = np.vstack([sample_vals, np.ones(len(sample_vals))]).T
        try:
            slope, intercept = np.linalg.lstsq(design, sample_pos, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        residual = np.abs(pos - (slope * vals + intercept))
        inliers = np.where(residual <= inlier_threshold)[0].tolist()
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    return best_inliers


def fit_axis(
    ticks: Sequence[Tick], min_n: int = 2, inlier_threshold: float = 15.0
) -> Optional[Tuple[float, float, List[Tick], bool]]:
    """Fit ``pixel = slope * v + intercept`` to tick pairs, auto-detecting log scale.

    Args:
        ticks: ``(value, pixel_position)`` pairs.
        min_n: Minimum usable ticks required to return a fit.
        inlier_threshold: Absolute pixel-residual RANSAC inlier cutoff.

    Returns:
        ``(slope, intercept, used, is_log)`` where ``used`` is the winning
        inlier ``(value, pixel_position)`` list, or ``None`` if fewer than
        ``min_n`` usable ticks remain, values are degenerate, or the fit is
        singular.
    """
    if len(ticks) < min_n:
        return None

    ticks_by_px = sorted(ticks, key=lambda t: t[1])
    deduped: List[Tick] = []
    for value, pixel in ticks_by_px:
        if deduped and abs(pixel - deduped[-1][1]) <= 5:
            if _log_roundness(value) < _log_roundness(deduped[-1][0]):
                deduped[-1] = (value, pixel)
        else:
            deduped.append((value, pixel))

    if len(deduped) < min_n:
        return None

    all_vals = np.array([t[0] for t in deduped], dtype=float)
    all_pos = np.array([t[1] for t in deduped], dtype=float)

    if len(set(all_vals)) < 2:
        return None

    linear_inliers = _ransac(all_vals, all_pos, inlier_threshold)

    log_inliers: List[int] = []
    if np.all(all_vals > 0):
        log_inliers = _ransac(np.log10(all_vals), all_pos, inlier_threshold)

    value_range_ratio = (
        float(all_vals.max() / all_vals.min()) if np.all(all_vals > 0) else 1.0
    )
    prefer_log = len(log_inliers) > len(linear_inliers) or (
        len(log_inliers) == len(linear_inliers)
        and len(log_inliers) >= min_n
        and value_range_ratio >= 10.0
    )

    if prefer_log and len(log_inliers) >= min_n:
        best_inliers, use_vals, is_log = log_inliers, np.log10(all_vals), True
    elif len(linear_inliers) >= min_n:
        best_inliers, use_vals, is_log = linear_inliers, all_vals, False
    else:
        return None

    inlier_vals = use_vals[best_inliers]
    inlier_pos = all_pos[best_inliers]
    if len(set(inlier_vals)) < 2:
        return None

    design = np.vstack([inlier_vals, np.ones_like(inlier_vals)]).T
    try:
        slope, intercept = np.linalg.lstsq(design, inlier_pos, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    used = [(float(all_vals[i]), float(all_pos[i])) for i in best_inliers]
    return float(slope), float(intercept), used, is_log


# Unit detection (T18): scans the y-axis label zone for a capacitance-unit
# token. Only explicit prefixed forms are matched (pF/nF/uF) — a bare "F" is
# deliberately NOT matched on its own, since real axis annotations use
# lowercase "f" as a frequency variable ("f = 1 MHz"), which would otherwise
# produce a false "Farad" detection.
_UNIT_PATTERNS = (
    ("pF", re.compile(r"(?i)\bp\s*f\b")),
    ("nF", re.compile(r"(?i)\bn\s*f\b")),
    ("uF", re.compile(r"(?i)\b[uµ]\s*f\b")),
)


def detect_y_axis_units(
    ocr_lines: Sequence[OcrLine], img_w: float, img_h: float
) -> Optional[str]:
    """Detect the capacitance unit (pF/nF/uF) from the y-axis label OCR text.

    Only scans the y-axis label zone (same ``cx / img_w < 0.30`` gate as
    :func:`parse_numeric_ticks`'s y-zone), so a stray unit-looking token
    elsewhere in the figure (caption, formula annotation) is never picked up.

    Args:
        ocr_lines: The figure's OCR lines.
        img_w: Figure crop width in pixels.
        img_h: Figure crop height in pixels.

    Returns:
        ``"pF"``, ``"nF"``, or ``"uF"`` if exactly one unit is found in the
        y-zone; ``None`` if none is found, or if more than one distinct
        unit is found (ambiguous — never guessed).
    """
    found = set()
    for line in ocr_lines:
        text = line.get("text", "")
        bbox = line["bounding_box"]
        cx = (bbox["x1"] + bbox["x2"]) / 2
        if img_w <= 0 or cx / img_w >= 0.30:
            continue
        for unit, pattern in _UNIT_PATTERNS:
            if pattern.search(text):
                found.add(unit)
                break

    if len(found) == 1:
        return next(iter(found))
    return None


def pixel_to_data(px: float, py: float, calibration: Dict[str, Any]) -> Tuple[float, float]:
    """Map a pixel coordinate to engineering-unit ``(x, y)`` via a calibration dict.

    ``calibration`` needs ``x_slope``, ``x_intercept``, ``x_log``,
    ``y_slope``, ``y_intercept``, ``y_log`` (as produced by
    :func:`derive_calibration`). Log-axis exponents are clamped to ±12 to
    avoid inf/NaN on wildly extrapolated pixels.
    """
    if calibration.get("x_log"):
        log_x = (px - calibration["x_intercept"]) / calibration["x_slope"]
        x = 10 ** max(min(log_x, 12.0), -12.0)
    else:
        x = (px - calibration["x_intercept"]) / calibration["x_slope"]

    if calibration.get("y_log"):
        log_y = (py - calibration["y_intercept"]) / calibration["y_slope"]
        y = 10 ** max(min(log_y, 12.0), -12.0)
    else:
        y = (py - calibration["y_intercept"]) / calibration["y_slope"]

    return float(x), float(y)


def data_to_pixel(
    x: float, y: float, calibration: Dict[str, Any]
) -> Optional[Tuple[float, float]]:
    """Exact inverse of :func:`pixel_to_data`: engineering ``(x, y)`` to pixels.

    Added for Stage 6's overlay drawing (T19): the viewer projects Stage 5's
    saved engineering-unit points back onto the source image using the
    STORED calibration dict — this function lives here, next to
    :func:`pixel_to_data`, precisely so no viewer ever grows its own drifted
    copy of the calibration math (a documented legacy bug).

    Args:
        x: Engineering-unit x value.
        y: Engineering-unit y value.
        calibration: Same dict shape :func:`pixel_to_data` consumes.

    Returns:
        ``(px, py)``, or ``None`` if a value is non-positive on a log axis
        (no pixel exists for it — the caller should skip that point).
    """
    if calibration.get("x_log"):
        if x <= 0:
            return None
        px = calibration["x_slope"] * np.log10(x) + calibration["x_intercept"]
    else:
        px = calibration["x_slope"] * x + calibration["x_intercept"]

    if calibration.get("y_log"):
        if y <= 0:
            return None
        py = calibration["y_slope"] * np.log10(y) + calibration["y_intercept"]
    else:
        py = calibration["y_slope"] * y + calibration["y_intercept"]

    return float(px), float(py)


def derive_calibration(
    ocr_lines: Sequence[OcrLine], img_w: float, img_h: float
) -> Optional[Dict[str, Any]]:
    """Parse ticks and fit both axes for one figure crop.

    Args:
        ocr_lines: The figure's OCR lines (same shape as :func:`parse_numeric_ticks`).
        img_w: Figure crop width in pixels.
        img_h: Figure crop height in pixels.

    Returns:
        ``{"plot_bbox": {"left","right","top","bottom"}, "x_slope",
        "x_intercept", "y_slope", "y_intercept", "x_log", "y_log"}``, or
        ``None`` if either axis failed to fit (logged with the reason).
    """
    x_ticks, y_ticks = parse_numeric_ticks(ocr_lines, img_w, img_h)
    x_fit = fit_axis(x_ticks)
    y_fit = fit_axis(y_ticks)
    if not x_fit or not y_fit:
        logger.info(
            "derive_calibration: failed (x_ticks=%d, y_ticks=%d, x_fit=%s, y_fit=%s)",
            len(x_ticks), len(y_ticks), x_fit is not None, y_fit is not None,
        )
        return None

    x_slope, x_intercept, x_used, x_log = x_fit
    y_slope, y_intercept, y_used, y_log = y_fit
    used_x_px = sorted(p for _, p in x_used)
    used_y_px = sorted(p for _, p in y_used)

    return {
        "plot_bbox": {
            "left": used_x_px[0], "right": used_x_px[-1],
            "top": used_y_px[0], "bottom": used_y_px[-1],
        },
        "x_slope": x_slope, "x_intercept": x_intercept,
        "y_slope": y_slope, "y_intercept": y_intercept,
        "x_log": x_log, "y_log": y_log,
    }
