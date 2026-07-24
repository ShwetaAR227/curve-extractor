"""Shared zth_vs_time "downstream" recipe: axis-reading -> Foster physics
fit -> Rth cross-check -> final result (CLAUDE.md §3, §4).

Extracted from :mod:`src.extraction.classical_zth` (2026-07-23, owner-
approved) — same shape/spirit as :mod:`src.extraction.curve_detection`'s
own extraction from :mod:`src.extraction.classical` a session earlier: a
PURE refactor, same math/gates/thresholds/messages, byte-for-byte. Proven
by classical_zth.py's own 67-test suite passing with the SAME count before
and after the refactor that wires this module in (see PROGRESS.md for that
before/after re-run).

Before this, the "we have the curve's pixel points -> convert to
engineering units -> sanity-check the shape -> cross-check against a
printed Rth value -> fit the 2-element Foster RC network -> build the
final schema result" sequence was NOT its own reusable piece — it was
inline inside ``classical_zth.run_classical_pipeline``, mixed together
with the OLD rule-based clustering/single-pulse-picker logic (which stays
in classical_zth.py, untouched — this module knows nothing about how the
pixel points were obtained, whether by that old clustering/tracing or by
an AI model's detected mask).

:func:`fit_and_validate_curve` is the single entry point both
``classical_zth.run_classical_pipeline`` and (a later addition)
``hybrid_zth.run_hybrid_pipeline`` call for everything past "we have the
curve's traced pixel points, in absolute image-pixel coordinates."

Deliberately NOT here: :func:`~src.extraction.classical_zth.fit_foster` and
:func:`~src.extraction.classical_zth.pick_rth_constraint` stay defined in
classical_zth.py — its existing test suite monkeypatches them directly by
module attribute name (``monkeypatch.setattr(classical_zth, "fit_foster",
...)``, etc.), and moving them would silently stop those tests from
patching anything. :func:`fit_and_validate_curve` takes them as plain
parameters instead (dependency injection): classical_zth.py passes its own
current module-global reference (so existing monkeypatching keeps working
exactly as before), and hybrid_zth.py imports the same two real functions
and passes them in too — same functions, verified by identity, never a
second implementation of either.
"""
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.common.log import get_logger
from src.extraction.schema import build_result

logger = get_logger(__name__)

STAGE3_ROOT_ENV_VAR = "LINEFORMER_STAGE3_ROOT"
CURVE_NAME = "single_pulse"
UNITS = "K/W"

# Physical-plausibility thresholds — same values classical_zth.py's inline
# block used before this extraction, unchanged.
_MIN_POINTS = 6
_RISE_RATIO_MAX = 1e6
_RISE_RATIO_MIN = 1.0
_MAX_PLAUSIBLE_LATE_VALUE = 100.0
_SCALE_RATIO_MIN = 0.05
_SCALE_RATIO_MAX = 3.0
_MIN_R_SQUARED = 0.5

FitFosterFn = Callable[..., Tuple[Optional[Dict[str, float]], Optional[float]]]
PickRthConstraintFn = Callable[[Dict[str, Any]], Tuple[Optional[float], Optional[str]]]


def pixel_to_data(px: float, py: float, cal: Dict[str, Any]) -> Tuple[float, float]:
    """zth's own pixel->data inverse (NOT src.calibration.ticks.pixel_to_data
    — a different, calibration-dict shape). Ported unchanged from
    classical_zth.py, including the 12-decade clamp; now shared by both
    classical_zth.py (which still imports it for its own axis_data_range
    computation) and hybrid_zth.py."""
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


def calibration_with_bonus_fields(cal_zth: Dict[str, Any]) -> Dict[str, Any]:
    """Map zth's own calibration dict onto our 6 required schema fields,
    keeping every extra legacy field as bonus detail alongside them."""
    calibration = dict(cal_zth)
    calibration["x_log"] = cal_zth["x_scale"] == "log"
    calibration["y_log"] = cal_zth["y_scale"] == "log"
    return calibration


def clamp_confidence(r_squared: Optional[float]) -> float:
    """Clamp a Foster fit's r-squared into [0, 1] (schema requirement) --
    ``None`` (no fit at all) clamps to 0.0."""
    if r_squared is None:
        return 0.0
    return max(0.0, min(1.0, r_squared))


def build_needs_review_result(
    device: str, curve_type: str, source_image: str, reason: str,
    calibration: Optional[Dict[str, Any]] = None,
    points: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Build a schema-validated ``needs_review`` result with the given
    plain-English reason. Confidence is always 0.0 here -- a
    ``needs_review`` result never claims a trustworthy number."""
    logger.warning("zth_fit(%s, %s): needs_review - %s", device, curve_type, reason)
    curves = [{"curve_name": CURVE_NAME, "confidence": 0.0, "points": points or []}]
    return build_result(
        device=device, curve_type=curve_type, source_image=source_image,
        status="needs_review", review_reason=reason, duplicates_removed=0,
        calibration=calibration, curves=curves, units=None if calibration is None else UNITS,
    )


def read_full_extraction_for_rth(
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
            "zth_fit(%s): no stage3_root and %s unset — "
            "proceeding without an Rth_JC table constraint", device, STAGE3_ROOT_ENV_VAR,
        )
        return None
    json_path = os.path.join(str(root), device, "full_extraction.json")
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.info(
            "zth_fit(%s): could not read %s (%s) — "
            "proceeding without an Rth_JC table constraint", device, json_path, exc,
        )
        return None


def fit_and_validate_curve(
    device: str,
    curve_type: str,
    source_image: str,
    pixel_points: Sequence[Tuple[float, float]],
    calibration: Dict[str, Any],
    stage3_root: Optional[str],
    fit_foster: FitFosterFn,
    pick_rth_constraint: PickRthConstraintFn,
) -> Dict[str, Any]:
    """Everything downstream of "we have the curve's traced pixel points":
    convert to engineering units, sanity-check the shape, cross-check
    against a printed Rth value, fit the Foster physics model, build the
    final schema result. Ported unchanged from classical_zth.py's own
    inline block (see module docstring) — same gates, same thresholds,
    same messages.

    Args:
        device: Device identifier.
        curve_type: Registry key (``"zth_vs_time"``).
        source_image: Figure image path/identifier, recorded in the result.
        pixel_points: Traced curve points, ``(x_img, y_img)`` in ABSOLUTE
            image-pixel coordinates (the caller applies any crop offset
            itself before calling this — this function knows nothing about
            crops).
        calibration: The schema-ready calibration dict (from
            :func:`calibration_with_bonus_fields`) — also used directly as
            ``pixel_to_data``'s calibration argument (it's a superset of
            the raw ``x_scale``/``y_scale``/... fields ``pixel_to_data``
            needs).
        stage3_root: Optional Stage-3 output root for the Rth_JC table
            lookup (falls back to ``LINEFORMER_STAGE3_ROOT``).
        fit_foster: Injected — the caller's own current
            :func:`~src.extraction.classical_zth.fit_foster` reference
            (see module docstring for why this is a parameter, not an
            import).
        pick_rth_constraint: Injected — the caller's own current
            :func:`~src.extraction.classical_zth.pick_rth_constraint`
            reference (same reason).

    Returns:
        A schema-validated Stage-5 result dict (``status`` "ok" or
        "needs_review").
    """
    eng_pts = [pixel_to_data(px, py, calibration) for px, py in pixel_points]

    if len(eng_pts) < _MIN_POINTS:
        return build_needs_review_result(
            device, curve_type, source_image,
            f"too_few_points: fewer than {_MIN_POINTS} digitized points ({len(eng_pts)} found)",
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

    full_extraction = read_full_extraction_for_rth(device, stage3_root)
    rth_constraint, rth_source = pick_rth_constraint(full_extraction or {})

    if rth_constraint is None and (
        rise_ratio > _RISE_RATIO_MAX or rise_ratio < _RISE_RATIO_MIN
        or right_q > _MAX_PLAUSIBLE_LATE_VALUE
    ):
        return build_needs_review_result(
            device, curve_type, source_image,
            f"calibration_disaster: rise_ratio={rise_ratio:.3g} out of "
            f"[{_RISE_RATIO_MIN:g}, {_RISE_RATIO_MAX:g}] or "
            f"right_q={right_q:.3g} > {_MAX_PLAUSIBLE_LATE_VALUE:g} K/W — "
            f"calibration is not physically plausible",
            calibration=calibration,
        )

    fit_constraint = None
    constraint_warning = None
    skip_fit = False
    if rth_constraint is not None:
        scale_ratio = right_q / float(rth_constraint)
        if _SCALE_RATIO_MIN <= scale_ratio <= _SCALE_RATIO_MAX:
            fit_constraint = rth_constraint
        else:
            skip_fit = True
            constraint_warning = (
                f"calibration_broken: y_obs_late={right_q:.3g} vs rth_table="
                f"{rth_constraint:.3g} (ratio {scale_ratio:.2g}); "
                f"curve trace unreliable, tau values not extracted"
            )

    if skip_fit:
        # Deliberate remap (see classical_zth.py's module docstring):
        # legacy calls this "clean", we call it needs_review — the message
        # itself describes something a reviewer should see.
        return build_needs_review_result(
            device, curve_type, source_image, constraint_warning, calibration=calibration,
        )

    fitted_params, r2 = fit_foster(sxs.tolist(), sys_.tolist(), rth_constraint=fit_constraint)
    if fitted_params is None or r2 is None or r2 < _MIN_R_SQUARED:
        r2_text = f"{r2:.3g}" if r2 is not None else "N/A"
        return build_needs_review_result(
            device, curve_type, source_image,
            f"foster_fit_failed: r_squared={r2_text} (< {_MIN_R_SQUARED:g} or fit did not converge)",
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
        "confidence": clamp_confidence(r2),
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
        "zth_fit(%s, %s): ok, r_squared=%.4f, rth_jc=%.4g", device, curve_type, r2, rth_steady,
    )
    return build_result(
        device=device, curve_type=curve_type, source_image=source_image,
        status="ok", review_reason=None, duplicates_removed=0,
        calibration=calibration, curves=[curve], units=UNITS,
    )
