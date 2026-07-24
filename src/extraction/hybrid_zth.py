"""Hybrid (AI model + rule-based) Stage-5 extraction front-end for
zth_vs_time (CLAUDE.md §1, §3, owner-approved design 2026-07-23).

Combines the AI model (LineFormer/mmdet, via the same shared
:func:`~src.extraction.inference.run_inference` primitive
:mod:`src.extraction.model_if_vsd` already uses — reused, not reinvented)
for the one thing it's actually needed for — finding the single-pulse
curve's pixel points on real, previously-unseen charts, including the
multi-duty-cycle-curve family charts where the OLD rule-based
clustering/picker (classical_zth.py's ``cluster_into_curves_zth`` /
``pick_single_pulse``) has known trouble — with classical_zth.py's
existing rule-based logic for everything else: the printed-table
shortcut, axis calibration, the ratio-axis guard, the Foster physics fit,
and the Rth-constraint cross-check.

Everything past "we have the curve's traced pixel points" is
:func:`~src.extraction.zth_fit.fit_and_validate_curve` — the SAME function
:mod:`src.extraction.classical_zth` itself now calls (extracted there in a
prior, separately-verified pure refactor: classical_zth.py's 67-test suite
passes with the identical count before and after, see PROGRESS.md). Every
function this module calls is imported directly, never reimplemented —
see ``TestIdentityNotDuplication`` in this module's own test suite for the
identity checks (``hybrid_zth.fit_foster is classical_zth.fit_foster``,
etc.).

Four rules, in order:

1. **Printed-table shortcut, unchanged, first.** If
   :func:`~src.extraction.classical_zth.parse_foster_table_from_ocr` finds
   a real printed Foster RC table, this module delegates the WHOLE result
   to :func:`~src.extraction.classical_zth.run_classical_pipeline` and the
   model never runs. Safe by construction, not merely by convention: that
   function's own internal call to the SAME
   ``parse_foster_table_from_ocr`` (same ``ocr_lines``, deterministic)
   necessarily finds the identical table and takes its own identical
   early-return branch — it can never reach its OLD clustering/picker
   code once a table exists.
2. **Ratio-axis detection still runs, unconditionally.**
   :func:`~src.extraction.classical_zth.detect_normalized_ratio_axis` is
   evaluated regardless of what the model/fit produce. Unlike
   classical_zth.py's own short-circuit-before-calibration guard, THIS
   module still runs the model, still traces the real curve, still does
   calibration and the physics fit — a ratio chart's numbers are still
   genuinely computed and shown to the reviewer. Only the FINAL
   ``status``/``review_reason`` are overridden to ``needs_review`` with
   the ratio reason, unconditionally (even if the underlying fit already
   independently decided ``needs_review`` for some other reason — the
   ratio message is the more actionable one for a reviewer either way).
   ``curves``/``calibration``/``units`` are left exactly as computed, so
   the reviewer sees the real traced shape and numbers; they just know it
   needs a manual multiply by the chart's own printed Rth value.
3. **Model detection-count gate.** Exactly one detection is required —
   this chart type only ever has one single-pulse curve. Zero detections
   or more than one both quarantine (``needs_review``), each with its own
   distinct, plain-English reason, with no fallback of any kind to the old
   rule-based clustering/picker (confirmed by
   ``TestNoOldRuleBasedFallback`` in this module's own tests: those two
   functions' names never even appear in this file's source).
4. **Everything else is delegated**, not reimplemented: axis calibration
   (:func:`~src.extraction.classical_zth.derive_calibration_zth`), the
   plot-bbox sanity check, and the whole Foster-fit/Rth-cross-check
   recipe (:func:`~src.extraction.zth_fit.fit_and_validate_curve`).

Checkpoint/config resolution follows the SAME env-var-root shape
classical_zth.py's own ``LINEFORMER_STAGE3_ROOT`` already established
(CLAUDE.md §3 — never a hardcoded machine path), under a new
``LINEFORMER_CHECKPOINTS_ROOT`` (no existing convention covers "where do
trained checkpoints live" today — confirmed by checking every real
``load_model`` call site in this repo before choosing this design).
Unlike ``LINEFORMER_STAGE3_ROOT`` (whose absence degrades gracefully — no
Rth constraint is a normal, common outcome), a missing checkpoints root
means the model cannot load AT ALL, so :func:`resolve_checkpoint_and_config`
raises ``RuntimeError`` loudly rather than pretending to proceed.

``run_hybrid_pipeline`` takes an already-loaded ``model`` (e.g. from
:func:`load_hybrid_model`), exactly mirroring
:func:`~src.extraction.model_if_vsd.run_model_pipeline`'s own convention —
the caller loads the (expensive) model once, this function never does.
This also keeps the module's own core logic importable and unit-testable
with no GPU and no network (CLAUDE.md §2): the only GPU-only import
(mmdet, inside :func:`~src.extraction.inference.load_model`/
:func:`~src.extraction.inference.run_inference`) stays lazily inside those
two functions, exactly as :mod:`src.extraction.inference` already keeps it.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.common.log import get_logger
from src.extraction.classical_zth import (
    derive_calibration_zth,
    detect_normalized_ratio_axis,
    fit_foster,
    parse_foster_table_from_ocr,
    pick_rth_constraint,
)
from src.extraction.inference import run_inference
from src.extraction.skeletonize import mask_to_points
from src.extraction.zth_fit import (
    build_needs_review_result,
    calibration_with_bonus_fields,
    fit_and_validate_curve,
)

import src.extraction.classical_zth as classical_zth

logger = get_logger(__name__)

CHECKPOINTS_ROOT_ENV_VAR = "LINEFORMER_CHECKPOINTS_ROOT"
CHECKPOINT_RELATIVE_PATH = "zth_vs_time_run2_patience2000/best_segm_mAP_50_iter_4000.pth"
CONFIG_RELATIVE_PATH = "src/training/configs/lineformer_zth_vs_time_patience2000.py"

_MIN_PLOT_DIMENSION_PX = 50


def resolve_checkpoint_and_config(checkpoints_root: Optional[str] = None) -> Tuple[str, str]:
    """Resolve the zth_vs_time hybrid checkpoint/config to absolute paths.

    ``checkpoints_root`` (or the ``LINEFORMER_CHECKPOINTS_ROOT`` env var
    when omitted) is where trained-run checkpoints live on this box —
    joined with the relative checkpoint identifier (CLAUDE.md §3, never a
    hardcoded machine path). A missing root raises ``RuntimeError``
    immediately: unlike a missing Rth table (a normal, common outcome), a
    missing checkpoint means the model cannot load at all.

    The config path is resolved against ``LINEFORMER_REPO_ROOT`` (the SAME
    env var :func:`~src.extraction.inference.load_model` already sets a
    default for), since the config file lives inside this repo, not on
    the checkpoints mount.

    Returns:
        ``(checkpoint_path, config_path)``, both absolute.
    """
    root = checkpoints_root or os.environ.get(CHECKPOINTS_ROOT_ENV_VAR)
    if not root:
        raise RuntimeError(
            f"{CHECKPOINTS_ROOT_ENV_VAR} is not set and no checkpoints_root was "
            f"given -- cannot locate the trained zth_vs_time checkpoint "
            f"({CHECKPOINT_RELATIVE_PATH})."
        )
    checkpoint = os.path.join(str(root), CHECKPOINT_RELATIVE_PATH)
    repo_root = os.environ.get("LINEFORMER_REPO_ROOT") or str(Path(__file__).resolve().parents[2])
    config = os.path.join(repo_root, CONFIG_RELATIVE_PATH)
    return checkpoint, config


def load_hybrid_model(checkpoints_root: Optional[str] = None, device: str = "cuda:0") -> Any:
    """Load the zth_vs_time hybrid checkpoint (GPU-only). Thin wrapper
    around :func:`~src.extraction.inference.load_model` (reused, not
    reinvented) with this module's own checkpoint/config resolution."""
    from src.extraction.inference import load_model

    checkpoint, config = resolve_checkpoint_and_config(checkpoints_root)
    return load_model(checkpoint, config, device=device)


def _run_model_and_fit(
    device: str,
    curve_type: str,
    source_image: str,
    image: np.ndarray,
    ocr_lines: Sequence[Dict[str, Any]],
    model: Any,
    stage3_root: Optional[str],
    score_thr: float,
) -> Dict[str, Any]:
    """Model detection -> calibration -> the shared downstream recipe.
    Everything except the table-shortcut and the ratio-axis override (both
    handled by :func:`run_hybrid_pipeline` itself, above this)."""
    fig_meta = {"ocr_lines": list(ocr_lines)}
    img_h, img_w = image.shape[:2]

    detections = run_inference(model, source_image, score_thr=score_thr)
    logger.info(
        "hybrid_zth(%s, %s): %d detection(s) from run_inference",
        device, curve_type, len(detections),
    )

    if len(detections) == 0:
        return build_needs_review_result(
            device, curve_type, source_image,
            "model_no_curves_found: the AI model detected 0 curves in this figure",
        )
    if len(detections) > 1:
        return build_needs_review_result(
            device, curve_type, source_image,
            f"model_ambiguous_multiple_curves: the AI model detected {len(detections)} "
            f"curves in a chart that should only ever have one single-pulse curve",
        )

    cal_zth = derive_calibration_zth(fig_meta, img_w, img_h)
    if cal_zth is None:
        return build_needs_review_result(
            device, curve_type, source_image,
            "calibration_failed: axis calibration failed (insufficient/degenerate tick marks)",
        )
    calibration = calibration_with_bonus_fields(cal_zth)
    bb = cal_zth["plot_bbox"]
    bb_width = bb["right"] - bb["left"]
    bb_height = bb["bottom"] - bb["top"]
    if bb_width < _MIN_PLOT_DIMENSION_PX or bb_height < _MIN_PLOT_DIMENSION_PX:
        return build_needs_review_result(
            device, curve_type, source_image,
            f"plot_bbox_too_small: plot area {bb_width:.0f}x{bb_height:.0f}px is too small to trust",
            calibration=calibration,
        )

    row_col_points = mask_to_points(detections[0].mask)
    if not row_col_points:
        return build_needs_review_result(
            device, curve_type, source_image,
            "model_mask_empty: the AI model's detected mask produced no traceable points",
            calibration=calibration,
        )
    pixel_points = [(col, row) for row, col in row_col_points]

    return fit_and_validate_curve(
        device, curve_type, source_image, pixel_points, calibration, stage3_root,
        fit_foster=fit_foster, pick_rth_constraint=pick_rth_constraint,
    )


def run_hybrid_pipeline(
    device: str,
    curve_type: str,
    source_image: str,
    image: np.ndarray,
    ocr_lines: Sequence[Dict[str, Any]],
    model: Any,
    stage3_root: Optional[str] = None,
    score_thr: float = 0.5,
) -> Dict[str, Any]:
    """Run the zth_vs_time hybrid (AI model + rule-based) pipeline. See
    the module docstring for the full design.

    Args:
        device: Device identifier.
        curve_type: Registry key (``"zth_vs_time"``).
        source_image: Figure image path, passed to ``run_inference`` and
            recorded as the result's ``source_image``.
        image: HxWx3 uint8 BGR figure crop.
        ocr_lines: The figure's OCR lines (dict-shaped:
            ``{"text": str, "bounding_box": {"x1","y1","x2","y2"}}``).
        model: A loaded model (e.g. from :func:`load_hybrid_model`).
        stage3_root: Optional Stage-3 output root for the Rth_JC table
            lookup (falls back to ``LINEFORMER_STAGE3_ROOT``, same as
            classical_zth.py).
        score_thr: Minimum model detection confidence to keep.

    Returns:
        A schema-validated Stage-5 result dict.
    """
    fig_meta = {"ocr_lines": list(ocr_lines)}

    # ---- Printed-table shortcut, unchanged, first: if found, done, the
    # model never runs (see module docstring for why delegating the WHOLE
    # result to classical_zth.run_classical_pipeline is safe here).
    table = parse_foster_table_from_ocr(fig_meta)
    if table is not None:
        logger.info(
            "hybrid_zth(%s, %s): printed Foster table found — delegating to "
            "classical_zth.run_classical_pipeline, model never runs", device, curve_type,
        )
        return classical_zth.run_classical_pipeline(
            device=device, curve_type=curve_type, source_image=source_image,
            image=image, ocr_lines=ocr_lines, stage3_root=stage3_root,
        )

    img_h, img_w = image.shape[:2]
    ratio_reason = detect_normalized_ratio_axis(ocr_lines, img_w, img_h)

    result = _run_model_and_fit(
        device, curve_type, source_image, image, ocr_lines, model, stage3_root, score_thr,
    )

    if ratio_reason is not None:
        # Unconditional override (see module docstring point 2): even if
        # the fit already independently produced its own needs_review
        # reason, the ratio message is the more actionable one -- but the
        # actual traced curves/calibration/units are left exactly as
        # computed, so the reviewer still sees the real shape and numbers.
        result = dict(result, status="needs_review", review_reason=ratio_reason)
    return result
