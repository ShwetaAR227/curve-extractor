"""Model-path (LineFormer) Stage-5 extraction front-end for if_vs_vsd
(body-diode forward/reverse current vs. source-drain voltage).

Unlike capacitance_vs_vds/id_vs_vgs (plain "model" entries that route
straight to :func:`src.extraction.pipeline.run_pipeline` with no
override), if_vs_vsd needs the same kind of expected-vs-detected safety
net already proven for vgsth_vs_tj's CLASSICAL wrapper
(:mod:`src.extraction.classical_vgsth`) — curves are named by temperature
label, not by a fixed position, so a naive fixed ``expected_curve_count``
gate inside ``process_detections`` can't express "2 or 4, whichever the
chart's own labels resolve to". This module is that safety net, built on
the MODEL side: :func:`~src.extraction.inference.run_inference` (the same
model-inference primitive ``run_pipeline`` itself calls) replaces
classical_vgsth's color/mono detection step; everything downstream
(counting, naming, the frozen core) mirrors that module's structure
exactly (CLAUDE.md §3 zero duplication; §4 frozen-stage integrity — no
frozen file is modified).

Naming/counting is entirely label-driven
(:mod:`src.extraction.naming.if_vs_vsd`) — like vgsth_vs_tj, if_vs_vsd has
no meaningful position-only naming (which curve is "25°C" vs "175°C" is
unknowable without OCR labels), so the naming-registry's own internal
lookup (``naming/__init__.py``'s ``curve_N`` placeholder) carries no real
authority. This wrapper ALWAYS overrides it with the real resolved names
on every non-quarantined path.

Core safety net — the expected-vs-detected curve-count comparison (never
silently merge curves at a crossing, never silently miss one): given
``N = count_expected_curves(ocr_lines)`` (how many curves the chart's own
temperature labels imply) and ``D = len(detections)`` (how many the model
actually found, already score-filtered by ``run_inference``),

    D == 0                    -> quarantine: no curves found
    N is not None and D < N   -> quarantine: likely merged at a crossing
    N is not None and D > N   -> quarantine: stray component / missed label
    N is None and D > 1       -> quarantine: no usable labels to disambiguate
    otherwise (D == N, or N is None and D == 1):
        name_curves_by_labels(...) is None -> quarantine: ambiguous naming
        else                               -> proceed, names override the
                                               registry placeholder

Every quarantine path still carries real calibration (same rule enforced
for vgsth_vs_tj after that gap was caught, and never reintroduced here):
calibration is derived purely from OCR axis tick labels, independent of
whether curve naming succeeded.

``ExtractionSpec.expected_curve_count`` for if_vs_vsd is ``None`` — this
module's REAL expected count always comes from
``count_expected_curves(ocr_lines)``, computed dynamically per chart,
exactly like vgsth_vs_tj's own registry field. Real charts show 2 curves
(two temperatures, the common case) or 4 (temp + a compound percentile
label, seen once) — but that shape lives in the corpus and in
``naming.if_vs_vsd``'s own docstring, not in a static registry field, since
nothing here checks a chart's curve count against a fixed set.
``run_model_pipeline`` accepts ``expected_curve_count`` purely for
call-signature symmetry with the generic ``run_pipeline`` (so
live_stages.py's dispatch can call either uniformly) — the parameter
itself is never read here (confirmed: grep for the bare
``expected_curve_count`` identifier in this file finds only its own
parameter declaration below; every ``process_detections(...)`` call
passes the locally computed detected count instead).
"""
from typing import Any, Dict, Sequence

from src.common.log import get_logger
from src.extraction.curve_detection import OcrLine
from src.extraction.inference import run_inference
from src.extraction.naming.if_vs_vsd import count_expected_curves, name_curves_by_labels
from src.extraction.pipeline import process_detections
from src.extraction.schema import build_result
from src.extraction.skeletonize import mask_to_points

logger = get_logger(__name__)


def _quarantine(
    device: str, curve_type: str, image_path: str, reason: str,
    detections: Sequence[Any], img_w: float, img_h: float,
    ocr_lines: Sequence[OcrLine],
) -> Dict[str, Any]:
    """Build a needs_review result for a gate failure before naming succeeds.

    Mirrors :func:`src.extraction.classical_vgsth._quarantine`: calibration
    doesn't depend on curve naming succeeding, so ``process_detections`` is
    called with ``expected_curve_count`` set to the ACTUAL detected count
    (its own internal count-gate becomes a trivial pass-through — it has no
    concept of the label-count mismatch or naming-tie this wrapper is
    quarantining for). Its status/review_reason verdict is always replaced
    with the wrapper's own, but its computed calibration/curves/units are
    kept.
    """
    logger.warning("model_if_vsd: quarantine - %s", reason)
    core_result = process_detections(
        device, curve_type, image_path, img_w, img_h, list(ocr_lines),
        detections, expected_curve_count=len(detections),
    )
    return build_result(
        device=device, curve_type=curve_type, source_image=image_path,
        status="needs_review", review_reason=reason,
        duplicates_removed=core_result["duplicates_removed"],
        calibration=core_result["calibration"], curves=core_result["curves"],
        units=core_result["units"],
    )


def run_model_pipeline(
    device: str,
    curve_type: str,
    image_path: str,
    ocr_lines: Sequence[OcrLine],
    img_w: float,
    img_h: float,
    model: Any,
    score_thr: float = 0.5,
    expected_curve_count: Any = None,
) -> Dict[str, Any]:
    """Run model inference then the full existing Stage-5 pipeline.

    The if_vs_vsd analogue of :func:`src.extraction.pipeline.run_pipeline`,
    with the same registry-driven-override role
    :func:`src.extraction.classical_vgsth.run_classical_pipeline` plays for
    vgsth_vs_tj: :func:`~src.extraction.inference.run_inference` replaces
    the generic ``run_pipeline``'s inline inference call. The detected
    count is compared against what the chart's own temperature labels
    imply (:func:`~src.extraction.naming.if_vs_vsd.count_expected_curves`)
    before anything else runs — a mismatch quarantines rather than ever
    silently merging curves at a crossing or dropping one. On a match,
    :func:`~src.extraction.naming.if_vs_vsd.name_curves_by_labels` resolves
    the real curve names, ``process_detections`` builds everything else
    (dedup, calibration, units, schema), and the resolved names overwrite
    whatever placeholder name the naming registry entry produced.

    Args:
        device: Device identifier.
        curve_type: Registry key (``"if_vs_vsd"``).
        image_path: Figure image path, passed to ``run_inference`` and
            recorded as the result's ``source_image``.
        ocr_lines: The figure's OCR lines, for counting, naming, and axis
            calibration.
        img_w: Figure crop width in pixels.
        img_h: Figure crop height in pixels.
        model: A loaded model (e.g. from
            :func:`src.extraction.inference.load_model`).
        score_thr: Minimum detection confidence to keep.
        expected_curve_count: Accepted for call-signature symmetry with
            ``run_pipeline`` only — never read (see module docstring).

    Returns:
        A schema-validated Stage-5 result dict — the exact shape Stage 6's
        gallery and Stage 7's orchestrator already consume from every other
        extraction path.

    Raises:
        KeyError: On structurally malformed inputs (e.g. an OCR line with
            no ``bounding_box``) — caller bugs are raised, never swallowed
            (CLAUDE.md §7).
    """
    detections = run_inference(model, image_path, score_thr=score_thr)
    logger.info(
        "model_if_vsd(%s, %s): %d detection(s) from run_inference",
        device, curve_type, len(detections),
    )

    detected = len(detections)
    expected = count_expected_curves(ocr_lines)

    if detected == 0:
        reason = "no curves found in the figure"
        return _quarantine(device, curve_type, image_path, reason, detections,
                           float(img_w), float(img_h), ocr_lines)

    if expected is not None and detected < expected:
        reason = (
            f"{detected} curve(s) detected but {expected} label(s) resolved "
            f"— likely merged at a crossing"
        )
        return _quarantine(device, curve_type, image_path, reason, detections,
                           float(img_w), float(img_h), ocr_lines)

    if expected is not None and detected > expected:
        reason = (
            f"{detected} curve(s) detected but only {expected} label(s) resolved "
            f"— stray component or missed label"
        )
        return _quarantine(device, curve_type, image_path, reason, detections,
                           float(img_w), float(img_h), ocr_lines)

    if expected is None and detected > 1:
        reason = "no usable labels to disambiguate multiple curves"
        return _quarantine(device, curve_type, image_path, reason, detections,
                           float(img_w), float(img_h), ocr_lines)

    # Reachable only here: (expected is None and detected == 1), or
    # (expected is not None and detected == expected).
    point_lists = [mask_to_points(d.mask) for d in detections]
    names = name_curves_by_labels(point_lists, ocr_lines)
    if names is None:
        reason = (
            "ambiguous naming despite a matched curve count "
            "(e.g. a genuine proximity tie)"
        )
        return _quarantine(device, curve_type, image_path, reason, detections,
                           float(img_w), float(img_h), ocr_lines)

    logger.info(
        "model_if_vsd(%s, %s): resolved names %s, proceeding to process_detections",
        device, curve_type, names,
    )
    result = process_detections(
        device, curve_type, image_path, float(img_w), float(img_h),
        list(ocr_lines), detections, expected_curve_count=detected,
    )
    curves = [dict(curve, curve_name=name) for curve, name in zip(result["curves"], names)]
    return build_result(
        device=device, curve_type=curve_type, source_image=image_path,
        status=result["status"], review_reason=result["review_reason"],
        duplicates_removed=result["duplicates_removed"],
        calibration=result["calibration"], curves=curves, units=result["units"],
    )
