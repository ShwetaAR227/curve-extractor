"""Stage-5 extraction routing registry (CLAUDE.md §1, §3).

Curve-type -> extraction-path routing is DATA, not code: one
:class:`ExtractionSpec` per curve type, held in a single dict — same
pattern as :mod:`src.classification.curve_registry`. Adding a new curve
type (or flipping a curve type from classical to model, or vice versa) is
a registry entry, never a new if/elif branch in the adapter
(:mod:`src.orchestrator.live_stages`).

Two extraction methods exist today:
- ``"classical"`` — ``checkpoint``/``config`` are ``None`` (nothing to
  load). ``classical_pipeline`` carries the ACTUAL function object for
  this curve type (e.g. :func:`src.extraction.classical.run_classical_pipeline`
  for rdson_vs_tj, :func:`src.extraction.classical_vgsth.run_classical_pipeline`
  for vgsth_vs_tj) — the adapter (:mod:`src.orchestrator.live_stages`)
  calls ``spec.classical_pipeline(...)`` directly, a genuine per-curve-type
  data lookup, not a single hardcoded import shared by every classical
  entry (fixed 2026-07-22; previously every ``"classical"`` entry
  unconditionally called rdson_vs_tj's own wrapper regardless of curve
  type — a real bug, since fixed, see PROGRESS.md).
- ``"model"`` — routes to :func:`src.extraction.pipeline.run_pipeline`
  (LineFormer/mmdet). ``checkpoint``/``config`` name the trained weights
  and its config file. ``classical_pipeline`` is ``None`` for every model
  entry — nothing classical to point at.

A third sentinel value, ``"none"``, marks a curve type that is registered
for classification-adjacent bookkeeping but deliberately has NO extractor
behind it yet (currently only ``vgs_vs_qg`` — gate-charge curves are a
different shape problem, staircase/plateau rather than a traced line, and
may need a third extraction approach not yet designed). This is a
different, calmer fact than "never registered at all": the adapter raises
a distinctly-named error for it
(:class:`src.orchestrator.live_stages.NoExtractorAvailable`) rather than
treating it as an unexpected crash.

Checkpoint/config values are relative identifiers, not absolute
machine-specific paths (CLAUDE.md §3 — the legacy repo's ``D:\\...``-path
lesson) — resolving them against wherever the trained weights actually
live on a given box is the caller's job.

``zth_vs_time`` has NO entry — the real current gap (no trained model or
classical detector yet); ``get_extraction_spec`` raises ``KeyError`` for
it, same as :func:`src.classification.curve_registry.get_spec` does for
its own unregistered types. ``vgsth_vs_tj`` (as of 2026-07-22) has a real
``"classical"`` entry, genuinely routed to its own wrapper. ``if_vs_vsd``
(as of 2026-07-22, follow-up session) has a real ``"model"`` entry,
genuinely routed to its own ``model_pipeline`` override
(:func:`src.extraction.model_if_vsd.run_model_pipeline`) — see
``model_pipeline`` below.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from src.extraction import classical, classical_vgsth, model_if_vsd


@dataclass(frozen=True)
class ExtractionSpec:
    """Data-only routing entry for one curve type. No behavior lives here."""

    curve_type: str
    method: str  # "classical" | "model" | "none"
    checkpoint: Optional[str]
    config: Optional[str]
    score_thr: float
    # int (e.g. 3), a tuple of allowed counts (e.g. (1, 2)), or None when the
    # curve type has no fixed count at all (e.g. vgsth_vs_tj, determined
    # dynamically per-chart inside its own classical wrapper) — this field
    # is only ever read by the "model" dispatch path (run_pipeline), never
    # by "classical", so a classical entry's value here is informational.
    expected_curve_count: Any
    # The actual classical-extraction function object for this curve type
    # (e.g. classical.run_classical_pipeline), or None for "model"/"none"
    # entries. The adapter calls this directly — a genuine per-curve-type
    # data lookup, never a hardcoded single import.
    classical_pipeline: Optional[Callable] = None
    # The "model"-dispatch analogue of classical_pipeline: the actual
    # override function object for a "model" entry that needs its own
    # expected-vs-detected safety net instead of the generic run_pipeline
    # call (currently only if_vs_vsd, via model_if_vsd.run_model_pipeline)
    # — None for every plain "model" entry (capacitance_vs_vds,
    # id_vs_vgs), which have no such override and keep routing straight
    # to run_pipeline exactly as before.
    model_pipeline: Optional[Callable] = None


_REGISTRY: Dict[str, ExtractionSpec] = {
    "capacitance_vs_vds": ExtractionSpec(
        curve_type="capacitance_vs_vds",
        method="model",
        # Run A production checkpoint (owner decision, 2026-07-08 — see
        # PROGRESS.md M2): mAP@50 0.88 on the frozen test set.
        checkpoint="run_a/best_segm_mAP_50_iter_1600.pth",
        config="src/training/configs/lineformer_run_a.py",
        score_thr=0.5,
        expected_curve_count=3,
    ),
    "id_vs_vgs": ExtractionSpec(
        curve_type="id_vs_vgs",
        method="model",
        # Run 3 production checkpoint (PROGRESS.md M8, 2026-07-13): mAP@50
        # 0.74 on the combined-batch test split. Trained directly on the
        # GPU box; no config file is committed to this repo yet for that
        # run — this path is a placeholder naming convention, flagged as a
        # follow-up, not an existing file.
        checkpoint="id_vs_vgs_run3_combined_8000iter/best_segm_mAP_50_iter_4800.pth",
        config="src/training/configs/lineformer_id_vs_vgs_run3.py",
        score_thr=0.5,
        expected_curve_count=3,
    ),
    "rdson_vs_tj": ExtractionSpec(
        curve_type="rdson_vs_tj",
        method="classical",
        checkpoint=None,
        config=None,
        # score_thr is unused by run_classical_pipeline (no such parameter)
        # — kept only for a uniform dataclass shape across every entry.
        score_thr=0.5,
        # 1 curve (IR single-curve template) or 2 (Infineon typ/max) are
        # both valid — see src/extraction/classical.py's own
        # EXPECTED_CURVE_COUNT/TWO_CURVE_COUNT handling.
        expected_curve_count=(1, 2),
        classical_pipeline=classical.run_classical_pipeline,
    ),
    "vgsth_vs_tj": ExtractionSpec(
        curve_type="vgsth_vs_tj",
        method="classical",
        checkpoint=None,
        config=None,
        # score_thr is unused by run_classical_pipeline (no such parameter)
        # — kept only for a uniform dataclass shape across every entry.
        score_thr=0.5,
        # Unlike rdson_vs_tj's fixed (1, 2), vgsth's curve count is NOT a
        # fixed set — count_expected_curves(ocr_lines) inside
        # src/extraction/classical_vgsth.py determines it dynamically per
        # chart (band scheme: 1-3; current-value scheme: unbounded). No
        # fixed value would be truthful here.
        expected_curve_count=None,
        classical_pipeline=classical_vgsth.run_classical_pipeline,
    ),
    "if_vs_vsd": ExtractionSpec(
        curve_type="if_vs_vsd",
        method="model",
        checkpoint="body_diode_run1/best_segm_mAP_50_iter_2200.pth",
        config="src/training/configs/lineformer_body_diode_run1.py",
        score_thr=0.5,
        # Real charts show 2 curves (two temperatures, the common case) or
        # 4 (temp + a compound percentile label, seen once) -- but unlike
        # capacitance_vs_vds/id_vs_vgs (plain "model" entries whose fixed
        # int here IS read, by run_pipeline -> process_detections's own
        # count gate), if_vs_vsd's model_pipeline override
        # (model_if_vsd.run_model_pipeline) NEVER reads this field at all
        # -- confirmed directly (grep for the bare `expected_curve_count`
        # identifier in model_if_vsd.py: it appears only as that
        # function's own parameter declaration; every actual
        # process_detections(...) call there passes the locally computed
        # detected count, never this parameter). Its REAL expected count
        # always comes from naming.if_vs_vsd.count_expected_curves(ocr_lines),
        # derived dynamically per chart from the labels themselves. A
        # fixed tuple here would misleadingly look like an enforced rule
        # when nothing checks it -- same reasoning as vgsth_vs_tj's own
        # expected_curve_count=None below.
        expected_curve_count=None,
        classical_pipeline=None,
        model_pipeline=model_if_vsd.run_model_pipeline,
    ),
    "vgs_vs_qg": ExtractionSpec(
        curve_type="vgs_vs_qg",
        method="none",
        checkpoint=None,
        config=None,
        score_thr=0.0,
        expected_curve_count=0,
    ),
}


def get_extraction_spec(curve_type: str) -> ExtractionSpec:
    """Look up the registered extraction spec for ``curve_type``.

    Args:
        curve_type: Registry key, e.g. ``"rdson_vs_tj"``.

    Raises:
        KeyError: If ``curve_type`` is not registered at all (``if_vs_vsd``,
            ``zth_vs_time``, ``vgsth_vs_tj`` today). The message lists every
            registered type so callers/logs are immediately actionable.
            Note this is different from a registered ``method="none"``
            sentinel (``vgs_vs_qg``), which does NOT raise here.
    """
    try:
        return _REGISTRY[curve_type]
    except KeyError:
        raise KeyError(
            f"Unknown curve_type '{curve_type}'. Registered types: {sorted(_REGISTRY)}"
        ) from None
