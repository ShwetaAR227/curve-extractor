"""Stage-5 extraction routing registry (CLAUDE.md §1, §3).

Curve-type -> extraction-path routing is DATA, not code: one
:class:`ExtractionSpec` per curve type, held in a single dict — same
pattern as :mod:`src.classification.curve_registry`. Adding a new curve
type (or flipping a curve type from classical to model, or vice versa) is
a registry entry, never a new if/elif branch in the adapter
(:mod:`src.orchestrator.live_stages`).

Two extraction methods exist today:
- ``"classical"`` — routes to
  :func:`src.extraction.classical.run_classical_pipeline` (OpenCV, no
  GPU). ``checkpoint``/``config`` are ``None`` (nothing to load).
- ``"model"`` — routes to :func:`src.extraction.pipeline.run_pipeline`
  (LineFormer/mmdet). ``checkpoint``/``config`` name the trained weights
  and its config file.

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

``if_vs_vsd``, ``zth_vs_time``, and ``vgsth_vs_tj`` have NO entry — the
real current gap (none of the three has a trained model or a classical
detector yet); ``get_extraction_spec`` raises ``KeyError`` for them, same
as :func:`src.classification.curve_registry.get_spec` does for its own
unregistered types.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ExtractionSpec:
    """Data-only routing entry for one curve type. No behavior lives here."""

    curve_type: str
    method: str  # "classical" | "model" | "none"
    checkpoint: Optional[str]
    config: Optional[str]
    score_thr: float
    expected_curve_count: Any  # int (e.g. 3) or a tuple of allowed counts (e.g. (1, 2))


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
