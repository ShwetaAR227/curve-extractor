"""Live Stage 4 -> Stage 5 wiring adapter (CLAUDE.md §1, stage 7 input).

Replaces :class:`src.orchestrator.pipeline.PrecomputedStage5` (which only
reads pre-made Stage-5 JSONs) with a real adapter matching the SAME
protocol (``run_classification(device)`` / ``run_extraction(device,
classification)``), so ``process_device``/``run_batch`` — frozen,
unmodified — work unchanged regardless of which adapter is injected.

Every stage this module touches is REUSED, never reimplemented:
``classify_device`` (Stage 4, :mod:`src.classification.classify`),
``load_figures_by_page`` (Stage-3 -> Stage-4 input,
:mod:`src.classification.stage3_loader`), ``run_classical_pipeline``
(Stage 5, classical/OpenCV path, :mod:`src.extraction.classical`),
``run_pipeline`` (Stage 5, model/LineFormer path,
:mod:`src.extraction.pipeline`), ``load_model`` (:mod:`src.extraction.inference`).
The classical-vs-model routing decision is a data lookup
(:func:`src.extraction.extraction_registry.get_extraction_spec`), never an
if/elif chain per curve type.

``ocr_lines`` conversion: ``FigureCandidate.ocr_lines`` (classification's
own output) holds :class:`src.classification.scoring.OcrLine` dataclass
instances (``.text``/``.bbox`` attributes). Every extraction-side consumer
(``classical.py``, ``pipeline.py``, ``ticks.py``, ``naming/rdson_vs_tj.py``)
does dict access instead (``{"text": str, "bounding_box": {"x1","y1","x2",
"y2"}}``) — :func:`_convert_ocr_lines` is the ONE conversion function used
identically before EITHER extraction path, never duplicated per route.
"""
import os
from typing import Any, Dict, List, Optional, Set, Union

import cv2

from src.classification.classify import ClassificationResult, ClassificationStatus, classify_device
from src.classification.scoring import OcrLine
from src.classification.stage3_loader import load_figures_by_page
from src.common.log import get_logger
from src.extraction.classical import run_classical_pipeline
from src.extraction.extraction_registry import get_extraction_spec
from src.extraction.inference import load_model
from src.extraction.pipeline import run_pipeline

logger = get_logger(__name__)

STAGE3_ROOT_ENV_VAR = "LINEFORMER_STAGE3_ROOT"


class NoExtractorAvailable(Exception):
    """Raised when a curve type is deliberately registered with NO
    extractor behind it yet (``ExtractionSpec.method == "none"``, e.g.
    ``vgs_vs_qg``) — a known, calm fact distinct from an unregistered
    curve type (which raises ``KeyError`` instead) or a genuine crash."""


class ClaimTracker:
    """Shared mutual-exclusion state across curve types AND adapter
    instances (owner-approved design, 2026-07-16). Wraps a
    ``claimed: Set[figure_id]`` per device, mutated only through explicit
    calls — no global/module-level state, so multiple ``LiveStages``
    instances (one per curve type) can share ONE tracker to prevent two
    curve types from claiming the same figure on the same device.
    """

    def __init__(self) -> None:
        self._claimed_by_device: Dict[str, Set[str]] = {}

    def get(self, device: str) -> Set[str]:
        """Return the figure_ids already claimed for ``device`` (empty set
        if none yet). A copy — callers can't corrupt internal state."""
        return set(self._claimed_by_device.get(device, set()))

    def update(self, device: str, new_claimed: Set[str]) -> None:
        """Add ``new_claimed`` to ``device``'s claimed set (accumulates,
        never overwrites — claims from earlier curve types are kept)."""
        self._claimed_by_device.setdefault(device, set()).update(new_claimed)


def _convert_ocr_lines(ocr_lines: List[OcrLine]) -> List[Dict[str, Any]]:
    """Convert classification's ``OcrLine`` dataclasses into the dict shape
    every extraction-side consumer actually reads (``line["bounding_box"]``
    dict access, confirmed by reading their real body code — see module
    docstring). Builds a NEW list; never mutates the input dataclasses.
    """
    converted = []
    for line in ocr_lines:
        if line.bbox is None:
            converted.append({"text": line.text, "bounding_box": None})
            continue
        x1, y1, x2, y2 = line.bbox
        converted.append({"text": line.text, "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}})
    return converted


class LiveStages:
    """Stage adapter wiring real classification + extraction together.

    One instance targets exactly ONE ``curve_type`` (matching
    ``PrecomputedStage5``'s existing implicit scoping —
    ``process_device`` never passes ``curve_type`` into stage calls).
    Mutual exclusion across MULTIPLE curve types running against the same
    device requires passing the SAME ``claim_tracker`` into every
    per-curve-type instance.
    """

    def __init__(
        self,
        curve_type: str,
        images_root: Union[str, "os.PathLike[str]"],
        stage3_root: Optional[Union[str, "os.PathLike[str]"]] = None,
        claim_tracker: Optional[ClaimTracker] = None,
    ) -> None:
        """
        Args:
            curve_type: Registry key this instance classifies/extracts for.
            images_root: Root directory the rendered figure PNGs live under
                (joined with a figure's ``image_path`` to read the image).
            stage3_root: Root directory of per-device Stage-3 output
                (``<stage3_root>/<device>/full_extraction.json``). Falls
                back to the ``LINEFORMER_STAGE3_ROOT`` env var when omitted
                (CLAUDE.md §3 — never a hardcoded path in source).
            claim_tracker: Shared claim state across curve types. A fresh,
                unshared tracker is created if omitted.

        Raises:
            RuntimeError: Neither ``stage3_root`` nor the env var is set.
        """
        self.curve_type = curve_type
        self.images_root = images_root
        if stage3_root is None:
            stage3_root = os.environ.get(STAGE3_ROOT_ENV_VAR)
            if not stage3_root:
                raise RuntimeError(
                    f"stage3_root not given and {STAGE3_ROOT_ENV_VAR} is not "
                    "set — LiveStages needs a Stage-3 output root."
                )
        self.stage3_root = stage3_root
        self.claim_tracker = claim_tracker if claim_tracker is not None else ClaimTracker()
        self._model: Any = None  # lazy-loaded, cached (one curve_type per instance)
        logger.info(
            "LiveStages(%s): stage3_root=%s images_root=%s claim_tracker=%s",
            curve_type, self.stage3_root, self.images_root,
            "shared" if claim_tracker is not None else "private",
        )

    def run_classification(self, device: str) -> ClassificationResult:
        """Classify ``device`` against this adapter's target curve type.

        Loads Stage-3 figures fresh each call (never cached — a device's
        figures don't change between calls), unwraps ``classify_device``'s
        ``(result, new_claimed)`` tuple into a plain ``ClassificationResult``
        (the shape ``process_device`` expects, via ``getattr(..., "status")``),
        and folds any newly matched claim into the shared tracker.

        Raises:
            KeyError: ``curve_type`` has no classification registry entry
                (propagated from ``classify_device``/``get_spec`` — NOT
                swallowed; ``process_device``'s own try/except is what
                turns this into ``failed_classification``).
        """
        figures_by_page = load_figures_by_page(device, self.stage3_root)
        claimed = self.claim_tracker.get(device)
        result, new_claimed = classify_device(figures_by_page, self.curve_type, claimed)
        self.claim_tracker.update(device, new_claimed)
        logger.info(
            "LiveStages.run_classification(%s, %s): %s (%d page(s) of figures)",
            device, self.curve_type, result.status, len(figures_by_page),
        )
        return result

    def run_extraction(self, device: str, classification: ClassificationResult) -> Dict[str, Any]:
        """Extract curves from the figure ``classification`` matched.

        Routes to the classical (OpenCV) or model (LineFormer) path per
        :func:`get_extraction_spec` — a data lookup, never a hardcoded
        if/elif per curve type.

        Raises:
            ValueError: ``classification.status`` isn't ``"matched"``
                (nothing to extract — callers must not reach here), or the
                registry's ``method`` is neither "classical", "model", nor
                "none".
            NoExtractorAvailable: The curve type is registered with
                ``method="none"`` (a deliberate, known gap — e.g.
                ``vgs_vs_qg`` — distinct from an unregistered curve type,
                which raises ``KeyError``).
            KeyError: ``curve_type`` has no extraction registry entry at all.
        """
        if classification.status != ClassificationStatus.MATCHED:
            raise ValueError(
                f"run_extraction called with non-matched classification "
                f"(status={classification.status}) for device {device!r} — "
                "nothing to extract."
            )
        figure = classification.figure
        spec = get_extraction_spec(self.curve_type)

        if spec.method == "none":
            logger.info(
                "LiveStages.run_extraction(%s, %s): no extractor registered "
                "(method='none') — a known gap, not a crash", device, self.curve_type,
            )
            raise NoExtractorAvailable(
                f"No extractor registered for curve_type '{self.curve_type}'"
            )

        ocr_lines = _convert_ocr_lines(figure.ocr_lines)
        image_full_path = os.path.join(str(self.images_root), figure.image_path)

        if spec.method == "classical":
            image = cv2.imread(image_full_path)
            if image is None:
                raise FileNotFoundError(
                    f"Could not read figure image for classical extraction: {image_full_path}"
                )
            logger.info(
                "LiveStages.run_extraction(%s, %s): routing to classical path (%s)",
                device, self.curve_type, image_full_path,
            )
            result = run_classical_pipeline(
                device=device, curve_type=self.curve_type, source_image=figure.image_path,
                image=image, ocr_lines=ocr_lines,
            )
        elif spec.method == "model":
            if self._model is None:
                logger.info(
                    "LiveStages.run_extraction(%s, %s): lazy-loading model "
                    "(checkpoint=%s, config=%s)", device, self.curve_type,
                    spec.checkpoint, spec.config,
                )
                self._model = load_model(spec.checkpoint, spec.config)
            logger.info(
                "LiveStages.run_extraction(%s, %s): routing to model path (%s)",
                device, self.curve_type, image_full_path,
            )
            result = run_pipeline(
                device=device, curve_type=self.curve_type, image_path=image_full_path,
                ocr_lines=ocr_lines, img_w=figure.figure_width, img_h=figure.figure_height,
                model=self._model, score_thr=spec.score_thr,
                expected_curve_count=spec.expected_curve_count,
            )
        else:
            raise ValueError(
                f"Unroutable extraction method {spec.method!r} for curve_type "
                f"'{self.curve_type}' — expected 'classical', 'model', or 'none'."
            )

        logger.info(
            "LiveStages.run_extraction(%s, %s): status=%s", device, self.curve_type, result["status"],
        )
        return result
