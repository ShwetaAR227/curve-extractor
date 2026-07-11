"""Stage-6 visual review gallery (CLAUDE.md §1, T19).

A pure VIEWER + approval-recorder over Stage 5's already-saved results.
It never recalculates calibration or re-derives values — a real legacy bug
(the old Stage 6 carried a drifted copy of the calibration math that could
silently disagree with Stage 5's actual output). Overlay projection uses
:func:`src.calibration.ticks.data_to_pixel` with the calibration dict
EXACTLY as Stage 5 stored it.

Buckets:
    needs_review    Stage 5's own status field says so. Always shown in full.
    low_confidence  status ok, but the weakest curve confidence (as stored
                    by Stage 5 — nothing recomputed) is below the threshold.
                    Always shown in full.
    confident       status ok and confident. May be capped via --sample-size
                    (deterministic, seeded) since these need the least
                    attention at scale.

Approve/Reject selections are exported to a review-state JSON (see
:mod:`src.review.review_state`) — never written into Stage 5's outputs.

CLI:
    python -m src.review.gallery <stage5_output_dir> <images_dir>
        --out <gallery_dir> [--sample-size N]
"""
import argparse
import html as html_mod
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2

from src.calibration.ticks import data_to_pixel
from src.common.log import get_logger
from src.dataset_tools.overlay_check import PALETTE, resolve_image
from src.review.review_state import decision_key, load_state

# When run as `python -m src.review.gallery`, __name__ is "__main__", which
# falls outside the "src" logger hierarchy and would silently drop every log
# line (the known Session-1 cvat_to_coco CLI defect — PROGRESS.md). Pin the
# real module path so CLI runs log identically to imported use.
logger = get_logger("src.review.gallery" if __name__ == "__main__" else __name__)

# A result is "low confidence" when its WEAKEST stored curve confidence is
# below this. Run A's real detections score 0.82-0.95 on good extractions
# (T8a); anything under 0.7 has historically been a duplicate/partial.
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.7
# Seed for the deterministic confident-bucket sample (same convention as the
# repo's other seeded sampling, e.g. split_dataset / overlay_check).
DEFAULT_SAMPLE_SEED = 42

BUCKETS = ("needs_review", "low_confidence", "confident")

Result = Dict[str, Any]


def bucket_result(result: Result, threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD) -> str:
    """Assign one Stage-5 result to a review bucket.

    Uses only what Stage 5 already stored: its ``status`` field and the
    per-curve ``confidence`` values. Nothing is recomputed.
    """
    if result["status"] == "needs_review":
        return "needs_review"
    confidences = [c["confidence"] for c in result["curves"]]
    if not confidences or min(confidences) < threshold:
        return "low_confidence"
    return "confident"


def sample_confident(
    items: List[Result], sample_size: Optional[int], seed: int = DEFAULT_SAMPLE_SEED
) -> List[Result]:
    """Deterministically cap the confident bucket at ``sample_size`` items.

    ``None`` means no cap (show everything). Only ever applied to the
    confident bucket — the caller never routes the other buckets here.
    """
    if sample_size is None or len(items) <= sample_size:
        return items
    rng = random.Random(seed)
    sampled = rng.sample(items, sample_size)
    # Keep the original (stable) ordering within the sample.
    order = {id(item): i for i, item in enumerate(items)}
    return sorted(sampled, key=lambda item: order[id(item)])


def _draw_overlay(result: Result, image_path: Path, out_path: Path) -> None:
    """Draw Stage 5's saved curve points onto the source image.

    Projects engineering-unit points back to pixels via the shared
    :func:`data_to_pixel` using the calibration EXACTLY as stored. Results
    without calibration (early needs_review) get a plain copy of the image
    so the reviewer still sees the chart.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    calibration = result.get("calibration")
    if calibration:
        h, w = img.shape[:2]
        for i, curve in enumerate(result["curves"]):
            color = PALETTE[i % len(PALETTE)]
            label_pos = None
            for point in curve["points"]:
                projected = data_to_pixel(point["x"], point["y"], calibration)
                if projected is None:
                    continue
                px, py = int(round(projected[0])), int(round(projected[1]))
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(img, (px, py), 1, color, -1)
                    if label_pos is None:
                        label_pos = (min(max(px + 4, 2), w - 60), min(max(py - 6, 12), h - 4))
            if label_pos is not None:
                # Same halo-label idiom as overlay_check.draw_overlay.
                cv2.putText(img, curve["curve_name"], label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, curve["curve_name"], label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def _item_html(result: Result, overlay_rel: Optional[str], state: Dict[str, Any]) -> str:
    """One gallery card: overlay (or missing-image note) + metadata + controls."""
    device = html_mod.escape(result["device"])
    key = decision_key(result["device"], result["curve_type"])
    key_esc = html_mod.escape(key)
    status = result["status"]
    reason = html_mod.escape(str(result.get("review_reason")))
    units = html_mod.escape(str(result.get("units")))
    curves_bits = ", ".join(
        f"{html_mod.escape(c['curve_name'])} ({c['confidence']:.2f})"
        for c in result["curves"]
    )
    if overlay_rel:
        visual = f'<img src="{html_mod.escape(overlay_rel)}" loading="lazy">'
    else:
        visual = '<div class="missing">source image not found</div>'

    decision = state.get(key, {}).get("decision")
    approve_checked = " checked" if decision == "approve" else ""
    reject_checked = " checked" if decision == "reject" else ""

    return f"""
    <div class="item" data-key="{key_esc}">
      <h3>{device} <span class="{status}">{status}</span></h3>
      {visual}
      <div class="meta">reason: {reason} | units: {units}<br>curves: {curves_bits}</div>
      <div class="controls">
        <label><input type="radio" name="{key_esc}" value="approve"{approve_checked}> Approve</label>
        <label><input type="radio" name="{key_esc}" value="reject"{reject_checked}> Reject</label>
      </div>
    </div>"""


_PAGE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Stage 6 review gallery</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #111; color: #eee; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 17px; border-bottom: 1px solid #444; padding-bottom: 4px; }}
h3 {{ font-size: 14px; margin: 0 0 6px 0; }}
.item {{ border: 1px solid #444; margin: 0 12px 16px 0; padding: 10px; border-radius: 6px;
        display: inline-block; vertical-align: top; max-width: 500px; }}
img {{ max-width: 480px; border: 1px solid #555; background: #fff; }}
.ok {{ color: #7CFC00; }} .needs_review {{ color: #FFD700; }}
.meta {{ font-size: 12px; margin: 6px 0; }}
.missing {{ color: #FF6B6B; padding: 40px; border: 1px dashed #FF6B6B; }}
.controls label {{ margin-right: 14px; }}
#exportbar {{ position: sticky; top: 0; background: #222; padding: 10px; z-index: 5;
             border-bottom: 1px solid #555; margin-bottom: 14px; }}
button {{ padding: 6px 14px; }}
</style></head><body>
<div id="exportbar">
  <button onclick="exportDecisions()">Export decisions JSON</button>
  <span id="tally"></span>
</div>
<h1>Stage 6 review gallery</h1>
<p>{summary_line}</p>
{sections}
<script>
function collect() {{
  const out = {{}};
  document.querySelectorAll('.item').forEach(item => {{
    const key = item.dataset.key;
    const sel = item.querySelector('input[type=radio]:checked');
    if (sel) out[key] = {{decision: sel.value,
                          decided_at: new Date().toISOString()}};
  }});
  return out;
}}
function exportDecisions() {{
  const blob = new Blob([JSON.stringify(collect(), null, 2)],
                        {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'review_state.json';
  a.click();
}}
function refreshTally() {{
  const n = Object.keys(collect()).length;
  document.getElementById('tally').textContent = ` ${{n}} decision(s) selected`;
}}
document.addEventListener('change', refreshTally);
refreshTally();
</script>
</body></html>
"""

_SECTION_TITLES = {
    "needs_review": "Needs review (flagged by Stage 5 — always shown in full)",
    "low_confidence": "Low confidence (ok, but weakest curve below threshold — always shown in full)",
    "confident": "Confident (ok)",
}


def build_gallery(
    stage5_dir: Path,
    images_dirs: Sequence[Path],
    out_dir: Path,
    sample_size: Optional[int] = None,
    threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> Dict[str, Any]:
    """Build the review gallery from a directory of Stage-5 result JSONs.

    Args:
        stage5_dir: Directory containing Stage-5 ``*.json`` results (flat, or
            the ``<device>/<curve_type>.json`` layout from ``result_path``).
        images_dirs: Directories to search for each result's source image.
        out_dir: Output directory (``gallery.html`` + ``overlays/``; an
            existing ``review_state.json`` here pre-fills the controls).
        sample_size: Optional cap on the CONFIDENT bucket only.
        threshold: Low-confidence boundary (min stored curve confidence).
        seed: Sampling seed (deterministic).

    Returns:
        Summary dict: ``counts`` (all results by bucket), ``shown`` (after
        sampling), ``missing_images``, ``html_path``.
    """
    stage5_dir, out_dir = Path(stage5_dir), Path(out_dir)
    result_files = sorted(stage5_dir.rglob("*.json"))
    results: List[Result] = []
    for f in result_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("build_gallery: skipping unreadable %s (%s)", f, exc)
            continue
        if isinstance(data, dict) and "curve_type" in data and "status" in data:
            results.append(data)
        else:
            logger.info("build_gallery: %s is not a Stage-5 result, skipped", f)

    buckets: Dict[str, List[Result]] = {b: [] for b in BUCKETS}
    for result in results:
        buckets[bucket_result(result, threshold)].append(result)
    counts = {b: len(buckets[b]) for b in BUCKETS}
    logger.info("build_gallery: %d result(s) in -> buckets %s", len(results), counts)

    shown: Dict[str, List[Result]] = {
        "needs_review": buckets["needs_review"],
        "low_confidence": buckets["low_confidence"],
        "confident": sample_confident(buckets["confident"], sample_size, seed),
    }
    if sample_size is not None:
        logger.info(
            "build_gallery: confident bucket sampled %d/%d (sample_size=%d, seed=%d)",
            len(shown["confident"]), counts["confident"], sample_size, seed,
        )

    state = load_state(out_dir / "review_state.json")

    out_dir.mkdir(parents=True, exist_ok=True)
    missing_images: List[str] = []
    sections = []
    for bucket in BUCKETS:
        cards = []
        for result in shown[bucket]:
            source_image = result["source_image"]
            image_path = resolve_image(source_image, [Path(d) for d in images_dirs])
            overlay_rel: Optional[str] = None
            if image_path is None:
                missing_images.append(source_image)
                logger.warning("build_gallery: source image not found: %s", source_image)
            else:
                overlay_name = f"{result['device']}__{result['curve_type']}.png"
                try:
                    _draw_overlay(result, image_path, out_dir / "overlays" / overlay_name)
                    overlay_rel = f"overlays/{overlay_name}"
                except ValueError as exc:
                    missing_images.append(source_image)
                    logger.warning("build_gallery: %s unreadable (%s)", image_path, exc)
            cards.append(_item_html(result, overlay_rel, state))
        sections.append(
            f'<h2>{html_mod.escape(_SECTION_TITLES[bucket])} — '
            f'{len(shown[bucket])} shown / {counts[bucket]} total</h2>\n'
            + "\n".join(cards)
        )

    summary_line = (
        f"{len(results)} result(s): "
        + ", ".join(f"{counts[b]} {b}" for b in BUCKETS)
        + (f"; confident bucket sampled to {len(shown['confident'])}"
           if sample_size is not None else "")
        + ". Select Approve/Reject, then Export decisions JSON "
          "(saved separately — Stage 5 outputs are never modified)."
    )
    html_path = out_dir / "gallery.html"
    html_path.write_text(
        _PAGE_TEMPLATE.format(summary_line=html_mod.escape(summary_line),
                              sections="\n".join(sections)),
        encoding="utf-8",
    )
    logger.info("build_gallery: wrote %s (%d missing image(s))",
                html_path, len(missing_images))

    return {
        "counts": counts,
        "shown": {b: len(shown[b]) for b in BUCKETS},
        "missing_images": missing_images,
        "html_path": str(html_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the Stage-6 visual review gallery from Stage-5 outputs.")
    parser.add_argument("stage5_dir", type=Path)
    parser.add_argument("images_dir", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Cap the confident bucket (default: show all)")
    parser.add_argument("--threshold", type=float,
                        default=DEFAULT_LOW_CONFIDENCE_THRESHOLD)
    args = parser.parse_args(argv)

    summary = build_gallery(args.stage5_dir, args.images_dir, args.out,
                            sample_size=args.sample_size, threshold=args.threshold)
    logger.info("gallery done: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
