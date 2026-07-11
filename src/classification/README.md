# src/classification — Stage 4: curve-type classification

Given a datasheet page's extracted figures (image crops + OCR text/bboxes,
from stage 3), decides which figure (if any) matches a target curve type,
with a confidence score and audit trail, and supports mutual exclusion so
two curve types can't claim the same figure.

Text/OCR-only — no visual/image classifier. Every curve type has a distinct
vocabulary (caption wording, axis labels), which is enough signal on its own.

## Modules

- `curve_registry.py` — data-only curve-type "fingerprints" (`CurveTypeSpec`).
  No logic lives here, only keyword/phrase data.
- `scoring.py` — one shared `score_figure(figure, spec)` that scores ANY
  figure against ANY spec. Combines caption keyword hits, position-aware
  axis-label keyword hits (a bbox-shape/edge-proximity heuristic separates
  y-axis labels — tall & narrow, near the left edge — from x-axis labels —
  wide & short, near the bottom edge), and weighted positive/negative phrase
  matches. Deterministic, auditable (`ScoreResult.matched_signals`).
- `classify.py` — `classify_page` ranks every unclaimed figure on one page;
  `classify_device` picks the single best result across all of a device's
  pages. Returns a `matched` / `quarantined` / `no_match` status — ambiguous
  or low-confidence results are quarantined for human review, never
  silently dropped or force-guessed. Mutual exclusion across curve types is
  explicit: a `claimed: set[figure_id]` is passed in and a new one is
  returned; there is no global state or monkey-patching.

## Adding a new curve type

Add one entry to `_REGISTRY` in `curve_registry.py`:

```python
"rdson_vs_tj": CurveTypeSpec(
    name="rdson_vs_tj",
    caption_keywords=[...],           # exact wording confirmed against real datasheet OCR
    axis_keywords={"x": [...], "y": [...]},
    positive_phrases=[("keyword", weight), ...],
    negative_phrases=[("keyword that indicates a DIFFERENT curve type", weight), ...],
)
```

That's it — `scoring.py` and `classify.py` need no changes. Before committing
the wording, confirm captions/axis-label text against a few real
`full_extraction.json` samples for that curve type (don't guess from the
curve-type name alone — see the `id_vs_vgs` entry for the pattern: it was
checked against ~90 real transfer-characteristics figures before being
written).

## Not yet wired up

This package is not yet connected to real stage-3 data or run on real
devices — see `PROGRESS.md`. That's a follow-up task once the owner
confirms the `id_vs_vgs` registry entry against real datasheets.
