# PROGRESS.md — Single Source of Truth

> Rules: update at the start and end of every working session. A task is **Done** only when its
> tests pass **and** the owner approved. Never delete history (see `CLAUDE.md` §8).

## M0 — Governance

| Task | Status | Notes |
|---|---|---|
| T0 — Governance files (CLAUDE.md, SETUP.md, PROGRESS.md, scaffolding) | ✅ Done (pending owner review) | 2026-07-07, commit `adcb2f2` |

## M1 — Data Foundation

| Task | Status | Notes |
|---|---|---|
| CVAT project set up | ✅ Done | cvat.ai cloud; label `line`, polyline + attribute `curve_name` |
| 7-image pilot batch annotated & validated | ✅ Done | 21 polylines; export format verified |
| 80-device batch annotated | ✅ Done | Pending conversion (blocked on T2) |
| T2 — CVAT XML → COCO converter | ✅ Done (pending owner review) | 2026-07-07; `src/cvat_to_coco.py`, **32 tests passing** (TDD: red confirmed before implementation). ⚠ T2 spec was reconstructed from the task outline — the verbatim spec from the earlier chat never reached this session; owner must confirm API/behavior match. |
| T3 — Merge mode (multi-export → one COCO) | ✅ Done | 2026-07-07; `merge_convert()` + multi-input CLI, **46 tests passing** (14 new, TDD red→green). Batch merged: `data/coco/batch_merged.json` = **164 images / 492 annotations** (matches owner's expected counts), 148 unannotated frames dropped, 0 validation errors. Duplicate-file_name hard error verified live against the pilot overlap. |
| Legacy review | ✅ Done | See `LEGACY_REVIEW.md` |
| T4 — Overlay visual check of buffered masks | 🔍 Awaiting owner review | 2026-07-07; `src/dataset_tools/overlay_check.py` (TDD waived by owner for this script). 6 overlays + contact sheet at `data/overlay_check/overlay_check.html`, rendered from `batch_merged.json` @ 3.0 px buffer. Owner decides: keep 3.0 px or adjust. Source figures resolved from the `datasheet-studio-v2` cache (160/164 covered; 4 batch images have no matching local figure crop — different OCR-run numbering in backup caches). |

### ⚠ OPEN SECURITY ITEM

The legacy repo contains a **committed AWS private key** (`aws_key/lineformer-key.pem`).
**Owner must rotate/revoke the key in AWS and scrub git history.**
Track this item until closed. Opened: 2026-07-07. Status: **OPEN**.

## Upcoming

- Train/val/test split
- LineFormer retraining
- Stages 4–7 rebuild

## Session log

### 2026-07-07 — Session 1
- **Start:** empty repo. Goals: T0 (governance) + T2 (CVAT→COCO converter, TDD).
- **End:** T0 complete (governance files, scaffolding, `.venv`, pinned deps, git init).
  T2 complete: `src/cvat_to_coco.py` (parse_cvat_xml, buffer_polyline, polygon_area,
  bbox_from_segmentation, convert, validate_coco, CLI) + `src/common/log.py` (shared
  logging per CLAUDE.md §7). 32 tests written first, confirmed red, then implemented
  to green (`pytest -v`: 32 passed). CLI smoke-tested end-to-end on the committed
  fixture. NOTE: the verbatim T2 spec from the design chat was not delivered to this
  session (placeholder in prompt); the implementation follows the task outline —
  owner review required to confirm it matches the approved design.
- **Later same day:** converted all five CVAT exports (four batch tasks + 7-image
  pilot job 4199936); all validate clean. Owner decisions applied: pilot XML is a
  committed test fixture only (never training data); four batch exports moved to
  `data/cvat_exports/` (git-ignored); legacy JSONs and the `Annotate` device-data
  dump moved under `data/`. T3 implemented TDD (14 new tests, suite now 46):
  `merge_convert()` renumbers ids, drops unannotated frames, hard-errors on
  duplicate file_names. Merged batch = 164 images / 492 annotations ✓.
- **Known open defect (owner approval needed to fix):** running the CLI as
  `python -m src.cvat_to_coco` names the module logger `__main__`, which bypasses
  the `src` logger hierarchy — INFO logs (conversion report) are lost on CLI runs
  and never reach `logs/pipeline.log`. Violates CLAUDE.md §7. Fix + regression
  test pending approval.
