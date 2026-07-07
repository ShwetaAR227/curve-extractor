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
| T4a — Recover annotated source images | ✅ Done | 2026-07-07; `src/dataset_tools/collect_images.py`, 13 tests (TDD red→green, suite now **59 passing**). **164/164 images** collected into `data/images/` from `D:/Extractor/data` (160) + `D:/LineFormerDataset_v2/categorised/1-30/` (final 4); zero hash conflicts across trees; legacy sources untouched. |
| T5 — Group-aware train/val/test split | ✅ Done — **TEST SET FROZEN 2026-07-07** | Owner approved option (b): template-true families via explicit `FAMILY_MERGE_MAP` ({IAUA,IAUAN,IAUC}→IAU, {AUIRF,AUIRFS,AUIRFP,AUIRFZ}→AUIRF, {AUIRL,AUIRLU}→AUIRL, {BSC,BSZ}→BSC-BSZ), BSC-BSZ pinned to train, eval invariant ≥2 families & ≥15 images each for val/test. Seed 42, source `batch_merged.json` (sha256 `76f6806f84a7…`). **Result: train 116 img/348 ann (70.7%), val 24/72 (14.6%), test 24/72 (14.6%)**; 12 families, all three parts validate clean, ids disjoint. Files: `data/coco/split/{train,val,test}.json` + `split_manifest.json` — **manifest sha256 `967aa00f87d6cf8be9b12caed8e0422a7c0365ab4bfa28023fc9a1f4e000180e`**. Changing the test set now requires owner approval (CLAUDE.md §4). Suite: **82 passing**. |
| T4 — Overlay visual check of buffered masks | ✅ Done (owner decided) | 2026-07-07; `src/dataset_tools/overlay_check.py` + three-radius comparison sheet (`data/overlay_check/overlay_check.html`, seed 42, sources from `data/images/`). **Owner decision: buffer radius = 4.5 px is canonical** — annotation jitter is 1–2 px and under-coverage of the stroke is worse than over-coverage. `DEFAULT_BUFFER_PX = 4.5` pinned by test (red→green); `data/coco/batch_merged.json` regenerated at 4.5 px — 164/492, validate clean; `compare_buf*.json` scratch files deleted. Suite: **60 passing**. |

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
