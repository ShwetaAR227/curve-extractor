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
| Legacy capacitance JSONs (`data/legacy/capacitance_*.json`) | ❌ REJECTED for training — permanent (owner, 2026-07-07) | Legacy annotations inspected, 0/5 samples correct, wrong-figure and offset errors confirmed — excluded from all training data. 49 non-overlapping images noted as low-priority re-annotation candidates (44 are BSC-BSZ, would inflate dominant family). Overlay evidence: `data/overlay_check/legacy_capacitance/`. |
| T4a — Recover annotated source images | ✅ Done | 2026-07-07; `src/dataset_tools/collect_images.py`, 13 tests (TDD red→green, suite now **59 passing**). **164/164 images** collected into `data/images/` from `D:/Extractor/data` (160) + `D:/LineFormerDataset_v2/categorised/1-30/` (final 4); zero hash conflicts across trees; legacy sources untouched. |
| T5 — Group-aware train/val/test split | ✅ Done — **TEST SET FROZEN 2026-07-07** | Owner approved option (b): template-true families via explicit `FAMILY_MERGE_MAP` ({IAUA,IAUAN,IAUC}→IAU, {AUIRF,AUIRFS,AUIRFP,AUIRFZ}→AUIRF, {AUIRL,AUIRLU}→AUIRL, {BSC,BSZ}→BSC-BSZ), BSC-BSZ pinned to train, eval invariant ≥2 families & ≥15 images each for val/test. Seed 42, source `batch_merged.json` (sha256 `76f6806f84a7…`). **Result: train 116 img/348 ann (70.7%), val 24/72 (14.6%), test 24/72 (14.6%)**; 12 families, all three parts validate clean, ids disjoint. Files: `data/coco/split/{train,val,test}.json` + `split_manifest.json` — **manifest sha256 `967aa00f87d6cf8be9b12caed8e0422a7c0365ab4bfa28023fc9a1f4e000180e`**. Changing the test set now requires owner approval (CLAUDE.md §4). Suite: **82 passing**. |
| T4 — Overlay visual check of buffered masks | ✅ Done (owner decided) | 2026-07-07; `src/dataset_tools/overlay_check.py` + three-radius comparison sheet (`data/overlay_check/overlay_check.html`, seed 42, sources from `data/images/`). **Owner decision: buffer radius = 4.5 px is canonical** — annotation jitter is 1–2 px and under-coverage of the stroke is worse than over-coverage. `DEFAULT_BUFFER_PX = 4.5` pinned by test (red→green); `data/coco/batch_merged.json` regenerated at 4.5 px — 164/492, validate clean; `compare_buf*.json` scratch files deleted. Suite: **60 passing**. |

### ⚠ OPEN SECURITY ITEM

The legacy repo contains a **committed AWS private key** (`aws_key/lineformer-key.pem`).
**Owner must rotate/revoke the key in AWS and scrub git history.**
Track this item until closed. Opened: 2026-07-07. Status: **OPEN**.

## M2 — LineFormer Retraining

| Task | Status | Notes |
|---|---|---|
| T6 — Training env on AWS GPU box | ✅ **DONE** (fp32-only) | 2026-07-08. GPU box (g4dn.xlarge, T4 16 GB) reached via `aws-lineformer`. Root EBS (25 GB) proved too small mid-install (`ENOSPC`); owner attached + we formatted/mounted a **116 GB volume at `/mnt/data`** (XFS, fstab-persisted by UUID), redirected conda envs/pkgs cache and the checkpoint there, root left untouched. `lineformer` conda env built to the exact owner-approved pins (python 3.8, torch 1.13.1+cu117, mmcv-full 1.7.1 via openmim, mmdet 2.28.2, scipy 1.9.3). Two real environment bugs found and fixed in the scripts: (1) miniconda PATH-detection false-reinstall in non-interactive SSH shells, (2) a pre-existing `~/.local` user-site torch+mmcv (leftover from an unrelated prior capacitance-training run on this box) silently shadowing/skipping the env's own pinned packages — fixed via `PYTHONNOUSERSITE=1`. **Smoke tests 1–3 pass**: GPU/torch visibility (Tesla T4, torch 1.13.1/cu117), mmdet 2.28.2 + mmcv-full 1.7.1 CUDA ops (`RoIAlign` import OK), real `inference_detector` on `AUIRF1010EZS__fig_p4_012.png` → **100 masks, peak GPU memory 1018 MiB**, visualization saved. **Smoke test 4 (fp16 autocast) is N/A by owner decision**: mmcv-full 1.7.1's deformable-attn CUDA op has no half-precision kernel (`RuntimeError: "ms_deform_attn_forward_cuda" not implemented for 'Half'`) — architecture limitation, not fixable by re-pinning. **Training will run fp32-only**; ~1 GiB peak inference memory on a 15 GiB GPU leaves ample headroom. Provenance: LineFormer commit `7952e27b4653dea025394618fbd655f41d82ab6b`; checkpoint `lineformer_pretrained_official_iter3000.pth` from the official README's "Inference" step-1 Drive link (file id `1cIWM7lTisd1GajDR98IymDssvvLAKH1n`), sha256 `ac03d7d52a11ce253350bf4bc73416e42ac68021c00bcce14d47fcc28ec65eb0` (verified after both the initial download and the box scp — no transit corruption). `envs/lineformer.lock.yml` + `envs/lineformer.commit` pulled back from the box and committed for reproducibility. |

| T7a — Eval script + baselines | ✅ Done — **numbers to beat below** | 2026-07-08; `src/training/eval_lineformer.py` (18 metric tests TDD red→green, suite **100 passing**; COCO mask AP via pycocotools, never hand-rolled). Evaluated on the frozen test split (24 img / 72 GT, sha256 `2a262e2c5c07…`). **Baseline 1 — official pretrained** (`lineformer_pretrained_official_iter3000.pth` + `lineformer_swin_t_config.py`): mAP@50 **0.1008**, mAP@75 **0.0000**, recall@(score≥.5,IoU≥.5) **0.2222** (Ciss 7/30, Coss 3/21, Crss 6/21). **Baseline 2 — legacy ckpt trained on rejected data** (`~/checkpoints/iter_10000.pth` → symlinked `legacy_flawed_data_iter10000.pth`, its own `lineformer_cap_config.py`): mAP@50 **0.0000**, mAP@75 **0.0000**, recall **0.0000** — sanity-probed: model outputs 100 confident preds/img (scores to 0.92) but masks are offset/ballooned blobs that never reach IoU 0.5 vs our 4.5px GT (visual evidence `data/eval/viz_legacy_BSP125.png`). Legacy 73% mAP@50 claim does not survive contact with a leakage-free test set. Reports: `data/eval/baseline{1,2}*.json` (git-ignored; numbers recorded here). |

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
