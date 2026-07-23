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
| T7b — Run A: fine-tune on real data (115 train / 25 val) | ✅ Done — **beats both baselines decisively** | 2026-07-08; `src/training/configs/lineformer_run_a.py` (owner-specified overrides on the upstream config via `_base_`), `src/training/early_stopping.py` + `checkpoint_retention.py` (24 tests TDD red→green; suite **124 passing**), custom `EarlyStoppingDivergenceHook` (`src/training/hooks/`), `src/training/train_lineformer.py` orchestrator. Init = official pretrained (sha256-verified `ac03d7d5…`, NOT the legacy checkpoint). LR = base×0.05 = 5e-6; batch size 1; fp32 only (no autocast/GradScaler anywhere — confirmed by code review, consistent with T6's fp16-kernel-gap finding); horizontal+vertical flip disabled via `RandomFlip(flip_ratio=0.0)` (curve-physics requirement — removing the transform entirely broke mmdet's `Collect` data contract, found via integration testing; `flip_ratio=0.0` is mmdet's own deterministic disable idiom, verified against its source); multi-scale resize + brightness/contrast jitter added; checkpoint retention = native `max_keep_ckpts=1` + `save_best='segm_mAP_50'`, audited post-run via the tested `checkpoint_retention.plan_retention` (0 unexpected files). **Training curve** (iter: train_loss / val mAP@50): 200: 38.4/0.190 · 400: 32.0/0.260 · 600: 29.4/0.301 · 800: 28.9/0.353 · 1000: 29.5/0.420 · 1200: 28.8/0.429 · 1400: 29.3/0.456 · **1600: 27.1/0.500 (best)** · 1800: 27.5/0.437 · 2000: 29.9/0.499. Ran the full 2000-iter ceiling — **early stopping did NOT trigger** (best iters_since_best reached 400 at the end, under the 600-iter patience); no divergence flag raised (the iter-1800 dip didn't co-occur with falling train loss, so it read as noise, not overfitting — logic verified separately via a mock-runner test forcing an actual stop, confirmed working). Wall-clock **742.6 s** (~12.4 min), peak GPU memory **2412.7 MiB**. **Frozen test set (24 img/72 GT) — best checkpoint (iter 1600):** mAP@50 **0.8774**, mAP@75 **0.0010**, recall@(score≥.5,IoU≥.5) **0.9306** overall (Ciss 27/30=0.900, Coss 20/21=0.952, Crss 20/21=0.952) — vs Baseline 1 (mAP@50 0.1008, recall 0.2222) and Baseline 2 (all zero). mAP@75 near-zero flags a real gap: masks are found and roughly localized with high confidence but rarely tight enough for IoU≥0.75 — worth watching in any future run, not a red flag on its own given the 4.5px GT buffer width. Full manifest (all hyperparameters, dataset/checkpoint hashes, seed) at `data/training_runs/run_a/run_manifest.json`; report at `data/eval/run_a_final.json` (both git-ignored; numbers recorded here). One integration-only finding, not yet acted on: LineFormer's own `install.sh` expects its *vendored, patched* mmdetection (`pip install -e mmdetection`), not the plain PyPI `mmdet==2.28.2` T6 pinned — same version string, but a patched `RandomCrop` (`crop_ratio` probability gate) that PyPI's lacks. Worked around by dropping `crop_ratio` (crop always applies) rather than switching environments; flagged for awareness, not a blocker. **STOPPED per owner instruction — no synthetic data work without review of these numbers.** |

| T8a — Pre-annotation quality check on unannotated figures | 🔍 Reported — **owner judges by eye** | 2026-07-08; ran Run A best checkpoint (iter 1600) on 13 real, unannotated figures from `D:\LineFormerDataset_v2\categorised\capacitance` (393 total, all Infineon-family — no onsemi/ST/Vishay/Toshiba/Nexperia/Rohm present, noted explicitly). Confirmed zero overlap with the 164 CVAT-annotated images or the frozen split (not assumed). Selection prioritized the 14-file "iS..." naming family (most visually distinct) plus one image per remaining prefix (13 distinct device families, none seen in training). **Result: 10/13 images got exactly 3 tight, high-confidence (0.82–0.93) detections matching Ciss/Coss/Crss**, including on chart styles/fonts not in training. **2 images over-detected** (4 and 5 preds) — both traced to a near-duplicate second mask specifically on the flat Ciss plateau, not a false curve elsewhere; all real curves still tightly tracked. **1 image under-detected** (2 preds, one weak at 0.40) — traced to a wrong-page pull (the figure is a breakdown-voltage plot, not capacitance, likely a folder-categorization/page-adjacency artifact, not a model failure) — model correctly avoided hallucinating 3 curves on a 1-curve chart. No clear garbage/complete failures. Contact sheet + overlays: `data/t8a_overlay/t8a_contact_sheet.html` (git-ignored). Owner to judge whether this is good enough to pre-annotate from. |

| T8b — CVAT pre-annotation import tool | ✅ Done — **393-image batch generated** | 2026-07-08; owner approved T8a predictions as good enough to pre-annotate from. `src/training/predict_to_cvat.py` (19 tests TDD red→green; suite **143 passing**; round-trips through the existing `parse_cvat_xml`, confirming CVAT-1.1 compatibility). Predicted masks (score ≥ 0.5) are emitted as CVAT **polygons** (mask contour, `cv2.findContours`+`approxPolyDP`) rather than skeletonized polylines — matches the "polygon for thick bands" half of our existing convention, avoids skeleton-branch artifacts at endpoints/crossings, and is just as easy to drag-correct in CVAT. Every polygon's `curve_name` is the literal placeholder `"TODO"` (model has no Ciss/Coss/Crss classes) — **annotators must replace every "TODO" before export**; noted that `cvat_to_coco.py` only checks for *empty* curve_name, not a fixed vocabulary, so a forgotten "TODO" would currently slip through — flagged, not fixed (frozen stage, needs owner sign-off per CLAUDE.md §4). **Known limitation documented, NOT auto-filtered per owner instruction:** duplicate masks on flat/low-texture curves. Ran on the full 393-image target (`D:\LineFormerDataset_v2\categorised\capacitance`, confirmed zero overlap with 164/split in T8a): **393/393 processed, 0 total failures, 1193 polygons, 346/393 (88%) got exactly the expected 3.** Spot-checked outliers and found 3 distinct patterns beyond simple "duplicate on flat curve": (1) a wrong-page/mis-categorized source figure for the whole IPT-family page-023 group (breakdown-voltage chart, not capacitance — confirmed on 3 separate devices, a data curation issue not a model failure); (2) spurious duplicate detections locking onto the figure's **caption text band** at the bottom of the image, one scoring as high as 0.86 — a new pattern distinct from the T8a flat-Ciss duplicate; (3) a genuinely-different chart template (IPD65R.../IPP65R... "Typ. capacitances" boxed-header family) with **more than 3 real traces** — the model correctly found extra real curves rather than hallucinating, but this breaks the 3-curves-per-chart assumption for that template; (4) one confirmed genuine miss — a flat Ciss curve completely undetected while Coss/Crss were found correctly. Output: `data/t8b_preannotations.xml` (397 KB, git-ignored). CVAT import: create a task with the 393 source images, then Actions → Upload annotations → format "CVAT 1.1" → select this file. |

| T8c — Validate semi-auto-corrected CVAT export (pre-merge check), pass 1 | ⚠ NOT READY (superseded by pass 2 below) | 2026-07-08; inspected `job_4208477` export v1 (393 images, 1185 shapes: 88 polylines, 1097 polygons) staged at `data/staging/job_4208477/` (git-ignored, not merged). Label check clean (100% `"line"`). Cross-checks verified: 0 duplicate file_names, 0 overlap with the 164 CVAT-annotated images, 0 overlap with the frozen split. **Blocking: 593/1185 shapes (50%) still "TODO", across 192/393 images.** 5 images with zero shapes. Not converted/merged — report only. |
| T8c — Validate semi-auto-corrected CVAT export, **pass 2 (re-export)** | 🔍 **Nearly merge-ready — 17 TODOs remain, owner go-ahead needed to merge** | 2026-07-08; re-inspected re-exported `job_4208477` v2 (393 images, 1145 shapes: 129 polylines, 1016 polygons) staged at `data/staging/job_4208477_v2/` (git-ignored). **Massive improvement**: TODOs down from 593→**17**, affected images down from 192→**6** (`IPD65R660CFDAATMA1__fig_p10_025.png`, `IPDQ60R040S7XTMA1__fig_p10_027.png`, `IPL60R285P7AUMA1__fig_p10_028.png`, `IPP030N06NF2SAKMA1__fig_p9_023.png`, `IPP65R150CFDAAKSA1__fig_p11_025.png`, `IRFB3006PBF__fig_p3_006.png`) — 3 of these are exactly the "genuinely >3 curves" / "caption-duplicate" atypical images already flagged in T8b, consistent with them being the hardest remaining cases rather than a regression. curve_name distribution: Crss 378, Coss 378, Ciss 372, TODO 17. Per-image distribution: 0→13, **3→378 (96%)**, 5→1, 6→1. Label check clean (100% `"line"`). Cross-checks re-verified clean: 0 duplicates, 0 overlap with 164/split. **The 13 zero-shape images (8 new since pass 1) visually confirmed genuinely non-capacitance charts** — 10 are "Diagram: Drain-Source breakdown voltage" (matches the known IPT-family wrong-page issue, now also caught on a few IPD/IPL/IPTG devices) and the original 2 IPPithreshold-voltage images are "Typ. gate threshold voltage" — annotator correctly emptied these rather than mislabeling; not an outstanding issue. **Verdict: NOT YET fully merge-ready** — 17 TODOs across 6 images still need real curve_name labels. Not converted/merged — report only, per brainstorm-first rule; awaiting owner go-ahead once TODOs hit zero. |

| T8d — Strip leftover TODO shapes + final re-validation | ✅ **MERGE-READY — awaiting owner go-ahead to convert/merge** | 2026-07-08; `src/dataset_tools/strip_todo_shapes.py` (8 tests TDD red→green; suite **151 passing**). Removes whole `<polyline>`/`<polygon>` elements whose `curve_name` is exactly `"TODO"`, leaves every other shape/image byte-identical, never touches the input file, writes a new file. Ran on `job_4208477` v2: **17 shapes removed from exactly the 6 flagged images** (`IPD65R660CFDAATMA1__fig_p10_025.png`, `IPDQ60R040S7XTMA1__fig_p10_027.png`, `IPL60R285P7AUMA1__fig_p10_028.png`, `IPP030N06NF2SAKMA1__fig_p9_023.png`, `IPP65R150CFDAAKSA1__fig_p11_025.png`, `IRFB3006PBF__fig_p3_006.png`) → `data/t8b_corrected_cleaned.xml` (git-ignored). **Re-validation, all clean: TODO count = 0** (Ciss 372, Coss 378, Crss 378, total 1128 shapes across 393 images); per-image distribution 0→16 (all confirmed genuinely non-capacitance charts across T8c passes 1–2), 2→3 (the 3 partially-TODO images, now correctly reduced), 3→374; 0 duplicate file_names; 0 overlap with the 164 CVAT-annotated images or the frozen train/val/test split. **Verdict: MERGE-READY.** Not converted/merged — stopped per the brainstorm-first rule, awaiting owner go-ahead. |

| T8e — Convert + merge semiauto batch1 into training data | ✅ **Done — merged, awaiting owner review before Run B** | 2026-07-08; owner approved `data/t8b_corrected_cleaned.xml` as merge-ready. Converted via `cvat_to_coco.merge_convert` (buffer 4.5px, same rules as always) → `data/coco/semiauto_batch1.json`: **377 images, 1128 annotations** (393 − 16 confirmed-non-capacitance images), `validate_coco` clean, curve_name breakdown Ciss 372/Coss 378/Crss 378. Combined with the four original batch exports → `data/coco/combined_pool.json`: **541 images, 1620 annotations** (164+377, 492+1128 — exact), clean. **New-batch split** (family-based, same `assign_family` heuristic as T5, reused not duplicated; test.json never touched by design): 29 families in the new batch, largest IPD at 82/377 (21.8%, no concentration flag); routed the 13 smallest families (37 images) to **val** for family-diversity in validation, the 16 larger families (340 images, 90.2%) to **train**. Family→side assignment fully recorded in `split_manifest.json` for reproducibility. **Final split after merge:** train **456 img / 1365 ann** (was 116/348), val **61 img / 183 ann** (was 24/72), **test UNCHANGED: 24 img / 72 ann, sha256 `2a262e2c5c077a3eb09c8bd6fd253b1b21d7d3f4c05c14b10da770c6c1cbccdc`** — verified identical to the T7a-recorded frozen-test hash, not just assumed. Cross-checked zero overlap between all three split files, total exactly 541. All three `validate_coco`-clean. **Not retrained yet — stopped per instruction; owner reviews these numbers before Run B (real + semiauto + synthetic) is scoped.** |

| T8f — Quick overlay check on owner-supplied image | 🔍 Reported — **owner judges by eye** | 2026-07-08; folder `D:\LineFormerDataset_v2\categorised\capacitance\test` contained **1 image** (`Screenshot 2026-07-08 184718.png`, no device name identifiable — a screenshot crop, not an extracted PDF figure), confirmed not in the 164 CVAT-annotated images or the frozen split. Ran Run A best checkpoint (iter 1600): **3/3 curves found, tight masks, high confidence (0.91/0.88/0.87)** — on a chart style genuinely unlike training data (log-log axes, no boxed diagram header, different font/layout, likely a non-Infineon manufacturer given the style, though unconfirmed since no device name is visible). Overlay: `data/t8f_overlay/t8f_contact_sheet.html`. No failures. |

| T9 — Promote ad-hoc new-batch split into a tested tool | ✅ Done | 2026-07-08; extended `src/dataset_tools/split_dataset.py` (same module, reuses `assign_family`/`extract_device`, no duplication) with `allocate_new_batch()` (pure: routes smallest families to val via `--val-families N` or `--val-images N` mode, exactly one required; families sorted ascending by size with alphabetical tie-break for determinism; hard invariant — no family straddles train/val, no path can write test) and `merge_new_batch_into_split()` (adds to existing train.json/val.json via id-renumbering-safe combine, copies test.json byte-for-byte and hard-errors if its hash ever changes). New CLI subcommand: `python -m src.dataset_tools.split_dataset allocate-new-batch <new_batch.json> <existing_split_dir> --val-families N|--val-images N --out <dir>`, dispatched without breaking the original CLI contract (12 new tests, TDD red→green; suite **162 passing**). **Retroactive verification: re-running the new tool on `semiauto_batch1.json` (same `val_image_target=38`, i.e. `round(0.10×377)`) reproduces the exact same family→side split as the ad-hoc script from commit `2c33a17`** — identical 13 val / 16 train families, identical 37/340 image counts. Confirmed, not assumed. |

| T10 — Run A2: fine-tune on expanded dataset (456/61) | ✅ Done — **underperforms Run A, flagged for review** | 2026-07-08; `src/training/configs/lineformer_run_a2.py` (chained from `run_a.py` via `_base_`, only schedule numbers changed), `status.txt` one-line progress file added (`format_status_line`, 4 new tests; suite **166 passing**). **Iteration scaling reasoned explicitly** (not copied blindly): train set grew 116→456 images (×3.93); kept Run A's raw iteration counts unchanged would give only ~4.4 total epochs / ~0.44 epochs-per-eval / ~1.3 epochs patience vs Run A's ~17.2/~1.72/~5.17 — so every schedule number was scaled by the same ×3.93 ratio: **max_iters 2000→8000, eval/checkpoint interval 200→800, patience 600→2400**, preserving Run A's relative schedule (same ~17.5 total epochs, same 10 eval checkpoints, same 3-eval-interval patience in epoch-equivalent terms). Same init weights as Run A (official pretrained, sha256 `ac03d7d5…` — confirmed identical, apples-to-apples), same LR (5e-6), fp32, batch 1, same augmentation. **Confirmed the new data was actually used**: train_ann_sha256 `dc2b80b1…` and val_ann_sha256 `56fac6ba…` differ from Run A's recorded `68e33fd6…`/`625ac185…`. **Training curve** (iter: loss/val mAP@50): 800: 35.4/0.478 · **1600: 30.5/0.5235 (best)** · 2400: 30.8/0.501 · 3200: 29.4/0.477 · 4000: 28.6/0.520. **Early stopping triggered at iter 4000/8000** (iters_since_best reached patience=2400 exactly) — confirmed via wall-clock (1434.5s, ≈half the 8000-iter budget), checkpoint audit (latest=iter_4000), and `status.txt` content, though the "EARLY STOPPING" warning-level log line itself didn't appear in the captured console output despite the same-logger info lines showing fine — a minor logging-visibility gap, not a functional one (the stop mechanism itself is independently confirmed correct). No divergence flag raised. **Frozen test set (best checkpoint, iter 1600):** mAP@50 **0.7011**, mAP@75 **0.0003**, recall **0.8333** (Ciss 22/30=0.733, Coss 18/21=0.857, Crss 20/21=0.952) — test.json hash re-verified `2a262e2c5c07…`, unchanged. Wall-clock **1434.5 s**, peak GPU memory **2413.6 MiB** (~same as Run A, as expected). **Headline finding: Run A2 underperforms Run A** (mAP@50 0.70 vs 0.88, recall 0.83 vs 0.93) **despite ~4x more training data with a proportionally-scaled schedule** — val mAP@50 peaked early (iter 1600, only ~3.5 epochs into an 8000-iter/~17.5-epoch budget) and never improved across the next 2400 iters until patience ran out. Hypotheses for the owner to weigh, not asserted causes: semi-auto-batch annotation quality (even post-TODO-stripping) may be noisier than the original hand-annotated 116; the new families (IPD/IPP/IRF/etc., mostly larger power packages) may differ stylistically enough from the test-set families (BSP/BSS/IRFL/BTS/BSF/BSR, mostly small-signal) that added exposure diluted rather than reinforced test-relevant features. Reassuring cross-run consistency: in BOTH runs, val mAP@50 tops out far below the test mAP@50 achieved on the same checkpoint (Run A: val 0.50 → test 0.88; Run A2: val 0.52 → test 0.70) — a repeatable pattern across runs, not a one-off anomaly, consistent with val simply being a harder/more diverse family mix than test in this split. Full manifest at `data/training_runs/run_a2/run_manifest.json`; report at `data/eval/run_a2_final.json` (git-ignored; numbers recorded here). **STOPPED per instruction — no synthetic-data work without owner review of this regression.** |

### Baseline / Run comparison (frozen test set, 24 images / 72 GT)

| Metric | Baseline 1 (official) | Baseline 2 (legacy) | Run A (116/24) | Run A2 (456/61) |
|---|---|---|---|---|
| mAP@50 | 0.1008 | 0.0000 | **0.8774** | 0.7011 |
| mAP@75 | 0.0000 | 0.0000 | 0.0010 | 0.0003 |
| Recall overall | 0.2222 | 0.0000 | **0.9306** | 0.8333 |
| — Ciss | 0.2333 | 0.0000 | 0.9000 | 0.7333 |
| — Coss | 0.1429 | 0.0000 | 0.9524 | 0.8571 |
| — Crss | 0.2857 | 0.0000 | 0.9524 | 0.9524 |
| Wall-clock | — | — | 742.6 s | 1434.5 s |
| Peak GPU mem | — | — | 2412.7 MiB | 2413.6 MiB |

| T11 — Diagnose Run A2 regression (bad labels vs. family mismatch) | 🔍 Reported — **not the labels; points to AUIRL-specific OOD collapse** | 2026-07-08; pure diagnosis, no retraining/data changes, per instruction. **(1) Per-family breakdown** (diagnostic script, not committed as permanent tooling — reused tested `compute_recall`/`mask_iou` from `eval_lineformer.py`): regression is **not uniform**. BSF/BSR/BTS stay perfect (1.0/1.0) in both runs; BSP 1.0→0.926, BSS 1.0→0.952 (small erosion); **AUIRL collapses: recall 0.583→0.250, mAP@50 0.383→0.089** — already the weakest family in Run A, far worse in Run A2. This one family (12/72 GT instances) accounts for most of the aggregate drop. **(2) GT visual audit**: 20-image sample from `semiauto_batch1.json` (including all 4 `IRFL`-family images — closest name to `AUIRL`, checked specifically), overlaid at `data/gt_audit/gt_audit_contact_sheet.html`. **All 20 clean**: exactly 3 shapes each, correctly labeled Ciss/Coss/Crss, tight masks, no leftover TODOs — semi-auto label quality is NOT the problem, at least not in this sample. **(3) Cross-generalization, corrected finding**: checked training-set membership before running anything — `AUIRL` is **not in train.json or val.json in either run** (test-only since the original T5 freeze; T9 only added new families, never moved test); `IRFL` landed in **val**, not train, so it never received a gradient update either. This rules out "IRFL training data directly confuses AUIRL" — revised explanation: `AUIRL` was already a fragile, out-of-distribution generalization case for Run A (recall 0.58 vs 1.0 elsewhere); Run A2's much larger, stylistically different training mix (mostly bigger IPD/IPP/IRF power packages) plausibly shifted the model's feature space further from whatever let it marginally generalize to `AUIRL`, while families closer to what's still trained held up fine. Also **flagged: T8a's 13 images and T8b's 393-image batch are now ~entirely part of Run A2's own training data** (12/13 T8a images confirmed) — using them for "does A2 generalize better" would be invalid (memorization, not generalization); only the T8f log-log screenshot remains genuinely uncontaminated. On that single image: **Run A finds exactly 3 clean curves; Run A2 finds 4 (a duplicate on Ciss)** — no evidence of better generalization on this (thin, n=1) sample. **Assessment for the owner:** not a labeling-quality problem (GT audit clean); likely an out-of-distribution generalization cost concentrated in the single weakest test family (`AUIRL`) as the training distribution broadened, rather than a uniform quality regression — reasoned, not fully proven; would need more `AUIRL`/`IRFL`-adjacent held-out images to confirm. Full per-family numbers, GT contact sheet, and screenshot comparison images saved under `data/` (git-ignored). No retraining or data changes made. |

| **Decision — production checkpoint: Run A** | ✅ Decided (owner, 2026-07-08) | **`run_a/best_segm_mAP_50_iter_1600.pth` (mAP@50 0.8774, recall 0.9306) is the production checkpoint** for the capacitance curve-tracing model going forward. **Run A2 (real + semi-auto combined, 456/61) is NOT discarded** — it's documented as a completed, informative experiment: it revealed that combining unevenly-sized/styled data can hurt underrepresented families specifically (the `AUIRL` case study — see T11), a finding worth keeping on record for future data-merge decisions, not a dead end. Both runs' manifests, checkpoints, and eval reports remain on the GPU box (`/mnt/data/my-datasheet/checkpoints/run_a/`, `run_a2/`) for reference. |
| **Decision — synthetic data deferred; next milestone is stages 4–7** | ✅ Decided (owner, 2026-07-08) | **Synthetic data generation for capacitance curves is DEFERRED** until all seven target curve types have working models in the pipeline. **Definitive list (owner-approved, 2026-07-08 — also recorded in `CLAUDE.md` §1):** (1) `capacitance_vs_vds` (Ciss/Coss/Crss) — ✅ DONE, Run A production checkpoint, mAP@50 0.88; (2) `rdson_vs_tj` — not started; (3) `if_vs_vsd` (body_diode) — not started; (4) `id_vs_vgs` (transfer_char) — not started; (5) `vgs_vs_qg` (gate_charge) — not started; (6) `vgsth_vs_tj` — not started; (7) `zth_vs_time` (thermal_impedance) — not started. Other curve-type folders in the legacy/local corpus (`avalanche_energy`, `breakdown_voltage`, `derating`, `output_char`, `soa`, `irrm_vs_didt`, `qrr_vs_didt`, `switching_energy`) are **OUT OF SCOPE**. **Next immediate milestone: build stages 4–7** (classify_curves, curve-extraction integration classical+LineFormer, visual review overlay/gallery, orchestrator/validation → MongoDB) for this repo, currently only scaffolded/frozen as a target in `CLAUDE.md`. **Stages 1–3 migration from the legacy `Extractor` folder into this repo is LAST on the roadmap**, after 4–7 are working — imported as-is per `CLAUDE.md`, not rebuilt, but deferred until the rebuilt stages have something real to consume/produce. |

## M3 — Stage 4: Curve Classification

| Task | Status | Notes |
|---|---|---|
| T12 — Stage 4 framework + `capacitance_vs_vds` + `id_vs_vgs` registry entries | ✅ Done (pending owner review) | 2026-07-09; `src/classification/` (`curve_registry.py`, `scoring.py`, `classify.py` + README), **43 new tests, TDD red→green confirmed** (suite now **209 passing**, `pytest tests/ -v`; `third_party/` vendored mmdetection collection errors are pre-existing/unrelated, scoped run used per repo convention — no `pytest.ini` exists to scope this automatically, left untouched as out of scope). Clean-room design: reused only the *idea* from `D:\Extractor\classify_curves\curve_scoring.py`/`score_curves.py::run_with_mutex` (caption+axis+keyword+position scoring, ranked selection, mutual exclusion) — no code copied, no `group_subfigures` duplication, no monkey-patching (explicit `claimed: set[figure_id]` passed in/out instead). Curve-type fingerprints are pure data (`CurveTypeSpec` registry entries); one shared `score_figure`/`classify_page`/`classify_device` handles any curve type. Ambiguous/low-confidence results are quarantined (not dropped, not force-guessed), with full per-candidate score audit trail (`ClassificationResult.all_scores`, `ScoreResult.matched_signals`) and CLAUDE.md §7 logging of every classification decision (figures considered, scores, winner rationale). **`id_vs_vgs` wording confirmed against real data before finalizing** (owner instruction: don't guess): scanned ~90 real `full_extraction.json` transfer-characteristics figures across `D:\Extractor\data\OCR1-OCR13` — captions are "Typical Transfer Characteristics" / "Typ. transfer characteristics" (not the task-outline's placeholder wording), axis labels "ID, Drain-to-Source Current (A)" / "VGS, Gate-to-Source Voltage (V)". Also corrected `capacitance_vs_vds`'s caption wording the same way — real data says "Typical Capacitance vs. Drain-to-Source Voltage", not "typ. capacitances"/"capacitance characteristics" as originally guessed in `CLAUDE.md` §1's table (registry now matches real OCR text; `CLAUDE.md`'s table description is a summary label, not verbatim spec, so left as-is unless owner wants it corrected too). **Not wired into real stage-3 data or run on real devices yet — deliberately deferred**, per instruction, until owner confirms the `id_vs_vgs` registry entry. |
| T13 — Dry-run: `classify_device()` on 19 real devices (report only, no pipeline wiring) | 🔍 Reported — superseded by T14's axis-completeness check | 2026-07-09; ad-hoc adapter script (throwaway, not committed — same convention as T11's diagnostic script; reads real `full_extraction.json` directly, builds `FigureCandidate`s, calls `classify_device` for both curve types per device, never touches `D:\Extractor` or this repo's `data/coco/*`). **19 devices** selected from 259 real devices confirmed to have both a capacitance and a transfer-characteristics figure (10 already in our training data, 9 genuinely new, seed 42). **Original result (before T14): capacitance_vs_vds matched 19/19; id_vs_vgs matched 18/19, quarantined 1/19, no_match 0/19** — but visual spot-check only covered 8/19 devices at the time, and (per T14) two of the *unchecked* 11 devices turned out to be silent bad picks (see T14). Two findings surfaced, both visually confirmed against the actual cropped images: (1) **Stage-3 caption/figure misattribution, not a classifier bug — deferred, no fix planned now.** On `IPP044N03LF2SAKSA1` (and independently confirmed on `IAUCN04S7L005ATMA1`, `BSC011N03LSATMA1`, `IPB029N06NF2SATMA1`, `IPP023N03LF2SAKSA1`, `IPD029N04NF2SATMA1`, plus two new instances found in T14: `IPQC60R010S7XTMA1`, `IPA60R380P6XKSA1`), a figure's `full_extraction.json` caption field is attached to the wrong figure — an off-by-one caption/figure-numbering assignment bug in stage-3's Azure OCR extraction, specific to the Infineon "Diagram N:" template family. **Owner decision (2026-07-09): known upstream issue, deferred until the stage 1–3 migration phase (see CLAUDE.md §1 roadmap) — no fix needed now.** Our classifier's OCR-content scoring (not caption-only) correctly ignores the bad caption when the figure's own content is unambiguous, but T14 showed this isn't always enough (see below) — worth revisiting when stages 1–3 are migrated into this repo. (2) Composite sub-figure fragmentation on `IRFP260MPBF` — see T14, now caught automatically. |
| T14 — Axis-completeness check on classification results (TDD) | ✅ Done (pending owner review) | 2026-07-09; added `figure_has_complete_axes()` to `src/classification/scoring.py` (reuses the existing `_classify_zone` bbox heuristic already used by `score_figure` — no new zone-detection logic invented) and wired it into `classify_page` in `src/classification/classify.py`: a would-be "matched" result whose winning figure has no OCR line in the x-axis zone AND no OCR line in the y-axis zone is downgraded to `quarantined` with an `incomplete_axes` reason string, rather than silently passed through. **8 new tests, TDD red→green confirmed** (suite now **217 passing**, `pytest tests/ -v`). **Re-ran the full T13 batch (19 devices) with the check active, per instruction, and visually verified every status change, not just IRFP260MPBF:** `capacitance_vs_vds` 19 matched → **17 matched / 2 quarantined**; `id_vs_vgs` 18 matched / 1 quarantined → **13 matched / 6 quarantined**. **Confirmed as requested: `IRFP260MPBF` now quarantines on both curve types** (composite sub-panel crops, missing one axis each — exactly the T13 finding). **Went further and visually audited every newly-quarantined case, surfacing a real true positive the original T13 spot-check had missed**: `IPQC60R010S7XTMA1` and `IPA60R380P6XKSA1` id_vs_vgs had been silently "matched" in T13 (never spot-checked) onto the **Infineon logo image**, not a chart — same stage-3 caption-misattribution bug landing a "Diagram N: Typ. transfer characteristics" caption on the page-header logo figure; the logo scored 6.0 from caption-keyword hits alone (also surfaced a minor registry double-count: `id_vs_vgs`'s two caption keywords "transfer characteristic"/"transfer characteristics" are substrings of each other and both matched the same phrase, inflating 3.0→6.0 — noted, not fixed, out of scope for this task). The axis check now correctly quarantines both. **Also found 3 false-positive downgrades** — genuinely complete, correct charts wrongly quarantined by the OCR-geometry heuristic, each with a different root cause, visually confirmed: `IPP60R190P6XKSA1` id_vs_vgs (y-axis label bbox height is *exactly* 2× width; `_classify_zone`'s tall/narrow check is a strict `>` against `ZONE_ASPECT_RATIO=2.0`, so an exact-threshold case falls through — a tuning issue), `BSC011N03LSATMA1` id_vs_vgs (the y-axis label text is entirely absent from this figure's stage-3 OCR lines, despite being visible in the rendered image — an upstream OCR extraction gap, not a heuristic problem), `BTS132E3129NKSA1` capacitance_vs_vds (old minimalist-style datasheet whose y-axis label is a single character "C", not a long rotated string — doesn't fit the tall/narrow-text assumption the heuristic was built on). **Net assessment: the check does real, valuable work** (2 confirmed bad picks caught that had been silently trusted) **but is an imperfect proxy** (3 confirmed good picks wrongly flagged) — reasonable as a "flag for human review" gate (nothing is silently dropped, quarantined figures still get a human look) but not yet precise enough to auto-reject. Updated gallery at `data/t13_dryrun/gallery.html` (10 devices, each finding labeled TRUE POSITIVE or FALSE POSITIVE with the visual evidence), full JSON at `data/t13_dryrun/dryrun_report.json` (both git-ignored). **Not wired into any pipeline; no production data touched.** Follow-ups for owner to prioritize, none fixed now: (a) `_classify_zone` aspect-ratio boundary (`>` vs `>=`) and its tall/narrow-only assumption, (b) `id_vs_vgs` registry's overlapping caption keywords double-counting, (c) whether stage-3's missing axis-label OCR on some figures needs a stage-3 fix (ties into the deferred caption-misattribution issue above — same migration-phase bucket). |
| T28 — Dry-run: `classify_device()` on the new Rohm batch, 612 real devices (report only, no pipeline wiring) | 🔍 Reported — **125 candidates found (98 matched / 27 quarantined), 1 data-integrity finding for owner** | 2026-07-13; ad-hoc adapter script (throwaway, not committed — same convention as T13), reading real `full_extraction.json` from the newly-transferred `data/rohm_raw/Rohm Semiconductor-20260713T052012Z-2-001/` (612 device folders, first real Rohm-family data run through this classifier — T8a had explicitly noted zero Rohm coverage until now). Ran `capacitance_vs_vds` only (the one DONE curve type). **Result: 98 matched, 27 quarantined, 487 no_match** (486 of the no_match have **zero figures at all** in their OCR extraction — confirmed these are genuine single-page condensed datasheets, `page_count: 1`, not a classifier miss; the remaining 1 had a figure that scored 0). **98 + 27 = 125**, matching the owner's own prior estimate of the batch's capacitance-relevant device count. Spot-checked 2 matched + 1 quarantined figure visually against the source PNG — all 3 genuine Ciss/Coss/Crss charts; the quarantined one was correctly flagged by the existing `incomplete_axes` check (T14) for a cropped-out x-axis label, same known pattern. **Data-integrity finding, not a classifier bug:** 19 of the 612 device folders (all in the `R60xx` "Y-suffix" family, e.g. `R6010YND3TL1`, `R6022YNXC7G`) have `full_extraction.json` entries referencing figure PNGs that don't exist on disk in this transfer — 589 missing files out of 5,459 figures referenced across the whole batch, most likely an incomplete OneDrive sync/export on the source side rather than an scp/rsync drop (the affected files are scattered non-contiguously within each device, not a clean tail-truncation). Scoring itself is unaffected (OCR text/bbox metadata was still present and complete), but **13 of the 98 "matched" and 4 of the 27 "quarantined" results have their winning figure's PNG missing** — those can't be visually verified yet and aren't usable by stage 5 (needs real pixels) until the affected 19 folders are re-transferred. Full device list in the saved report. **Outputs (git-ignored under `data/`, per repo convention):** `data/rohm_classification/rohm_capacitance_classification.json` (full per-device matched/quarantined/no_match breakdown + the data-integrity finding), `data/rohm_classification/quarantined_gallery.html` (all 27 quarantined figures, embedded images, reviewer checkboxes; the 4 with missing PNGs shown as flagged placeholders instead of broken images). **Not wired into any pipeline; no production data touched. No full-corpus run beyond this one classification pass** — owner to eyeball the 27-card gallery and decide on the 19 R60xx folders needing re-transfer. |

## M4 — Stage 5: Curve Extraction Pipeline

| T29 — First real end-to-end Stage 5→7 run on real, never-annotated devices (Rohm) | 🔍 Reported — **86 finalized / 22 needs_review / 0 failures on 108 devices** | 2026-07-13; first time stages 5-7 ran with the actual trained model (prior T15-T20 runs all used frozen-test-set GT masks as detection stand-ins, no GPU). **Checkpoint note:** `/mnt/data/my-datasheet/checkpoints/run_a/best_segm_mAP_50_iter_1600.pth` transferred by owner is technically from the `run_a_reconstructed` lineage (the original `run_a` checkpoint + its EBS volume were lost in an untracked 2026-07-12 incident — see `data/coco/split/split_manifest.json`'s own reconstruction note); verified equivalent on the same frozen test set (mAP@50 0.8789 vs original 0.8774, recall 0.9306 identical). **GPU box rebuilt**: `/mnt/data` had no volume attached (fstab UUID pointed at a volume no longer present) — formatted+mounted a fresh 116GB volume per SETUP.md's own "if this box is ever rebuilt" section, then ran `scripts/setup_training_env.sh` + `verify_training_env.sh` (conda env `lineformer`, torch 1.13.1+cu117/mmcv-full 1.7.1/mmdet 2.28.2, all 4 smoke tests pass — found and worked around a pre-existing `pipefail`+`head -1` SIGPIPE bug in `verify_training_env.sh`'s default-image lookup, not fixed, not in scope). **Input**: the 108 devices from T28 (98 matched + 27 quarantined, minus the 17 with missing PNGs). Ad-hoc driver script (throwaway, not committed) calls `run_pipeline()` per device unmodified — no stage logic reimplemented. **Result: 86 ok / 22 needs_review / 0 crashes** — needs_review reasons: wrong post-dedup curve count (several devices), axis calibration failure (2), units undetected (several). Stage 6 gallery: 84 confident / 2 low_confidence / 22 needs_review, 0 missing images. Stage 7 (`--auto-approve`, no human review loop set up for this test): 86 finalized / 22 needs_review / 0 pending_review / 0 rejected / 0 failed_classification / 0 failed_extraction. **Accuracy spot-check (9 random finalized devices, seed 42), extracted curve range vs. the printed chart, all visually verified**: all 9 devices' Ciss/Coss/Crss ranges matched their printed axes correctly to within normal chart-reading precision. **One real, specific finding, not fixed**: on `SCT3030ARHRC15`, Coss's extracted upper bound (2590 pF) is inflated to match Ciss's height — traced to the two curves visually touching/converging at the low-voltage end of the chart, where the mask/skeleton tracing picks up a few of the wrong curve's pixels at the point of contact. Not a systemic failure (only surfaced once in 9 spot-checked devices, all other curves on all other devices were clean), but a known real limitation worth watching at scale. Outputs (git-ignored): `data/t25_stage5_real/results/` (flat `<device>.json` per device), `data/t25_stage6_gallery/gallery.html`, `data/t25_stage7_orchestrator/` (`final/`, `followup_queue.json`, `batch_summary.json`). **Not a production run — 108 devices, no MongoDB write, no review-state loop.** |

| Task | Status | Notes |
|---|---|---|
| T15 — Stage 5 pipeline (capacitance_vs_vds reference implementation, TDD) | ✅ Framework done (pending owner review) — **⚠ calibration finding needs a decision before trusting "ok" results** | 2026-07-10; new packages `src/calibration/` (`ticks.py`) and `src/extraction/` (`inference.py`, `dedup.py`, `skeletonize.py`, `naming/` [registry + `capacitance_vs_vds.py`], `schema.py`, `pipeline.py`). **101 new tests, TDD red→green confirmed at every step** (suite now **312 passing**, `pytest tests/ -v`). Three new pinned deps added to `requirements.txt`/`SETUP.md`: `scikit-image==0.26.0` (skeletonize), `pycocotools==2.0.11` (mask decode, matches what `eval_lineformer.py`/`predict_to_cvat.py` already lazy-import — installed locally so their tested-but-previously-untested lazy path could actually be exercised). **Calibration lift** (`src/calibration/ticks.py`): `parse_numeric_ticks`/`fit_axis`/`pixel_to_data`/`derive_calibration` ported line-for-line from the canonical `D:\Extractor\5_opencv_extract\cv_curve_extract.py` (lines 116–402) per `LEGACY_REVIEW.md` §3 — the one piece of legacy code deliberately lifted, not reinvented; 4-tuple contract preserved, RANSAC/log-auto-detect/compound-token behavior verified against known tick examples (21 tests), not trusted blindly. **Inference wrapper** (`src/extraction/inference.py`) reuses `eval_lineformer._pred_to_bool_masks` and `predict_to_cvat.filter_by_score` directly rather than a third mask-decode/score-filter implementation; GPU-only calls stay lazy, unit-tested by injecting a fake `mmdet.apis` module (8 tests). **Dedup** (`src/extraction/dedup.py`): named/tested `dedup_detections()` reusing `eval_lineformer.mask_iou`, plus a same-vertical-band + x-span-overlap heuristic for the documented T8a/T8b flat-curve near-duplicate case that IoU alone misses (9 tests). **Skeletonize** (`src/extraction/skeletonize.py`): `skimage.skeletonize` + per-column averaging on the thin skeleton (not the raw mask — avoids the documented legacy "averaging window on raw mask" flaw) with no x-deduplication (avoids the documented legacy `interpolate()` data-loss flaw) (10 tests). **Naming** (`src/extraction/naming/`): Stage-4-style pluggable registry (`get_naming_fn(curve_type)`); `capacitance_vs_vds.py` sorts 3 curves by mean pixel row -> Ciss/Coss/Crss top-to-bottom, ties handled via stable sort, wrong count raises clearly (11 tests). **Schema** (`src/extraction/schema.py`): fresh design, NOT legacy's flat/keyed fork — `curve_type` always present/non-empty (hard-rejected before any write), **one file per device per curve type** (chosen over a keyed single file specifically so the merge-corruption bug class is structurally impossible, not just guarded against), atomic tmp+`os.replace` write, validated before disk, refuses to overwrite a path already holding a different curve_type's result, full calibration/curve/point structural validation including NaN/Inf rejection (25 tests). **Pipeline** (`src/extraction/pipeline.py`): `process_detections()` (GPU-free orchestration core, fully unit-tested) implements the exact curve-count logic specified — exactly 3 -> proceed; 4+ -> attempt `dedup_detections`, proceed only if reduced to exactly 3; fewer than 3 -> `needs_review`, never guessed; naming still runs (real Ciss/Coss/Crss names) even when calibration fails, so a `needs_review` record is maximally useful for human review, not just an empty shell; `run_pipeline()` is the thin GPU-dependent wrapper adding inference (11 tests). **Real-data sample run** (T15 continued, `data/t15_stage5_sample/`, git-ignored): ran on **8 real test-set images** (frozen test split, known-correct GT curve identity) — **using real ground-truth masks as detection stand-ins (score=1.0), NOT the trained model**, because torch/mmdet are not installed on this Windows dev box (GPU inference runs on the AWS box per T6/T7) — clearly labeled as such in the gallery, not silently passed off as a model run; a true model-inference sample run is a follow-up once this framework is reviewed. Every stage downstream of detection (dedup gate, skeletonize, naming, calibration, schema validation, atomic write) is genuinely real. **Result: 6/8 "ok", 2/8 correctly `needs_review`** (`AUIRL1404ZS`, `BTS132E3129NKSA1` — both calibration failures, correctly caught, not guessed). Output JSON + overlay gallery: `data/t15_stage5_sample/gallery.html` (color-coded per-curve overlay dots on the source image, Ciss=blue/Coss=green/Crss=orange). **⚠ Major finding, requires an owner decision, not fixed:** of the 6 "ok" results, only **1 (`AUIRLU3114Z`) is fully correct end-to-end** (verified visually: overlay traces match the printed curves tightly, and the calibrated values — Ciss 3798–4635 pF, Coss 604–2154 pF, Crss 327–899 pF — are physically sane and correctly ordered). **The other 5 "ok" results have silently WRONG calibration**: Azure OCR splits log-axis superscript tick labels like "10³" into two separate OCR tokens, "10" and "3"; the ported `fit_axis` then finds a plausible-looking *linear* fit through the bare exponent digits (1, 2, 3...) instead of recognizing the axis as log-scale, so `y_log` comes back `False` when it should be `True` — every returned "capacitance" value is off by orders of magnitude (e.g. `BSF050N03LQ3G` returned Ciss/Coss/Crss all ≈ 101–103, when the real chart shows ≈ 100–2500 pF; confirmed by reading the actual axis image and cross-checking the raw parsed ticks). Curve tracing, dedup, and naming are all independently confirmed CORRECT in every one of these 5 cases (overlay dots track the real curves tightly) — **only the calibration units are wrong, and the result schema currently has no way to flag this**: the mask/pixel-level pipeline is exactly right, but the answer written to `curves[].points` is numerically false while still passing schema validation (all values are finite, so the "not NaN/inf" DoD bar was met even though the numbers are wrong) and the result claims `status: "ok"`. This was not one of the 8 documented `LEGACY_REVIEW.md` §3 caveats — a new discovery from real-data testing, not a porting bug (the ported function is faithfully reproducing what the original legacy code would do on this exact OCR input). **Not fixed** — per instruction, `ticks.py` preserves the legacy function's documented behavior verbatim; a fix (e.g., merging OCR-adjacent "10"+digit token pairs into `10^digit` before tick parsing, or a downstream plausibility check) needs an owner decision since it touches the one piece of legacy logic that was deliberately not supposed to be reinvented without approval. **Definition of done:** all tests green ✅; sample run + overlay produced for owner eyeball-check ✅; **NOT wired into a full-corpus batch run** ✅ (per instruction) — owner reviews the sample output, and specifically this calibration finding, before that happens. |

| T16 — Fix log-axis exponent OCR splitting + plausibility safety net (TDD) | ✅ Done (pending owner review) — **zero silently-wrong results on the T15 sample** | 2026-07-10; owner-approved fix to the T15 calibration finding — the first (and only) approved deviation from the verbatim `ticks.py` port, scoped to OCR-token repair; `fit_axis` itself untouched. **(1) Exponent repair in `src/calibration/ticks.py`**: real corpus inspection showed Azure OCR renders superscript decade labels ("10³") three ways — `"10 3"` (one token with a space; the legacy compound-token path split it into two bogus ticks 10 and 3), `"103"`/`"102"`/`"101"` (digits concatenated into 103/102/101), and `"10º"`/`"10³"` (ordinal/unicode superscript chars). Fixes: space/superscript forms parsed directly as 10^d (checked before the compound path); bare concatenated forms (values 101–109) reinterpreted as 10^d **only with ≥2 such ticks on the same axis** (or ≥1 alongside an unambiguous exponent label) so a lone genuine 104 is never corrupted; on any axis identified as log-decade-labeled, non-positive junk ticks are dropped (a stray "0" otherwise blocks `fit_axis`'s all-positive log detection); y-zone wins over x-zone for exponent labels (the bottom-most y decade label sits inside the x-band). 10 new tests built from the real failing devices' exact OCR geometry (BSP125/BSF050/BSP88/BSS138), incl. regression tests that genuine compound rows ("0 20 40 60") and plain "10" ticks still parse identically. **(2) Plausibility check in `src/extraction/pipeline.py`**: data-only `PLAUSIBILITY_SPECS` registry (Stage-4 pattern — new curve types add an entry, no logic); for `capacitance_vs_vds`: `require_y_log=True` (every capacitance chart in the corpus has a log y-axis — this is the primary catch for the T15 failure signature) + y-range (0.1, 1e6) pF (corpus axis labels span 10^0..10^5 pF, one decade margin each side — bounds derived from data, not guessed). A would-be "ok" failing either check downgrades to `needs_review` with reason `implausible_calibration`, **keeping** the traced curves and suspect calibration for the reviewer (not an empty shell); curve types without a spec entry are simply unchecked. 5 new tests. **15 new tests total, TDD red→green confirmed; suite 327 passing.** **(3) T15 8-image re-run — the acceptance bar ("zero silently-wrong ok") is met**: 6/8 "ok", all with `y_log=True` and values matching their printed axes (spot-verified: `BSS87H6327FTSA1` extracted Ciss 72–258 / Coss 8.9–216 / Crss 4.3–186 pF vs. the actual chart image's 10^0..10^3 axis — matches; `BSF050N03LQ3G` went from the silently-wrong ~101–103 to Ciss 2122–2443 / Coss 801–2639 / Crss 14–337 pF on its printed 10^1..10^4 axis). 2/8 `needs_review`, both verified as **genuine upstream OCR failures, correctly flagged**: `AUIRL1404ZS` (OCR mangled the "10000" y-label into "4.10000") and `BTS132E3129NKSA1` (axis uses *negative* exponents "10-1"/"10-2" = 10⁻¹/10⁻² in nF — deliberately not guessed: a "10-1" token is too ambiguous to auto-repair; noted as a possible future extension if the corpus shows enough of these). Updated gallery: `data/t15_stage5_sample/gallery.html` (git-ignored). **No full-corpus run — awaiting owner review of these results per instruction.** |

| T17 — Wider sample check: 30 real images (report only, no code changes) | 🔍 Reported — **zero silently-wrong statuses; 2 systemic findings for owner** | 2026-07-10; same GT-masks-as-detections approach as T15/T16 (ad-hoc script, clearly labeled — no model involved, so split leakage doesn't apply). Coverage: **all 24 frozen-test-split images + 6 val-split extras** chosen from families absent from test for style diversity (BUZ Siemens-era, 94-3316, IPF/IPTG/IQD modern "Diagram N:" Infineon, IRFL International Rectifier) — the test split alone has only 24 images, short of the requested 25–30. **Result: 21/30 ok, 9/30 needs_review.** Spot-verified "ok" correctness visually across styles (IQD005 matches its printed 10¹..10⁵ pF axis, BUZ111S its 10²..10⁴ pF axis; T16-fixed devices still correct). **All 9 quarantines are correct outcomes**: 6 flagged for exactly the right reason — `BSP324H`/`BSP324L` (OCR garbled exponent labels "10 21"/"10 1L."/"10 OL." beyond repair → linear mis-fit caught by `require_y_log`), `BSS127H`/`BSS127I` ("100"=10⁰ + negative exponent "10-1"=10⁻¹ labels, both outside T16's deliberate repair scope), `BTS132` (negative exponents, known), `AUIRL1404ZS` (OCR-mangled "4.10000", known); 3 flagged for a **partially wrong reason but still correctly quarantined** — `AUIRL3705N`, `94-3316`, `IRFL014NTRPBF` are an International-Rectifier template with a **genuinely LINEAR y-axis** (values extracted correctly!) where `require_y_log` misfires, BUT their log x-axes (printed 1..100 V) simultaneously mis-calibrate to ~0.8–16 V linear (a stray "0" tick blocks `fit_axis`'s all-positive log detection), so needs_review is the right status either way. **Two systemic findings for the owner:** (1) **the "capacitance y-axis is always log" assumption (from the T16 task spec, encoded as `require_y_log`) is FALSE for the IR linear-y template family** — the plausibility spec needs rethinking (e.g. per-axis consistency checks rather than a hard y-log requirement) before these charts can pass; they currently fail safe (quarantined, not silently wrong). (2) **Axis units are not captured**: `BTS247ZE3062ANTMA1` passed "ok" with values 0.19–3.64 that exactly match its printed axis — but that axis is in **nF**, not pF (verified visually); the schema has no `units` field, so a pF-assuming consumer would be 1000× off. Same nF style as BTS132. Unit detection (the "nF"/"pF" OCR token is right there in the figure) is a needed schema/pipeline extension. Extension candidates from this run, none implemented: "100"→10⁰ and "10-d"→10⁻ᵈ repair (would recover BSS127H/I and BTS132), stray-"0" dropping on log-labeled x-axes (would recover the 3 IR charts' x-axes), units field. Outputs: `data/t17_stage5_wide/` (per-device JSON, overlays, `gallery.html`; git-ignored). **No full-corpus run.** |

| T18 — Fix log/linear-axis assumption + add units detection (TDD) | ✅ Done (pending owner review) — **27/30 ok on the T17 sample (was 21/30), zero silently-wrong results** | 2026-07-10; **(1) Removed the T16 `require_y_log` hard rule** from `src/extraction/pipeline.py::PLAUSIBILITY_SPECS` — T17 found real capacitance charts (an International Rectifier template: `AUIRL3705N`, `94-3316`, `IRFL014NTRPBF`) with a genuinely linear y-axis and correctly-extracted values that the rule wrongly force-rejected. `fit_axis` (`src/calibration/ticks.py`) already picks log vs. linear via its own RANSAC inlier-count comparison — that judgment is now trusted rather than second-guessed; the y-range plausibility check (0.1–1e6 pF) remains as the real end-to-end fit-quality signal, since it tests the actual converted values rather than assuming one axis scale is always correct. **(2) Stray-zero-tick dropping for log-labeled axes**: new `_drop_stray_zero_on_log_axis()` in `ticks.py`, fires when a lone "0" tick sits alongside ≥2 other positive ticks whose range ratio is ≥10x — the same threshold `fit_axis` itself already uses to prefer a log fit (reused, not reinvented), so the heuristic can't disagree with `fit_axis`'s own judgment. Recovers the 3 IR charts' log x-axis (1..100 V), previously blocked by a stray "0" tick. **(3) Negative-exponent + widened bare-exponent token repair**: `"10-1"`/`"10-2"` (no space) now parsed as 10⁻¹/10⁻² (new `_EXPONENT_NEGATIVE_RE`); bare-digit reinterpretation range widened from 101–109 to 100–109 so bare `"100"` can resolve to 10⁰ — but **only when corroborated** by an unambiguous 101–109 sibling or explicit exponent notation on the same axis (a lone bare "100" is far too likely to be a genuine 100V/100pF tick to reinterpret on its own). Recovers `BSS127H6327XTSA2`/`BSS127IXTSA1` (bare 102/101/100 + 10-1) and `BTS132E3129NKSA1` (10-1/10-2). **(4) Units field added to the output schema**: `src/extraction/schema.py` gains a required `units` key (`str | None`; `build_result` now requires it explicitly, no default — every call site updated); new `detect_y_axis_units()` in `ticks.py` scans the y-axis-label zone for pF/nF/uF tokens (deliberately excludes bare "F" alone, since real axis annotations use lowercase "f" as a frequency variable — "f = 1 MHz" — which would otherwise false-positive as a Farad unit). Undetected or ambiguous (2+ distinct units found) → `units=None`, status downgrades to `needs_review` with reason `"units_undetected"` — schema hard-rejects a non-None units value alongside that reason (never guessed). **28 new tests, TDD red→green confirmed; suite 361 passing.** **A real regression was found and fixed before completing this task**: the widened bare-"100" repair initially broke `AUIRLU3114Z`'s x-axis — two literal duplicate "100" OCR tokens (one genuine tick, one an OCR artifact at a different pixel position) both got reinterpreted to 10⁰, destroying a tick set that `fit_axis`'s own RANSAC was already correctly handling by rejecting the duplicate as an outlier. Root-caused via direct `fit_axis` inspection, fixed by requiring 101–109 corroboration before touching a bare "100" (item 3 above already reflects the fix), and a permanent regression test added (`test_lone_bare_100_with_duplicate_pixel_position_not_reinterpreted`) — caught during this session's own re-run, not after the fact. **T17 sample re-run, full accounting**: **27/30 ok (was 21/30), 3/30 needs_review (was 9/30)**. All 6 devices T17 flagged for the *wrong* axis-type reason now correctly pass (the 3 IR-template linear-y charts + `BSS127H`/`BSS127I`/`BTS132`), each spot-verified against real values (e.g. `BSS127H` 1–37 pF matches its printed 10⁻¹..10² pF axis). `BTS247ZE3062ANTMA1` — the T17 finding that a pF-assuming consumer would be 1000× off — now correctly reports `units: "nF"` in the schema. The 3 remaining `needs_review` are all genuine, visually re-verified catches: `BSP324H6327XTSA1`/`BSP324L6327` (visually confirmed a normal clean log-axis chart whose OCR *label text itself* was too garbled to repair — `"10 21"`, `"10 1L."` — now caught via the range check producing impossible negative pF values, rather than a blind y-log rule) and `AUIRL1404ZS` (known OCR-mangled "10000"→"4.10000", unrelated to this task, unaffected). Updated gallery: `data/t17_stage5_wide/gallery.html` (git-ignored), per-device JSON updated in place. **No full-corpus run.** |

## M4 — Stage 6: Visual Review

| Task | Status | Notes |
|---|---|---|
| T19 — Stage 6 visual review gallery (TDD) | ✅ Done (pending owner review) | 2026-07-11; `src/review/` (`gallery.py`, `review_state.py`) + `data_to_pixel()` added to `src/calibration/ticks.py`. **35 new tests, TDD red→green confirmed; suite 396 passing.** **Pure-viewer contract enforced, not just promised**: Stage 6 never recalculates calibration or re-derives values (the documented legacy bug — a drifted Stage-6 copy of calibration math silently disagreeing with Stage 5). Overlay projection uses the new `data_to_pixel()` — the exact inverse of `pixel_to_data`, living in `ticks.py` beside it so viewer code *can't* grow its own copy — applied to Stage 5's STORED calibration dict; a dedicated test monkeypatches `derive_calibration` to explode if the gallery ever calls it. Round-trip verified on real data: projected points land exactly on the printed curves (spot-checked `BTS247ZE3062ANTMA1` overlay). **Bucketing** uses only what Stage 5 already stored: `needs_review` status → flagged (always shown in full, regardless of confidence); ok + weakest stored curve confidence < 0.7 (`DEFAULT_LOW_CONFIDENCE_THRESHOLD`, from T8a's observed 0.82–0.95 good-detection range) → `low_confidence` (always shown in full); rest → `confident`, cappable via `--sample-size N` (deterministic, seed 42; default no cap). **Review state** (`review_state.py`): approve/reject decisions in a separate JSON keyed `device::curve_type` — never written into Stage 5's outputs (stage outputs immutable); forgiving load (missing/malformed file starts fresh, logged, never crashes a session), strict save (validated + atomic tmp+`os.replace`); gallery pre-fills controls from an existing state file, and the HTML's "Export decisions JSON" button downloads the same schema. Image lookup reuses `resolve_image` and the halo-label idiom from `overlay_check.py` (no reimplementation); missing/unreadable source images are flagged in the gallery and logged, never crash the batch. **One real bug caught by TDD during this task**: `python -m src.review.gallery` initially lost ALL its log lines to the `__main__`-logger-name trap — the exact known Session-1 `cvat_to_coco` CLI defect (still open there, that module being frozen) — reproduced here in new code, fixed (logger name pinned to the real module path), and locked in with a subprocess regression test asserting bucket counts reach the console. **Real run on the T17/T18 30-image Stage-5 output** (`--sample-size` unset, show all, per instruction): **3 needs_review / 0 low_confidence / 27 confident, 0 missing images** — gallery at `data/t19_review_gallery/gallery.html` (git-ignored). Note: 0 low_confidence is expected for THIS batch — its detections are GT-mask stand-ins with confidence 1.0 (T15–T18 convention, no GPU on this box); the low-confidence bucket becomes meaningful once real Run-A model scores flow through Stage 5 on the GPU box. CLI: `python -m src.review.gallery <stage5_dir> <images_dir>... --out <dir> [--sample-size N]`. |

## M5 — Stage 7: Orchestrator

| Task | Status | Notes |
|---|---|---|
| T20 — Stage 7 pipeline orchestrator (TDD) | ✅ Done (pending owner review) — **first real finalized records: 27/30** | 2026-07-11; `src/orchestrator/` (`pipeline.py`, `validation.py`, `queue.py`) + `get_expected_names()` added to the naming registry (sourced from the same per-curve-type module as the naming function itself, so the two can't drift). **30 new tests, TDD red→green confirmed; suite 426 passing.** Pure orchestration — stages are called through a small injected adapter protocol (`run_classification`/`run_extraction` + review-state dict), no stage logic reimplemented. **Six mutually-exclusive statuses** (finalized / pending_review / rejected / needs_review / failed_classification / failed_extraction), each covered by tests including failure isolation (a stage-4 or stage-5 exception becomes failed_classification/failed_extraction and the batch continues — never crashes). **NO auto-pass, enforced by test**: a perfectly clean ok result with no review decision stays pending_review; `require_manual_approval=True` is the default (an `--auto-approve` flag exists for the future flip, still validation-gated, and explicit REJECT always wins). **Final validation** (`validation.py`) reuses Stage 5's `validate_result` (never reimplements schema checks) and adds only finalization-specific gates: all expected curve names present per the naming registry (no missing, no unexpected), `units` non-None, `calibration` non-None; returns a reason string, never raises. **Finalized record** (fresh design, one file per device per curve type under `final/`): Stage 5's data + provenance (stage5 status, review decision + `decided_at`, `finalized_at` timestamp). **Follow-up queue** (`followup_queue.json`): actionable statuses only (pending_review / needs_review / failed_classification / failed_extraction; rejected and finalized are terminal), full atomic regeneration each run so re-runs can't duplicate or half-update. CLI (`python -m src.orchestrator.pipeline <stage5_dir> --out <dir> --review-state <json>`) ships with a `PrecomputedStage5` adapter (reads existing Stage-5 result JSONs; live end-to-end classify→extract wiring lands with the stage 1–3 migration / GPU-box integration — documented in the module). Same `__main__`-logger fix as T19 applied from the start. **Real run — owner actually reviewed the T19 gallery** (30 exported decisions: 29 approve, 1 reject) — orchestrator run over the T17/T18 Stage-5 outputs with that real review state: **finalized 27, rejected 1 (`AUIRL1404ZS`, owner decision), needs_review 2, pending_review/failed_* 0.** The 2 needs_review are the validation gate working exactly as designed: the owner had APPROVED `BSP324H6327XTSA1`/`BSP324L6327` in the gallery, but their Stage-5 records carry `units=None` (+ known-implausible calibration) — final validation refused to ship them, downgrading to needs_review with the reason in the queue file ("units missing — cannot finalize without units"), demonstrating approval is not a validation bypass. Outputs: `data/t20_orchestrator/` (`final/<device>/<curve_type>.json` ×27, `followup_queue.json`, `batch_summary.json`; git-ignored). Owner's exported decisions installed at `data/t19_review_gallery/review_state.json` (also git-ignored). **All four rebuilt stages (4–7) now demonstrated end-to-end on real data.** |

| T29 — Live Stage 4→5 wiring adapter (TDD) | ✅ Done (pending owner review) | 2026-07-21; three new modules, each red→green in isolation (owner-approved sequence — stage3_loader → extraction_registry → live_stages, full suite re-run green after each before moving on): **`src/classification/stage3_loader.py`** (`load_figures_by_page(device, stage3_root)`, 11 tests) — the piece that never existed as tested code (README said so explicitly): parses a device's real `full_extraction.json` into `Dict[int, List[FigureCandidate]]`, the exact input `classify_device` needs. Figure width/height come from actually reading the referenced PNG (not the JSON's own page-relative bounding_box) since OCR-line bboxes are recorded in the rendered PNG's pixel space; a missing/unrendered image (the T25/T27-documented Stage 1-3 gap) degrades that one figure to `width=height=None` rather than failing the device, but a missing/malformed `full_extraction.json` itself raises. **`src/extraction/extraction_registry.py`** (`ExtractionSpec` + `get_extraction_spec`, 17 tests) — data-only classical-vs-model routing table, same pattern as `curve_registry.get_spec`: `capacitance_vs_vds`/`id_vs_vgs` → model (Run A / Run 3 checkpoints, distinct per type), `rdson_vs_tj` → classical (`expected_curve_count=(1,2)`, matching its own 1-or-2 handling), `if_vs_vsd`/`zth_vs_time`/`vgsth_vs_tj` → no entry (the real current gap, raises `KeyError`). `vgs_vs_qg` gets a deliberate sentinel (`method="none"`) rather than being absent — "known gap, not an oversight" — distinct from the other three. **`src/orchestrator/live_stages.py`** (`ClaimTracker`, `LiveStages`, `NoExtractorAvailable`, 44 tests) — replaces `PrecomputedStage5` behind the SAME adapter protocol (`process_device`/`run_batch` untouched, unmodified): unwraps `classify_device`'s `(result, new_claimed)` tuple into a plain `ClassificationResult`; `ClaimTracker` (shared across curve-type-scoped instances, owner-approved design) makes mutual exclusion work across MULTIPLE curve types classifying the same device; routes extraction via the registry lookup (never an if/elif); converts `FigureCandidate.ocr_lines` (scoring's own `OcrLine` dataclass) into the dict shape both `run_classical_pipeline` and `run_pipeline` actually read (`line["bounding_box"]` — confirmed by reading their real body code, not type hints) via ONE shared conversion function, used identically before either path; lazy-loads and caches the model per instance (one instance = one curve_type, matching `PrecomputedStage5`'s own implicit scoping); `stage3_root` falls back to `LINEFORMER_STAGE3_ROOT` env var (raises if neither given — no hardcoded path). **One real bug found and fixed during implementation, not swallowed**: 9 of the 44 live_stages tests initially failed with `FileNotFoundError` — a genuine test-fixture gap (classical-path tests never wrote a real image file, yet the adapter correctly does a real `cv2.imread` pre-check), not a wrong implementation call; fixed by mocking `cv2.imread` itself in the shared test fixture (owner-approved, confirmed patching the real `cv2.imread` attribute, not a fake stand-in module) while the one test that deliberately wants a genuine missing-file failure keeps its real `tmp_path`, unmocked. **72 new tests total, suite 781 passing** (was 709), zero regressions. No frozen file modified — `orchestrator/pipeline.py`, `classification/classify.py`, `classification/curve_registry.py`, `extraction/pipeline.py`, `extraction/classical.py`, `extraction/naming/*` all untouched, only imported. **Not wired into the CLI yet** (`orchestrator/pipeline.py`'s `main()` still defaults to `PrecomputedStage5`) — swapping the CLI's default adapter is a separate, smaller follow-up needing its own owner go-ahead (frozen-file edit, §4). |

## M6 — Stage 1–3 migration

| Task | Status | Notes |
|---|---|---|
| T21 — Stage 1 CSV ingest (TDD, clean-room) | ✅ Done (pending owner review) | 2026-07-11; `src/ingest/` (`column_mapping.py`, `parsers.py`, `unit_conversion.py`, `model_name_utils.py`, `csv_reader.py`). **113 new tests, TDD red→green confirmed; suite 539 passing.** Clean-room reimplementation of legacy `D:\Extractor\1_csv_input\` (all 7 legacy files read for behavior; no code copied). **Column registry is DATA** (same pattern as stage 4's curve registry): `{device_type: {canonical_field: [ordered header aliases]}}`, first match wins, case/whitespace-insensitive; unknown headers ignored gracefully; unknown device_type raises with the valid list. Header aliases are real DigiKey export column names lifted from the legacy tables (no real device CSVs exist anywhere on disk — searched `D:\Extractor` + siblings — so test fixtures were built from those real header names: two files with deliberately different header conventions + malformed rows). **Legacy behavior preserved:** canonical units (V / A / mΩ / pF / nC / W / mJ / ns / °C/W / USD), (Ta)-preferred current/power parsing, temperature range → (Tjmin, Tjmax), DigiKey-only suffix stripping (`-ND`/`-DKR`/`-TRDKR`; manufacturer suffixes like `-TR`/`-PbF` kept — real variants), bare-number Rdson assumed mΩ, unparseable optional values → None with row kept. **Legacy behavior deliberately changed:** (1) unknown source unit in `normalize_unit()` returns None + WARNING instead of legacy's silent 1:1 guess; (2) rows never silently dropped — `read_csv()` returns `IngestResult(records, skipped)`, each `SkippedRow` with row number + reason, summarized in the log; (3) missing ModelName *column* is a hard ValueError (legacy logged and returned nothing); (4) fixed a latent legacy regex bug where `"5Ohm @ …"` parsed to None despite the docstring claiming 5000 mΩ (a `(?!hm)` lookahead blocked the "O" of "Ohm"). **Out of scope, flagged:** legacy `param_mapper.py` is PDF-table (Azure OCR) param extraction, not CSV ingest — it imports `ocr_correction` (not among the 7 files) and belongs with the Stage-3 consumer migration; not ported. Legacy README drift noted (`normalize_status()` mentioned but doesn't exist). Legacy optional `chardet` encoding sniff dropped in favor of `utf-8-sig` + `errors="replace"` (BOM covered by test) — revisit only if a real non-UTF-8 vendor CSV appears. |

| T22 — Stage 2 PDF download (TDD, clean-room) | ✅ Done (pending owner review) — **no real download run yet by design** | 2026-07-12; `src/pdf_download/` (`downloader.py`, `exceptions.py`, `sources/{base,csv_url,direct_mfr,mouser_api}.py`). **38 new tests, TDD red→green confirmed; suite 577 passing — zero real network calls (urlopen + sleep mocked).** Clean-room from legacy `D:\Extractor\2_pdf_download\`. **Investigation surprise:** legacy `sources/` was mislabeled — it held stage-1 CSV configs, and the README's `resolve_url()` interface was never built (onsemi.py an empty TODO). This rebuild implements the promised pluggable interface for real: `PdfSource` ABC + registry (`register`/`get_source`/`iter_sources` by priority) — csv_url (P0, the stage-1 record's own URL), direct_mfr (P10, manufacturer URL patterns as a DATA table; ST/onsemi/Microchip/Nexperia/Diodes from the legacy 2026-05-02 probe; Infineon deliberately absent — unpredictable version slug), mouser_api (P20, fires only when `MOUSER_API_KEY` env is set — zero-network no-op otherwise, verified by test). **Legacy behavior preserved:** `%PDF` magic + ≥100-byte verification before success, retry ×3 exponential backoff with UA rotation, full browser header set (empirically needed vs Cloudflare 403s), gzip/deflate decoding, 1.0s/host throttle (env-tunable), `<out>/<safe_name>/<safe_name>.pdf` layout, skip-if-exists. **Deliberately changed:** (1) typed exceptions (`PdfNetworkError`/`PdfNotFoundError`/`InvalidPdfError`) instead of True/False; (2) 404/410 never retried; (3) TLS verification ON by default (legacy: always `CERT_NONE`), env opt-out `PIPELINE_SSL_NO_VERIFY=1` logged loudly; (4) per-host throttle is in-process monotonic (legacy `fcntl.flock` was a silent no-op on Windows — where this runs); (5) verified payload written tmp→`os.replace`, garbage never lands on the final name; (6) batch returns `BatchReport` (input dicts never mutated; legacy wrote `_pdf_download_failed` into them); (7) legacy checkpoint JSON dropped — skip-if-exists is the resume mechanism; (8) brotli optional-dep support dropped (not requested in Accept-Encoding). **Not ported, flagged:** `alldatasheet` scraper (legacy's own notes: low-quality scans, untested at scale) — one registry entry away if the owner wants it; legacy dead code confirmed dead (Mouser scraping, DuckDuckGo). **Per task instruction, no real download attempted yet** — that sanity check is a follow-up after owner design review. |

| T23 — Stage 3 Azure OCR: figures + OCR attachment + composite splitting (TDD, clean-room) | ✅ Done (pending owner review) — **no real Azure run yet by design** | 2026-07-12; `src/azure_ocr/` (`config.py`, `figures.py`, `figure_ocr.py`, `composite.py`). **37 new tests, TDD red→green confirmed; suite 614 passing — zero Azure/network calls (OCR callable injected + mocked; synthetic images).** Clean-room from legacy `D:\Extractor_azure_ocr\` (16 files read). **Scope**: the three task-named concerns; azure_client/batch/assembler/tables/metadata/CLI are follow-ups. **Investigation finds:** (1) legacy snapshot is broken — every module imports a `..config` that exists nowhere in `D:\Extractor` (never copied from the original repo); constants reconstructed, `DEFAULT_DPI=200` verified against real page PNGs (A4 1654×2339). (2) Azure creds correctly env-based (4 vars), no hardcoded paths ✓. **The composite-split bug, diagnosed:** legacy pixel-gap detection needed ~97% white columns over the FULL image height, so caption text spanning under both charts crossed every gap and killed detection; squarish 2-ups also fell under the 2.2 candidate aspect threshold; failures passed through silently at DEBUG. **Fixes built + tested:** (1) **OCR-text masking** — all OCR line boxes painted white before gap detection (spanning captions are OCR text and vanish; chart ink isn't OCR'd and still blocks false gaps) with the whiteness bar raised to 0.995 (masking removes the reason legacy needed 3% tolerance) — dedicated regression test draws the exact spanning-caption composite and asserts it splits; (2) ≥2 "Figure N" sub-captions is now an independent candidate signal (catches squarish 2-ups); (3) unsplittable candidates get `composite_suspect: True` instead of passing silently; (4) OCR-cluster column fallback kept (axis-title keyword+unit/length/position, 30% cluster gap, 0.55 band validation). **Legacy behavior preserved:** inches→pixels polygon cropping at render DPI with padding+clamping, contour fallback with legacy gates, package-outline keyword flagging incl. the `[mm]`-misread guard + graph-keyword override, OCR idempotency/budget/rate-limit/failure-isolation, split-at-⅓-of-gap positioning, min sub-size 200×150, max 8 subs, sub-caption assignment + local OCR re-coordinates, partial-split file cleanup. **OCR callable is injected** (`ocr_fn`) — production wires Azure Read API later with client/batch migration; tests mock it. Per task, no real Azure OCR run yet — follow-up after owner review. |

## M7 — rdson_vs_tj: classical (non-AI) extraction

| Task | Status | Notes |
|---|---|---|
| T24 — Classical extraction path for `rdson_vs_tj` (TDD) | ✅ Done (pending owner review) — **no real rdson corpus run yet (no rdson figures on disk)** | 2026-07-13; `src/extraction/classical.py` (`detect_curve_classical`, `run_classical_pipeline`, `detect_rdson_units`) + `src/extraction/naming/rdson_vs_tj.py`. **16 new tests, TDD red→green confirmed** (owner approved the test file before any implementation existed; red = `ModuleNotFoundError` verified) — suite now **630 passing, no regressions** (was 614). rdson_vs_tj is a single solid-colored curve, so no LineFormer/GPU: curve pixels are isolated classically (chromatic-pixel segmentation — axes/grid/text are achromatic; morphological close bridges small print/compression gaps; too-small/too-short components like legend swatches dropped with logged reasons) and emitted as the SAME `Detection` objects the AI path produces, then handed to the **existing frozen Stage-5 core unchanged**: `process_detections(..., expected_curve_count=1)` runs dedup, count gating, skeletonize, naming (registry), calibration (`parse_numeric_ticks`/`fit_axis`), and schema validation — zero reimplementation, output dict is byte-for-byte the Stage-6/Stage-7 contract. Never-guess preserved: 2+ credible curves are ALL returned and quarantine via the count gate; no curve → `needs_review`, not a crash; malformed OCR input raises (→ Stage 7 `failed_extraction`). New rdson y-axis unit detection (mΩ→`mOhm`, Ω→`Ohm`, `normalized`; ASCII-canonical, milli pattern case-sensitive so "MΩ" never reads as milli) fills units when the core's capacitance-only unit detector comes back `units_undetected` — already-computed curves/calibration reused verbatim. **Files: new code only**, except the owner-authorized registry entry in `src/extraction/naming/__init__.py` (import + 2 data entries — the designed extension point). No frozen Stage-5 file modified. **Open items for owner:** (1) **monochrome/black-curve charts not handled yet** — a chroma-free figure correctly quarantines rather than extracting; a mono fallback needs the real rdson corpus (none in `data/` yet — needs a Stage 1–3 batch) to size properly; (2) **plausibility range check for `rdson_vs_tj` deferred** (would be a `PLAUSIBILITY_SPECS` entry in frozen `pipeline.py` — separate approval) until tested on real data. |

| T25 — Stage-4 `rdson_vs_tj` registry entry + Infineon "Diagram"-template classification fix (TDD) | ✅ Done (pending owner review) — **27→43 matched of 50, 0 regressions** | 2026-07-13; permanent `rdson_vs_tj` entry added to `src/classification/curve_registry.py` (owner-approved; previously runtime-injected in the T24 runs, validated at 11/13 then 27/50). **21 new tests, TDD red→green confirmed** (red demonstrated by stashing the entry: 19/20 fail; the `"R DS(on)"` case red-first separately), every caption/OCR string verbatim from the real corpus — `tests/test_curve_registry_rdson.py`. Suite **651 passing, no regressions.** **Quarantine diagnosis (15 devices):** two shared root causes — (1) the known Stage-3 caption off-by-one shifts "Diagram N: Drain-source on-state resistance" onto the WRONG figure (page-header Infineon logo, output characteristics, or RDS(on)=f(ID)), so the true chart is captionless; (2) the true chart's axis text is OCR-mangled (y `RDS(on) [m_2]`/`[22]`/`[mQ]`/`[mW]`/`R DS(on)`, x `Tj[C]`/`T [C]`/`T, [º℃]`), leaving it at 2.5–3.5 vs. the 5.0 threshold. **Fix is registry DATA only** (no scoring/classify logic touched): mangled-Tj x keywords (`"tj ["`/`"tj["`/`"t, ["`/`"t [c"` — deliberately NOT `"tc ["`; a test locks that case-temperature axes gain no credit), `"r ds(on)"` y variant, and two evidence-backed negatives (`"parameter: v"` for multi-Vgs-curve footers, `"output characteristic"` for shifted captions); every keyword commented with the device that motivated it. **50-device before/after: 27→43 matched, 15→0 quarantined** (all 15 now match, each verified as the TRUE chart), plus `BSB012N03LX3G` (previously no_match). **All 27 original matches unchanged (same device → same figure).** Remaining 7 no_match verified as **upstream Stage 1–3 data gaps, not classifier issues** — their `figures/` dirs are incomplete on disk (e.g. `AUIRFZ34N`: 22 figures in `full_extraction.json`, 1 PNG rendered, rdson chart missing); flagged for the stage 1–3 migration. **Open item for owner:** several Infineon rdson charts plot **TWO curves ("typ" and "max")**, unlike the single-curve IR template — decide how `rdson_vs_tj`'s expected-curve-count (currently 1) should treat these before finalizing the mono detector design. |

## M8 — id_vs_vgs: LineFormer retraining

> Recorded 2026-07-14 from owner-supplied results (runs executed 2026-07-13 on the
> GPU box; manifests/eval reports live there, not in this repo's `data/`).

| Task | Status | Notes |
|---|---|---|
| Run 3 — fine-tune on combined batch1+batch2 | ✅ Done — **PRODUCTION CHECKPOINT for id_vs_vgs** (owner, 2026-07-13) | Dataset: combined id_vs_vgs pool, **train 299 / val 70 / test 30 images** (the owner-fixed split, test sha256 `6e8e96d1b4ce…`). Config: 8000-iteration ceiling, patience 600, buffer radius 4.5 px. Best checkpoint `best_segm_mAP_50_iter_4800.pth` — GPU box `/mnt/data/my-datasheet/checkpoints/id_vs_vgs_run3_combined_8000iter/` (local copy under `checkpoints/id_vs_vgs/`, git-ignored). **Frozen test set: mAP@50 0.7414, mAP@75 0.0021, overall recall 0.8889.** Per-temperature recall: TJ_25C 0.92, TJ_150C 1.00, TJ_175C 0.91, TJ_-40C 0.50 (low sample size — only 4 TJ_-40C instances in test; TJ_-55C has zero val/test instances by split construction, noted 2026-07-13). mAP@75 near-zero matches the pattern seen in capacitance Runs A/A2 (4.5 px GT buffer wider than the printed stroke). |
| 2px buffer-radius experiment | ❌ Abandoned (owner decision, 2026-07-13) — **4.5 px stays; revisit in the diversity pass** | Finding: id_vs_vgs had inherited the 4.5 px default buffer from capacitance (owner-tuned there, T4) without ever being tuned for transfer-char charts; visual comparison showed **2 px matches the real printed line width better**. Retrained with 2 px masks: **training repeatedly got stuck at 0.0 score** — early stopping too aggressive for thinner masks needing more localization precision. A longer patience (1200) was attempted but a command-line mixup prevented confirming it actually applied. **Decision: reverted to the working 4.5 px Run-3 checkpoint (74%) as production; 2 px abandoned for now** — not worth the debugging time. Revisit during the "diversity pass" phase after all 7 curve types are built. |
| T26 — Real-world spot check: Rohm id_vs_vgs figures | 🔍 Reported | **10/12 spot-checked images looked correct** despite different chart styles — the model generalizes reasonably well even though training data is all-Infineon so far. Overlays: `data/t26_id_vs_vgs_rohm/` (git-ignored). |

| T27 — Monochrome (black-curve) detector for `rdson_vs_tj` (TDD) | ✅ Done (pending owner review) — **9/11 previously-quarantined charts now extract cleanly; 2 have a diagnosed spurious-branch artifact** | 2026-07-14; `detect_curve_monochrome(image, ocr_lines=None)` added to `src/extraction/classical.py` as a **fallback** to the color path (color runs first; only when it finds nothing — every real rdson chart, 0 chromatic pixels — does mono run). **23 new tests, TDD red→green confirmed** (red = ImportError; suite **697 passing**, was 674, no regressions). Emits the SAME `Detection` objects, so the frozen Stage-5 core is untouched (no frozen file modified; only new code + the already-authorized `classical.py`). **Two ideas adopted from the reviewed legacy `cv_curve_extract.py` (never copied):** OCR-label-box **inpainting** (not white-out — a white hole splits a curve a label sits on; inpaint reconstructs it; proven by a white-out-baseline test that splits where inpaint stays one) and a **density+width-span** component filter for text. **The legacy flat-curve failure is guarded:** only near-full-span straight runs (≥50% of the dimension) are treated as gridlines/axes, so a partially-flat curve survives — locked by a dedicated test. Design driven by the real corpus (`data/t24_mono_survey/MONO_DETECTOR_REQUIREMENTS.md`): ink = gray<128 (curve AND dark grid), gridlines/axes removed by STRUCTURE not intensity (Infineon axes can be lighter than the curve), crossing nicks reconnected by a small **dilation** (not a close — the close's erosion severs the thin diagonal bridge; skeletonize thins the extra width away downstream), two stacked typ/max curves never merged (tested at 18px separation). **Real-data run — the 11 T24 charts that ALL quarantined before (0 color detections):** now **11/11 status ok** with correct calibration and units (`normalized`), values physically sane (x −55→175 °C, y ~0.65→2.5 rising, strictly increasing). **BUT visual overlay check (`data/t27_mono_rdson_run/overlays/`, git-ignored) found 2/11 are silently partly-wrong** (`AUIRF7675M2TR`, `AUIRF7736M2TR`): a spurious upper branch drifts above the true curve over the warm half. **Diagnosed:** a non-full-width horizontal line (partial gridline/scan streak) running parallel just above the curve survives gridline removal (its run is <50% width) and the dilation bridge MERGES it into the curve component (mask mean column-thickness ~20px / max 33px vs ~14/19 on the 9 clean charts); the per-column mean then rides up. **9/11 are clean end-to-end** (overlays track the printed curve tightly). Same "ok-but-wrong slips the schema" class as the T15 capacitance finding — reported, not patched (adding more heuristics to the near-frozen `classical.py` needs owner review per §4/§5). **Open items for owner:** (a) the merged-parallel-line artifact — options are a max-column-thickness plausibility gate (downgrade abnormally thick masks to needs_review) or a smarter line-vs-curve separation; (b) still no `PLAUSIBILITY_SPECS` entry for rdson value ranges (deferred from T24, frozen-file approval). |
| T27 follow-up — Max-column-thickness safety gate for the mono merge artifact | ✅ Done (pending owner review) — **owner-approved fix confirmed on both real known-bad devices, zero false positives on the 9 clean ones** | 2026-07-14; owner approved option (a) from T27's open items. `MONO_MAX_MEDIAN_COL_THICKNESS_PX = 18` + `_median_col_thickness()` added to `src/extraction/classical.py` (new code only, no frozen file touched — `pipeline.py`/`PLAUSIBILITY_SPECS` deliberately left alone, same as T24's rdson-specific unit/name overrides already living in `run_classical_pipeline`). **Threshold corpus-calibrated, not guessed:** measured MEDIAN (not mean/max — median proved the cleanest separator) column thickness across all 11 real T24 charts — the 9 genuinely single-stroke extractions cluster at 12–16px, the 2 known merged-line cases at 21–22px; 18px sits at the midpoint with margin both sides. Gate fires only when `used_monochrome` is true (the color path is exempt by design — the merge mechanism is specific to monochrome gridline-removal + gap-bridging) and only downgrades a currently-"ok" result, keeping the traced curves/calibration/units for the reviewer (same "never an empty shell" pattern as the existing implausible-calibration gate) rather than dropping them. **3 new tests, TDD red→green confirmed** (red = the merged-streak fixture stayed "ok"; a synthetic partial-width streak 10px above the curve was tuned empirically — same iterative-fixture approach used earlier in the T27 session — to reproduce the diagnosed mechanism, measured at 20.0px, safely clear of the 11.0px clean control). Suite **700 passing** (was 697), no regressions. **Real-data confirmation (owner-specified acceptance test) — re-ran the exact same 11-device corpus:** `AUIRF7675M2TR` now `needs_review` (median 21.0px, was silently "ok"), `AUIRF7736M2TR` now `needs_review` (median 22.0px, was silently "ok") — **both known-bad devices caught, exactly as diagnosed.** All 9 genuinely clean devices (`94-3316`, `AUIRF1010EZS`, `AUIRF1010ZL`, `AUIRF1404`, `AUIRF1404STRL`, `AUIRF1405-INF`, `AUIRF2804STRL7P`, `AUIRF7640S2TR`, `AUIRF7739L2TR`) **still `ok`** — zero false positives. **Net result: 9/11 correctly `ok`, 2/11 correctly `needs_review` — zero silently-wrong results on this sample**, closing the gap T27 flagged. Updated outputs: `data/t27_mono_rdson_run/` (git-ignored). Still open: no `PLAUSIBILITY_SPECS` entry for rdson value ranges (unrelated to this fix). |
| T28 — rdson_vs_tj plausibility ranges (owner-approved frozen-file addition) | ✅ Done (pending owner review) — **9 new tests, corpus re-run unchanged (9 ok / 2 caught), zero false positives** | 2026-07-14; closes the last T24 open item. **(1) Frozen `pipeline.py` (this specific addition owner-approved):** `PLAUSIBILITY_SPECS["rdson_vs_tj"] = {"x_range": (-75.0, 200.0)}` — junction temperature; real corpus spans −57..176 °C, silicon Tj(max) is 150–200 °C — plus `x_range` support in `_implausibility_reason` (exact mirror of the existing `y_range` logic, still data-driven; needed because the approved check is on the temperature axis). Capacitance's entry byte-identical, pinned by a new data-pin test. **(2) Unit-aware y-range gate in `classical.py`** (`RDSON_Y_PLAUSIBLE_RANGES`): the plausible rdson y bound depends on the detected unit, which only `run_classical_pipeline` knows (the core's unit detector is capacitance-only) — `normalized (0.3, 5.0)` (owner-specified; the whole real corpus, all normalized, spans 0.52–2.5), `mOhm (0.05, 1e5)` / `Ohm (5e-5, 100)` (physical bounds + the capacitance spec's one-decade-margin idiom; **flagged: no mOhm/Ohm chart exists in the tested corpus yet, so those two ranges are corpus-untested**). Runs last (needs final units), applies to BOTH color and mono paths, downgrades ok→needs_review keeping curves/calibration/units (reason `implausible_rdson_values: …`). **9 new tests, TDD red→green confirmed** (red: 5 failed, the 4 in-range controls passing as expected); suite **709 passing** (was 700). **One fixture correction, assertions untouched:** the new gate exposed that `test_rdson_two_curve.py`'s synthetic `two_curve_chart` drew its lower curve to row 260 — below the 0 mΩ axis row, i.e. **negative resistance** — so the fixture (not the gate) was wrong; lower-curve offset changed +60→+35 px (stays above the axis), "typ" label and the three-curve variant's third curve moved accordingly. **Real 11-device corpus re-run: identical outcome** — 9 ok (all pass both new gates), 2 needs_review (still caught by the T27 thickness gate, which fires first; their y values are in-range so the new gates correctly stay silent), **zero false positives, no new catches** (expected — the corpus was already visually verified clean). |

## Upcoming

- **Next: owner decision on T15's log-axis calibration finding**, then a real model-inference sample run on the GPU box (T15 used GT masks as a stand-in), then stage 6 (visual review) and stage 7 (orchestrator/validation → MongoDB)
- Stage-3 follow-ups: azure_client/batch/assembler/tables/metadata migration (T23 covered figures+OCR+composite), then a real Azure OCR sanity run (owner-gated; costs money)
- Synthetic data generation for capacitance — deferred until all 7 target curve types have working models (list above; 6 of 7 not started)
- Extend LineFormer training/eval pipeline to the remaining 6 curve types once stages 4–7 exist to drive that work
- rdson_vs_tj follow-ups (T24): produce a real rdson figure corpus via stages 1–3, sanity-run the classical path on it, then decide the mono/black-curve fallback and the `PLAUSIBILITY_SPECS` entry (frozen-file approval needed)

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

### 2026-07-11 — Session: T21 Stage-1 CSV ingest
- **Start:** stages 4–7 complete (T12–T20); goal: migrate Stage 1 (CSV ingest) clean-room, TDD.
- Investigation first: all 7 legacy `1_csv_input` files read and summarized; two legacy
  bugs identified and deliberately NOT ported (silent 1:1 unit guess, silent row drops);
  `param_mapper.py` identified as Stage-3 material, not CSV ingest. No real device CSVs
  found on disk — fixtures built from the real DigiKey header names in the legacy tables.
- **End:** T21 complete (pending owner review). `src/ingest/` ×5 modules, 113 tests
  (red→green), full suite **539 passing**. PROGRESS.md M6 section added. Details in T21 row.

### 2026-07-12 — Session: structural cleanup (owner-approved)
- `src/cvat_to_coco.py` → `src/dataset_tools/cvat_to_coco.py` (data-prep tool, per
  inventory recommendation); imports updated in `split_dataset.py` + 4 test files,
  CLI is now `python -m src.dataset_tools.cvat_to_coco` (SETUP.md updated).
  Historical PROGRESS.md references left as written (§8: never delete history).
- Git-ignored scratch renames to the two-digit tNN convention: `data/t8a_overlay` →
  `t08a_overlay`, `t8f_overlay` → `t08f_overlay`, `t8b_*.xml` → `t08b_*.xml`.
- Full suite 539 passing after the move.

### 2026-07-12 — Session: T22 Stage-2 PDF download
- Investigation: legacy `sources/` package was mislabeled stage-1 CSV configs; the
  README's pluggable `resolve_url` interface was never actually built. Real logic
  lived in `pdf_downloader.py` + `pdf_sources.py`. Flags: TLS verification disabled
  unconditionally; flock rate-limiter dead on Windows; 404s retried; bool returns.
  No hardcoded creds (MOUSER_API_KEY correctly via env) ✓.
- **End:** T22 complete (pending owner review). `src/pdf_download/` (typed exceptions,
  source registry, verified downloads), 38 tests all-mocked (red→green), suite
  **577 passing**. No real download run yet — awaits owner design review. Details in T22 row.

### 2026-07-12 — T22 follow-up: owner approval + real download sanity check
- **Owner approved T22.** Infineon direct-mfr question answered: not needed now —
  CSV-URL covers Infineon in practice (DigiKey exports carry a Datasheet URL for every
  vendor); a direct Infineon pattern isn't constructible (unpredictable version slug,
  same reason legacy skipped it); Mouser API is the better fallback if recovery
  becomes an issue. Decision: no Infineon source unless a real gap shows up.
- **Real sanity run (6 attempts, 3 downloaded):**
  - ✅ `csv_url` — TI CSD18536KCS via ti.com symlink (922,031 bytes, PDF 1.4, EOF ok, ~13 pages)
  - ✅ `direct_mfr` ST — STP55NF06 (812,175 bytes, PDF 1.3; text extracted and read — genuine datasheet)
  - ✅ `direct_mfr` onsemi — NTD5867NL (272,801 bytes, PDF 1.4)
  - ❌ onsemi 2N7002 + Nexperia BSS138: servers returned HTML → **InvalidPdfError correctly
    raised, nothing written to disk — the HTML-as-.pdf guard proven on real traffic**
  - ❌ hand-guessed Vishay URL: 404 → typed PdfNotFoundError, single attempt (no-retry-on-404 confirmed live)
- **No credentials in logs:** MOUSER_API_KEY unset (source no-op'd); log lines contain
  only public URLs + byte counts; the mouser_api module never logs its request URL
  (the key rides in the query string by Mouser's design — kept out of logs by construction).

### 2026-07-12 — Session: T23 Stage-3 figures/OCR/composite
- Investigation: legacy `..config` module missing from the snapshot (imports broken);
  DPI=200 reconstructed from real page renders. Azure creds env-based ✓, no hardcoded paths ✓.
  Composite-split bug root-caused: full-height whiteness requirement defeated by
  captions spanning under both charts + candidate misses + silent pass-through.
- **End:** T23 complete (pending owner review). `src/azure_ocr/` ×4 modules; OCR-text
  masking fix + composite_suspect flagging + sub-caption candidate signal; 37 tests
  (red→green, spanning-caption regression included), suite **614 passing**. No real
  Azure call made. Details in T23 row.

### 2026-07-13 — Session: T24 rdson_vs_tj classical extraction
- Design proposed and discussed first (CLAUDE.md §5): classical detector as a
  drop-in `Detection` front-end so the frozen Stage-5 core is reused, not copied.
- Owner approved the 16-test file (red phase confirmed: module absent) BEFORE any
  implementation; scenarios owner-specified (clean/gridline/axis-edge/gaps/legend/
  faint/border/blank-quarantine/bad-calibration/single-valued-x/axis-range +
  Stage-6 output-format match).
- **End:** T24 complete (pending owner review). New: `src/extraction/classical.py`,
  `src/extraction/naming/rdson_vs_tj.py`; authorized edit: naming-registry entry in
  `src/extraction/naming/__init__.py`. 16/16 new tests green, full suite
  **630 passing**, no regressions. Open items: mono/black-curve fallback and
  rdson plausibility bounds — both deferred until a real rdson corpus exists
  (requires a Stage 1–3 batch run). Details in T24 row.

### 2026-07-13 — Session: T24 runs + T25 classification fix
- 13-device pipeline run (Stage 4→5→6): 11 rdson figures matched, 0 ok / 11
  needs_review / 0 errors — all quarantines from the documented mono limitation
  (every chart black-and-white). Gallery at `data/t24_rdson_run/gallery/`.
- 50-device mono survey: 27 matched charts measured (all solid, 4–9 px, two grid
  templates); requirements doc at `data/t24_mono_survey/MONO_DETECTOR_REQUIREMENTS.md`.
- T25: permanent registry entry + Infineon quarantine fix (details in T25 row).
  21 tests red→green; suite **651 passing**; 50-device re-run 27→43 matched, 0 regressions.

### 2026-07-13 — id_vs_vgs training data: batch 2 converted + merged into split
- CVAT job 4235434 export (`data/cvat_exports/id_vs_vgs_batch2_cleaned.xml`, 200 images)
  pre-validated: labels 100% `line`, 0 duplicate file_names, **0 overlap with batch 1**,
  no degenerate shapes. Two findings, both handled: (1) **2 leftover "TODO" shapes** on
  `IPA60R180P7SXKSA1__fig_p9_022.png` — stripped via the tested `strip_todo_shapes` tool;
  the image then had ZERO real shapes (an unfinished annotation of a real chart, NOT a
  verified negative), so it was **dropped entirely** rather than kept as a false empty
  negative — needs re-annotation if it should be recovered (staged intermediates:
  `data/staging/id_vs_vgs_batch2_{stripped,final}.xml`). (2) **New temperature label
  `TJ_-55` (46 shapes)** — not in batch 1's vocabulary (TJ_25C/150C/175C/-40C) and
  missing the "C" suffix; kept verbatim (never-guess), **owner to decide** whether to
  normalize to `TJ_-55C`.
- Converted via frozen `convert()` (buffer 4.5 px, empty negatives kept, same as batch 1)
  → `data/coco/id_vs_vgs_batch2.json`: **199 images / 420 annotations, validate_coco
  clean; 11 zero-shape images kept as empty negatives** (TJ_25C 182 / TJ_150C 110 /
  TJ_175C 72 / TJ_-55 46 / TJ_-40C 10).
- Combined pool `data/coco/id_vs_vgs_combined.json`: **399 images / 744 annotations**
  (200+199, 324+420 — exact), clean, 0 duplicate file_names.
- Merged into the existing split via the T9 tool (`merge_new_batch_into_split`,
  `val_image_target=20` = round(10% × 199), greedy landed on 12): 11 batch-2 families,
  6 → train (BSC-BSZ, IAU, IAUCN, IPA, IPB, IPD = 187 img), 5 smallest → val
  (BUZ, IAUTN, IAUZ, IMZA, IPAN = 12 img). **Final split: train 328 img / 651 ann
  (22 empty), val 41 / 39 (24 empty), test UNCHANGED 30 / 54 — sha256
  `6e8e96d1b4ce7633ca6d3f17044db1916db39bf44c85698d011c8e1a45107616` verified
  identical before/after.** All three parts validate clean, pairwise disjoint,
  total exactly 399. Batch-1-only split backed up at
  `data/staging/split_id_vs_vgs_backup_batch1_only/`. Data files git-ignored per policy.
- **Two flags for owner before retraining:** (1) val is 59% empty images (24/41) and
  annotation-light (39 ann vs train's 651) — same imbalance that motivated batch 1's
  val/test swap; consider whether val is still a meaningful early-stopping signal.
  (2) `assign_family` treats IAU / IAUCN / IAUTN / IAUZ as four distinct families
  (only IAUA/IAUAN/IAUC merge into IAU per the T5 map), so IAU+IAUCN sit in train
  while IAUTN/IAUZ sit in val — if these subseries share one chart template, val
  style-leaks from train; extending `FAMILY_MERGE_MAP` is an owner call.
- **STOPPED before retraining per instruction — owner reviews these numbers first.**

### 2026-07-14 — Session: T27 monochrome rdson detector
- Design provided/approved by owner in-task (color-first, mono-fallback; adopt
  legacy inpaint + density filter; guard the flat-curve failure). Legacy
  `cv_curve_extract.py` reviewed first (read-only) — technique is the same
  "remove straight lines, remove text, keep bendy" idea, more fully built;
  lifted two ideas, kept our schema discipline.
- TDD: 23 tests written first (red = ImportError), then `detect_curve_monochrome`
  + color→mono fallback wiring. Suite **697 passing** (was 674), no regressions.
  Two fixture/impl tunings during green: dilation-bridge (not close) to
  reconnect crossings without severing them; tight OCR box for the on-curve
  inpaint-vs-whiteout test (TELEA only reconstructs a stroke-tight box).
- Real run on the 11 T24 charts that were 0/11 before: **11/11 status ok, sane
  values**, but overlay check found **2/11 silently partly-wrong** (spurious
  upper branch from a merged parallel line — diagnosed via mask thickness).
  **9/11 clean end-to-end.** Reported, not patched (frozen-module change needs
  owner review). Outputs: `data/t27_mono_rdson_run/` (git-ignored).

### 2026-07-21 — Session: T29 live Stage 4→5 wiring adapter
- CLAUDE.md read fully first; design (module names/locations) proposed and
  confirmed with the owner BEFORE any file was created (§5), including 3
  clarifying design questions (claim-state sharing mechanism, stage3_root
  configuration, model-loading strategy) resolved before writing tests.
- TDD strictly sequential per module, owner-approved order: write tests →
  confirm red (`ModuleNotFoundError`, not a fixture bug) → implement → green
  → full suite re-run → next module. Red phase for all three was reviewed
  and approved by the owner before any implementation began; one correction
  came back mid-review (the `ocr_lines` conversion is identical for both
  extraction paths, not split classical-vs-model as first assumed) and was
  applied to the tests, re-confirmed red, before implementation started.
- Implementation order: `stage3_loader.py` (11 tests green) → full suite
  720 passing → `extraction_registry.py` (17 tests green) → full suite 737
  passing → `live_stages.py` (initially 35/44 green, 9 failing on a real
  test-fixture bug — classical-path tests needed a real image file on disk
  that was never written; owner approved the fix, confirmed to patch the
  real `cv2.imread` attribute specifically and keep the one genuine
  missing-file test unmocked) → all 44 green → full suite **781 passing**.
- Zero regressions, zero frozen files touched (only imported from).
  `orchestrator/pipeline.py`'s CLI still defaults to `PrecomputedStage5` —
  swapping it is a deliberately separate follow-up (touches a frozen file).
  Full details in the new T29 row (M5 — Stage 7: Orchestrator).

### 2026-07-16 — zth_vs_time multi-curve: training config prepared (not yet run)
- Owner approved: do NOT merge with the single-curve `zth_vs_time` split;
  train on `data/coco/split_zth_multicurve_batch1/` alone.
- New config `src/training/configs/lineformer_zth_multicurve_run1.py`,
  chained via `_base_` from `lineformer_run_a.py` — same pattern Run A2
  used, only the DATA (new split/image-folder paths, `data/zth_vs_time_images/`
  — already fully populated, all 51 needed images confirmed present) and
  schedule numbers overridden. Inherits unchanged: official pretrained init,
  fp32 (no fp16 key anywhere), LR 5e-6, no horizontal/vertical flip,
  multi-scale resize + brightness/contrast jitter, batch size 1, "best +
  latest only" retention. `classes=("line",)` and `num_queries=100` both
  inherited as-is — correct for this dataset unchanged (label is "line"
  here too; 100 queries comfortably covers the max 7 instances/image).
  **Schedule, owner-specified directly this time (not re-derived from
  dataset size like Run A2 was): max_iters=8000, patience_iters=2000**;
  eval_interval kept at 800 (10% of max_iters, same ratio as Run A/A2) —
  patience therefore ~2.5 eval intervals, looser than Run A/A2's clean 3,
  since the owner gave the number directly. **Flagged, not resolved:**
  train is only 41 images, so 8000 iters ≈ 195 epochs — far more than Run A
  (~17.2) or Run A2 (~17.5) saw on their much larger splits; may be
  intentional for a small first batch, noted for owner awareness.
- **Could not be validated locally** — no mmcv/torch on this Windows box
  (GPU-only training env lives on the AWS box per T6); real validation
  happens when `train_lineformer.py` resolves and dumps the full config to
  `run_manifest.json` before training starts, same as every prior run.
- **Not yet run** — owner runs it on the GPU box. Exact command:
  `python -m src.training.train_lineformer --config src/training/configs/lineformer_zth_multicurve_run1.py --work-dir /mnt/data/my-datasheet/checkpoints/zth_multicurve_run1 --seed 42`
- **Transfer checklist for the GPU box** (none of this is git-tracked —
  `data/` is git-ignored repo-wide): the new config file itself (tracked in
  git, so a normal pull/push covers it), `data/coco/split_zth_multicurve_batch1/{train,val}.json`,
  and `data/zth_vs_time_images/` (51 PNGs this run actually needs, out of
  the folder's 315).
- Suite **709 passing**, unchanged.

### 2026-07-16 — zth_vs_time: first multi-curve annotation batch (all duty-cycle curves traced)
- Same CVAT job (4244123), re-exported after progressive re-annotation: shapes
  now traced with a GENERIC label/attribute (`label="line"`, `curve_name="Curve"`)
  instead of the old single-curve `"single_pulse"` style. **The attribute
  VALUE alone can't distinguish old vs. new** — CVAT reset ALL 436 shape
  attribute values (old single-curve ones included) to the new schema's
  default when the label's value list changed; the reliable signal is
  **shape count per image**: 0 (41, unannotated, unchanged), 1 (117, still
  old single-curve style, not yet re-annotated), 4/5/6/7 (54, the new
  multi-curve tracing). Cross-checked against the 2026-07-14 export:
  **all 54 multi-shape images were single-shape there** — confirms this is
  a progressive upgrade of a subset of the same 171 previously-annotated
  images, not a new device batch (the task's "~70" was an estimate; actual
  is 54, then 51 after excluding 3 incomplete — see below).
- **Completeness check — two automated heuristics tried and rejected, visual
  ground truth used instead.** (1) OCR-label-count-vs-shape-count: too
  unreliable (Azure OCR misses duty-factor labels inconsistently — flagged
  a chart I'd already visually confirmed complete). (2) Geometric
  "untraced dark component" scan (reused `_remove_straight_lines` from the
  monochrome detector): also over-triggered, mostly on unmasked
  caption/footer text and normal black-stroke "fringe" around a thin
  colored polyline overlay (an artifact of the QC overlay rendering, not a
  real gap) — both false-positived on images later confirmed complete.
  **Settled on full-resolution colored-overlay visual review**, tiled into
  5 contact sheets (`zth_bigsheet_0..4`, git-ignored scratch) covering all
  54 images at legible size, each shape drawn in a distinct color: a
  genuinely untraced curve reads as a solid uncolored stroke running the
  chart's full span, unambiguous once the fringe artifact is understood.
  **Result: the shape-count anomaly WAS the correct signal all along** —
  every device family has a fixed template curve count (IAU*/IAUC*/IAUCN*/
  IAUTN*/IAUZ* families: 5 curves always; IPA* family: 7 always, confirmed
  across every image reviewed), so any image whose shape count falls short
  of its family's fixed count is incomplete, no exceptions found either way
  in the full visual pass. **3 confirmed incomplete, excluded:**
  `IAUC80N04S6N036ATMA1__fig_p4_010.png` (4/5 traced, missing D=0.01),
  `IPA60R120C7XKSA1__fig_p7_015.png` (6/7, missing D=0.05),
  `IPA60R190P6XKSA1__fig_p8_015.png` (6/7, missing D=0.02) — each verified
  by overlaying the traced curves in color and confirming exactly one
  fully-uncolored duty-cycle curve remains. **51 images / 303 annotations
  kept** (27×5 + 24×7 — exact), 0 TODOs, 0 duplicate file_names.
- Filtered set staged at `data/cvat_exports/annotations_zth_multicurve_batch1.xml`
  (git-ignored; source XML in Downloads untouched).
- **Buffer radius re-measured for this export** (not assumed from the
  single_pulse batch): same distance-transform method, 60 sampled points
  across the 51 images gave **median 4.0px** full stroke width (single_pulse
  batch: 4.4px) — consistent, same source rendering. **Kept buffer_px=2.5**
  (the established `zth_vs_time` value) rather than retuning to 2.0, so
  both annotation batches of this curve type share one mask convention;
  visually confirmed on a 7-curve chart that the buffered outlines hug each
  of the closely-packed curves without materially bleeding into neighbors.
- Converted via `convert()` (buffer 2.5px) → `data/coco/zth_multicurve_batch1.json`:
  **51 images / 303 annotations**, `validate_coco` clean, `curve_name` 100%
  `"Curve"` (generic — Stage 4/5 naming for zth_vs_time doesn't exist yet,
  same as every other zth batch so far; no per-curve-name uniqueness is
  enforced at this conversion stage, confirmed against `validate_coco`'s
  actual checks before relying on it).
- **Simple 80/20 split** (owner-specified: no family-grouping this time,
  small dataset) — seeded random per-image shuffle: **train 41 img / 247
  ann, val 10 img / 56 ann**, both `validate_coco` clean, 0 overlap, total
  exactly 51. Both the 5-curve and 7-curve templates present on each side
  (train 20×5/21×7, val 7×5/3×7). Files at
  `data/coco/split_zth_multicurve_batch1/` (git-ignored).
- **Not yet combined with the single-curve `zth_vs_time` split** (T-batch
  from 2026-07-15) — that dataset has 1 curve/image (`single_pulse` only);
  this one has 5-7 curves/image (all duty factors, generic name) from a
  partially-overlapping set of source images (all 51 devices here were
  previously in that single-curve split, now upgraded) — combining them
  needs an owner decision on how to handle the same image appearing with
  two different annotation styles (e.g., does the multi-curve version
  supersede the single-curve one for that image, or do both coexist under
  different curve_type scoping?), not decided here.
- Suite **709 passing**, unchanged (no source code touched this session).
- **STOPPED before training per instruction — owner reviews these numbers first.**

### 2026-07-15 — zth_vs_time: image collection + 6-device Stage 1-3 gap dropped (owner-approved)
- `collect_images` (T4a tool, unchanged) run against the combined 321-image
  COCO, searching the only local source root
  (`D:\Extractor\data\OCR1-OCR13`): **315/321 copied** to
  `data/zth_vs_time_images/` (flat, git-ignored). 6 missing, all confirmed
  as a genuine **Stage 1-3 rendering gap, same class as the T25 finding**
  (not a naming mismatch): each figure IS listed in its device's
  `full_extraction.json` with a caption matching the annotated chart
  ("Maximum Effective Transient Thermal Impedance" / "Max. transient
  thermal impedance"), but the PNG was never rendered to disk —
  `AUIRFZ34N__fig_p6_014.png`, `AUIRFZ48Z__fig_p6_013.png`,
  `AUIRLU3114Z-701TRL__fig_p6_014.png`, `AUIRLU3114Z__fig_p5_018.png`,
  `BSB165N15NZ3GXUMA2__fig_p6_012.png`, `BSC009N04LSSCATMA1__fig_p6_012.png`.
  One near-miss checked and rejected: `AUIRLU3114Z`'s folder has a
  `validated_fig_p5_018.png` — inspected visually, confirmed to be a
  **legacy tool's curve-fit-overlay output** (colored curve_1..7 traces +
  an RC-network Ri/τi table baked into the pixels), not the raw chart;
  `collect_images`'s variant-name exclusion correctly refused to match it,
  and it was NOT substituted in.
- **Owner approved dropping the 6** (same resolution as T25's rdson gap).
  Excised directly from `zth_vs_time_combined.json` and the three split
  files (train -4, val -2, test -0 — no family lost its last image: AUIRF
  19→17, AUIRL 4→2, BSB 3→2, BSC-BSZ 104→103, all still non-empty on their
  side) — **no re-run of the split algorithm**, so every one of the other
  315 images keeps its original train/val/test placement exactly. All four
  files re-`validate_coco`-clean; `split_manifest.json` counts/ratios/
  `source_coco_sha256` updated in place, with a new `missing_images_dropped`
  field recording the reason, the 6 dropped names, and which side each came
  from, for permanent traceability (CLAUDE.md §8: never delete history).
- **Final: 315 images / 315 annotations total** — train 221/221 (was 225),
  val 46/46 (was 48), test 48/48 (unchanged). All three re-verified
  pairwise disjoint, `validate_coco` clean, and now byte-for-byte matching
  the 315 images actually present in `data/zth_vs_time_images/`.
- Suite **709 passing**, unchanged (no source code touched this session).

### 2026-07-15 — zth_vs_time (thermal_impedance): first annotation batch combined + split
- Two raw CVAT exports from Downloads (annotation-only, no images bundled —
  source figures resolved from the known local corpus `D:\Extractor\data\OCR1-OCR13`).
  **Task's file-to-format mapping was swapped vs. what's actually on disk** —
  flagged and resolved by content, not filename, before touching anything:
  - `job_4244123_annotations_...` (212 images / 171 shapes) is the file
    already using `label="line"`/`curve_name="Single pluse"` (typo) —
    matches the task's "File 2" description, not "File 1".
  - `Zth_Archit_200` (declared job size 207, but only **155** `<image>`
    elements actually exported — CVAT omits never-opened frames; 150 shapes)
    is the file using `label="Curve_Zth_single"` directly / attribute
    `Zth_curve="Single_Pulse"` — matches the task's "File 1" description.
  - Both counts match the task's expected `~155/150` and `~212/171` exactly
    once paired correctly.
- Standardized both to `label="line"`, `curve_name="single_pulse"` (XML
  rewritten via ElementTree, meta `<labels>` block + every shape's label/
  attribute; **originals in Downloads never touched**, output staged at
  `data/cvat_exports/annotations_zth_{job4244123,archit200}.xml`,
  git-ignored). Re-validated with `parse_cvat_xml`: 0 TODOs, 0 duplicate
  file_names within either export, **0 cross-export overlap**, uniform
  `curve_name="single_pulse"` in both.
- **Buffer radius measured, not assumed** (per instruction, same rigor as
  the id_vs_vgs 2px check): a naive perpendicular-scan gave a misleading
  ~8–9px median (inflated by sampling through the chart's convergence zone,
  where the multi-duty-cycle Zth family visually merges — these are
  standard 6-7-curve Zth-vs-time datasheet charts, D=0.5/0.2/0.1/0.05/0.02/
  0.01 + single-pulse, and only single-pulse is in scope/annotated). A
  robust distance-transform measurement at 40 sampled vertices across 20
  images per source gave **combined median 4.4px full stroke width**
  (job_4244123 4.0px, Zth_Archit_200 4.4px). Visually confirmed on an
  isolated (non-convergence) segment: the CVAT polyline sits correctly at
  the stroke's visual centerline, true width ~5–6px there — the
  convergence-zone crop initially looked "off-center" only because it was
  sampling a spot where 2–3 curves visually merge, not this curve's own
  edge. **Buffer radius set to 2.5px** (total mask ≈5.0px, matching the
  measured width) — deliberately NOT the inherited 4.5px default, which
  would have overshot by ~2×. Visual overlay check on both a convergence-
  zone and an isolated segment confirms tight, correct coverage at 2.5px.
  **Flag for future extraction work (not this task):** the multi-duty-cycle
  convergence zone has curve-to-curve gaps as low as 1px in places — no
  buffer radius can avoid ambiguity there; classical/AI extraction will need
  its own handling, same class of issue as rdson's typ/max proximity.
- Converted via frozen `convert()`/`merge_convert()` (buffer 2.5px):
  `zth_job4244123.json` (212/171), `zth_archit200.json` (155/150), combined
  `zth_vs_time_combined.json` — **321 images / 321 annotations** (171+150
  exact; `merge_convert` drops the 46 unannotated frames), `validate_coco`
  clean, 0 duplicate file_names, curve_name 100% `single_pulse`.
- **Group-aware split** (same family tool as every prior batch, seed 42,
  `BSC-BSZ` pinned to train per the T5 map): proposal landed clean on the
  first try, **no manual override needed** (unlike id_vs_vgs's val/test
  swap) — both val and test clear the eval-diversity invariant (≥2
  families/≥15 images) with 6 families each. **train 225 img/225 ann
  (70.1%), val 48/48 (15.0%), test 48/48 (15.0%)** — every image has
  exactly 1 shape (no empty-negative convention here, since `merge_convert`
  already dropped unannotated frames), all three parts `validate_coco`
  clean, pairwise disjoint, total exactly 321. Files at
  `data/coco/split_zth_vs_time/` (git-ignored).
- **STOPPED before training per instruction — no GPU/AWS work, owner reviews
  these numbers first.** No naming/classification module exists yet for
  `zth_vs_time` (Stage 4/5 registry work, separate from this dataset-prep
  task) — `curve_name="single_pulse"` is carried as a plain COCO attribute
  only, same as id_vs_vgs's `TJ_*` values before its own registry entry
  existed.

### 2026-07-14 — Session: T28 rdson plausibility ranges
- Owner approved the frozen-file addition (last T24 open item). TDD: 9 tests
  first (red: 5 failed / 4 in-range controls green), then the rdson
  `x_range` entry + generic `x_range` support in `pipeline.py` and the
  unit-aware `RDSON_Y_PLAUSIBLE_RANGES` gate in `classical.py` (y bound
  depends on detected units, so it lives in the wrapper, both paths).
- The new gate caught a genuinely unphysical TEST FIXTURE: the two-curve
  chart drew its lower curve below the 0 mΩ axis (negative resistance).
  Fixture geometry corrected (+60→+35 px offset), assertions untouched.
- Suite **709 passing** (was 700). Real 11-device corpus re-run: identical
  9 ok / 2 needs_review, zero false positives, no new catches. mOhm/Ohm
  ranges flagged corpus-untested (no such charts on disk yet).

### 2026-07-14 — Session: T27 follow-up, max-column-thickness safety gate
- Owner approved the merged-parallel-line fix flagged at the end of the T27
  session (option (a): a max-column-thickness plausibility gate).
- Measured MEDIAN column thickness across the real 11-device corpus to pick a
  non-guessed threshold: clean cluster 12–16px, the 2 known-bad devices
  21–22px — set the gate at 18px.
- TDD: 3 tests written first (red — merged-streak fixture stayed "ok"),
  synthetic fixture tuned empirically to reproduce the diagnosed mechanism
  (streak 10px above the curve → measured 20.0px, vs 11.0px clean control),
  then wired the gate into `run_classical_pipeline` (mono-only, "ok"-only,
  keeps curves/calibration/units for the reviewer). Suite **700 passing**
  (was 697), no regressions.
- Re-ran the exact real 11-device corpus per the owner's acceptance test:
  both known-bad devices now correctly `needs_review`; all 9 clean devices
  still `ok`. Zero false positives, zero silently-wrong results remaining
  on this sample.

### 2026-07-13 — id_vs_vgs Run 3 + 2px experiment + T26 Rohm check (logged 2026-07-14)
- Executed on the GPU box 2026-07-13; numbers supplied by the owner 2026-07-14 and
  recorded in the new M8 section (Run 3 details, 2px abandonment decision, T26 result).
- **Run 3 (combined 299/70/30) is the id_vs_vgs production checkpoint**: mAP@50 0.7414,
  recall 0.8889, `best_segm_mAP_50_iter_4800.pth`.
- 2px buffer experiment abandoned (training stuck at 0.0 score; owner reverted to
  4.5 px) — revisit in the diversity pass after all 7 curve types are built.
- T26: 10/12 Rohm figures spot-checked correct — first cross-manufacturer evidence.

### 2026-07-13 — id_vs_vgs batch 2: three owner-approved fixes + split re-run
- Owner approved all three flags from the first pass; split rebuilt from the
  batch-1-only backup with the fixes applied:
- **(1) `TJ_-55` → `TJ_-55C`** renamed at the source (exact-match, XML-parsed,
  46 shapes — verified nothing else touched) in
  `data/staging/id_vs_vgs_batch2_final.xml`; `id_vs_vgs_batch2.json` and
  `id_vs_vgs_combined.json` regenerated (199/420 and 399/744, both clean).
  Vocabulary is now uniform: TJ_25C / TJ_150C / TJ_175C / TJ_-40C / TJ_-55C.
- **(2) `FAMILY_MERGE_MAP` extended (TDD red→green):** IAUCN/IAUTN/IAUZ → IAU
  (one Infineon automotive template) in `src/dataset_tools/split_dataset.py`;
  tests updated with real batch-2 device names, red confirmed (2 failures)
  before the 3-line map change. Suite **674 passing**.
- **(3) Weak-val fix — owner-approved manual allocation** (same precedent as
  the batch-1 capacitance ad-hoc split, commit `2c33a17`): the greedy
  smallest-first tool would have routed owner-pinned BSC-BSZ to val, so
  allocation was set explicitly — val = IMZA (1/2) + BUZ (2/0) + IPAN (4/8)
  + **annotation-rich IPA (34 img / 68 ann, 0 empty)**; train = BSC-BSZ, IPD,
  IAU (merged, 43 img / 129 ann), IPB. Decision + rationale recorded in
  `split_manifest.json` (`allocation_mode: manual`).
- **Final split: train 299 img / 598 ann (22 empty), val 70 / 92 (24 empty,
  34% — was 59%), test UNCHANGED 30 / 54 — sha256 `6e8e96d1b4ce…` verified
  identical.** All three validate clean, pairwise disjoint, total exactly 399.
- Note for training review: all 46 `TJ_-55C` annotations sit in train (they
  belong to the IAU family); val/test contain none, so per-temperature eval
  on -55 °C curves isn't possible with this split.
- **STOPPED before retraining — combined dataset ready for owner review / Run scoping.**

### 2026-07-12 — id_vs_vgs training data: batch 1 converted + split
- CVAT job 4225954 (200 images, temperature-labeled TJ_25C/150C/175C/-40C) converted
  via the frozen `cvat_to_coco` tool → `data/coco/id_vs_vgs_batch1.json`:
  **200 images / 324 annotations, validate_coco clean; the 40 zero-shape images are
  intentional (misclassified charts) and kept as empty negatives.**
- Group-aware split (same family tool as capacitance, seed 42) with **owner-approved
  val/test swap** (original test side was 76% empty / 14 annotations):
  `data/coco/split_id_vs_vgs/` → train 141 img / 256 ann (13 empty),
  val 29 / 14 (22 empty), test 30 / 54 (5 empty). Decision recorded in the
  split manifest. Data files git-ignored per policy.
### 2026-07-17 — Session: `zth_multicurve_run1` config recovery + diagnostic overlay
- **Config recovery:** `src/training/configs/lineformer_zth_multicurve_run1.py` was
  created on the owner's local machine and used to launch training on the GPU box
  (checkpoints already exist under `/mnt/data/my-datasheet/checkpoints/zth_multicurve_run1/`)
  but the file itself was never committed/pushed. Recreated it directly on the GPU box
  from the established `lineformer_zth_vs_time*.py` pattern (`_base_` inheritance,
  official-pretrained init, fp32, no flip, 8000-iter ceiling, patience_iters=2000),
  overriding only `data.{train,val,test}.ann_file` to
  `data/coco/split_zth_multicurve_batch1/{train,val}.json` (`img_prefix` unchanged —
  same `data/images_zth_vs_time/` folder as the single-curve split). Matches the
  training command already in `~/.bash_history` verbatim.
- **Diagnostic overlay, reusing tested code (`eval_lineformer._pred_to_bool_masks`/
  `mask_iou`, `predict_to_cvat.DEFAULT_SCORE_THR`), not reimplemented** — same
  GREEN=GT/RED=pred convention as the earlier `zth_vs_time` single_pulse diagnostic.
  Ran the `zth_multicurve_run1` best checkpoint (`best_segm_mAP_50_iter_400.pth` —
  training early-stopped at iter 2400, patience 2000 from iter 400) on all 10
  `split_zth_multicurve_batch1/val.json` images (no held-out test split exists for
  this 51-image batch). **Raw metric looks like a failure**: `status.txt` shows
  segm_mAP_50 cratering to ~0.002–0.03 across the run, val mAP never recovering
  after its early iter-400 peak. **Visual overlay tells a different story**: on
  8/10 images the model finds every GT curve family (5–7 per chart — the 0.01–0.5
  duty-cycle family plus single_pulse) as a separate, high-confidence (0.75–0.89)
  prediction, correctly tracking each one individually through the crossing/
  converging regions rather than collapsing them into one blob — kept_preds counts
  are within ±1–2 of n_gt on every image. Best-IoU-vs-GT per curve sits in the
  0.2–0.5 band, below the eval script's 0.5 match threshold, which plausibly
  explains the nearly-zero mAP@50 as a metric artifact (the GT masks in this batch
  look visibly wider/more buffered than the thin, tightly-traced prediction masks,
  so a well-centered thin prediction still scores a low IoU against a fat GT band)
  rather than genuine misses — not confirmed via buffer_px measurement, just the
  visual read; would need an actual buffer-width comparison to be sure. One image
  (`IPA60R230P6XKSA1`, compressed x-axis, curves crammed close together) showed
  visibly noisier/wobblier tracking in the dense diagonal crossing region — the one
  case that looked more genuinely imprecise rather than just under-scored. **Net
  read for the owner: the model is finding the right curves in roughly the right
  places on most images, not failing outright** — but mAP@50 as currently computed
  is not a trustworthy signal for this run and needs the GT-buffer-vs-mask-width
  question resolved before it can be used to judge future multicurve runs.
  Outputs (git-ignored, throwaway script not committed — same convention as
  T11/T13): `data/zth_multicurve_diag_overlay/contact_sheet.html` (10 overlays +
  per-image GT/pred counts, scores, best-IoU breakdown).

### 2026-07-17 — Session: new thermal_impedance batch — upload/organize prep only
- **Goal (owner request):** stage new, un-annotated `zth_vs_time` images from the
  owner's local Windows machine (`D:\LineFormerDataset_v2\categorised\thermal_impedance\200`
  and `\400`) on the GPU box, ahead of a semi-auto inference + CVAT-correction pass
  using the `zth_multicurve_run1` checkpoint. **Upload/organize only, per instruction
  — no training or inference run.**
- **Blocker surfaced, not worked around:** this session runs on the AWS GPU box
  (Linux), which has no mount, SMB/sshfs client, or inbound path into the owner's
  local `D:\` drive (checked: `mount`, `/etc/fstab`, `which sshfs smbclient` — none
  present). The established transfer pattern for this project (seen in this box's
  own shell history) is `scp` initiated **from** the Windows machine **to** this box
  via the `aws-lineformer` host alias — the reverse direction of what a session
  running here can do. Flagged to the owner rather than assumed away.
- **Two naming/scope ambiguities flagged and resolved by owner before any folder
  was finalized:** (1) owner's literal suggested name `zth_multicurve_batch2` echoes
  `split_zth_multicurve_batch1`, but that's a COCO-split output folder under
  `data/coco/`, not a raw-image staging folder — the repo's actual existing
  convention for raw pre-annotation staging is `images_<curve_type>_batchN` (e.g.
  `images_id_vs_vgs_batch2`, already on disk). **Owner chose `images_zth_vs_time_batch2`**,
  matching the existing convention. (2) duplicate-filename check scope — batch1's
  51 images only, vs. the full 315-image `data/images_zth_vs_time/` working pool
  batch1 was drawn from. **Owner chose the full 315-image pool** (more thorough;
  catches collisions against the other 264 images too, not just batch1's subset).
- **Prepared, nothing copied yet:** created empty
  `data/images_zth_vs_time_batch2_staging/{200,400}/` (destination for the owner's
  own `scp` push, subfolders kept separate so 200-vs-400 collisions can be checked
  before any merge) and empty `data/images_zth_vs_time_batch2/` (final flat
  destination, matching `images_zth_vs_time/`'s own flat layout). Wrote (not yet
  run — no source files exist here yet) a throwaway dedup/merge script: flags any
  filename collision between `200/` and `400/`, or against the existing 315-image
  pool, and **skips copying that file rather than silently overwriting** (left in
  staging for manual review), per instruction. No existing file touched or modified.
- **Next step (owner action required):** run `scp -r` from the Windows machine for
  both source folders into the two staging subfolders above via the `aws-lineformer`
  alias; ping back so the dedup/merge + count report can run.
- **Owner ran the two `scp -r` commands; dedup/merge executed.** Staged:
  `200/` = 207 files, `400/` = 212 files (419 total), **0 filename collisions
  between `200/` and `400/` themselves**. Against the existing 315-image
  `images_zth_vs_time/` pool: **315/315 duplicates — every existing pool
  filename has a same-named match somewhere in this upload** (144 matched
  inside `200/`, 171 inside `400/`), leaving **104 genuinely new-to-corpus
  filenames**, copied to `data/images_zth_vs_time_batch2/`
  (`/home/ec2-user/my-datasheet/data/images_zth_vs_time_batch2/`, verified 104
  files). **Not the DPI-resolution-variant theory floated earlier in this
  entry** — that would predict the *same* filenames appearing in both `200/`
  and `400/` (one per resolution), which isn't what happened (zero overlap
  between the two folders); more likely `200`/`400` are two non-overlapping
  historical export batches from the owner's local organization, both of
  which happen to mostly re-cover devices already in the working pool.
  Existing `images_zth_vs_time/` pool confirmed untouched (still exactly 315
  files) — no duplicate was copied over an existing file, all were skipped
  and left in place per instruction.
- **Owner decision on the 315 duplicate-named files:** don't delete, don't
  leave in a `_staging` folder either — moved to a permanent, clearly-labeled
  location for a later decision: `data/images_zth_vs_time_dpi_variants/{200,400}/`
  (207 + 212 files, straight rename/move of the staging folders, nothing
  copied/deleted/modified within them). **Status: parked, no action taken on
  content** — a real bucket, not limbo, but still an open decision for a
  future session (are these worth a pixel-level compare against the existing
  pool's versions, or safe to discard).
- **Final state, all under `data/`:**
  `images_zth_vs_time_batch2/` — 104 new-to-corpus images, ready for the
  semi-auto `zth_multicurve_run1` inference + CVAT pass (still not run — out
  of scope for this task per instruction).
  `images_zth_vs_time_dpi_variants/{200,400}/` — 207 + 212 duplicate-named
  images, parked pending a future owner decision.
  `images_zth_vs_time/` (original 315-image pool) — untouched.

### 2026-07-17 — Session: zth_multicurve_run1 curve-COUNT recognition check
- **Goal (owner request):** before deciding retrain vs. annotate-more, check
  whether `zth_multicurve_run1` finds *every curve present* per image (count
  only — mask/IoU quality explicitly out of scope, already covered by the
  earlier overlay diagnostic). **Read-only, no training, no file changes.**
- **Image-set ambiguity flagged and resolved by owner before running
  anything:** owner asked for "~60 fully-annotated images"; the only fully
  multi-curve-annotated zth dataset on this box is
  `data/coco/split_zth_multicurve_batch1/` (train 41 + val 10 = **51** images,
  not 60) — searched for a closer-matching ~60-image set (other coco jsons,
  CVAT exports, staging dirs) and found none. **Owner confirmed: use the
  51-image batch1 set.** Important caveat carried into the result below: this
  is the model's own train+val data, so this measures recognition on data it
  was fit to, not generalization to unseen images.
- **Method:** throwaway script (not committed, same convention as prior
  diagnostics), reusing `eval_lineformer._pred_to_bool_masks` and
  `predict_to_cvat.DEFAULT_SCORE_THR` (0.5) — no reimplementation. Best
  checkpoint (`best_segm_mAP_50_iter_400.pth`), count of kept (score≥0.5)
  predictions vs. count of GT annotations, per image, no IoU/mask comparison
  at all.
- **Result: 21/51 images (41.2%) exact count match** — but the mismatches are
  almost entirely **over**-detection, not misses: **50/51 images (98.0%)
  detected count ≥ GT count**; only **1/51** under-detected (`IPA060N06NXKSA1`,
  GT 7 / detected 6). Diff histogram (detected − GT): −1×1, 0×21, +1×20,
  +2×7, +3×1, +4×1. **Answer to the owner's actual question: yes — the
  current weak model already finds essentially every curve present in nearly
  every image; the low exact-match rate is a duplicate/over-detection
  problem (consistent with the earlier overlay diagnostic's kept_preds
  sometimes running 1–2 over n_gt on the denser 7-curve charts), not a
  recognition gap.** Full per-image table (51 rows: image, split, gt_count,
  detected_count, match) at
  `data/zth_multicurve_recognition_check/recognition_check.csv`
  (git-ignored). No training run, no existing file modified.

### 2026-07-17 — Session: zth_multicurve dedup post-processing (standalone, not wired in)
- **Goal (owner request):** cut the over-detection found by the recognition
  check (27/51 images with +1 to +4 extra boxes) so CVAT semi-auto review has
  fewer spurious boxes to delete. **Post-processing only — no checkpoint/model
  change; standalone script, deliberately NOT wired into `predict_to_cvat.py`
  yet**, per instruction, pending this review.
- **Reused, not reinvented:** `src/extraction/dedup.py::dedup_detections()`
  already exists — built in T15 for the identical "near-duplicate curve mask"
  problem on capacitance charts, tested, greedy keep-highest-score. Per
  CLAUDE.md's zero-duplication rule, called it directly via
  `src/extraction/inference.py` (`load_model`/`run_inference`) rather than
  writing a second dedup implementation. **Threshold note:** the owner
  suggested >0.7 IoU as a starting point; `dedup_detections()`'s existing
  constant is `IOU_DUPLICATE_THRESHOLD=0.5` **plus** an OR'd flat-curve
  heuristic (same vertical band ±6px AND ≥70% x-span overlap, tuned for
  capacitance's plateau-shaped duplicates) — used as-is, unmodified, for the
  first pass rather than silently picking a new number.
- **Result on the same 51-image set (`split_zth_multicurve_batch1`
  train+val) as the recognition check — `dedup_detections()` as-is:**
  exact-count match **21/51 (41.2%) → 34/51 (66.7%)**, 55 duplicate
  detections removed, over-detected images 29→3. **But: 4 previously-exact
  images regressed** (dedup removed a real, distinct curve, not a
  duplicate) — `IAUC100N04S6N028ATMA1` 5→4, `IAUC60N06S5L073ATMA1` 5→4,
  `IPA50R800CEXKSA2` 7→6, `IPA60R120P7XKSA1` 7→6 (all GT counts unchanged).
  Under-detected images overall: 1→14 (net +13, including those 4
  regressions) — **the single previously under-detected image
  (`IPA060N06NXKSA1`, GT7/raw6) was NOT made worse** (0 removed, stayed at
  6), directly answering the owner's specific regression question. Full
  per-image before/after at `data/zth_multicurve_dedup_check/dedup_check.csv`.
- **Root-caused the 4 regressions, not just reported them:** ran a
  diagnostic threshold sweep (`data/zth_multicurve_dedup_check/threshold_sweep.txt`),
  reusing only the already-tested `eval_lineformer.mask_iou` in the same
  greedy-keep-highest-score shape as `dedup_detections` — no new dedup
  module — testing **pure mask-IoU-only** dedup (flat-curve heuristic
  disabled) at IoU 0.5/0.6/0.7/0.8. **IoU-only @0.5 is clearly the best
  performer: 40/51 exact (78.4%), over-detected 3, under-detected 8, only
  2 regressions** — beats both `dedup_detections()`'s default (66.7%, 4
  regressions) and the owner's suggested >0.7 (68.6% exact, 5 under, 11
  over at 0.7). **Conclusion: the flat-curve vertical-band/x-span heuristic
  — not the mask-IoU check — is the source of the extra regressions on
  zth_multicurve's near-parallel duty-cycle curve families**; it was tuned
  for capacitance's flat-plateau near-duplicates and doesn't transfer. 2
  residual regressions persist at every threshold tested
  (`IPA50R800CEXKSA2`, `IPA60R120P7XKSA1`) — real IoU≥0.5 overlap between
  genuinely distinct curves (likely the converged high-t_p region seen in
  the earlier overlay diagnostic), not fixable by threshold tuning alone.
- **Not done, needs owner sign-off before proceeding:** `dedup_detections()`
  is shared, already-used-in-production code (the approved
  `capacitance_vs_vds` Stage-5 pipeline calls it) — CLAUDE.md §4 pipeline-
  integrity rule means it was **not modified** to add a
  disable-flat-curve-heuristic option, even though the sweep shows that
  would clearly help zth_multicurve. Recommendation for the owner: add an
  opt-in parameter to `dedup_detections()` (default preserves current
  capacitance behavior) rather than fork a second dedup module, once
  approved. **Not wired into `predict_to_cvat.py` or any live path** — both
  scripts here are standalone/throwaway, not committed, no existing file
  touched.

### 2026-07-17 — Session: dedup_detections() opt-in parameter + standalone review tool
- **Owner approved** the recommendation above: add an opt-in parameter to
  `dedup_detections()` to disable the flat-curve heuristic (default
  unchanged, `capacitance_vs_vds` unaffected), then wire the IoU-only@0.5
  version into a standalone dedup step — **still explicitly NOT into
  `predict_to_cvat.py`**, one more review first.
- **`src/extraction/dedup.py` — TDD, red→green.** Added
  `use_flat_curve_heuristic: bool = True` to both `_is_duplicate()` and
  `dedup_detections()`; `False` skips the same-vertical-band/x-span OR-branch
  entirely, comparing on mask IoU alone. **5 new tests written first**
  (confirmed red — `TypeError: unexpected keyword argument` against the old
  signature — then green): default-matches-explicit-True, heuristic-off
  keeps genuinely-distinct near-parallel curves apart, heuristic-off still
  dedupes high-IoU/exact duplicates (2 cases), and a 3-detection case mixing
  a true duplicate with a near-parallel distinct curve. All 9 pre-existing
  tests still pass unmodified — default behavior, and therefore the
  approved `capacitance_vs_vds` Stage-5 pipeline, is unaffected.
- **New standalone tool, `src/training/dedup_review.py`** (+
  `tests/test_dedup_review.py`, 11 tests for the pure summarization logic —
  the inference-driving path reuses `src.extraction.inference`/`dedup`
  directly, not retested). CLI mirrors `predict_to_cvat.py`'s/
  `eval_lineformer.py`'s conventions; defaults to
  `use_flat_curve_heuristic=False` (the investigation's finding), writes a
  before/after CSV, and prints exact-match %, regressions, and any
  under-detected-image-made-worse cases explicitly. **Deliberately not
  called by `predict_to_cvat.py`** — a separate CLI, reviewed here, wiring
  is a future step.
- **Full suite: 630 passing** (614 at last count in T23 → +5 for the
  `dedup.py` parameter → 619 → +11 for `dedup_review.py`'s pure-logic tests
  → 630; confirmed by running `pytest tests/ -q --ignore=third_party` after
  each addition, not just at the end). **Environment note:** this GPU box's
  `.venv` (the documented "pipeline venv") was
  missing `pytest`/`shapely`/`pycocotools`/`scikit-image` entirely and its
  pip's available package index doesn't reach several of `requirements.txt`'s
  exact pins (e.g. `numpy==2.4.6`, `pytest==9.1.1` aren't resolvable from
  this box's index) — worked around by using the system `/opt/pytorch`
  Python (already had a matching `pytest==9.1.1`) and `pip install`ing the
  four missing packages into it to get a real full-suite signal; the
  project's own `.venv` was left untouched. Flagged for the owner: this
  box's pipeline venv isn't actually usable for the full suite as set up —
  worth a real fix later, out of scope for this task.
- **Ran the new tool for real** on the same 51-image
  `split_zth_multicurve_batch1` set (train+val), confirming the earlier
  sweep's numbers exactly: **exact match 21/51 → 40/51 (78.4%)**, 45
  duplicates removed, **2 regressions** (`IPA50R800CEXKSA2`,
  `IPA60R120P7XKSA1` — same 2 residual cases identified earlier, not fixed,
  real IoU≥0.5 overlap between distinct curves), **0 under-detected images
  made worse**. Report: `data/zth_multicurve_dedup_check/dedup_review_iou_only.csv`.
- **Still not wired into `predict_to_cvat.py`** — awaiting the owner's next
  review before that happens, per instruction.

### 2026-07-17 — Session: dedup wired into predict_to_cvat.py (zth_vs_time only) + .venv fixed
- **Owner approved both**: (1) wire the reviewed IoU-only dedup into
  `predict_to_cvat.py`, gated strictly to `zth_vs_time` — every other curve
  type (`capacitance_vs_vds`, `rdson_vs_tj`, `if_vs_vsd`, `id_vs_vgs`,
  `vgs_vs_qg`, `vgsth_vs_tj`) must be byte-for-byte unaffected; (2) fix this
  GPU box's broken `.venv` so the full suite runs standalone, no
  `/opt/pytorch` workaround needed.

**Part 1 — dedup wired in, zth_vs_time only.**
- **Files changed:** `src/training/predict_to_cvat.py` (modified — added
  `apply_curve_type_dedup()`, `ZTH_VS_TIME_CURVE_TYPE` constant, a
  `curve_type` parameter on `run_predict_to_cvat()`, and a `--curve-type`
  CLI flag; the pre-existing `filter_by_score(...)` call line and the entire
  polygon-extraction loop body are untouched — only a new line immediately
  after replaces `kept` conditionally); `tests/test_predict_to_cvat.py`
  (modified — 12 new tests). No other file touched for Part 1.
- **Design:** `apply_curve_type_dedup(kept, curve_type)` is a single gate —
  `curve_type != "zth_vs_time"` (covers the default `None`, i.e. every
  pre-existing call site, plus every other curve type explicitly) returns
  `kept` as an unmutated copy, 0 removed; only `"zth_vs_time"` runs
  `dedup_detections(..., use_flat_curve_heuristic=False)` (reused from
  `src.extraction.dedup`, not reimplemented). Wired in as one added line
  (`kept, n_deduped = apply_curve_type_dedup(kept, curve_type)`) right after
  the existing `filter_by_score` line; everything downstream (polygon
  extraction, XML building) is identical code for every curve type. Also
  updated the CLI's printed "KNOWN LIMITATION" message to branch on
  `curve_type` (zth_vs_time now correctly says dedup WAS applied; every
  other curve type sees the exact original unmodified sentence).
- **Real circular-import bug found and fixed during this task** (not
  present before, introduced by the naive first attempt at this wiring):
  `src.extraction.inference` already imports `DEFAULT_SCORE_THR`/
  `filter_by_score` FROM `predict_to_cvat.py`; importing `dedup_detections`/
  `Detection` back at `predict_to_cvat.py`'s module level created an import
  cycle. Fixed by moving those two imports inside
  `apply_curve_type_dedup()` (function-local, same lazy-import idiom already
  used elsewhere in this file for GPU-only deps) — documented inline so a
  future editor doesn't move it back to module level and reintroduce the
  cycle.
- **TDD, red→green:** 12 new tests in `tests/test_predict_to_cvat.py`
  (`TestApplyCurveTypeDedup`) — confirmed red first (`ImportError` for the
  not-yet-existing `ZTH_VS_TIME_CURVE_TYPE`/`apply_curve_type_dedup`), then
  green. Explicitly parametrized over **all 6 other real curve_type strings
  plus `None`** (not just capacitance) to directly satisfy the hard rule:
  each must return `kept` byte-for-byte unchanged, 0 removed. Separate tests
  confirm zth_vs_time dedupes high-IoU duplicates, does NOT apply the flat-
  band heuristic (the specific case that would wrongly merge two genuinely
  distinct near-parallel curves), is a no-op when there's nothing to dedup,
  handles an empty list, and never mutates its input for a non-zth
  curve_type. All pre-existing tests in the file (mask_to_polygon,
  filter_by_score, build_cvat_xml — the actual capacitance-path logic)
  pass **unmodified**.
- **Real GPU-box integration smoke test** (not just unit tests): ran the
  actual CLI end-to-end. `--curve-type` omitted, real Run A capacitance
  checkpoint, 3 images → 9 polygons (exactly 3/image), original "not
  filtered here" message printed verbatim — **identical to pre-task
  behavior**. `--curve-type zth_vs_time`, `zth_multicurve_run1` checkpoint,
  5 never-annotated `images_zth_vs_time_batch2` images → dedup fired on
  every image (25/57 raw detections removed across the 5), new accurate
  message printed. Both runs written to `/tmp/` scratch paths, nothing in
  the repo touched by the smoke test itself.
- **Full suite: 642 passing** (630 → 642, +12 for this task).

**Part 2 — `.venv` fixed.**
- **Root cause (not a broken index, an interpreter mismatch):** `SETUP.md`
  (line 16) requires **Python 3.10+** for the pipeline venv; the `.venv` on
  this box was actually built with **Python 3.9.25**. `requirements.txt`'s
  newer pins (`numpy==2.4.6`, `pytest==9.1.1`) only ship wheels for 3.10+,
  so pip's refusal to install them was 3.9 correctly rejecting incompatible
  wheels, not a stale/misconfigured index — confirmed by checking `pip
  config list` (empty, default PyPI, on both `.venv` and the `/opt/pytorch`
  workaround env) and by successfully installing every one of those exact
  pins under this box's other Python 3.13 install. How `.venv` ended up on
  3.9 in the first place isn't recoverable from what's on disk — no record
  of which command created it — but it was never compliant with SETUP.md's
  own documented requirement.
- **Fix:** deleted `.venv` (confirmed git-ignored, untracked, nothing lost —
  owner approved the delete/recreate explicitly first) and recreated it
  with `/usr/bin/python3.13` (`python3.10`/`3.11`/`3.12` aren't installed on
  this box; 3.13 satisfies "3.10+" and is already proven compatible via the
  `/opt/pytorch` workaround). `pip install -r requirements.txt` then
  resolved and installed **every exact pin with no substitutions**
  (`numpy==2.4.6`, `shapely==2.1.2`, `pytest==9.1.1`, `opencv-python==5.0.0.93`
  — the real pinned package, not the `-headless` substitute the earlier
  workaround used since only that was needed for the workaround —
  `scikit-image==0.26.0`, `pycocotools==2.0.11`).
- **Confirmed: `.venv/bin/python -m pytest tests/ -q --ignore=third_party`
  → 642 passed**, run standalone via `.venv` directly, no `/opt/pytorch`
  fallback needed anymore.
- **Files/state changed:** `.venv/` only (deleted + recreated; git-ignored,
  not a tracked file). **No `requirements.txt` or other dependency-file
  edit was needed** — the pins were already correct, only the interpreter
  building the venv was wrong. No pipeline source code touched for Part 2.

### 2026-07-18 — Session: CVAT pre-annotations for `images_zth_vs_time_batch2` (104 images)
- **Goal (owner request):** generate CVAT pre-annotations for the 104
  never-annotated `zth_vs_time` images staged in the prior session
  (`data/images_zth_vs_time_batch2/`), using the `zth_multicurve_run1`
  checkpoint with the curve-type-gated dedup wired into `predict_to_cvat.py`
  in the immediately preceding session. Generation only — no CVAT upload
  (manual, owner-side).
- **Ran (GPU box, `lineformer` conda env):**
  `python -m src.training.predict_to_cvat --checkpoint
  /mnt/data/my-datasheet/checkpoints/zth_multicurve_run1/best_segm_mAP_50_iter_400.pth
  --config src/training/configs/lineformer_zth_multicurve_run1.py
  --images-dir data/images_zth_vs_time_batch2/ --out
  data/zth_vs_time_batch2_preannotations.xml --curve-type zth_vs_time
  --device cuda:0`.
- **Result: 104/104 images processed, 580 polygons written after dedup**,
  format "CVAT for images 1.1" (not COCO — COCO would drop polylines/lose
  the per-curve polygon shape). Verified programmatically (not eyeballed):
  the XML's 104 `<image name=...>` entries are an exact set-match against
  the 104 filenames on disk in `images_zth_vs_time_batch2/` — 0 missing,
  0 extra. **0 images with 0 detections** — none need to be routed to
  from-scratch manual annotation instead of correction. Per-image polygon
  count distribution: 1×6, 3×9, 4×8, 5×11, 6×38, 7×25, 8×6, 9×1 — the 6
  images with only 1 polygon (`BSS225H6327FTSA1__fig_p4_007`,
  `BSS225H6327XTSA1__fig_p4_007`, `IAUA250N04S6N008AUMA1__fig_p6_011`,
  `IPB093N04LG__fig_p4_009`, `IPB200N25N3GATMA1__fig_p4_011`,
  `IPD60R1K4C6ATMA1__fig_p8_012`) are flagged for extra attention during
  correction — plausible for a single-curve chart but also consistent with
  under-detection on a multi-curve one; not distinguishable without eyeballing
  each image against its source figure, not done here.
  Every polygon's `curve_name` attribute is still the literal placeholder
  `TODO` (unchanged tool behavior) — annotators must fill in the real name
  before this ever round-trips through `cvat_to_coco.py`.
- **Output:** `data/zth_vs_time_batch2_preannotations.xml` (new file).
- **Scope respected:** read-only on `images_zth_vs_time/` (confirmed still
  315 files, untouched), `images_zth_vs_time_dpi_variants/`, and
  `data/coco/split_zth_multicurve_batch1/` — none written to. No existing
  file modified or deleted. No CVAT upload performed (manual, owner-side,
  per instruction).

### 2026-07-18 — Session: `body_diode_annotations.xml` inspection (if_vs_vsd, not started)
- **Goal (owner request):** count-only inspection of a newly-uploaded CVAT
  export for `if_vs_vsd` (body_diode) — training for this curve_type hasn't
  started (still "not started" in the §1 scope table); this is pre-work
  reconnaissance, no processing. **Read-only, nothing converted/moved/trained.**
- **File:** `data/body_diode_annotations.xml` (159,839 bytes, CVAT-for-images
  1.1, parsed with `xml.etree.ElementTree`).
- **Result:**
  - **200** `<image>` entries.
  - **427** curve annotations total, **all `<polyline>`** (0 polygons, 0
    boxes/points/ellipses — single shape type throughout).
  - **Average 2.135 curves/image** (427/200).
  - **8 images with 0 annotations** (need re-export or manual annotation
    before use): `BSB012N03LX3G__fig_p6_013_sub1.png`,
    `BSC009N04LSSCATMA1__fig_p8_020_left.png`,
    `BSC009NE2LSATMA1__fig_p8_021_left.png`,
    `BSC010N04LS6ATMA1__fig_p8_021_left.png`,
    `BSC011N03LSIATMA1__fig_p8_021_left.png`,
    `BSC034N03LSGATMA1__fig_p8_021_left.png`,
    `BSR802NL6327HTSA1__fig_p6_014.png`, `BSZ065N06LS5ATMA1__fig_p8_022.png`.
  - Per-image count distribution: 0×8, 2×169, 3×3, 4×20.
  - **Label names used are literal temperature values, not a generic
    `line`/`curve` label:** `25 degree C` (192), `150 degree C` (109),
    `175 degree C` (83), `-55 degree C` (20), `125 Degree C` (18, note the
    capitalization mismatch vs. the other five — same value, inconsistent
    case, would collide if labels are lowercased somewhere downstream),
    `-40 degree C` (3), `100 degree C` (2). **This differs from the CVAT
    project convention recorded in the M1 table** (`label "line", polyline +
    attribute "curve_name"`, used by every other in-scope curve_type's CVAT
    exports/`predict_to_cvat.py` output) — here the temperature IS the
    label and there are no `<attribute>` elements on any polyline at all.
    Flagged, not resolved — a downstream converter for this curve_type would
    need to either treat label-as-curve_name directly or expect this project
    to re-map it; not decided here, no code written.
- **Not done (out of scope per instruction):** no conversion to COCO, no
  move/copy of the file, no training, no code changes. File and all other
  repo state left exactly as found.

### 2026-07-18 — Session: `body_diode_annotations.xml` label normalization (new file, original untouched)
- **Goal (owner request):** normalize the prior session's inspected labels
  (literal temperature strings) to the project's standard CVAT convention —
  label `"line"` + a `curve_name` attribute — matching what `capacitance`,
  `id_vs_vgs`, etc. already use (see M1 table row 1, and
  `data/id_vs_vgs_batch2_corrected.xml`'s label block, used as the format
  reference). **New output file only — original never opened for writing.**
- **Pre-check before merging anything (per instruction, not assumed):**
  confirmed programmatically that `'125 degree C'` (lowercase) does **not**
  exist anywhere in the source — only `'125 Degree C'` (capital D, 18
  occurrences across 18 distinct images) is present. So this was a pure
  capitalization normalization of one label, not a real merge of two
  populations that might have carried different meaning — verified, not
  guessed.
- **Normalization convention chosen: `[-]NC`** (e.g. `25C`, `150C`, `-55C`,
  `-40C`) — sign preserved, no decimal point (source values are all whole
  degrees), no unit-name text, no prefix (deliberately not the `TJ_`-prefixed
  style seen in `id_vs_vgs_batch2_corrected.xml`, since that prefix is
  specific to that curve_type's junction-temperature semantics and the owner's
  own example (`"25C"`) in this task's instructions matches the no-prefix form).
  Regex: `^(-?\d+)\s+[Dd]egree\s+C$` → `f"{N}C"`, applied to all 7 original
  labels with **zero non-matches** (verified — every label fit the pattern,
  nothing silently dropped).
- **Script:** throwaway `normalize_body_diode.py` (not committed, same
  convention as prior one-off diagnostic scripts), reusing no pipeline code
  (pure stdlib `xml.etree.ElementTree`) — the transform itself is
  intentionally trivial (rename `label` attr, add one `<attribute>` child)
  so a dedicated module wasn't warranted for a one-time file conversion.
  Also rewrote the stale `<meta><job><labels>` block (which listed the old
  seven per-temperature label definitions) to describe the new single
  `"line"` label + `curve_name` select-attribute, in the same shape as
  `id_vs_vgs_batch2_corrected.xml`'s meta block — purely descriptive
  metadata, not consumed by `parse_cvat_xml` (confirmed: it only reads
  `<image>`/`<polyline>`/`<attribute>`, never `<meta>`), so this doesn't
  affect downstream parsing either way but avoids leaving a misleading
  artifact. Added one XML comment at the top of the output documenting the
  transform and the untouched-original guarantee.
- **Result — verified programmatically (not eyeballed), twice:**
  - **Total polylines: 427 in, 427 out** — exact match, nothing dropped or
    duplicated.
  - **200/200 `<image>` entries preserved** (name/width/height untouched);
    the **8 zero-annotation images carried through as empty `<image>`
    elements**, no fabricated annotations, no drop.
  - Per-image `points`, `occluded`, `source`, `z_order` on every polyline
    byte-identical to the source (diffed element-by-element between old and
    new files) — **only** `label` and the added `curve_name` attribute
    changed.
  - **Final `curve_name` counts:** `25C` 192, `150C` 109, `175C` 83,
    `-55C` 20, `125C` 18, `-40C` 3, `100C` 2 (sums to 427). All polylines'
    `label` is now the single string `"line"` (confirmed: `{'line'}` is the
    only value across all 427).
  - **Round-tripped through the project's real, unmodified
    `src/dataset_tools/cvat_to_coco.parse_cvat_xml`** (not a hand-rolled
    check): 200 images, 427 shapes kept, 0 unsupported shapes skipped, 0
    errors — including its hard `ValueError` gate on missing/empty
    `curve_name`, which every shape passed. `curve_name` distribution read
    back through the real parser matches the write-side counts exactly.
- **Files:** `data/body_diode_annotations_normalized.xml` (new, 174,983
  bytes). **`data/body_diode_annotations.xml` confirmed untouched**
  (sha256 `d413e5be25f371823f64c19d431ec6572ef1c53777f8514e624aadb6e5ac558c`,
  same size/mtime as upload). No other file touched. No COCO conversion,
  no training — normalization only, per instruction.

### 2026-07-18 — Session: `body_diode` (if_vs_vsd) → COCO + group-aware train/val split
- **Goal (owner request):** convert `body_diode_annotations_normalized.xml`
  to COCO, exclude the 8 zero-annotation images, build a family-based
  train/val split (~85/15, ~170/~22) mirroring the
  `split_zth_multicurve_batch1/` folder convention (train.json + val.json,
  no test.json — this is a fresh training batch, not the frozen T5 eval
  set). **Reuse-only, per instruction** — no new conversion/splitting
  algorithm written.
- **Conversion — 100% reused, unmodified:**
  `src.dataset_tools.cvat_to_coco.merge_convert(["data/body_diode_annotations_normalized.xml"], "data/coco/body_diode_batch1.json")`.
  Chose `merge_convert` over the single-file `convert()` specifically
  because `merge_convert`'s existing, already-tested behavior drops any
  image with zero shapes — this **is** the "exclude the 8 zero-annotation
  images" requirement, satisfied by an existing code path rather than new
  filtering logic. Result: **192 images, 427 annotations** kept, 8 dropped
  (log line confirms exactly 8) — buffered @ the project default 4.5 px,
  `validate_coco` clean.
- **Split — reused the existing family-aware allocator, not the frozen
  3-way `group_split`:** `group_split`/`propose_split` (the canonical T5
  method) hardcodes `SIDES = ("train","val","test")` and can't produce a
  2-way split without editing that shared, frozen function — not done, per
  CLAUDE.md §4. Instead reused
  `src.dataset_tools.split_dataset.allocate_new_batch` (T9's existing
  train/val-only family allocator — originally written to route a *new
  batch* into an *existing* split, but its allocation logic doesn't touch
  or require one, so it applies directly to a fresh dataset) together with
  the same `assign_family`/`extract_device` family-key heuristic (+
  `FAMILY_MERGE_MAP`) the frozen split uses. `val_image_target=22` lands
  exactly on a family boundary: **170 train / 22 val**, matching the
  owner's stated target precisely. **16 families total** — the dominant
  `BSC-BSZ` family (117/192 images, same dominant Infineon template family
  pinned to train in the frozen T5 split) fell into train automatically
  (smallest-families-first-to-val ordering), no explicit pin needed.
  **Train: 5 families** (`AUIRF`, `BSC-BSZ`, `BSP`, `IAU`, `IAUCN`).
  **Val: 11 families** (`94-3316`, `AUIRL`, `BSB`, `BSF`, `BSR`, `BSS`,
  `BTS`, `BUZ`, `IAUTN`, `IAUZ`, `IMZA`). **Zero family overlap** — verified
  independently, twice (once inside the build script, once in a fresh
  from-disk re-parse of the written `train.json`/`val.json` using a
  hand-checked re-implementation of the family heuristic as a cross-check).
- **Result — verified independently from the written files, not just the
  in-process build:**
  - **train.json: 170 images, 383 annotations.**
  - **val.json: 22 images, 44 annotations.**
  - **Totals: 192 images / 427 annotations combined** — matches the
    expected effective set exactly.
  - **0 of the 8 excluded zero-annotation filenames appear in either
    output file** (set-intersection check against the known excluded-name
    list, empty).
  - **0 images in either output have zero annotations** (every image id in
    `train.json`/`val.json` has at least one matching annotation) —
    confirms `merge_convert`'s drop behavior worked as expected, not just
    trusted.
  - Each part passed `validate_coco` (same gate `write_split()` itself
    uses) before being written.
- **Files (all new, nothing pre-existing touched):**
  `data/coco/body_diode_batch1.json` (192/427, intermediate merged COCO,
  naming mirrors `data/coco/zth_multicurve_batch1.json`'s convention),
  `data/coco/split_body_diode_batch1/{train.json,val.json,split_manifest.json}`.
  Manifest records method, `val_image_target`, `families_by_side`, counts,
  achieved ratios, excluded-filename list, source XML/COCO sha256s, and
  `buffer_px`.
- **Confirmed untouched:** `data/body_diode_annotations.xml` and
  `data/body_diode_annotations_normalized.xml` (sha256 unchanged from the
  prior session), and every other curve type's existing `data/coco/split*`
  directory (`split`, `split_id_vs_vgs*`, `split_zth_vs_time*`,
  `split_zth_multicurve_batch1`, `split_a2_456`) — file mtimes inside those
  directories confirmed unchanged (only the shared parent `data/coco/`
  directory's own mtime moved, from the new sibling files being created,
  which is expected and not a modification of any existing file).
- **Script:** throwaway `build_body_diode_split.py` (not committed, same
  one-off convention as the earlier normalization script) — the only
  original code in it is ~15 lines of glue (family-dict construction from
  COCO output, writing the two JSON parts + manifest); every actual
  conversion/splitting decision is delegated to the reused functions named
  above.
- **Not done (out of scope / needs a decision, not assumed):** no
  `test.json` (matches the requested folder convention, but means this
  batch has no held-out test split of its own yet — flagging in case that
  matters before training); no training run started; `if_vs_vsd` naming
  module (`src/extraction/naming/`) for stage-5 curve identification by
  temperature doesn't exist yet either — out of scope for this task.

### 2026-07-18 — Session: body_diode (if_vs_vsd) config + first training run launched
- **Goal (owner request):** create a training config for `body_diode` and
  launch a run, following the `zth_multicurve_run1` pattern.
- **Config:** `src/training/configs/lineformer_body_diode_run1.py` — a full
  standalone config based on `third_party/lineformer/lineformer_swin_t_config.py`
  (same relationship `id_vs_vgs.py`/`zth_vs_time.py` have to each other,
  **not** chained onto `lineformer_zth_vs_time.py` the way
  `zth_multicurve_run1.py` is, since body_diode is a genuinely different
  curve_type/dataset, not another batch of the same one). Hyperparameters
  copied from `zth_multicurve_run1`/`zth_vs_time` verbatim (official-
  pretrained init, LR 5e-6, fp32, no flip, multi-scale + brightness/contrast
  jitter, best+latest checkpoint retention): `max_iters=8000`,
  `patience_iters=2000` (owner instruction — use zth_multicurve_run1's
  settings as the starting point; its 41-train-image dataset is the closest
  precedent for a small batch, and 170 train images here is larger, so
  patience=2000 is a generous, not tight, starting point). `classes=("line",)`
  — no class-count change needed, confirmed the converted COCO has exactly
  one category (`"line"`), same as every other curve_type. Data:
  `data/coco/split_body_diode_batch1/{train,val}.json`,
  `img_prefix=data/images_body_diode/`. Config load-tested via
  `mmcv.Config.fromfile` before launch (parses clean, `load_from` checkpoint
  path verified to exist).
- **Blocker hit and resolved before launch, not worked around:** none of
  the 192 image filenames referenced by `split_body_diode_batch1/{train,val}.json`
  existed anywhere on this box — checked `data/images`, `images_id_vs_vgs*`,
  `images_zth_vs_time*`, and a full `/mnt/data` search, 0 matches everywhere
  (these are different source-figure pages than capacitance/id_vs_vgs for
  the same devices, e.g. `fig_p4_014` vs `fig_p4_012` — never staged on this
  box). **Stopped and asked rather than guessing or launching a doomed
  run.** Created the empty staging dir `data/images_body_diode/` (same
  precedent as the earlier `images_zth_vs_time_batch2` staging) and the
  checkpoint dir, but did not start training. **Owner uploaded the images
  to `data/images_body_diode/`** — verified before launch, not assumed:
  200 files on disk, all 192 required filenames present (0 missing), the 8
  extra files matched exactly the known 8 zero-annotation exclusions (set
  comparison), and all 192 required images individually integrity-checked
  (non-zero size, opens cleanly, pixel dimensions match the COCO record) —
  0 bad files.
- **Checkpoint collision check:** `/mnt/data/my-datasheet/checkpoints/`
  listing confirmed no existing `body_diode_run1` directory before creating
  it — no risk of overwriting `run_a`, `id_vs_vgs_run3_combined_8000iter`,
  `zth_multicurve_run1`, or any `zth_vs_time_run*` checkpoint.
- **Launched:** `python -m src.training.train_lineformer --config
  src/training/configs/lineformer_body_diode_run1.py --work-dir
  /mnt/data/my-datasheet/checkpoints/body_diode_run1 --seed 42` (GPU box,
  `lineformer` conda env, background process, PID 7819).
- **Confirmed working end-to-end, not just "started":** watched the log
  through the first full checkpoint+eval cycle. Training loss decreasing
  normally through iter 200 (no NaN/divergence). `Iter [200/8000]` →
  checkpoint saved (`iter_200.pth` + `best_segm_mAP_50_iter_200.pth`,
  569 MB each, `latest.pth` symlinked) → val eval ran cleanly on all 22 val
  images → **segm_mAP_50 = 0.1680 at iter 200** (expected-noisy this early,
  not a quality judgment — first eval point only) → training resumed to
  iter 220 automatically. GPU utilization confirmed active (~55%,
  ~2.4 GB/15.4 GB) during training.
- **Timing:** steady-state **~0.35 s/iteration** (first iter 0.499s is
  warmup-inflated; iters 20–180 averaged 0.35s before the first eval
  pause). With `samples_per_gpu=1` and 170 train images, **1 epoch = 170
  iterations ≈ 60 s (~1 minute/epoch)**. Full 8000-iter ceiling ≈
  **47–52 minutes** if it runs to completion without early stopping (the
  in-training ETA field read 0:47–1:06 across the observed iters, settling
  in that band after warmup). Early stopping (patience 2000) may cut this
  shorter, same as `zth_multicurve_run1`'s actual run (early-stopped at
  iter 2400).
- **Files:** `src/training/configs/lineformer_body_diode_run1.py` (new),
  `data/images_body_diode/` (new, 200 images, owner-uploaded), checkpoint
  output `/mnt/data/my-datasheet/checkpoints/body_diode_run1/` (new dir).
  No other curve type's config, checkpoint, or data touched.
- **⚠ REMINDER FLAGGED — do not forget:** per standing instruction,
  checkpoints must eventually be backed up to **3 places (AWS, local,
  Google Drive)**. Not done yet — training was just launched. **Flag again
  when this run completes** (best checkpoint at
  `/mnt/data/my-datasheet/checkpoints/body_diode_run1/best_segm_mAP_50_iter_*.pth`)
  so the backup step isn't skipped.
- **Not done (still open):** run not yet complete as of this entry
  (in progress, background PID 7819) — final metrics/checkpoint TBD next
  session. No eval against a held-out test set (none exists for this batch
  yet, noted in the prior session's entry).

### 2026-07-18 — Session: body_diode_run1 — training completed (early stopped)
- **Run finished on its own**, PID 7819 exited cleanly, no crash. **Early
  stopping fired at iter 4199**: no improvement in `segm_mAP_50` for 2000
  iters (patience_iters=2000, same value as `zth_multicurve_run1`), best
  was `0.6298` at iter 2199 — same early-stopping mechanics/threshold as
  every other run in this project, not a new decision.
- **Wall clock: 1635.9 s (≈27.3 minutes)** for 4200 iters — well under the
  ~47–52 min estimate given for the full 8000-iter ceiling, because early
  stopping cut the run short at iter 4200. **Peak GPU memory: 2418.6 MiB**
  (in line with every other LineFormer run on this box, 15.4 GB card).
- **Training curve** (iter: val `segm_mAP_50`, eval every 200 iters):
  200: 0.168 · 400: 0.340 · 600: 0.389 · 800: 0.406 · 1000: 0.401 ·
  1200: 0.435 · 1400: 0.412 · 1600: 0.523 · 1800: 0.534 · 2000: 0.598 ·
  **2200: 0.630 (best)** · 2400: 0.604 · 2600: 0.585 · 2800: 0.564 ·
  3000: 0.602 · 3200: 0.593 · 3400: 0.590 · 3600: 0.605 · 3800: 0.584 ·
  4000: 0.627 · 4200: 0.611 (last, patience exhausted here). Train loss
  fell from 38.4 at iter 0 to the 17–24 range by iter 3000+, noisy but
  trending down throughout — no divergence flagged.
  **Read: a real, substantial improvement from the noisy iter-200 starting
  point (0.168) to a stable ~0.58–0.63 band from iter 2000 onward** — the
  best checkpoint (iter 2200, mAP@50 0.630) is a reasonable pick, though
  the metric plateaus/wobbles rather than cleanly converging, similar in
  shape to other small-dataset runs in this project (`zth_multicurve_run1`).
  **Caveat, not resolved here:** this mAP@50 is measured against the 22
  `val` images only — **no held-out `test.json` exists for this batch**
  (flagged in the prior session's entry), so there is no fully-independent
  number yet, same open item as before.
- **Checkpoint retention audited automatically** (`run_result.json`,
  reusing the tested `checkpoint_retention` logic, not eyeballed): present
  files exactly match expected keep-list — `best_segm_mAP_50_iter_2200.pth`,
  `iter_4200.pth`, `latest.pth` (symlink to `iter_4200.pth`) — **0
  unexpected files**.
- **Final checkpoint dir:** `/mnt/data/my-datasheet/checkpoints/body_diode_run1/`
  — `best_segm_mAP_50_iter_2200.pth` (569 MB, the checkpoint to use going
  forward) + `iter_4200.pth`/`latest.pth` (last iter, same size) +
  `run_manifest.json`/`run_result.json`/`status.txt`/`None.log.json` (full
  per-iter train+val log).
- **⚠ BACKUP STILL OUTSTANDING — flagging again per standing instruction:**
  `best_segm_mAP_50_iter_2200.pth` (and/or `iter_4200.pth`) needs to be
  copied to all **3 required places (AWS, local, Google Drive)**. Not done
  in this session — no backup action was requested or taken. Carry this
  forward until confirmed done.
- **Not done / open decisions for the owner:** (1) no test split — is a
  held-out test set needed for this batch before trusting these numbers
  the way `zth_vs_time`/`id_vs_vgs`/Run A's frozen test splits are trusted?
  (2) mAP@50 plateauing in the 0.58–0.63 band rather than climbing further
  — worth a longer/different-LR follow-up run, or is this considered a
  usable first checkpoint? (3) `if_vs_vsd` naming module
  (`src/extraction/naming/`, temperature-based curve identification for
  stage 5) still doesn't exist — needed before this checkpoint can be used
  in the full extraction pipeline. None of these were decided or acted on
  in this session.

### 2026-07-18 — Session: body_diode_run1 — final results report (read-only re-verification)
- **Goal (owner request):** report final training results for
  `body_diode_run1`, re-verified fresh from source files (log JSON,
  `status.txt`, `run_result.json`, checkpoint dir listing), not recalled
  from the prior session's summary. **Read-only — nothing copied, moved,
  or modified.**
- **Final iteration: 4200 of the 8000 ceiling — did NOT reach 8000.**
  Stopped via early-stopping patience (2000 iters, same value used
  throughout this run), confirmed via the exact log line: `EARLY STOPPING
  at iter 4199: no improvement in segm_mAP_50 for 2000 iters (best 0.6298
  at iter 2199)`.
- **segm_mAP_50 start → mid → end:** iter 200 (start) = **0.168** → iter
  2000 (mid) = **0.598** → iter 4200 (end/last) = **0.611**. **Best was
  0.630 at iter 2200** (peak, not the end value) — re-confirmed against
  all 21 logged eval points (every 200 iters, `None.log.json`), matching
  the prior session's record exactly, no discrepancy found.
- **Best checkpoint confirmed by eval score, not recency:**
  `best_segm_mAP_50_iter_2200.pth` (segm_mAP_50 0.630) — higher than the
  final/last checkpoint `iter_4200.pth` (segm_mAP_50 0.611 at that point).
- **Checkpoint files re-verified on disk**
  (`/mnt/data/my-datasheet/checkpoints/body_diode_run1/`):
  `best_segm_mAP_50_iter_2200.pth` — 569,702,329 bytes;
  `iter_4200.pth` — 569,702,329 bytes; `latest.pth` — symlink to
  `iter_4200.pth`. Retention audit in `run_result.json` still shows exactly
  these 3 files present, 0 unexpected — unchanged since the completion
  entry above.
- **No new action taken** — this session only re-read and re-verified
  existing state. Backup-to-3-places reminder from the prior entry still
  stands, still not done.

### 2026-07-18 — Session: body_diode_run1 checkpoint — Google Drive backup done, AWS skipped by owner decision
- **Goal (owner request):** back up `best_segm_mAP_50_iter_2200.pth` to
  AWS, then prepare local + Google Drive backup. Investigated the "check
  how zth_multicurve_run1's best was backed up to AWS, follow the same
  pattern" instruction **before acting** — found the premise didn't hold.
- **AWS backup — investigated, then explicitly SKIPPED (owner decision,
  not silently dropped):** no S3 bucket, backup destination, or backup
  script exists anywhere in this repo, `SETUP.md`, or `PROGRESS.md`,
  and **no prior checkpoint (including `zth_multicurve_run1`'s) was ever
  actually backed up to AWS** — there is no established pattern to follow.
  This box also has **no AWS credentials configured** (`aws sts
  get-caller-identity` → `NoCredentials`) and no attached IAM instance
  role (metadata endpoint empty) — an S3 upload wasn't possible from here
  even with a bucket name. **Flagged to the owner rather than guessing a
  bucket or skipping silently; owner decided:** treat the existing
  `/mnt/data` EBS volume (already durable AWS-resident storage, where
  every checkpoint on this box already lives) as the AWS copy, and do
  local + Drive instead of a separate S3 upload.
- **Google Drive backup — DONE, verified byte-for-byte.** Found an
  already-configured `rclone` remote (`gdrive:`) on this box with a real
  established convention: `Run A`'s best checkpoint is already at
  `gdrive:models/checkpoints/run_a/best_segm_mAP_50_iter_1600.pth` — no
  other run had been backed up there before this. Uploaded to
  **`gdrive:models/checkpoints/body_diode_run1/best_segm_mAP_50_iter_2200.pth`**
  (matching that exact folder-per-run convention; confirmed no pre-existing
  `body_diode_run1` folder there, so nothing was overwritten). **Verified
  after upload, not assumed:** `rclone lsl` shows the Drive copy at
  **569,702,329 bytes**, and `rclone md5sum` on the Drive copy vs. local
  `md5sum` on the original both give **`4e3dc3b46830933a768911b7366263ea`**
  — exact match. **Original checkpoint confirmed untouched**: same mtime
  (`Jul 18 07:54`) and same sha256
  (`0097974dc602c2f2862a21cd453c609e57503f5b8e9539ebb0a3d085eb2810e6`) as
  before the upload.
- **Local backup — path reported, not executed from here.** Same
  established constraint as the earlier image-transfer sessions: this box
  can't push files out to a local machine, only receive via `scp` initiated
  *from* the local machine (the `aws-lineformer` alias pattern already in
  use). **Exact path for Kimo to `scp` from the local machine:**
  `/mnt/data/my-datasheet/checkpoints/body_diode_run1/best_segm_mAP_50_iter_2200.pth`
  (host: `ip-172-31-66-143.ec2.internal`, reachable via the existing
  `aws-lineformer` SSH alias) — 543 MB (569,702,329 bytes), sha256
  `0097974dc602c2f2862a21cd453c609e57503f5b8e9539ebb0a3d085eb2810e6`.
- **Backup status: 2 of 3 places done.** AWS — skipped by owner decision
  (EBS volume counted as the AWS copy). Google Drive — done, verified.
  Local — outstanding, needs Kimo's own `scp` pull using the path above.
- **Files/state changed:** one new file added to Google Drive
  (`gdrive:models/checkpoints/body_diode_run1/...`) via `rclone`. **Nothing
  on this box was deleted, modified, or moved** — original checkpoint
  untouched, no other curve type's checkpoint touched.

### 2026-07-20 — Session: `zth_vs_time_batch2_corrected.xml` — investigation only, no merge
- **Goal (owner request):** figure out which of the 104
  `images_zth_vs_time_batch2` images in the newly-uploaded
  `data/zth_vs_time_batch2_corrected.xml` (CVAT job 4258522, created
  2026-07-18 07:03, updated 2026-07-20 04:03, exported/dumped 2026-07-20
  06:44) were actually manually corrected, before merging anything into
  `split_zth_multicurve_batch1/`. **Read-only — no merge, no conversion, no
  existing file touched**, per instruction.
- **Found a reliable, non-guessed signal — CVAT's own per-shape `source`
  attribute** (`"file"` = imported from the uploaded pre-annotation XML and
  never touched; `"manual"` = created or edited by the annotator in the
  UI), cross-checked against `data/zth_vs_time_batch2_preannotations.xml`
  by diffing actual point geometry, not just trusting the label: every
  `source="file"` shape's points are **byte-identical** to some polygon in
  the original pre-annotation file; every `source="manual"` shape's points
  are **different** from all of them (also frequently a different point
  count, and 26 of them are `<polyline>` — the project's manual-annotation
  shape convention — vs. the pre-annotation's `<polygon>`-only output).
  `curve_name`/attribute-based signal was **not usable**: the CVAT job's
  `line` label has an empty `<attributes>` schema, so the `curve_name=TODO`
  attribute written by `predict_to_cvat.py` was dropped on import (only 1
  stray `<attribute` tag exists in the whole file, inside the label
  definition, not on any shape).
- **Result — 3 categories, verified to sum to 104:**
  - **Category A — genuinely corrected, 8 images** (≥1 `source="manual"`
    shape, geometry-diff-confirmed real edits):
    `AUIRFZ34N__fig_p6_014.png`, `AUIRFZ48Z__fig_p6_013.png`,
    `AUIRLU3114Z-701TRL__fig_p6_014.png`, `AUIRLU3114Z__fig_p5_018.png`,
    `BSB165N15NZ3GXUMA2__fig_p6_012.png`,
    `BSC009N04LSSCATMA1__fig_p6_012.png`,
    `IAUA250N04S6N005AUMA1__fig_p6_012.png`,
    `IAUA250N04S6N007EAUMA1__fig_p6_012.png`.
  - **Category B — untouched raw pre-annotation still present, 5 images**
    (nonzero shapes, all `source="file"`, geometry identical to the
    original): `IAUA120N04S5N014AUMA1__fig_p6_012.png`,
    `IAUA180N04S5N012AUMA1__fig_p6_012.png`,
    `IAUA200N04S5N010AUMA1__fig_p6_012.png`,
    `IPDQ60R040S7AXTMA1__fig_p7_013.png`,
    `IPDQ60R040S7XTMA1__fig_p7_013.png`.
  - **Category C — empty, 91 images** (0 shapes — every original
    pre-annotation polygon deleted; full list in this session's chat log,
    not duplicated here). Ambiguous by construction: CVAT's export carries
    no per-image "visited" marker, so an empty image can't be distinguished
    between "reviewed, correctly found nothing" and "never opened."
- **Discrepancy flagged, not assumed away:** the task context expected
  "~20 images actually corrected"; the reliable signal found only **8**,
  well short of that — surfaced explicitly rather than stretching the
  definition of "corrected" to close the gap (e.g. by counting category B's
  5 untouched images as reviewed, which the geometry diff shows they are
  not).
- **Owner decision, asked and confirmed before doing anything further:**
  only the **8 category-A images** count as genuinely corrected. Categories
  B (5) and C (91) are **excluded from any merge for now**.
- **Not done (explicitly out of scope this session, per instruction):** no
  merge into `split_zth_multicurve_batch1/`, no COCO conversion, no file
  written or modified anywhere — `split_zth_multicurve_batch1/`,
  `zth_vs_time_batch2_preannotations.xml`, and every other existing file
  confirmed untouched. Next step (separate task): convert + merge the 8
  category-A images only.

### 2026-07-20 — Session: merged the 8 corrected zth_vs_time images into a new `split_zth_multicurve_batch2/`
- **Goal (owner request):** extract only the 8 Category-A images from
  `zth_vs_time_batch2_corrected.xml`, convert to COCO, merge with the
  existing 51-image pool, and build a fresh family-based train/val split
  into a **new** folder — `split_zth_multicurve_batch1/` must not be
  touched or overwritten.
- **Blocker found and resolved before converting, not worked around:**
  every shape in `zth_vs_time_batch2_corrected.xml` has **zero
  `<attribute>` elements** — no `curve_name` at all — because the CVAT
  job's `line` label was never given an attribute schema, so the
  `curve_name="TODO"` written by `predict_to_cvat.py` was dropped on
  import. `cvat_to_coco.parse_cvat_xml` hard-requires a non-empty
  `curve_name` on every shape (deliberate, frozen validation — not
  weakened). **Checked the existing pool for precedent before deciding
  anything:** all 303 existing annotations in
  `split_zth_multicurve_batch1/{train,val}.json` already use the literal,
  non-semantic value `"Curve"` (no naming module exists for `zth_vs_time`
  in `src/extraction/naming/`, only `capacitance_vs_vds.py` does) — so
  injecting `curve_name="Curve"` on the 8 new images' shapes reproduces an
  established precedent already used for every annotation in the pool
  being merged into, not a new invented convention. Not asked as a
  separate question since the precedent was unambiguous.
- **Extraction:** `data/zth_vs_time_batch2_corrected_category_a.xml` (new
  file, throwaway `extract_category_a.py` script, not committed) — exactly
  the 8 named `<image>` elements copied verbatim (all shapes regardless of
  `source="manual"`/`source="file"`, since a reviewed image legitimately
  keeps already-correct predictions untouched — e.g.
  `BSC009N04LSSCATMA1__fig_p6_012.png` has 1 manual + 6 file shapes, all 7
  kept) with `curve_name="Curve"` injected per shape. **52 shapes across 8
  images**, matching the prior session's count exactly.
- **Conversion — 100% reused, unmodified:**
  `cvat_to_coco.merge_convert(["data/zth_vs_time_batch2_corrected_category_a.xml"], "data/coco/zth_multicurve_batch2_new8.json")`
  → 8 images, 52 annotations, `validate_coco` clean.
- **Merge — reused `split_dataset._combine_and_renumber` (existing
  function, not reimplemented), twice:** `data/coco/zth_multicurve_batch1.json`
  (the original unsplit 51-image source) **no longer exists on disk**, so
  the existing pool was reconstituted from `train.json`+`val.json` first
  (confirmed zero id overlap between them beforehand) → 51 images / 303
  annotations, then combined with the new 8-image batch →
  **`data/coco/zth_multicurve_batch2.json`: 59 images, 355 annotations**
  (303 + 52), `validate_coco` clean.
- **Split — reused `assign_family`/`extract_device` + `allocate_new_batch`
  (same approach as the `body_diode` split session), a fresh family-based
  split over the FULL combined 59 images, not an incremental append:**
  10 families total (`IPA` 23, `IAU` 19, `IAUCN` 6, `IAUTN` 2, `IAUZ` 2,
  `AUIRF` 2, `AUIRL` 2, `IMZA` 1, `BSB` 1, `BSC-BSZ` 1).
  `val_image_target=11` landed exactly on a family boundary — **train 48 /
  val 11 (81.4%/18.6%)**, close to batch1's original 41/10 (80.4%/19.6%)
  ratio. **Train families (3):** `IAU`, `IAUCN`, `IPA`. **Val families
  (7):** `AUIRF`, `AUIRL`, `BSB`, `BSC-BSZ`, `IAUTN`, `IAUZ`, `IMZA`. Zero
  overlap.
- **Result — verified independently from the written files, twice (once
  in-script, once in a fresh from-disk re-parse):**
  - **train.json: 48 images, 286 annotations. val.json: 11 images, 69
    annotations. Total: 59 images / 355 annotations** — exact match to
    expected.
  - All 8 Category-A filenames confirmed present exactly once across the
    two files; 0 duplicate filenames anywhere in the output.
  - Spot-checked 3 excluded filenames (1 from Category B, 2 from Category
    C) — confirmed **absent** from both output files.
  - 0 family overlap between train and val, re-verified with an
    independent re-implementation of the family heuristic against the
    written files (not just trusted from the build script).
- **Files (all new; nothing pre-existing touched):**
  `data/zth_vs_time_batch2_corrected_category_a.xml`,
  `data/coco/zth_multicurve_batch2_new8.json`,
  `data/coco/zth_multicurve_batch2.json`,
  `data/coco/split_zth_multicurve_batch2/{train.json,val.json,split_manifest.json}`.
  Manifest records method, family assignment, counts, achieved ratios, full
  merge provenance (source file hashes, the curve_name precedent note), and
  `buffer_px`.
- **Confirmed untouched (hash + mtime):** `split_zth_multicurve_batch1/train.json`
  (sha256 `39e8ea42…`) and `val.json` (sha256 `17cb7431…`), both still
  timestamped `Jul 16 13:11` — unchanged. `zth_vs_time_batch2_corrected.xml`
  itself also confirmed unchanged (sha256 `8fd6693f…`).
- **Not done / open:** no `test.json` for this batch either (same open
  item as batch1); the 5 Category-B (untouched pre-annotation) and 91
  Category-C (empty) images remain excluded, available for a future batch
  once actually reviewed.

### 2026-07-20 — Session: `images_body_diode_batch2` staged (308 new) — inference run + typ/max ("98%"/"max") pattern CONFIRMED real
- **Goal (owner request):** stage new body_diode images
  (`D:\LineFormerDataset_v2\categorised\body_diode\200`, uploaded to
  `data/images_body_diode_batch2_staging/`), dedup against the existing
  200-image pool, run `body_diode_run1` inference, flag likely typ/max
  charts, and produce a CVAT pre-annotation XML — no merge/training yet.
- **Duplicate check: 0 duplicates.** Compared all 308 staged filenames
  against the full existing `data/images_body_diode/` pool (200 files —
  192 annotated + 8 excluded zero-annotation, same full-pool convention
  used for the earlier `zth_vs_time_batch2` dedup check) — **0 collisions,
  all 308 genuinely new.** Copied all 308 into a new
  `data/images_body_diode_batch2/` (flat destination, matches the
  `images_<curve_type>_batchN` convention already used for `zth_vs_time`
  and `id_vs_vgs`); `_batch2_staging/` left as-is, untouched, still 308
  files.
- **Inference — reused `predict_to_cvat.py`, unmodified:**
  `python -m src.training.predict_to_cvat --checkpoint
  /mnt/data/my-datasheet/checkpoints/body_diode_run1/best_segm_mAP_50_iter_2200.pth
  --config src/training/configs/lineformer_body_diode_run1.py --images-dir
  data/images_body_diode_batch2/ --out
  data/body_diode_batch2_preannotations.xml --device cuda:0`. **308/308
  images processed, 654 polygons.** Verified programmatically: XML's 308
  `<image>` entries exactly match the 308 files on disk (0 missing, 0
  extra). Per-image count distribution: 0→6, 1→9, 2→245, 3→37, 4→11.
  **6 images with 0 detections** (need manual annotation, not correction):
  `IPD90N03S4L02ATMA1__fig_p6_014.png`, `IPD90N04S402ATMA1__fig_p6_014.png`,
  `IPD90N06S405ATMA2__fig_p6_014.png`, `IPI120N04S401AKSA1__fig_p6_015.png`,
  `IPP100N06S2L05AKSA2__fig_p6_014.png`, `IPP80N04S304AKSA1__fig_p6_014.png`.
  **`curve_name` deliberately left as the tool's standard `"TODO"`
  placeholder for every polygon — NOT auto-filled with a guessed
  temperature.** The task asked for "curve_name defaulting to the detected
  temperature where confident," but no code anywhere in this project can
  infer temperature from a curve's pixel position/appearance (no
  `if_vs_vsd` naming module exists, confirmed again this session) — the
  segmentation model itself has zero notion of temperature, same as it has
  zero notion of Ciss/Coss/Crss for capacitance. Inventing a position-based
  guess here specifically would be reckless given this session's own
  finding below (same-temperature duplicate curves are common in this
  batch — a naive "sort top-to-bottom" guess would actively mislabel
  them). Flagged rather than guessed, matching `predict_to_cvat.py`'s
  existing, tested behavior for every other curve type it's ever run on.
- **"IMPORTANT NEW FINDING" from the task — investigated, not assumed,
  and found to be TRUE for this batch specifically (note: this directly
  contradicts the immediately preceding session's finding of 0 typ/max
  cases in the *original* 192-image pool — both are correct for their
  respective image sets; this is new information from a different upload,
  not a correction of the earlier one).** Flagged all 11 images with 4+
  detected curves (the task's own suggested heuristic) and **visually
  inspected all 11, not a sample — 11/11 (100%) confirmed** showing two
  temperatures, each with a plain curve AND a second variant curve at the
  *same* temperature, labeled either `", max"` (`IPF026N15NM6ATMA1`,
  `IQD005N04NM6SCATMA1`, `IPD350N06LGBTMA1` region) or `", 98%"`
  (`IPI60R099CPXKSA1`, `IPP110N20N3GXKSA1`, etc. — a distribution
  percentile bound, same structural issue as "max," not literally the
  same word). Full flagged list (all need a distinct second curve_name per
  temperature during CVAT correction, e.g. `25C` vs `25C_max`/`25C_98pct`):
  `IPD350N06LGBTMA1__fig_p6_015.png`, `IPD600N25N3GATMA1__fig_p6_018.png`,
  `IPF026N15NM6ATMA1__fig_p8_022.png`, `IPF129N20NM6ATMA1__fig_p8_024.png`,
  `IPI60R099CPXKSA1__fig_p6_015.png`, `IPP110N20N3GXKSA1__fig_p6_019.png`,
  `IPP200N25N3GXKSA1__fig_p6_021.png`, `IPP50R250CPXKSA1__fig_p6_014.png`,
  `IPT007N06NATMA1__fig_p8_024.png`, `IPW60R099CPAFKSA1__fig_p6_017.png`,
  `IQD005N04NM6SCATMA1__fig_p9_024.png`. **Heuristic validated as clean,
  not just assumed:** also spot-checked 3 random 2-curve images
  (`IPZA60R037P7XKSA1`, `IPDQ60R065S7XTMA1`, `IPD60R950C6ATMA1`) — all 3
  show two genuinely distinct temperatures, no hidden variant — no
  evidence of false negatives at low curve counts. **No real OCR run**
  (Azure OCR needs API credentials/costs money per call — out of scope for
  this ad-hoc flagging pass); the "max"/"98%" text was read by eyeballing
  the 11 flagged images directly, not automated — a real OCR-based
  legend-parse would still be needed before this could be trusted
  unsupervised.
- **Files:** `data/images_body_diode_batch2/` (new, 308 images),
  `data/body_diode_batch2_preannotations.xml` (new, 308 images/654
  polygons). `data/images_body_diode_batch2_staging/` left untouched (308
  files, unchanged). **Existing 192-image annotated pool, its
  `data/images_body_diode/` folder, and `body_diode_annotations_normalized.xml`
  — not touched at all this session.**
- **Not done:** no merge into any COCO/split, no training, no CVAT
  upload — pre-annotation generation and flagging only, per instruction.

### 2026-07-20 — Session: `body_diode_batch2_corrected.xml` correction-completeness audit — MAJOR finding beyond the literal ask
- **Goal (owner request):** verify which of the 308 `body_diode_batch2`
  images are genuinely corrected (real `curve_name`, not `TODO`), categorize
  fully/partially/untouched, and check the 11 previously-flagged typ/max
  images specifically. **Read-only — no merge, no file changes.**
- **Structural blocker found immediately, flagged before anything else:**
  every shape's temperature attribute is named **`"curve"`**, not
  `"curve_name"` as `cvat_to_coco._extract_curve_name` hard-requires (100%
  consistent — checked all 664 shapes, zero exceptions, zero shapes with 0
  or >1 attribute elements). As-is, this file would fail conversion on
  every single shape via the existing (unmodified) converter. Not fixed
  here (read-only); needs a rename pass before any future merge.
- **Literal categorization (0 `TODO`/empty anywhere in the file):**
  **Fully corrected (no TODO/empty): 294. Partially corrected: 0.
  Untouched (all TODO): 0. Zero-shape (emptied): 14.** Sum verified = 308.
  Of the 14 zero-shape images, 6 were the original zero-detection images
  from the pre-annotation run, and **8 more had real pre-annotation
  detections that the annotator deliberately emptied** (cross-checked
  against `body_diode_batch2_preannotations.xml`'s per-image counts, not
  assumed) — full list in this session's chat log.
- **Went beyond the literal TODO check, because "not TODO" ≠ "correct":
  found that 161 of the 294 "fully corrected" images have a `curve` value
  repeated 2–4× within the same image** — the exact ambiguity this whole
  line of investigation exists to prevent, just hiding behind a
  non-empty-string check. **Only 133/294 are genuinely clean** (every
  curve name distinct within its image). Verified this is real, not
  noise: visually inspected 7 of the 161 across different sub-patterns
  (`IRF7815TRPBF__fig_p4_008.png`, `IRF640NSTRLPBF__fig_p4_010.png`,
  `IPW60R017C7XKSA1__fig_p9_024.png`, `IPT60R022S7XTMA1__fig_p9_022.png`,
  `IPD320N20N3GATMA1__fig_p6_016.png`, `IPT054N15N5ATMA1__fig_p8_022.png`)
  — **7/7 confirmed genuinely distinct temperature curves that both got
  the same label** (e.g. a chart plotting `T_J=175°C` and `T_J=25°C` where
  *both* curves were labeled `"25C"`). This is the **dominant** failure
  mode (~100+ of the 161) — a plain wrong-temperature mislabel, not a
  typ/max issue. **Tried a bbox-IoU geometric heuristic to auto-triage
  "benign duplicate mask" vs. "real mislabel" across all 166
  duplicate-label groups, then caught it giving a false result on
  cross-check** (`IRF7815TRPBF` scored IoU=0.565, bucketed as "likely
  benign," but is visually a confirmed real mislabel) — **discarded that
  classification rather than reporting it as trustworthy**; flagging the
  161 as needing real per-image visual review, not an automated split.
- **The 11 previously-flagged typ/max images, checked individually:**
  **5/11 correctly resolved** with distinct `_max` labels
  (`IPF026N15NM6ATMA1`, `IPF129N20NM6ATMA1`, `IPP110N20N3GXKSA1`,
  `IPP200N25N3GXKSA1`, `IPT007N06NATMA1`). **6/11 still broken:**
  `IPD350N06LGBTMA1` (both temps duplicated, no `_max` at all),
  `IPD600N25N3GATMA1` (`25C`×3), `IPI60R099CPXKSA1` (`25C`×2, `150C`×2),
  `IPP50R250CPXKSA1` (partial — `150C_max` correct but `25C`×2 still
  duplicated), `IPW60R099CPAFKSA1` (all 4 shapes = `25C`),
  `IQD005N04NM6SCATMA1` (all 4 shapes = `25C`).
- **Net assessment for the owner:** correction quality is meaningfully
  worse than "294 fully corrected" suggests at face value — **realistically
  only 133/308 (43%) images are both non-empty and unambiguous**. The
  `curve`-vs-`curve_name` attribute-name mismatch is a separate hard
  blocker on top of that. Neither issue was fixed this session (read-only,
  per instruction) — both need owner decisions (re-open the CVAT job for a
  second correction pass on the 161? rename the attribute globally before
  conversion?) before any merge.
- **Files touched:** none. `data/body_diode_batch2_corrected.xml` read
  only; every other file (including `data/images_body_diode/`,
  `body_diode_annotations_normalized.xml`, and everything from prior
  sessions) untouched.

### 2026-07-20 — Session: merged the 133 clean body_diode_batch2 images into a new `split_body_diode_batch2/`
- **Goal (owner request):** attribute-rename + extract only the 133
  genuinely-clean images from `body_diode_batch2_corrected.xml` (excluding
  the 161 duplicate-label images and 14 zero-shape images from the prior
  audit), convert, merge with the existing 192-image pool, and build a
  fresh family-based train/val split into a **new** folder —
  `split_body_diode_batch1/` must not be touched.
- **133-image list re-verified fresh this session** (not reused from
  memory) by re-running the same duplicate/empty check against
  `body_diode_batch2_corrected.xml` — exact same 133 filenames as the
  prior audit, confirming that session's finding.
- **Attribute rename — done only in a new extracted copy, source never
  touched:** `data/body_diode_batch2_corrected_clean133.xml` (new file,
  throwaway `extract_category_a`-style script, not committed) — the 133
  named `<image>` elements copied verbatim (geometry, `source`, all shape
  attrs unchanged) with each shape's `curve` attribute renamed to
  `curve_name` (value unchanged, just the attribute `name`). **133 images,
  297 shapes.** Verified source file byte-identical before/after (sha256
  `f5a96a9a…`, same mtime `Jul 20 09:51`).
- **Conversion — 100% reused, unmodified:**
  `cvat_to_coco.merge_convert(["data/body_diode_batch2_corrected_clean133.xml"], "data/coco/body_diode_batch2_new133.json")`
  → 133 images, 297 annotations, `validate_coco` clean.
- **Merge — reused `split_dataset._combine_and_renumber` (existing
  function, not reimplemented):** `data/coco/body_diode_batch1.json` (the
  original unsplit 192-image source) **still exists on disk this time**
  (unlike the zth_vs_time_batch2 session, where it had to be reconstructed
  from train/val) — combined directly with the new 133-image batch →
  **`data/coco/body_diode_batch2.json`: 325 images, 724 annotations**
  (427 + 297), `validate_coco` clean.
- **Split — reused `assign_family`/`extract_device` + `allocate_new_batch`
  (same approach as both prior merge sessions), a fresh family-based split
  over the FULL combined 325 images:** 28 families total (`BSC-BSZ` 117,
  `IPP` 44, `IPL` 18, `AUIRF` 17, `IAUCN` 16, `IPDQ` 15, `IAU` 13, `IPD` 11,
  `IPDD` 10, `IPN` 10, `IPQC` 9, plus 17 smaller families).
  `val_image_target=38` landed on a stable family boundary (identical
  result across target=36–40) — **train 287 / val 38 (88.3%/11.7%)**,
  close to batch1's original 170/22 (88.5%/11.5%) ratio. **Train families
  (12):** `AUIRF`, `BSC-BSZ`, `BSP`, `IAU`, `IAUCN`, `IPD`, `IPDD`, `IPDQ`,
  `IPL`, `IPN`, `IPP`, `IPQC`. **Val families (16):** `94-3316`, `AUIRL`,
  `BSB`, `BSF`, `BSR`, `BSS`, `BTS`, `BUZ`, `IAUTN`, `IAUZ`, `IMZA`, `IPF`,
  `IPI`, `IPLK`, `IPT`, `IPZ`. Zero overlap.
- **Result — verified independently from the written files, twice (once
  in-script, once in a fresh from-disk re-parse):**
  - **train.json: 287 images, 633 annotations. val.json: 38 images, 91
    annotations. Total: 325 images / 724 annotations** — exact match to
    expected.
  - All 133 clean filenames confirmed present exactly once across the two
    files; 0 duplicate filenames anywhere in the output.
  - **Re-derived the excluded sets fresh from `body_diode_batch2_corrected.xml`
    (161 duplicate-label images + 14 zero-shape images) and checked every
    single one against the output — 0 of the 161 and 0 of the 14 appear
    anywhere in `train.json`/`val.json`.** Not a sample check — the full
    175-name exclusion list was checked exhaustively.
  - 0 family overlap between train and val, re-verified with an
    independent re-implementation of the family heuristic against the
    written files.
- **Files (all new; nothing pre-existing touched):**
  `data/body_diode_batch2_corrected_clean133.xml`,
  `data/coco/body_diode_batch2_new133.json`,
  `data/coco/body_diode_batch2.json`,
  `data/coco/split_body_diode_batch2/{train.json,val.json,split_manifest.json}`.
  Manifest records method, family assignment, counts, achieved ratios, full
  merge provenance (source file hashes, the attribute-rename note, the
  exclusion-reason note), and `buffer_px`.
- **Confirmed untouched (hash + mtime):** `split_body_diode_batch1/train.json`
  (sha256 `02ffc4d9…`), `val.json` (sha256 `6781fc3a…`), and
  `data/coco/body_diode_batch1.json` (sha256 `91cf8b19…`) — all unchanged.
  `body_diode_batch2_corrected.xml` itself also confirmed unchanged (sha256
  `f5a96a9a…`, mtime `Jul 20 09:51`, same as before this session started).
- **Not done / open:** no `test.json` for this batch (same open item as
  every other curve type's batch2 split so far); the 161 duplicate-label
  and 14 zero-shape images remain excluded, available for a future batch
  once the CVAT job gets a second correction pass (per the prior session's
  flagged recommendation — not decided or acted on here).

### 2026-07-20 — Session: `body_diode_run2` — retrain launched on the expanded 325-image dataset
- **Goal (owner request):** create a config for the combined 325-image
  `split_body_diode_batch2/` pool (287 train / 38 val, 724 annotations)
  following `body_diode_run1`'s pattern, and launch training.
- **Blocker found and resolved before launch, not worked around:** the 325
  images span **two existing folders** — `data/images_body_diode/` (the
  original 192) and `data/images_body_diode_batch2/` (the new 133) —
  confirmed via the split JSONs (192 filenames resolve only in the first,
  133 only in the second, 0 overlap, 0 missing). mmdet's `CocoDataset`
  takes a single `img_prefix`, so neither folder alone works. **Created
  `data/images_body_diode_batch2_combined/`** — 325 symlinks (192 + 133)
  pointing back into the two originals, nothing copied/moved. Verified: 0
  broken symlinks, spot-checked resolution targets and file sizes, and
  confirmed both source folders' file counts unchanged (200 and 308) after
  creating it.
- **Config:** `src/training/configs/lineformer_body_diode_run2.py` — full
  standalone config (same relationship to `run1` that `run1` has to
  `zth_multicurve_run1`: its own complete file, not chained via `_base_`
  onto `run1`). Hyperparameters kept **identical to run1** per explicit
  instruction (`max_iters=8000`, `patience_iters=2000`, official-pretrained
  init, LR 5e-6, fp32, no flip, multi-scale + jitter, best+latest
  retention). Data: `data/coco/split_body_diode_batch2/{train,val}.json` +
  the new combined-symlink image dir. Config load-tested via
  `mmcv.Config.fromfile` before launch (parses clean, `load_from` and
  `img_prefix` both verified to exist on disk).
- **Flagged, not applied, per instruction ("flag if you think anything
  should change" rather than silently changing it):** this project has a
  direct precedent for this exact situation — **T10/Run A2** scaled every
  schedule number by the train-size ratio when Run A's dataset grew ~4x
  (`max_iters` 2000→8000, eval interval 200→800, patience 600→2400),
  reasoning that fixed iteration counts on more data silently shrink the
  number of epochs actually seen. Train set here grew 170→287 images
  (×1.688); applying the same method would give `max_iters≈14000`,
  `eval_interval≈350`, `patience≈3500` (preserves run1's ~47-epoch-total /
  10-eval-patience shape almost exactly: 14000/287≈48.8 epochs vs
  8000/170≈47.1). **Not applied** — used run1's numbers as literally
  instructed. Worth a follow-up config with the scaled numbers if this
  run's val curve still looks like it's improving when patience runs out
  (run1 itself early-stopped at iter 4199, short of its 8000 ceiling, so
  under-training here is a real but unconfirmed risk, not a proven one).
- **Checkpoint collision check:** `/mnt/data/my-datasheet/checkpoints/`
  listing confirmed only `body_diode_run1` existed before creating
  `body_diode_run2/` — no collision risk with run1 or any other curve
  type's checkpoint.
- **Launched:** `python -m src.training.train_lineformer --config
  src/training/configs/lineformer_body_diode_run2.py --work-dir
  /mnt/data/my-datasheet/checkpoints/body_diode_run2 --seed 42` (GPU box,
  `lineformer` conda env, background process, PID 12473).
- **Confirmed working end-to-end through the first full cycle:** loss
  decreasing normally through iter 200 (no NaN/divergence) → checkpoint
  saved (`iter_200.pth` + `best_segm_mAP_50_iter_200.pth`, 569 MB each,
  `latest.pth` symlinked) → val eval ran cleanly on all 38 val images →
  **segm_mAP_50 = 0.3931 at iter 200** — notably higher than run1's
  first-eval value (0.168), plausibly reflecting the larger/more diverse
  training set already at this very early point, not a confirmed trend →
  training resumed automatically to iter 220.
- **Timing:** steady-state **~0.35–0.36 s/iteration** (near-identical to
  run1's, as expected — same architecture/batch size). With 287 train
  images at `samples_per_gpu=1`, **1 epoch ≈ 287 iters ≈ 100–105 s
  (~1.7 min/epoch)**. Full 8000-iter ceiling ≈ **48–50 minutes** if run to
  completion without early stopping; likely shorter given run1's own
  precedent of early-stopping well short of its ceiling.
- **Files:** `src/training/configs/lineformer_body_diode_run2.py` (new),
  `data/images_body_diode_batch2_combined/` (new, 325 symlinks — no real
  image data duplicated), checkpoint output
  `/mnt/data/my-datasheet/checkpoints/body_diode_run2/` (new dir). **Not
  touched:** `body_diode_run1`'s config, checkpoint, or logs; either source
  image folder (`images_body_diode/`, `images_body_diode_batch2/`);
  `split_body_diode_batch1/`; any other curve type's files.
- **Reminder carried forward:** once this run completes, its best
  checkpoint will need the same 3-place backup (AWS/local/Drive) that
  `body_diode_run1` already received — not done yet, flagging now so it
  isn't missed.
- **Not done (still open):** run not yet complete as of this entry — final
  metrics/checkpoint TBD next session.

### 2026-07-20 — Session: `body_diode_run2` — training completed (ran full 8000-iter ceiling, did NOT early-stop)
- **Run finished on its own**, PID 12473 exited cleanly, no crash. **Did
  NOT early-stop this time** — ran the full `max_iters=8000` ceiling;
  `iters_since_best` was only 1600/2000 patience when it hit the ceiling,
  meaning the schedule ran out before the patience mechanism would have
  stopped it. **Wall clock: 3319.3 s (≈55.3 minutes)**, peak GPU memory
  2418.6 MiB (same as every other LineFormer run on this box).
- **Training curve** (iter: val `segm_mAP_50`, eval every 200 iters):
  200: 0.393 · 400: 0.385 · 600: 0.432 · 800: 0.460 · 1000: 0.499 ·
  1200: 0.442 · 1400: 0.465 · 1600: 0.482 · 1800: 0.480 · 2000: 0.532 ·
  2200: 0.492 · 2400: 0.505 · 2600: 0.483 · 2800: 0.490 · 3000: 0.522 ·
  3200: 0.521 · 3400: 0.519 · 3600: 0.501 · 3800: 0.528 · 4000: 0.533 ·
  4200: 0.527 · 4400: 0.532 · 4600: 0.552 · 4800: 0.557 · 5000: 0.554 ·
  5200: 0.614 · 5400: 0.615 · 5600: 0.601 · 5800: 0.617 · 6000: 0.660 ·
  6200: 0.609 · **6400: 0.695 (best)** · 6600: 0.623 · 6800: 0.629 ·
  7000: 0.672 · 7200: 0.682 · 7400: 0.683 · 7600: 0.639 · 7800: 0.681 ·
  8000: 0.651 (last).
  **Read: unlike run1 (which plateaued in a 0.58–0.63 band from iter 2000
  onward and comfortably early-stopped), this run was still climbing/
  fluctuating near its peak (0.65–0.68 across the final 2000 iters) when
  the iteration budget ran out — the schedule cut it off before
  convergence, not after.** This directly confirms the scaling concern
  flagged in the launch session (Run A2/T10 precedent): `max_iters=8000`
  unscaled for the larger 287-train-image set left real headroom on the
  table. **A follow-up run with the scaled schedule
  (`max_iters≈14000`, `eval_interval≈350`, `patience≈3500`, flagged but
  not applied at launch) would very plausibly climb further** — not done
  this session, a recommendation for the owner to decide on.
- **Best checkpoint: `best_segm_mAP_50_iter_6400.pth`, segm_mAP_50 =
  0.6947 at iter 6400** — **beats `body_diode_run1`'s best (0.630 at iter
  2200) by +0.065 (+10.3% relative)**, on ~1.7x more training data (287 vs
  170 train images) and a comparable/larger val set (38 vs 22).
- **Checkpoint retention audited automatically** (`run_result.json`,
  reusing the tested `checkpoint_retention` logic): present files exactly
  match expected keep-list — `best_segm_mAP_50_iter_6400.pth`,
  `iter_8000.pth`, `latest.pth` (symlink to `iter_8000.pth`) — **0
  unexpected files**.
- **Final checkpoint dir:** `/mnt/data/my-datasheet/checkpoints/body_diode_run2/`
  — `best_segm_mAP_50_iter_6400.pth` (569 MB, the checkpoint to use going
  forward) + `iter_8000.pth`/`latest.pth` (same size) +
  `run_manifest.json`/`run_result.json`/`status.txt`/`None.log.json`.
- **Same caveat as run1, carried forward:** this mAP@50 is against the 38
  `val` images only — no held-out `test.json` exists for this batch, so
  it's not an independently-verified number.
- **⚠ BACKUP STILL OUTSTANDING** (same standing instruction as run1):
  `best_segm_mAP_50_iter_6400.pth` needs to go to all 3 required places —
  AWS/EBS, local, Google Drive. Not done this session — no backup action
  taken or requested yet.
- **Not done / open decisions for the owner:** (1) launch the scaled
  follow-up run (`max_iters≈14000`) given the evidence this run was still
  improving at the ceiling? (2) same no-test-split caveat as every batch2
  split so far — needed before trusting these numbers unsupervised? (3)
  the 161 duplicate-label images from the correction-completeness audit
  are still sitting unused, pending a second CVAT correction pass. None
  decided or acted on here.

### 2026-07-20 — Session: `body_diode_run2` checkpoint — Google Drive backup done, AWS/EBS reconfirmed, local path reported
- **Goal (owner request):** back up `best_segm_mAP_50_iter_6400.pth`,
  same 3-place pattern already applied to `body_diode_run1`'s checkpoint.
- **Google Drive — DONE, verified byte-for-byte.** Checked the destination
  first: no `body_diode_run2/` folder existed yet under
  `gdrive:models/checkpoints/` (only `body_diode_run1/` and `run_a/`) — no
  overwrite risk. Uploaded via the same `rclone` remote used for run1 to
  **`gdrive:models/checkpoints/body_diode_run2/best_segm_mAP_50_iter_6400.pth`**.
  **Verified after upload:** `rclone lsl` shows the Drive copy at
  **569,702,329 bytes** (matches local exactly), and `rclone md5sum` on
  the Drive copy vs. local `md5sum` on the original both give
  **`60a15b14377f2c696e92788190f56c4b`** — exact match. **Original
  checkpoint confirmed untouched**: same mtime (`Jul 20 10:59`) and same
  sha256 (`a6238c252a95e7a22def4adb7f06e513ea380413c2a359e47c5c01a52d95cd3c`)
  as before the upload.
- **AWS — reconfirmed as already covered, no separate action, same
  reasoning as run1:** the checkpoint already lives on the `/mnt/data` EBS
  volume (durable AWS-resident storage) at
  `/mnt/data/my-datasheet/checkpoints/body_diode_run2/`; no S3 bucket or
  credentials exist on this box (same finding as the run1 backup session),
  and no new investigation was needed since the owner's prior decision
  (treat EBS as the AWS copy) already covers this.
- **Local — path reported, not executed from here** (same pull-only
  constraint as before): **`ip-172-31-66-143.ec2.internal:/mnt/data/my-datasheet/checkpoints/body_diode_run2/best_segm_mAP_50_iter_6400.pth`**
  — 543 MB (569,702,329 bytes), sha256
  `a6238c252a95e7a22def4adb7f06e513ea380413c2a359e47c5c01a52d95cd3c`,
  reachable via the existing `aws-lineformer` SSH alias for Kimo's own
  `scp` pull.
- **Backup status: 2 of 3 places done** (AWS/EBS counted, Drive done),
  local outstanding pending Kimo's pull — identical pattern to run1.
- **Files/state changed:** one new file added to Google Drive
  (`gdrive:models/checkpoints/body_diode_run2/...`) via `rclone`. **Nothing
  on this box was deleted, modified, or moved** — `body_diode_run2`'s
  checkpoint confirmed untouched (hash+mtime), `body_diode_run1`'s
  checkpoint also re-verified untouched (sha256
  `0097974dc602c2f2862a21cd453c609e57503f5b8e9539ebb0a3d085eb2810e6`,
  unchanged), no other curve type's checkpoint touched.

### 2026-07-20 — Session: `body_diode_run3` — scaled follow-up run launched
- **Goal (owner request):** launch the scaled follow-up flagged at
  `run2`'s launch and confirmed necessary by `run2`'s actual result (ran
  the full 8000-iter ceiling without early stopping, still climbing near
  its peak when cut off).
- **Config:** `src/training/configs/lineformer_body_diode_run3.py` — full
  standalone config, identical to `run2` in data (same
  `split_body_diode_batch2/` 287-train/38-val split,
  `images_body_diode_batch2_combined/` symlink dir), init (official
  pretrained), LR (5e-6), fp32, no-flip — **only the schedule changed**,
  per the T10/Run A2 scaling precedent (train set 170→287, ×1.688):
  `max_iters` 8000→**14000**, `evaluation`/`checkpoint_config` interval
  200→**350**, `EarlyStoppingDivergenceHook.patience_iters` 2000→**3500**
  (preserves run1's ~47-epoch-total/10-eval-patience shape: 14000/287≈48.8
  epochs, 3500/287≈12.2 epochs patience). Config load-tested via
  `mmcv.Config.fromfile` before launch — parsed clean, all 4 scaled values
  confirmed (`max_iters=14000`, eval interval=350, checkpoint interval=350,
  `patience_iters=3500`), `load_from` and `img_prefix` both verified to
  exist on disk.
- **Checkpoint collision check:** `/mnt/data/my-datasheet/checkpoints/`
  listing confirmed `body_diode_run1` and `body_diode_run2` both present
  and untouched before creating `body_diode_run3/` — no collision risk,
  and both prior runs' checkpoints re-verified present afterward.
- **Launched:** `python -m src.training.train_lineformer --config
  src/training/configs/lineformer_body_diode_run3.py --work-dir
  /mnt/data/my-datasheet/checkpoints/body_diode_run3 --seed 42` (GPU box,
  `lineformer` conda env, background process, PID 16596).
- **Confirmed working end-to-end through the first full cycle:** loss
  decreasing normally through iter 350 (no NaN/divergence) → checkpoint
  saved (`iter_350.pth` + `best_segm_mAP_50_iter_350.pth`, 569 MB each,
  `latest.pth` symlinked) → val eval ran cleanly on all 38 val images →
  **segm_mAP_50 = 0.3808 at iter 350** (first eval point, comparable in
  shape to run2's first eval of 0.393 at iter 200 — expected-noisy this
  early) → training resumed automatically to iter 380.
- **Timing:** steady-state **~0.35 s/iteration** (identical to run1/run2,
  as expected — same architecture/batch size/dataset). With 287 train
  images at `samples_per_gpu=1`, **1 epoch ≈ 287 iters ≈ 100 s
  (~1.7 min/epoch)**, same as run2. Full **14000-iter ceiling ≈
  81–82 minutes** if run to completion; may finish sooner if patience
  (3500 iters without improvement) triggers first.
- **Files:** `src/training/configs/lineformer_body_diode_run3.py` (new),
  checkpoint output `/mnt/data/my-datasheet/checkpoints/body_diode_run3/`
  (new dir). Reused `run2`'s `images_body_diode_batch2_combined/` symlink
  dir and `split_body_diode_batch2/` split unchanged, no new data
  artifacts needed. **Not touched:** `body_diode_run1` or `body_diode_run2`
  configs/checkpoints/logs, either source image folder, any other curve
  type's files.
- **Reminder carried forward:** once this run completes, its best
  checkpoint will need the same 3-place backup (AWS/local/Drive) already
  applied to run1 and run2 — not done yet, flagging now so it isn't
  missed.
- **Not done (still open):** run not yet complete as of this entry — final
  metrics/checkpoint TBD next session.

### 2026-07-20 — Session: `body_diode_run3` — training completed (ran full 14000-iter ceiling, did NOT early-stop)
- **Run finished on its own**, PID 16596 exited cleanly, no crash. **Did
  NOT early-stop** — ran the full `max_iters=14000` ceiling;
  `iters_since_best` was only 350/3500 patience when it hit the ceiling.
  **Wall clock: 5467.3 s (≈91.1 minutes)**, peak GPU memory 2418.6 MiB
  (same as every other run).
- **Training curve** (iter: val `segm_mAP_50`, eval every 350 iters, 40
  points total): rose fairly steadily from 0.381 (iter 350) to ~0.70 by
  iter 9450, then **oscillated in a 0.66–0.72 band for the remaining
  ~4500 iters** without a clean plateau — notably noisier/flatter than a
  clear convergence signal. Selected points: 3500: 0.555 · 5250: 0.640 ·
  7000: 0.678 · 9100: 0.697 · **9450: 0.701** · 11550: 0.713 · 12250:
  0.703 · **13650: 0.719 (best, near the very end)** · 14000: 0.693
  (last). Full 40-point curve in this session's chat log.
- **Best checkpoint: `best_segm_mAP_50_iter_13650.pth`, segm_mAP_50 =
  0.7193 at iter 13650** — **beats run2's best (0.6947) by +0.0246
  (+3.5% relative)**, and beats run1's original (0.630) by +0.089 (+14.2%
  relative) overall across the 3 runs.
- **Diminishing-returns read, not asserted as proven:** the run2→run3
  scaling gain (+3.5%) is much smaller than the run1→run2 data-size gain
  (+10.3%), and the late-run curve is bouncing in a band rather than
  cleanly rising — suggests this may be approaching the ceiling of what
  more iterations alone (same 287-image training set) can buy, though the
  very-late new peak (iter 13650, right before the ceiling) means it
  hadn't fully flattened either — genuinely ambiguous, not a confirmed
  plateau.
- **Checkpoint retention audited automatically**: present files exactly
  match expected keep-list — `best_segm_mAP_50_iter_13650.pth`,
  `iter_14000.pth`, `latest.pth` (symlink to `iter_14000.pth`) — **0
  unexpected files**.
- **Final checkpoint dir:** `/mnt/data/my-datasheet/checkpoints/body_diode_run3/`
  — `best_segm_mAP_50_iter_13650.pth` (569 MB, the checkpoint to use going
  forward) + `iter_14000.pth`/`latest.pth` (same size) +
  `run_manifest.json`/`run_result.json`/`status.txt`/`None.log.json`.
- **Same caveat as run1/run2, carried forward:** this mAP@50 is against
  the 38 `val` images only — no held-out `test.json` exists for this
  batch, so it's not an independently-verified number.
- **⚠ BACKUP STILL OUTSTANDING** (same standing instruction as run1/run2):
  `best_segm_mAP_50_iter_13650.pth` needs to go to all 3 required places —
  AWS/EBS, local, Google Drive. Not done this session.
- **Not done / open decisions for the owner:** (1) is 0.7193 good enough
  to call this the production body_diode checkpoint, or push further
  (more data via the 161 un-reviewed duplicate-label images, or yet more
  iterations given the ambiguous late-run trend)? (2) same no-test-split
  caveat as every batch2 split so far. (3) the 161 duplicate-label images
  from the correction-completeness audit are still unused, pending a
  second CVAT correction pass. None decided or acted on here.

### 2026-07-20 — Session: `body_diode_run3` checkpoint — Google Drive backup done, AWS/EBS reconfirmed, local path reported
- **Goal (owner request):** back up `best_segm_mAP_50_iter_13650.pth`,
  same 3-place pattern already applied to `body_diode_run1` and
  `body_diode_run2`'s checkpoints.
- **Google Drive — DONE, verified byte-for-byte.** Checked the destination
  first: no `body_diode_run3/` folder existed yet under
  `gdrive:models/checkpoints/` (only `body_diode_run1/`, `body_diode_run2/`,
  and `run_a/`) — no overwrite risk. Uploaded via the same `rclone` remote
  used for run1/run2 to **`gdrive:models/checkpoints/body_diode_run3/best_segm_mAP_50_iter_13650.pth`**.
  **Verified after upload:** `rclone lsl` shows the Drive copy at
  **569,702,329 bytes** (matches local exactly), and `rclone md5sum` on
  the Drive copy vs. local `md5sum` on the original both give
  **`e75a7e804b733c06573316e31ff24fbe`** — exact match. **Original
  checkpoint confirmed untouched**: same mtime (`Jul 20 13:09`) and same
  sha256 (`946134b8c60ad7e7a0564fce531a687cd0677abe30f799c4f44f8161980bfb45`)
  as before the upload.
- **AWS — reconfirmed as already covered, no separate action, same
  reasoning as run1/run2:** the checkpoint already lives on the
  `/mnt/data` EBS volume (durable AWS-resident storage) at
  `/mnt/data/my-datasheet/checkpoints/body_diode_run3/`; no new
  investigation needed since the owner's standing decision (treat EBS as
  the AWS copy) already covers this.
- **Local — path reported, not executed from here** (same pull-only
  constraint as before): **`ip-172-31-66-143.ec2.internal:/mnt/data/my-datasheet/checkpoints/body_diode_run3/best_segm_mAP_50_iter_13650.pth`**
  — 543 MB (569,702,329 bytes), sha256
  `946134b8c60ad7e7a0564fce531a687cd0677abe30f799c4f44f8161980bfb45`,
  reachable via the existing `aws-lineformer` SSH alias for Kimo's own
  `scp` pull.
- **Backup status: 2 of 3 places done** (AWS/EBS counted, Drive done),
  local outstanding pending Kimo's pull — identical pattern to run1/run2.
  **All three body_diode checkpoints (run1/run2/run3) now have Drive
  backups.**
- **Files/state changed:** one new file added to Google Drive
  (`gdrive:models/checkpoints/body_diode_run3/...`) via `rclone`. **Nothing
  on this box was deleted, modified, or moved** —
  `body_diode_run3`'s checkpoint confirmed untouched (hash+mtime),
  `body_diode_run1`'s (sha256 `0097974d…`) and `body_diode_run2`'s (sha256
  `a6238c25…`) checkpoints re-verified untouched, no other curve type's
  checkpoint touched.

### 2026-07-20 — Session: Stage-5 extraction planning for `body_diode` — investigation only, no code
- **Goal (owner request):** investigate the existing Stage-5 extraction
  pipeline (calibration + naming) to plan `body_diode`'s own extraction
  stage, ahead of the detector (`run3`, mAP@50 0.719) being usable
  end-to-end. **Read-only — no code written, per instruction.**
- **`capacitance_vs_vds`'s real pipeline, traced in full:**
  - **Calibration** (`src/calibration/ticks.py`, ported verbatim from
    legacy `cv_curve_extract.py` per `LEGACY_REVIEW.md` — not reinvented):
    `parse_numeric_ticks` (OCR-line → tick candidates by position zone,
    with extensive log-decade exponent-repair logic) → `fit_axis`
    (RANSAC linear/log fit, frozen 4-tuple contract) → `pixel_to_data`/
    `data_to_pixel` → `derive_calibration` (orchestrates both axes). **All
    of this is 100% curve-type-agnostic** — pure OCR-text + pixel-geometry
    math, nothing capacitance-specific. The one exception:
    `detect_y_axis_units` is hard-coded to `pF`/`nF`/`uF` patterns only.
  - **Mask→points**: `src/extraction/skeletonize.py::mask_to_points` —
    `skimage.skeletonize` + per-column averaging. Fully generic.
  - **Naming — position-based, capacitance-specific**:
    `src/extraction/naming/capacitance_vs_vds.py::name_curves` — sorts
    exactly 3 curves by mean pixel row, assigns Ciss→Coss→Crss
    top-to-bottom; hard-requires exactly 3. Looked up via a pluggable
    registry (`src/extraction/naming/__init__.py`) whose own docstring
    **already anticipates this won't generalize**: *"future types like
    id_vs_vgs will need a different rule, e.g. temperature-based"* —
    designed-for conceptually, never built for any curve type yet.
  - **Orchestration**: `src/extraction/pipeline.py::process_detections` —
    dedup → exact curve-count gate (caller-supplied
    `expected_curve_count`, default 3) → skeletonize → naming lookup →
    calibration → pixel-to-data → curve-type-keyed plausibility check
    (`PLAUSIBILITY_SPECS`, currently only a capacitance entry) → unit
    detection → schema-validated result. Any gate failure →
    `needs_review` with a logged reason, never guessed. **Already
    curve-type-parametric — doesn't need to change for a new curve type,
    only registry/spec entries do.**
  - **Dedup** (`src/extraction/dedup.py::dedup_detections`): IoU +
    optional flat-band/x-span heuristic. **Relevant prior finding
    surfaced here, not new this session:** that heuristic was tuned for
    capacitance's flat plateaus and was found (in the earlier
    `zth_multicurve_run1` investigation) to over-merge genuinely distinct
    near-parallel curves on multi-curve charts — `zth_vs_time` now runs
    with `use_flat_curve_heuristic=False` for exactly this reason.
    **Flagged as a real risk for body_diode's typ/max charts**, which have
    the same near-parallel-curve shape.
- **`rdson_vs_tj`'s "classical extractor" — confirmed, again, not to
  exist.** Same false premise already flagged in an earlier session (the
  typ/max investigation task) — re-confirmed from a different angle this
  time: `CLAUDE.md`'s scope table still lists it "not started," no
  PROGRESS.md entry exists, and a full-repo grep found zero
  extraction/calibration/naming code for it (the only "rdson" hits are
  unrelated Stage-1 CSV-ingest parsing of the *Rdson spec parameter*, a
  single datasheet-table number, nothing to do with curve tracing). Also
  checked `LEGACY_REVIEW.md` for a documented classical/OpenCV extractor —
  none referenced there either. The "classical OpenCV path" in
  `CLAUDE.md`/`SETUP.md` is a stage-5 architectural concept, not built for
  any curve type yet — `capacitance_vs_vds`'s real pipeline is 100%
  LineFormer-based.
- **Reuse plan for `body_diode` (nothing implemented, planning only):**
  **Reuse as-is**: `src/calibration/ticks.py` (all of it), `mask_to_points`,
  `dedup_detections` (with `use_flat_curve_heuristic=False`),
  `process_detections`/`run_pipeline` orchestration, `schema.py` — none of
  these need to change. **Reuse the pattern, add an entry**: the naming
  registry mechanism. **Genuinely new, needs real design before any code
  (per CLAUDE.md §5 Brainstorm-First):**
  1. A body_diode naming module — cannot be position-based like
     capacitance; needs temperature (and now typ/max-variant)
     identification, likely via OCR-matched legend text or
     line-style/label-proximity matching — no existing template in this
     repo for this class of problem.
  2. Variable expected curve count — capacitance is always exactly 3;
     body_diode is 2 in most charts, 4 in confirmed typ/max cases (see
     the `body_diode_batch2` correction-completeness audit) — doesn't fit
     `process_detections`'s single fixed-int gate cleanly.
  3. `detect_y_axis_units` needs new current-unit patterns (A/mA) —
     currently capacitance-only.
  4. `PLAUSIBILITY_SPECS` needs a new body_diode entry (current-range
     bounds) — just data, same existing pattern.
- **Not done (explicitly out of scope, per instruction):** no code
  written — no naming module, no registry entry, no pipeline changes. No
  file touched anywhere in the repo this session (pure investigation).

### 2026-07-20 — Session: `rdson_vs_tj` "classical extractor" discrepancy RESOLVED — root cause: stale local checkout, not lost work
- **Goal (owner request):** resolve the discrepancy between two prior
  sessions' finding ("`rdson_vs_tj` not started, no code exists anywhere")
  and the owner's expectation that a classical rdson extractor
  (color/monochrome detector, thickness gate, typ/max naming) already
  exists. **Read-only — no pull, merge, or checkout performed.**
- **Root cause found: this AWS box's local git checkout was frozen at
  commit `d19f268` ("T23...", 2026-07-12) — `git fetch` had not been run
  on this box in a long time.** Running `git fetch --all --prune` this
  session moved `origin/main` from `d19f268` to `3eb889f`, revealing
  **7 commits that exist on GitHub (`origin/main`) but were never fetched
  into this working directory**, most importantly:
  - `7f3b1b5` (2026-07-14): **"T24+T25: rdson_vs_tj classical extraction,
    Stage-4 registry entry, two-curve typ/max naming"** — adds
    `src/extraction/classical.py` (269 lines, the classical/non-AI curve
    detector), `src/extraction/naming/rdson_vs_tj.py` (162 lines,
    **label-anchored two-curve typ/max naming with top/bottom fallback** —
    real prior art for the typ/max naming problem, not the README-template
    stub found in an earlier session), a real `curve_registry.py` entry,
    and 3 test files (16+21+13 tests).
  - `3eb889f` (2026-07-14, same day): **"T27: monochrome rdson detector +
    max-thickness safety gate (26 tests)"** — adds `detect_curve_monochrome`
    (grayscale fallback: OCR-box inpainting, gridline removal with a
    flat-curve guard, dilation gap-bridging, text filtering) and a
    **median-column-thickness > 18px safety gate** (corpus-calibrated,
    downgrades `ok`→`needs_review` on the mono path only) to the same
    `classical.py` file. "Suite 700 passing" per the commit message.
  - `5d2be20` (2026-07-14): updated `CLAUDE.md`'s curve-type scope table
    to reflect `rdson_vs_tj`'s real status — **this is the exact table
    both this session and an earlier one cited as evidence "not
    started."** That citation was accurate for the *local* file at the
    time; the local file was simply outdated.
  - Also present but not `rdson`-related: `id_vs_vgs` batch1/batch2
    conversion+split commits and a Run 3 production-checkpoint log entry —
    flagging their existence too, since this box's `id_vs_vgs` state may
    also be stale (not verified in depth this session, out of scope).
- **Verified no risk of lost/conflicting local work:** `git log
  origin/main..main` is empty — local has **zero commits that aren't
  already on origin**, so there is no divergent history a future pull
  would need to reconcile at the commit level. `git branch -a` (post-fetch)
  shows only `main`/`origin/main`, no other branches or tags exist
  anywhere. There IS uncommitted local work on this box right now
  (modified `PROGRESS.md`, `envs/lineformer.lock.yml`, `src/extraction/dedup.py`,
  `src/training/predict_to_cvat.py`, 2 test files, plus several untracked
  files including this whole session arc's training configs) — worth
  being deliberate about when a `git pull` eventually happens (stash or
  commit first), but nothing indicates any of it will hard-conflict.
- **On the Windows `D:\` question:** not checked directly (no access to
  that machine from this AWS box) — not needed, since the GitHub finding
  fully explains the discrepancy on its own: the work was pushed to
  `origin/main` (`github.com/ShwetaAR227/curve-extractor`) on 2026-07-14
  from wherever it was authored; this box's checkout just never fetched
  past 2026-07-12.
- **Confirmed via direct filesystem check:** `src/extraction/classical.py`,
  `src/extraction/naming/rdson_vs_tj.py`, `tests/test_classical.py`,
  `tests/test_monochrome.py` all genuinely absent from this box's current
  working tree right now (`ls` → "No such file or directory") — consistent
  with "on origin, not yet fetched," not "exists locally but was
  overlooked."
- **Not done (read-only per instruction):** no `git pull`/`merge`/`checkout`
  performed — this session only ran `fetch`, `log`, `branch`, `status`,
  `show --stat`, and `ls` (all non-mutating). **Flagging for the owner:**
  this local checkout needs a `git pull` (after handling the uncommitted
  local changes above) before any further `rdson_vs_tj` or `id_vs_vgs`
  work on this box, and this very `PROGRESS.md` edit will itself need
  reconciling against origin's own newer `PROGRESS.md` content once that
  pull happens.

### 2026-07-20 — Session: pre-pull sync audit — full uncommitted/untracked inventory, categorized (read-only)
- **Goal (owner request):** before pulling `origin/main`'s 7 missing
  commits (see the session above), fully inventory every uncommitted and
  untracked file on this box, categorize each against `.gitignore`
  (checked, not assumed) as project-code-to-commit vs. local-only data,
  and report before touching git state. **Read-only — only `status`,
  `diff`, `log`, `show --stat`, `cat-file -e`, `check-ignore`, and a
  dry-run `add -n` were run; no `pull`/`commit`/`stash`/`add` performed.**
- **🔴 Real finding needing an owner decision — a genuine task-number
  collision in `PROGRESS.md`, not just an additive conflict.** Local
  uncommitted `PROGRESS.md` has its own `T24`/`T25` table rows ("Dry-run:
  `classify_device()` on the new Rohm batch, 612 real devices" / "First
  real end-to-end Stage 5→7 run on real, never-annotated devices (Rohm)",
  both dated 2026-07-13) — **describing entirely different work** than
  origin's committed `T24`/`T25` ("Classical extraction path for
  `rdson_vs_tj`" / "Stage-4 `rdson_vs_tj` registry entry + Infineon
  Diagram-template classification fix", also dated 2026-07-13). Both
  streams independently continued from the same base (`d19f268`, T23) and
  picked up numbering at T24 without visibility into each other — this
  local box's own uncommitted Rohm-classification/extraction work (a real,
  substantial session: 612-device classify dry-run, then the **first ever
  real Stage 5→7 run with the actual trained model on 108 real devices**,
  86 finalized/22 needs_review — plus a **noteworthy standalone finding**:
  *"the original `run_a` checkpoint + its EBS volume were lost in an
  untracked 2026-07-12 incident"*, mentioned in-line in that entry but not
  otherwise flagged before now) never got renumbered or reconciled against
  origin's parallel `rdson_vs_tj` stream. **Confirmed the collision is
  confined to T24/T25 only** — local has no `T26`/`T27` entries at all, so
  origin's `T26` (id_vs_vgs Rohm spot-check) and `T27`+follow-up
  (monochrome rdson detector + thickness gate) have no local counterpart
  to collide with. **This needs an editorial decision before merging**
  (e.g. renumber local's Rohm entries to `T24b`/`T25b` or slot them after
  `T27`) — not a mechanical `git pull`, and not decided or acted on here.
- **🟡 Gitignore gap found and verified, not fixed:** `data` at the repo
  root is a **symlink** (`data -> /mnt/data/my-datasheet/data`), and
  `.gitignore`'s `data/` rule (trailing slash) does **not** match a
  symlink — confirmed via `git check-ignore -v data` (no match, exit 1)
  and a **dry-run `git add -n -A`, which showed `data` WOULD actually be
  staged** if anyone ever ran `git add -A`/`git add .` on this box. Not
  currently committed only because that hasn't happened yet — the gap is
  real, not hypothetical. Flagged for the owner to decide the fix (e.g. a
  non-trailing-slash `/data` or `data` entry), not changed here.
- **Full categorized inventory — verified against `.gitignore` and
  cross-checked path-by-path against `origin/main`'s tree, not assumed:**
  - **Modified (tracked), 6 files, all legitimate project code/config,
    should be committed:** `PROGRESS.md` (conflict, see above),
    `envs/lineformer.lock.yml` (trivial `libglib`/`tzdata` version-bump
    diff from a routine `conda env export`), `src/extraction/dedup.py`
    (adds `use_flat_curve_heuristic` param, the 2026-07-17
    `zth_multicurve_run1` dedup investigation), `src/training/predict_to_cvat.py`
    (curve-type-gated dedup wiring), `tests/test_dedup.py`,
    `tests/test_predict_to_cvat.py`. **Only `PROGRESS.md` has an origin
    path collision** (`git cat-file -e origin/main:<path>` checked for
    every one of the other 5 — none exist on origin, so none can
    conflict).
  - **Untracked, 13 items:** `data` (symlink, local-only, should NOT be
    committed — see gitignore gap above); 3 `lineformer_body_diode_run{1,2,3}.py`
    configs (this conversation's work); 3 `lineformer_id_vs_vgs*.py`
    configs (pre-existing local, referenced during body_diode sessions);
    4 `lineformer_zth_*.py` configs (pre-existing local); `src/training/dedup_review.py`
    + `tests/test_dedup_review.py` (real, tested standalone tool from the
    2026-07-17 dedup investigation). **All 12 non-`data` items verified
    via `git cat-file -e origin/main:<path>` to have zero path collision
    with origin** — every one is safe to commit with no merge conflict at
    the file-path level.
  - **Origin-only changes (7 files not touched locally at all) will pull
    in cleanly**: `.gitignore`, `CLAUDE.md`,
    `src/classification/curve_registry.py`,
    `src/dataset_tools/split_dataset.py` (a small `FAMILY_MERGE_MAP`
    extension — worth noting several sessions in this conversation called
    `assign_family`/`allocate_new_batch` from this exact file for the
    `zth_multicurve_batch2` and `body_diode_batch2` splits; the extension
    doesn't retroactively change already-written split JSONs, only future
    calls after the pull), plus the new `classical.py`/`rdson_vs_tj.py`/
    `naming/__init__.py` files and their 4 test files.
- **Confirmed clean at the commit level:** `git log origin/main..main` is
  empty — local has zero commits that aren't already on origin, so no
  commit-level divergence exists; the only real risk is the **working-tree**
  collision on `PROGRESS.md`'s content (both the bulk-append conflict and
  the T24/T25 numbering collision) plus the `data` symlink gitignore gap.
- **Not done (read-only per instruction):** no `pull`/`commit`/`stash`/`add`
  run — this session only inspected state. Recommended next steps for the
  owner to approve before any git-state-changing action: (1) decide how to
  renumber/reconcile the T24/T25 collision, (2) decide the `.gitignore`
  fix for the `data` symlink, (3) then stage+commit the 5 safe modified
  files + 12 safe untracked files (everything except `PROGRESS.md` and
  `data`), (4) resolve `PROGRESS.md` by hand (append origin's version
  under the reconciled numbering, not a blind merge), (5) `git pull`.

---

**✅ MERGE CONFLICT RESOLVED (2026-07-20, not yet committed) — kept BOTH
sides, no content dropped.** The narrative log above is now ordered:
origin's `3eb889f` T24-T27 rdson_vs_tj sessions + id_vs_vgs entries first
(chronologically earlier, continuing straight from T23), then this box's
own T28/T29 (renumbered from the collision) plus its full session log,
unchanged. Resolved via exact line-slicing (not manual retyping) — line
counts before/after reconciled precisely, verified no lines gained or
lost. Conflict markers removed. **Not committed yet** — awaiting review
per instruction before `git add PROGRESS.md && git commit` finalizes the
merge (everything else the merge touched is already staged).

### 2026-07-20 — Session: committed local work, `git pull origin main` — ONE CONFLICT, stopped per instruction
- **Goal (owner request):** commit the 19 confirmed-safe local
  files (from the prior sync-audit + renumbering sessions), then pull
  origin's 7 missing commits (the `rdson_vs_tj` work).
- **Commit — succeeded cleanly.** Staged exactly the 19 files confirmed
  safe (7 modified + 12 untracked; explicit paths, not `git add -A`) and
  committed as `f141848` ("body_diode run2/run3, zth_vs_time batch2
  merge, dedup review tool, PROGRESS.md renumbering"). **Verified `data`
  is NOT in this commit** — `git show --stat f141848` lists no `data`
  entry, confirming the `.gitignore` fix (previous session) worked as
  intended, not just in the dry-run.
- **`git pull origin main` — refused without a strategy** (branches had
  diverged: 1 local commit vs. 7 remote; no default `pull.rebase`/`pull.ff`
  configured on this box, and per standing instruction global git config is
  never changed). Used an explicit one-off `git pull --no-rebase origin main`
  (merge, not rebase) rather than setting a persistent default.
- **Result: exactly ONE conflict, exactly where predicted by the earlier
  sync audit — `PROGRESS.md`.** Every other file **auto-merged cleanly and
  is already staged**: `.gitignore`, `CLAUDE.md`,
  `src/classification/curve_registry.py`,
  `src/dataset_tools/split_dataset.py`, and the real `rdson_vs_tj` code —
  `src/extraction/classical.py` (new), `src/extraction/naming/rdson_vs_tj.py`
  (new), `src/extraction/naming/__init__.py`, plus 4 new test files
  (`test_classical.py`, `test_curve_registry_rdson.py`, `test_monochrome.py`,
  `test_rdson_two_curve.py`) and `tests/test_split_dataset.py`.
  **Confirmed on disk, not just staged**: `ls` shows all 4 rdson files
  genuinely present with today's mtime.
- **STOPPED per explicit instruction — did NOT attempt to resolve the
  PROGRESS.md conflict.** No `git commit`, no `git merge --abort`, no
  edits inside the conflict markers. `git status` confirms the repo is
  mid-merge with exactly one unmerged path (`PROGRESS.md`); the
  auto-merged files above are already in the merge's staging area, ready
  to commit once `PROGRESS.md` is resolved by hand.
- **Test suite run anyway (read-only, doesn't touch the conflict or any
  git state) — confirms the merged CODE is healthy independent of the
  documentation conflict:** `.venv/bin/python -m pytest tests/ -q
  --ignore=third_party` → **728 passed, 0 failed** (up from this box's
  last known 700ish-plus-local-additions baseline — consistent with
  origin's own "suite 700 passing" claim in the T27 commit plus this box's
  own `test_dedup_review.py` and other locally-committed tests).
- **What's left (owner decision, not done here):** resolve the
  `PROGRESS.md` conflict by hand — this isn't a simple pick-one-side
  case, since **both sides added real, wanted content** (local's full
  session log for today's work + the earlier renumbered T28/T29 entries;
  origin's T24-T27 rdson work + T26 id_vs_vgs entry). The right resolution
  is almost certainly "keep both, concatenated in a sensible order" (e.g.
  origin's T24-T27 block first since it's chronologically earlier content
  continuing straight from T23, then local's T28/T29 Rohm entries, then
  local's remaining narrative session log) — not a `--ours`/`--theirs`
  discard of either side. Once resolved: `git add PROGRESS.md` then
  `git commit` to finalize the merge (no `--no-edit` needed; a real merge
  commit message describing the reconciliation would be appropriate).

### 2026-07-20 — Session: `id_vs_vgs` chart-structure investigation — naming module planning, no code
- **Goal (owner request):** investigate real `id_vs_vgs` chart images/data
  to plan its Stage-5 naming module (detection is done — Run 3 production
  checkpoint, mAP@50 0.7414 — but no naming/pixel-to-value conversion
  exists yet, same gap already flagged for `body_diode`). **Read-only —
  no naming code written.**
- **Distinguishing variable confirmed: junction temperature (`T_j`), not
  `V_DS`.** Visually inspected images from both `images_id_vs_vgs`
  (batch1) and `images_id_vs_vgs_batch2`, spanning IR/AUIRF, Infineon
  "Diagram N" (BSZ/IPA/IAU families), and Rohm (T26 overlays). Two
  Infineon charts carry an explicit on-chart annotation confirming this
  directly: `"I_D=f(V_GS), |V_DS|>2|I_D|R_DS(on)max; parameter: T_j"` and
  `"I_D=f(V_GS); V_DS=6V: parameter: T_j"` — **`V_DS` is a fixed test
  condition per chart, `T_j` is the swept parameter that produces
  multiple curves.**
- **Ground truth already encodes the exact target naming convention.**
  Grepped `curve_name` attributes out of both CVAT XMLs directly (not
  guessed): batch2 uses `TJ_25C` (182), `TJ_150C` (110), `TJ_175C` (72),
  `TJ_-55` (46), `TJ_-40C` (10); batch1 raw uses the same
  `TJ_25C`/`TJ_150C`/`TJ_175C`/`TJ_-40C` set. **This is the same
  `TJ_<value>C` format M8's per-temperature recall table already reports
  against** — the naming module's target output format is not a design
  choice, it already exists in the annotations.
- **Curve count is NOT fixed like capacitance's exact-3** — three real
  patterns coexist, confirmed via a full XML tally (batch2, 200 images):
  **2 curves — 143/200 (dominant)**, typically `TJ_25C`+one hot temp
  (`150C`/`175C`); **3 curves — 45/200**, a distinct template family
  (`IAUA*`/`IAUC*`/`IAUT*`/`IAUZ*`, always page `p7_015`/`p7_016`) fixed
  at exactly `{TJ_25C, TJ_175C, TJ_-55}`; **0 curves — 11/200** (misses);
  **1 curve — 1/200**. Batch1 raw: 2 curves 156/200, 0 curves 40/200
  (pre-correction misses), 3 curves 4/200. **Rohm charts run higher
  still** — a directly-viewed sample (`2SK2463T100`, `R6004KNXC7G`) shows
  **4 curves** (`Ta = 125°C/75°C/25°C/-25°C`, ambient not junction temp,
  different label token), and T26's own detection-count distribution over
  612 Rohm figures peaks at 4 but reaches 5/6/7 — the model is
  Infineon-only-trained (per T26), so these higher counts are genuine
  chart complexity, not confirmed to be all correct detections.
- **Labeling style varies by chart, both OCR-recoverable, neither
  reusable verbatim from `rdson_vs_tj`'s label dict:** (a) dominant
  Infineon style — temperature text sits directly adjacent to/touching
  each curve at its diverging region, no leader line (e.g.
  `AUIRF1010EZS`: `"T_J = 175°C"` / `"T_J = 25°C"`; `BSZ034N04LSATMA1`:
  `"150 °C"` / `"25 °C"`); (b) Rohm style — a stacked legend block with
  leader/pointer lines to each of 4 curves (`"Ta = 125°C"` down to
  `"Ta = -25°C"`). Both are proximity-anchorable the same way
  `rdson_vs_tj.py::_nearest_curve_index` already anchors "max"/"typ"
  labels — but `id_vs_vgs` needs an **open-ended numeric temperature
  parse** (regex over `T_?J|T_?j|Ta`-style OCR text), not rdson's small
  fixed `{max, 98%, typ}` dict.
- **Curves visually cross/converge — position-based fallback is unsafe
  here, unlike rdson's cleaner top/bottom split.** Directly confirmed on
  two viewed charts: `AUIRF1010EZS` (2-curve) — the `T_J=175°C` and
  `T_J=25°C` traces cross and nearly overlap from `V_GS≈6V` onward;
  `IAUC120N04S6N006ATMA1` (3-curve) — `175°C`/`25°C`/`-55°C` all converge
  to the same point near `V_GS≈5V, I_D≈1000A` before re-diverging.
  **Label-proximity must be the primary signal, not a fallback** — a
  mean-pixel-row ranking (rdson's approach) would be unreliable exactly
  in the region where curves are hardest to tell apart, and there is no
  safe generic ordering to fall back to when labels are missing/ambiguous
  (same "never guess, quarantine instead" rule rdson's
  `name_curves_by_labels` already follows on conflict).
- **Typ/max-style duplicate-curve issue checked — exists, but rare, and
  looks like mask fragmentation, not a real chart-design duplicate (unlike
  body_diode's confirmed intentional typ/max pattern).** Tallied
  same-`curve_name`-twice-in-one-image across both XMLs: **4/200 in
  batch2** (`IAUC120N06S5N022ATMA1` dup `TJ_-55`,
  `IAUCN04S7N015ATMA1` dup `TJ_175C`, `IPA50R500CEXKSA2` dup `TJ_-40C`,
  plus one dup `TODO` placeholder), **3/200 in batch1 raw**. Visually
  inspected `IAUC120N06S5N022ATMA1` (the `TJ_-55` dup): the chart shows
  what reads as one visually-broken/discontinuous stroke near the
  crossing region split into two separate polygon instances carrying the
  same label — consistent with a mask-fragmentation artifact for Stage
  5's existing dedup to catch, not a second physical curve at the same
  nominal temperature. **Not confirmed with pixel-level polygon overlap
  math this session** (visual read only) — worth a real dedup-IoU check
  before assuming it's non-naming, but nothing here looks like
  body_diode's genuine same-temperature-twice design.
- **Naming strategy this data supports (planning only, nothing
  implemented):** primary = OCR-text temperature-value extraction
  (regex over `T_j`/`T_J`/`Ta`-prefixed `°C` text) anchored to the
  nearest curve by pixel proximity, reusing the proximity mechanism (not
  the label dict) from `rdson_vs_tj.py`; output names in the
  already-annotated `TJ_<value>C` convention. No safe position-only
  fallback — an unresolvable/ambiguous label set should quarantine
  (`needs_review`), matching CLAUDE.md's never-guess rule, not attempt a
  top-to-bottom guess the crossing-curve evidence above shows would be
  unreliable. **Open design questions for the owner, not decided here:**
  (1) variable expected curve count (1/2/3, and Rohm's 4+) doesn't fit
  `process_detections`'s single fixed-int gate any more cleanly than
  body_diode's 2-vs-4 case already flagged; (2) whether Rohm's
  ambient-temp (`Ta`) charts are in scope now or deferred (different
  label token than Infineon's `T_j`, and higher/uncertain curve counts on
  an Infineon-only-trained detector); (3) the fragmented-duplicate finding
  above should probably be root-caused via dedup IoU math before the
  naming module is built, so it isn't mistaken for a naming problem later.
- **Not done (explicitly out of scope, per instruction):** no naming
  module, no registry entry, no pipeline changes, no dedup-IoU
  verification of the fragmentation theory. No file touched outside this
  PROGRESS.md entry.

### 2026-07-21 — Session: T30 — Stage-7 live-mode device discovery (TDD)
- **Goal:** the orchestrator CLI (`src/orchestrator/pipeline.py`) has only
  ever run in "precomputed mode" (`PrecomputedStage5`, reads a directory of
  already-produced Stage-5 result JSONs). Live mode — discovering devices
  directly from real Stage-3 OCR output — is new groundwork toward the
  eventual stage 1-3 migration; this session scoped narrowly to just device
  discovery, not the classify->extract wiring (still correctly deferred,
  per the module's own docstring, until stages 1-3 land).
- **Design decision (owner-confirmed):** precomputed mode's CLI excludes
  known non-device stems by a fixed name blocklist
  (`{"summary", "dryrun_report", "batch_summary"}`) — confirmed **not
  safe** for the real Stage-3 output root, since `data/` accumulates
  non-device folders constantly (training image batches, COCO splits,
  overlays, raw downloads, ...) and a blocklist would silently miss new
  ones as they appear. **Fix: self-verifying discovery** — a subfolder
  counts as a device iff it directly contains `full_extraction.json` (and
  it must be a file, not e.g. a same-named directory), never a name-based
  exclude list.
- **Implementation:** new `LiveStages` class in `src/orchestrator/pipeline.py`
  (same file as `PrecomputedStage5`, the designed extension point already
  noted in its docstring). `__init__(stage3_root)` stores
  `self.stage3_root = Path(stage3_root)` once; `discover_devices()` reads
  off `self.stage3_root` rather than re-deriving it, so there is a single
  source of truth for the root path once `run_classification`/
  `run_extraction` are added later. `run_classification`/`run_extraction`
  are deliberately NOT implemented yet — out of scope until real Stage-3
  output exists.
- **Tests first (TDD red->green confirmed):** 13 new cases in
  `tests/test_orchestrator.py` (red = `ImportError: cannot import name
  'LiveStages'`, verified before implementation) — folder-with-json found,
  folder-without-json skipped, sorted output, empty root, all-decoys root,
  loose files at root ignored, nested json two levels down NOT promoted
  (only direct children are candidates), `full_extraction.json` as a
  directory doesn't count, extra files in a device folder don't disqualify
  it, missing `stage3_root` raises `FileNotFoundError` (a config error,
  not silently "zero devices"), not cached (re-scans live), plus a
  realistic-shape test mixing real device folders with the exact kind of
  decoy folder names called out above. Suite **741 passing** (was 728),
  no regressions.
- **No real Stage-3 output exists on this machine yet** (that OCR run
  hasn't happened — separate task, noted already in T29). Tested only
  against constructed `tmp_path` fixtures, as instructed, until real data
  is available.
- **Not done / open for owner:** CLI wiring (a `--live`/mode-select flag
  in `main()`) and the actual `run_classification`/`run_extraction`
  methods on `LiveStages` are both still future work, gated on the stage
  1-3 migration per the existing module docstring — this session only
  covers device discovery.

### 2026-07-21 — Session: T30 correction — merged into the real `LiveStages`, duplicate removed
- **T30 above was written against a stale local checkout** (this box had
  fallen behind `origin/main`) — the real `LiveStages` already existed at
  `src/orchestrator/live_stages.py` (owner's separate T29 work: `classify`
  -> `extract` wiring, `ClaimTracker`, `NoExtractorAvailable`, 44 tests,
  committed to origin before this session started). T30's `LiveStages`
  inside `pipeline.py` was an unintentional duplicate class, discovered
  only after a `git pull` merge (own merge commit: three-way merge of
  `origin/main`'s T29 with this box's T28/local commits, one unrelated
  add/add conflict in `lineformer_zth_multicurve_run1.py` resolved in
  favor of this box's version — confirmed via `run_manifest.json` +
  training log + existing checkpoint dir that it's the version that
  actually produced the completed run; origin's version was never
  executed here).
- **Fix:** `discover_devices()` (the self-verifying "does this subfolder
  contain `full_extraction.json`" check, unchanged logic from T30) moved
  onto the real `LiveStages` in `live_stages.py` as an added method — one
  behavior difference from T30's version: reads `Path(self.stage3_root)`
  at call time rather than normalizing to `Path` in `__init__`, matching
  the real class's existing contract (`stage3_root` stays whatever
  type — `str` or `PathLike` — the caller passed, same as
  `load_figures_by_page`'s own `Union[str, Path]` signature). The
  duplicate `LiveStages` class deleted from `pipeline.py` entirely —
  `pipeline.py` is now byte-identical to its pre-T30 state (`git diff`
  against the merge commit is empty), `main()`/CLI untouched.
- **Tests migrated, not discarded:** all 13 of T30's device-discovery
  tests carried over into `tests/test_live_stages.py` (new
  `TestDeviceDiscovery` class, section K), adapted to the real
  constructor (`LiveStages(curve_type, images_root=..., stage3_root=...)`
  instead of T30's single-arg one) — confirmed red first
  (`AttributeError: 'LiveStages' object has no attribute
  'discover_devices'`) before implementing. One test swapped for a
  better-fitting equivalent: T30's "stores stage3_root as Path" (asserted
  `__init__` normalizes the type) doesn't hold for the real class, which
  passes `stage3_root` through unchanged — replaced with
  `test_accepts_string_stage3_root_not_just_path`, which tests the same
  real concern (a plain string root works) against the real class's
  actual contract instead. The stale duplicate tests removed from
  `tests/test_orchestrator.py` (13 cases + its own `_make_device_dir`
  helper + the `LiveStages` import), which is now also back to its
  pre-T30 state.
- **Suite: 822 passing** (44 pre-existing `live_stages` tests + 13
  migrated = 57 there; net unchanged elsewhere). `grep -rn "class
  LiveStages"` across the repo returns exactly one hit
  (`src/orchestrator/live_stages.py`).
- **Still not done** (unchanged from T30, now correctly scoped to the
  real module): CLI mode-select wiring in `pipeline.py`'s `main()` — it
  still only knows `PrecomputedStage5`; swapping in `LiveStages` (which
  already has `run_classification`/`run_extraction` wired, per the real
  T29) plus `discover_devices()` for the device list is a separate
  follow-up task, not done here.

### 2026-07-21 — Session: T30 follow-up — `--mode live`/`precomputed` CLI switch
- Implements the CLI wiring flagged as not-done above. `main()` now takes
  `--mode {live,precomputed}` (default `live`), `--stage3-root` and
  `--images-root` (both `type=Path`, default `None`), and `stage5_dir`
  changed to optional (`nargs="?", default=None`) since it's only
  meaningful for `--mode precomputed` now.
- **Manual post-parse validation, only where the adapters themselves don't
  already validate:** `--images-root` required for live mode (`LiveStages.
  __init__` stores it unchecked — no default, but a `None` value passes
  through silently since there's no type validation on it), `stage5_dir`
  required for precomputed mode. **Deliberately NOT validated in `main()`:**
  `--stage3-root` — `LiveStages.__init__` already falls back to the
  `LINEFORMER_STAGE3_ROOT` env var (or raises `RuntimeError`) on its own;
  duplicating that check in `main()` would be the exact kind of two-places-
  same-logic drift risk CLAUDE.md's zero-duplication rule exists to avoid.
- **Live-mode device discovery goes through `stages.discover_devices()`**
  (the self-verifying `full_extraction.json`-per-folder check, now on the
  real `LiveStages` per the T30 correction above) — `main()` does not
  re-derive `stage3_root` or re-implement folder listing for live mode.
  Precomputed mode's own device listing (glob + the
  `{"summary","dryrun_report","batch_summary"}` stem exclusion) is
  completely unchanged.
- **12 new tests, TDD red→green confirmed** in `tests/test_orchestrator.py`
  (new `TestCliModeSelection` class) — red was two distinct, correct
  failure modes: `SystemExit: unrecognized arguments: --mode` for every
  test using the new flags (confirmed `--mode` didn't exist yet), and
  `AttributeError: module 'pipeline' has no attribute 'LiveStages'` for
  every test monkeypatching it (confirmed `LiveStages` wasn't imported
  into `pipeline.py` yet) — 9 of 12 failed red, the other 3
  (`SystemExit`-expecting validation tests) incidentally passed red too
  since an unrecognized `--mode` flag also raises `SystemExit`, just not
  for the intended reason; all 3 still exercise the real validation path
  now that it exists. `LiveStages` itself is faked in these tests (records
  constructor args, returns canned devices/results) — this file's job is
  `main()`'s routing/wiring, not classify/extract correctness (that's
  `live_stages.py`'s own 57-test suite). Coverage: default-mode-is-live,
  both required-arg errors, invalid `--mode` value rejected, correct args
  passed to `LiveStages` (`curve_type` positional + `images_root`/
  `stage3_root` keywords), `stage3_root=None` passed through unvalidated,
  `discover_devices()` drives the batch device count, a full live-mode
  finalize-via-fake-stages run, a spy confirming `PrecomputedStage5` is
  never constructed in live mode (routing is a real branch, not "run
  both"), precomputed mode's blocklist behavior unchanged, a precomputed
  backward-compatibility full run, and a positional-`stage5_dir`-without-
  `--mode` case (confirms the positional alone doesn't switch mode away
  from the `live` default).
- **Suite: 834 passing** (was 822), no regressions. `grep -rn "class
  LiveStages"` still exactly one hit (`pipeline.py` now only imports it).
- **Not done / not pushed** — owner review pending before push, per
  instruction.

### 2026-07-21 — Session: extract generic curve detection into `curve_detection.py`
- **Pure refactor (owner-approved), zero behavior change for rdson_vs_tj.**
  Moved `detect_curve_classical`, `detect_curve_monochrome`,
  `_inpaint_ocr_boxes`, `_remove_straight_lines`, `_median_col_thickness`
  out of `src/extraction/classical.py` into a new
  `src/extraction/curve_detection.py` — verified none of them took
  `curve_type` or contained any rdson-specific logic (image/ocr_lines in,
  `Detection` list out, only). `classical.py` now imports them and keeps
  only what's genuinely rdson-specific: `run_classical_pipeline`,
  `detect_rdson_units`, `RDSON_Y_PLAUSIBLE_RANGES`, the unit regex table,
  plus `EXPECTED_CURVE_COUNT`/`TWO_CURVE_COUNT`/
  `MONO_MAX_MEDIAN_COL_THICKNESS_PX` (rdson-specific gating constants used
  inside `run_classical_pipeline`, not among the moved generic ones).
- **Design change while moving:** the 10 tuning constants these functions
  used as module-level globals (`CHROMA_MIN_SPREAD`, `MIN_CURVE_AREA_PX`,
  `MIN_COL_SPAN_FRAC`, `GAP_CLOSE_KERNEL`, `MONO_INK_MAX_GRAY`,
  `MONO_GRID_MIN_SPAN_FRAC`, `MONO_BRIDGE_KERNEL`, `MONO_MAX_FILL_DENSITY`,
  `MONO_DENSITY_EXEMPT_SPAN_FRAC`, `MONO_INPAINT_RADIUS` — all corpus-
  tuned against rdson_vs_tj's real chart corpus, T24/T25/T27) are now
  KEYWORD-ONLY parameters on the moved functions, defaulting to the exact
  same values. `run_classical_pipeline`'s call sites pass zero overrides,
  so rdson_vs_tj's behavior is unchanged; a future classical-path curve
  type can override any of them (untested whether rdson's corpus-tuned
  values transfer — that's exactly why this parameterization exists)
  without duplicating these functions.
- **TDD, red confirmed properly despite writing the module first:** wrote
  `tests/test_curve_detection.py` (tests moved verbatim from
  `test_classical.py::TestDetectCurveClassical` and
  `test_monochrome.py::TestDetectCurveMonochrome`, adjusted for the new
  keyword-only signatures) before removing the functions from
  `classical.py`; since the new module already existed when the test file
  was written, red was confirmed by temporarily moving
  `curve_detection.py` aside and re-running — `ModuleNotFoundError`,
  confirmed for the right reason — then restoring it and confirming green
  before proceeding to remove the functions from `classical.py`.
- **3 NEW tests** (`TestParameterOverridesActuallyApply`, not migrated
  from anywhere) prove the actual point of the refactor — that overriding
  a tunable changes the outcome, not just that it exists in the
  signature: raising `chroma_min_spread` makes a faint curve invisible,
  lowering `min_curve_area_px` admits a previously-dropped swatch,
  raising `ink_max_gray` detects a lighter-gray "black" curve the default
  threshold misses.
- **Fixture ownership, minimally disrupted:** `test_curve_detection.py`
  imports its chart-drawing fixtures from `test_classical.py` (geometry:
  `blank_chart`, `draw_axes`, `standard_chart`, etc.) and `test_monochrome.py`
  (mono-specific: `mono_chart`, `draw_mono_curve`, `draw_hline`,
  `draw_vline`, `BLACK`) rather than relocating fixture definitions —
  same cross-file-import convention this codebase already uses
  (`test_monochrome.py`/`test_rdson_two_curve.py` already import from
  `test_classical.py`). `test_classical.py` and `test_monochrome.py` keep
  everything except the moved test classes and now-unused import names
  (`detect_curve_classical` dropped from `test_classical.py`'s import;
  `detect_curve_monochrome`, `IMG_H`, `IMG_W`, `ocr_line` dropped from
  `test_monochrome.py`'s — confirmed genuinely unused, not just moved,
  via grep before removing).
- **rdson_vs_tj behavior provably unchanged:** `test_classical.py`
  (12 tests, was 21 — the 9 moved out), `test_monochrome.py` (9 tests,
  was 26 — the 17 moved out), `test_rdson_two_curve.py` (23 tests,
  UNTOUCHED — never called the moved functions directly, only
  `run_classical_pipeline`/naming functions) — every one of these,
  same assertions, same fixtures, same pixel-level expected values,
  **all still pass**. `test_curve_detection.py`: 29 tests (9 + 17 moved
  + 3 new). Accounting: 70 pre-existing rdson-related tests, ALL
  preserved (26 relocated, 44 untouched in place), plus 3 new = 73.
- **Bug caught before it shipped:** removing `numpy`'s import from
  `classical.py` (no longer directly used, moved functions took it with
  them) would have broken `run_classical_pipeline`'s own signature
  (`image: np.ndarray` type hint, evaluated at def-time) — caught by a
  syntax/import check before running tests, `numpy` re-added since
  `classical.py` still needs it for that one type hint.
- **Full suite: 837 passing** (was 834), no regressions anywhere
  (confirmed `live_stages.py`'s `from src.extraction.classical import
  run_classical_pipeline` — its only import from this module — still
  resolves correctly).
- **Not pushed** — owner review pending, per instruction; this session
  also came with an explicit "do not touch any existing file without
  asking first" instruction — the files touched (`classical.py`,
  `test_classical.py`, `test_monochrome.py`) were exactly and only the
  ones the task itself required editing (removing moved code / adjusting
  imports for the move), nothing beyond that scope.

### 2026-07-21 — Session: T31 — vgsth_vs_tj naming module + shared `nearest_curve_index` (TDD)
- **Red phase (owner-approved, strict "no implementation" turn):** two new
  test files, touching zero existing files —
  `tests/test_vgsth_naming.py` (30 tests) and
  `tests/test_nearest_curve_index.py` (3 tests). Confirmed red for the
  right reason (`ModuleNotFoundError: ...vgsth_vs_tj`,
  `ImportError: cannot import name 'nearest_curve_index'`), rest of suite
  unaffected (837 passing via `--continue-on-collection-errors`). A
  separate, explicit regression-baseline check (not new test code, since
  it would require touching `rdson_vs_tj.py`, out of scope for red phase):
  ran `tests/test_rdson_two_curve.py` as-is — 23/23 passing, confirmed
  zero diff via `git status` — the baseline the future `_nearest_curve_index`
  rewiring must reproduce.
- **Owner review caught a real spec ambiguity before it shipped:**
  scenarios B.7 and C.13 both described "two current-value labels that
  normalize to the identical value" with opposite expected outcomes
  (count=1 vs None). First draft resolved this with a two-tier rule
  ("harmless when it's the whole label set, ambiguous otherwise") flagged
  explicitly in the test file for owner review. **Owner rejected the
  leniency**: revised to a single unconditional rule — any duplicate
  normalized value (band OR current-value scheme) always -> `None`, no
  exceptions. B.7's test revised to expect `None`; the B.9 micro-sign-glyph
  test (same leniency-dependent shape, not explicitly called out by the
  owner but caught during the revision to avoid a self-contradicting
  suite) revised too. Docstring rewritten with the unconditional rule plus
  a new "KNOWN, DELIBERATELY OUT-OF-SCOPE LIMITATION" note: a duplicate
  could legitimately be OCR re-detecting one physical label twice rather
  than two colliding curves, but telling those apart needs label-position
  reasoning this module doesn't attempt — always quarantines rather than
  guessing which case it is. Re-confirmed red for the same two reasons
  after the revision, zero other files touched.
- **Implementation, one piece at a time, full suite after each (owner
  instruction), zero regressions at every step:**
  1. `nearest_curve_index(curves, cx, cy) -> int` added to
     `curve_detection.py` — the exact algorithm moved from
     rdson_vs_tj.py's private `_nearest_curve_index` (same squared-distance
     comparison, same implicit tie-break: strict `<` means the first
     curve/point encountered at the minimum distance wins, never
     overwritten by a later exact tie — now made an explicit, documented
     contract instead of an implementation accident). New `Point = Tuple[
     float, float]` type alias added alongside the module's existing
     `OcrLine`. **rdson_vs_tj.py deliberately NOT touched** — still has
     its own private copy; rewiring it to import the shared version is an
     explicit separate follow-up (touches an existing file). 3 tests green,
     suite 840 passing (was 837).
  2. `src/extraction/naming/vgsth_vs_tj.py` — `count_expected_curves`
     (band-scheme role classification + current-value regex parsing
     `I_?D\s*=\s*([\d.]+)\s*(u|µ|μ|m)?A`, normalizing to µA; both u/µ/μ
     glyphs and A/mA/µA magnitudes recognized; unconditional
     duplicate-value-> `None` rule; mixed-scheme detection) and
     `name_curves_by_labels` (1 curve always `["vgsth"]`; multi-curve
     requires EVERY curve to get its own independently-resolved label —
     no elimination-completion at any curve count, a deliberate departure
     from rdson_vs_tj's 2-curve elimination trick; genuine nearest-curve
     ties refused via a new `_resolve_curve_index` wrapper built ON TOP OF
     the shared `nearest_curve_index` — reuses it for the actual nearest-
     curve search, adds its own tie-detection pass since the shared
     function's contract is "always answer deterministically," not "flag
     ties," which naming needs but a low-level geometry helper
     legitimately shouldn't provide). Both `Point`/`OcrLine` imported from
     `curve_detection.py`, not redefined. **All 30 tests passed on the
     first implementation run** (every hand-derived expectation against
     the finalized rules matched, including the input-order-alignment,
     tie, and quarantine cases). Suite **870 passing** (was 840).
- **rdson regression check: 23/23 still passing**, `git diff --stat
  src/extraction/naming/rdson_vs_tj.py` empty (confirmed untouched).
  **Nothing else touched** — verified via `git status`/`git diff --stat`
  that every other file in the working tree (the still-uncommitted T29/T30
  work from earlier sessions) has byte-identical diffs to before this
  session started.
- **Not done (explicitly out of scope, per instruction):** rewiring
  `rdson_vs_tj.py` to import the shared `nearest_curve_index` instead of
  its own private copy — separate follow-up task. No registry entry for
  `vgsth_vs_tj` added either (not asked for; `curve_registry.py`/
  `extraction_registry.py` wiring follows the same pattern rdson_vs_tj
  used, as its own later, separately-approved task, per T25's precedent).

### 2026-07-22 — Session: T31 follow-up — vgsth_vs_tj `PLAUSIBILITY_SPECS` entry + naming-registry placeholder (TDD)
- **Two owner-approved frozen-file pure additions**, resolving the two
  gaps flagged in the prior session's `classical_vgsth.py` red-phase
  report — same precedent as T28's rdson x_range addition. **Dependency
  order mattered for a clean TDD sequence, discovered while writing
  tests**: `process_detections` unconditionally calls
  `get_naming_fn(curve_type)` before anything else, so the plausibility
  tests (which call `process_detections` directly for `vgsth_vs_tj`)
  can't even reach the plausibility gate — let alone go red for a
  plausibility-specific reason — until the naming-registry entry exists.
  Implemented in dependency order (registry first, then plausibility),
  confirmed at each step.
- **1. Naming-registry placeholder** (`src/extraction/naming/__init__.py`):
  `_vgsth_vs_tj_placeholder_names(curves) -> List[str]` returns
  `["curve_0", "curve_1", ...]` aligned to input order, never raises
  (doesn't inspect point content at all, so empty curves/zero curves are
  both safe). Registered in `_NAMING_REGISTRY["vgsth_vs_tj"]` only —
  `_EXPECTED_NAMES` deliberately NOT touched (vgsth's curve count isn't
  fixed, so a fixed expected-names list doesn't apply the way it does for
  capacitance/rdson). Docstring states in caps that this must never be
  trusted as final output and that `classical_vgsth.py`'s wrapper is
  required to override it on every ok-status path — a dedicated test
  (`test_vgsth_placeholder_docstring_flags_itself_as_non_authoritative`)
  locks in that the warning language itself stays present, not just that
  the function works. **6 new tests in `tests/test_naming.py`**, red
  confirmed first (`KeyError: No naming function registered for
  curve_type 'vgsth_vs_tj'`), green after implementing.
- **2. `PLAUSIBILITY_SPECS["vgsth_vs_tj"]`** (`src/extraction/pipeline.py`):
  `{"x_range": (-75.0, 200.0)}`, rdson_vs_tj's exact bound reused verbatim
  (same axis — junction temperature — already confirmed to cover every
  real chart reviewed). Deliberately no `y_range`: real vgsth charts with
  negative y-values exist in the sample corpus and there isn't enough
  data yet to set a safe bound anywhere for this curve type. **4 tests in
  `tests/test_pipeline.py`** (1 existing data-pin test extended with a
  third assertion — same idiom already used for capacitance/rdson, not a
  new test — plus 3 new tests mirroring rdson's exact three-test pattern:
  out-of-range downgrades with the specific `implausible_calibration`
  reason, in-range keeps curves/calibration, in-range control passes
  through to `units_undetected`). **Red confirmed in two stages**,
  matching the dependency-order discovery above: before the registry fix,
  3 of the 4 failed with the registry's `KeyError` (a real but non-
  specific reason); after the registry fix, re-ran and got the correctly
  *specific* red (`assert 'implausible_calibration' in 'units_undetected'`
  — proving the plausibility gate itself, not just the naming lookup, was
  the remaining gap) before implementing.
- **New safety-net test** (`tests/test_classical_vgsth.py`, owner-
  requested addition, not part of the original 31): confirms across all
  6 of that file's existing `status == "ok"` scenarios (single curve/no
  labels, single curve/I_D= label present, 3- and 2-curve band scheme,
  2- and 4-curve current-value scheme) that no final `curve_name` ever
  matches the placeholder pattern `curve_\d+` — each scenario's mocked
  `process_detections` result is configured with the REAL placeholder's
  exact naming pattern first, so the test genuinely proves the wrapper's
  override replaces it, not just that the mock happened to use a
  different string. Still red via the same pre-existing collection error
  as the rest of that file (`classical_vgsth.py` itself remains
  unimplemented — a separate future task, unaffected by this session).
- **Confirmed pure additions**: `git diff` on both files shows only new
  lines — every existing dict entry/line byte-identical to before.
- **Full suite: 879 passing** (was 870: +6 naming, +3 pipeline; the pin
  test's extension doesn't add a test, it extends an existing one).
  Zero regressions.

### 2026-07-22 — Session: T31 follow-up — `classical_vgsth.py` implemented (green)
- **All prerequisites now in place** (naming-registry placeholder +
  `PLAUSIBILITY_SPECS["vgsth_vs_tj"]`, prior session) — implemented
  `src/extraction/classical_vgsth.py`'s `run_classical_pipeline` to make
  the full red-phase suite pass. **All 32 tests in
  `tests/test_classical_vgsth.py` passed on the first implementation
  run** (31 original scenarios + the placeholder-leak safety-net test),
  no fixture/design mismatches found — the multi-session TDD design work
  (decision tree, exact reason-string wording, mock boundaries) held up
  exactly as specified.
- **How it works:** color-then-mono detection via
  `curve_detection.py`'s shared functions (imported directly — identity-
  checked by the delegation tests, never wrapped or copied), default
  tunables throughout. `N = count_expected_curves(ocr_lines)` and
  `D = len(detections)` are compared BEFORE anything else runs — the
  actual crossing/merge safety net: `D == 0` quarantines regardless of N
  ("no curves found"); `D < N` quarantines ("likely merged at a
  crossing"); `D > N` quarantines ("stray component or missed label" —
  each reason string embeds both numbers and is substring-distinct from
  the others, checked directly by the test suite, not just by
  inspection); `N is None and D > 1` quarantines ("no usable labels").
  Only when the count is safely resolved (`D == N`, or `N is None` with
  exactly 1 curve) does it call `name_curves_by_labels` — if THAT also
  refuses (a genuine proximity tie), quarantines with a reason
  distinguishable from every count-mismatch reason above, and critically
  `process_detections` is never even called on that path (verified
  directly: `process_detections` mock asserted not-called across every
  quarantine branch). Only on success does `process_detections` (frozen
  core) run, and its result's `curve_name` fields are unconditionally
  overwritten with the real resolved names before returning — this is
  what makes the naming-registry placeholder ("curve_0", "curve_1", ...)
  safe to have registered at all.
- **Quarantine results reuse `pipeline.py`'s own `_placeholder_curves`**
  (imported, not reimplemented — same "unnamed_N" shape the frozen core's
  own early-gate failures already use) so a reviewer always sees how many
  components were found, never an empty shell; `calibration=None` since
  it's never attempted before these gates run.
- **Units**: no vgsth-specific table — the frozen core's own generic
  y-axis detector (which already recognizes "V") runs inside
  `process_detections`; `units_undetected` passes straight through
  unmodified, confirmed by test.
- **Safety-net test passes**: `TestPlaceholderNeverLeaksIntoTrustedResult`
  — every one of the file's 6 `status == "ok"` scenarios configured with
  the REAL placeholder's exact `curve_N` pattern as the mocked
  `process_detections` return, and confirmed the final result never
  carries that pattern — proving the override is unconditional on every
  ok path, not just usually-triggered.
- **Full suite: 911 passing** (was 879, +32), zero regressions. Only new
  file: `src/extraction/classical_vgsth.py`; nothing else touched this
  session (confirmed via `git status`).

### 2026-07-22 — Session: T31 follow-up — quarantine-path calibration gap found and fixed
- **Owner audit, before closing Stage 3**: checked all 5 of
  `classical_vgsth.py`'s quarantine paths (no curves found / detected<N
  crossing-merge / detected>N stray-missed-label / N-is-None+multi-curve
  no-usable-labels / naming-tie) for whether calibration gets computed.
  **Confirmed a real, uniform gap**: all 5 routed through the same
  `_quarantine()` helper, which hardcoded `calibration=None` and used
  `_placeholder_curves` (empty points) — `process_detections` was never
  called on any of them. Calibration (tick parsing) doesn't depend on
  curve naming succeeding, so there was no structural reason for this;
  rule 27 (mirroring rdson's precedent) requires every quarantined result
  to carry curves AND calibration.
- **Fixed, TDD (tests first):** updated 10 existing test assertions across
  6 test methods in `tests/test_classical_vgsth.py` — the ones asserting
  `process_detections.assert_not_called()` (now `assert_called_once()`,
  since it's now genuinely called) and one asserting
  `calibration is None` (now `is not None`) — confirmed these went red
  against the still-unfixed implementation first (10 failures, all for
  the expected reason) before touching `classical_vgsth.py`.
- **Fix**: `_quarantine()` now calls `process_detections(..., expected_curve_count=len(detections))`
  — using the ACTUAL detected count (not the mismatched N), so
  `process_detections`'s own internal count-gate is a trivial pass-through
  and it proceeds all the way through calibration. Its status/review_reason
  are always overridden by the wrapper's own verdict (it has no concept of
  the label-count mismatch or naming-tie this wrapper cares about), but its
  calibration/curves/units are kept as-is. Curve names are never
  fabricated — they stay whatever the naming-registry placeholder produced
  (`curve_0`, `curve_1`, ...), which is exactly the case that placeholder's
  own docstring already says is legitimate: a quarantined result a human
  is expected to inspect via the overlay, not trust the name of. Removed
  the now-unused `_placeholder_curves` import.
- **All 10 updated assertions pass**, plus the other 22 (including the
  placeholder-leak safety-net test, unaffected since it only covers the
  "ok" paths, which were never part of this gap). **Full suite: 911
  passing** (same count — no new tests added, existing ones strengthened),
  zero regressions.

### 2026-07-22 — Session: T31 follow-up — rdson rewired to the shared `nearest_curve_index`
- **Closes the T30-correction-era deferred item**: `rdson_vs_tj.py`'s
  private `_nearest_curve_index` replaced with an import of
  `curve_detection.py`'s shared `nearest_curve_index` (built in Stage 2,
  intentionally left unwired at the time since it touched an existing
  file). Pure refactor, no design changes.
- **Tie-break equivalence confirmed before editing, not assumed**: read
  both implementations side by side. Byte-for-byte identical core logic —
  same `best_index, best_d2 = 0, float("inf")` init, same nested
  `enumerate(curves)` / `for row, col in points` iteration order, same
  strict `d2 < best_d2` comparison. Because the comparison is strict
  (`<`, not `<=`), a later curve's EQUAL distance never overwrites an
  earlier one — the lower curve index deterministically wins any exact
  tie, in both versions identically. Signature match too:
  `(curves, cx, cy) -> int`, same parameter names. No mismatch found —
  confirmed genuinely equivalent, not just "probably fine."
  (Unsurprising: the shared version was moved verbatim FROM this exact
  function in Stage 2 — this session closes the loop rdson's own copy was
  deliberately left open at the time.)
- **Swap**: added `from src.extraction.curve_detection import
  nearest_curve_index` to `rdson_vs_tj.py`, removed the private function
  definition entirely (no leftover duplicate), updated the one call site
  (`name_curves_by_labels`). `curve_detection.py`'s own module docstring
  updated to drop the now-stale "rdson still has its own private copy"
  note.
- **Regression check: rdson's 23-test suite (`tests/test_rdson_two_curve.py`)
  passes byte-for-byte identically** — same test file, zero edits, all 23
  green post-swap (positional naming, label-anchored naming including the
  4 real-device parametrized cases, and the end-to-end classical-pipeline
  two-curve tests).
- **Full suite: 911 passing**, unchanged — zero regressions anywhere.
  `grep` confirms exactly one `nearest_curve_index` implementation exists
  repo-wide (`curve_detection.py`); `rdson_vs_tj.py` now imports it,
  confirmed via direct grep of its import line and call site.

### 2026-07-22 — Session: T31 follow-up — vgsth_vs_tj registered in both Stage-4/5 registries (TDD)
- **Flagged before proceeding, per instruction (§4 rigor)**: read
  `src/orchestrator/live_stages.py` first and found its "classical"
  dispatch is a single HARDCODED import
  (`from src.extraction.classical import run_classical_pipeline`, rdson's
  own wrapper) with no per-curve-type routing at all — confirmed there's
  even a pre-existing test
  (`tests/test_live_stages.py::test_a_future_classical_entry_routes_the_same_way_as_rdson`,
  written before `classical_vgsth.py` existed) that explicitly encoded
  this as intentional at the time ("vgsth_vs_tj... must reuse the SAME
  routing code as rdson"). **Registering `vgsth_vs_tj` as `method=
  "classical"` today does NOT make the live pipeline call
  `classical_vgsth.py` — it would silently call rdson's wrapper instead
  if exercised end-to-end.** This is now documented prominently in
  `extraction_registry.py`'s own module docstring. Out of scope for this
  task's literal ask (a registry data addition) and for the given test
  instructions (which only test `get_extraction_spec` itself, not
  `live_stages.py`'s dispatch) — did NOT touch `live_stages.py`. Wiring
  real per-curve-type classical dispatch is a separate, necessary
  follow-up before vgsth_vs_tj can actually extract for real, needing its
  own explicit approval (touches an existing file).
- **1. `src/extraction/extraction_registry.py`**: `vgsth_vs_tj` entry,
  `method="classical"`, `checkpoint=None`, `config=None` (mirrors
  rdson's exact shape). `expected_curve_count=None` (not `(1,2)`-style —
  vgsth's count genuinely isn't a fixed set, determined dynamically by
  `count_expected_curves` inside `classical_vgsth.py`; confirmed this
  field isn't even read by the classical dispatch path at all, only by
  `run_pipeline`'s model branch, so there's no functional risk either
  way, just an honesty question about the field's documented value). The
  dataclass's own field comment extended to note `None` as a legitimate
  third shape. Replaced the now-obsolete
  `test_vgsth_vs_tj_not_yet_registered_raises_keyerror` with 3 new tests.
  **19 tests total in `tests/test_extraction_registry.py`, red confirmed
  first** (`KeyError: ... Registered types: [...]`, no `vgsth_vs_tj`),
  green after implementing.
- **2. `src/classification/curve_registry.py`**: new `vgsth_vs_tj`
  `CurveTypeSpec`. Real-corpus grounding: ONE confirmed real example
  (BSC010N04LSATMA1 page 9, Infineon "Diagram" template) — the exact
  figure already embedded in `test_curve_registry_rdson.py`'s own
  end-to-end fixture as rdson's "must NOT match" distractor; imported
  from there rather than re-typed so both files stay pinned to the same
  real OCR text (`y="VGS(th) [V]"`, `x="Tj [ºC]"`, caption wrongly
  shifted to `"Typ. capacitances"` by the same Stage-3 off-by-one bug
  that shifts rdson's own caption). x-axis keywords reused VERBATIM from
  rdson_vs_tj's own T25 battle-tested list (same physical quantity —
  junction temperature — same OCR pipeline, same manglings, not
  re-derived, pinned by its own test). Deliberately NO `"capacitance"`
  negative phrase (unlike every other entry in this registry) — it would
  fire on vgsth's own real mis-shifted caption and defeat the match; axis
  keywords alone already keep a genuine capacitance chart at zero without
  it — same caption-unreliability lesson rdson's own entry already
  learned, applied here.
- **Real judgment-call finding, caught by the tests themselves, not by
  inspection**: an initial draft added `"gate threshold voltage"`
  (spelled out, JEDEC-style) as a caption_keyword/positive_phrase, for a
  hypothetical unconfirmed IR/AUIRF-style verbose template — analogous to
  rdson_vs_tj having both an IR and Infineon template.
  `test_end_to_end_page9_lineup_matches_vgsth_true_chart` (a 3-figure
  `classify_page` test using the real page-9 lineup) caught it scoring
  rdson_vs_tj's OWN true chart at 7.5 — HIGHER than the real vgsth
  chart's 7.0 — because on this exact real page, `"gate threshold
  voltage"` is precisely the caption text wrongly shifted onto rdson's
  figure (not vgsth's). **Reverted**: only the one corpus-confirmed
  signal (`"vgs(th)"`) remains across caption/axis/positive-phrase.
  Documented in both the registry entry's own comment and the test file's
  module docstring so it isn't re-added without re-verifying against a
  real example first.
- **Cross-type contamination explicitly checked, not assumed**: `"VGS(th)"`
  contains `"vgs"` as a substring — id_vs_vgs's own x-axis keyword — a
  genuine risk, not a trivial one. Confirmed safe: the real vgsth chart
  scores negative-to-low against `capacitance_vs_vds`, `id_vs_vgs`, and
  `rdson_vs_tj`'s specs, and (the harder direction) rdson's own two real
  true charts and reference capacitance/id_vs_vgs charts all score well
  below threshold against the new vgsth spec.
- **14 new tests in `tests/test_curve_registry_vgsth.py`**, red confirmed
  first (`KeyError`, unregistered) — 3 of the 14 (checking OTHER curve
  types' EXISTING specs against the real vgsth chart) passed even in red,
  independently confirming the cross-contamination risk was already
  covered from the other curve types' side before this entry existed.
  11 genuinely new-signal tests green after implementing (14 total, since
  the caption-keyword revert didn't remove any test, just changed which
  assertions in the already-written suite passed).
- **Confirmed pure additions**: `git diff` — `extraction_registry.py` has
  zero data-entry lines removed (only doc-comment rewording);
  `curve_registry.py` is 100% pure insertion, ZERO lines removed at all.
  `capacitance_vs_vds`/`id_vs_vgs`/`rdson_vs_tj` byte-identical in both
  files, confirmed directly, not inferred. rdson's own 21-test suite
  (`test_curve_registry_rdson.py`) re-run and unaffected.
- **Full suite: 927 passing** (was 911, +16: +2 net in
  `test_extraction_registry.py` [-1 obsolete, +3 new], +14 in the new
  `test_curve_registry_vgsth.py`), zero regressions.

### 2026-07-22 — Session: T31 follow-up — classical-dispatch routing fix (TDD, closes the live_stages.py gap flagged two sessions ago)
- **Fixes the real bug flagged when vgsth_vs_tj was first registered in
  `extraction_registry.py`**: `live_stages.py` hardcoded a single
  top-level `from src.extraction.classical import run_classical_pipeline`
  and called it for EVERY `method="classical"` curve type regardless of
  which one — meaning vgsth_vs_tj would have silently run rdson_vs_tj's
  wrapper if ever exercised end-to-end.
- **Circular-import check done first, not assumed** (per instruction):
  traced the full transitive closure of `classical.py`'s and
  `classical_vgsth.py`'s imports down to leaves — confirmed neither
  imports `extraction_registry`, `live_stages`, or `orchestrator.pipeline`
  at any depth; `grep -rln extraction_registry src/` found only
  `live_stages.py` as a current importer. No cycle exists or would be
  created by `extraction_registry.py` importing `classical.py`/
  `classical_vgsth.py` at module level. Confirmed safe before writing
  any test.
- **Red phase**: `ExtractionSpec` gained `classical_pipeline:
  Optional[Callable] = None`. 6 new tests in `test_extraction_registry.py`
  (5 red on `AttributeError`/`TypeError: unexpected keyword argument`,
  1 — the "imports cleanly" guard — correctly green both before and
  after). Retired `test_a_future_classical_entry_routes_the_same_way_as_
  rdson` (had encoded the bug as an intentional assumption, written
  before `classical_vgsth.py` existed), replaced with
  `test_vgsth_and_rdson_route_to_independent_classical_functions` in a
  new `TestClassicalDispatchRoutingFix` class (7 tests, 6 red, 1 —
  "no extraction_registry reference in classical modules" — correctly
  green both before and after). 939 collected, 11 failed for the
  intended reason, 928 passed, zero unexpected breakage.
- **Green phase, three steps, full suite after each:**
  1. `ExtractionSpec.classical_pipeline` set to `classical.run_classical_pipeline`
     for rdson_vs_tj, `classical_vgsth.run_classical_pipeline` for
     vgsth_vs_tj, left `None` for model entries. **25/25 in
     `test_extraction_registry.py` green**; full suite 934 passing (was
     928) — exactly the 6 extraction-registry-level tests flipped, the 5
     live_stages-level tests (which need live_stages.py itself) correctly
     still red.
  2. `live_stages.py`: removed the hardcoded import entirely; classical
     dispatch now calls `spec.classical_pipeline(...)` directly, with an
     explicit `ValueError` (mentioning "classical_pipeline" by name) if a
     `method="classical"` entry has none configured — never a silent
     fallthrough. Discovered immediately that `build_adapter`
     (test_live_stages.py's shared fixture helper) still tried to
     monkeypatch the now-gone `live_stages_mod.run_classical_pipeline`
     attribute — fixed it to inject `classical_pipeline` onto WHATEVER
     registry entry is active for the adapter's `curve_type` (real or
     already-faked by the calling test) via `dataclasses.replace`
     (preserves every other field), only when that entry's method is
     "classical". This single shared-helper fix transparently propagated
     to every test using `build_adapter`.
  3. **The 3 named `TestExtractionRouting` tests
     (`test_rdson_routes_to_classical_never_touches_model`,
     `test_model_routed_curve_types_call_run_pipeline`,
     `test_routing_is_data_driven_not_hardcoded_if_elif`) needed ZERO
     body edits** — fixing `build_adapter` alone made all 3 pass again,
     preserving their exact original intent under the new mechanism:
     - `test_rdson_routes_to_classical_never_touches_model`: `mocks[
       "run_classical_pipeline"]` is still the function actually invoked
       for rdson_vs_tj — now wired via `spec.classical_pipeline` instead
       of a hardcoded name, same observable behavior, same assertions.
     - `test_model_routed_curve_types_call_run_pipeline`: capacitance_vs_vds/
       id_vs_vgs are model-routed, so `build_adapter`'s classical-injection
       is skipped for them (method != "classical") — the classical mock
       stays un-wired-in and genuinely never gets called, exactly as before.
     - `test_routing_is_data_driven_not_hardcoded_if_elif`: its own
       `monkeypatch.setitem`-injected `fake_classical_type`/
       `fake_model_type` entries are picked up by `build_adapter`'s
       `_REGISTRY.get(curve_type)` lookup at call time — the fake
       classical entry gets its `classical_pipeline` field replaced
       (via `dataclasses.replace`, so its OTHER fields stay exactly as
       that test set them) and correctly gets called; the fake model
       entry is left alone — still proving the routing decision is a
       genuine dict lookup, zero curve-type-specific adapter code, now
       through the real mechanism instead of the old hardcoded one.
     4 of my own new `TestClassicalDispatchRoutingFix` tests needed
     adjusting once `build_adapter` itself was fixed (they'd pre-injected
     their OWN mocks, which `build_adapter`'s now-correct logic
     immediately overwrote) — simplified to use `build_adapter`'s own
     mocks directly, consistent with the rest of this file's convention.
  - **One genuine regression found outside the 15 planned scenarios**:
    `TestErrorIsolationBatchSafety::test_malformed_ocr_input_for_one_
    device_does_not_crash_others` also directly patched the now-gone
    attribute. Confirmed it was vestigial — that test only exercises
    `run_classification`, never `run_extraction` — so the line was dead
    setup; removed it, zero coverage lost.
- **Full suite: 939 passing** (was 927 pre-red-phase; net +12 tests
  across both files). Confirmed directly (not just "tests pass"):
  `grep` for `live_stages_mod.run_classical_pipeline` and the old
  hardcoded import string across `src/`+`tests/` returns nothing except
  the test that asserts its ABSENCE and a docstring comment describing
  the historical bug. Live registry check: `get_extraction_spec(
  "rdson_vs_tj").classical_pipeline is classical.run_classical_pipeline`
  → True; `get_extraction_spec("vgsth_vs_tj").classical_pipeline is
  classical_vgsth.run_classical_pipeline` → True; the two are NOT the
  same object; capacitance's is `None`. vgsth_vs_tj now genuinely routes
  to its own function when exercised — no longer a documented gap.

### 2026-07-22 — Session: T32 — if_vs_vsd registered end-to-end (classification + extraction), TDD throughout
- **Scope**: `if_vs_vsd` (body_diode) goes from completely unregistered
  (raised `KeyError` in both `curve_registry.py` and
  `extraction_registry.py`) to genuinely reachable end-to-end:
  classification finds it, extraction routes to its OWN wrapper (not a
  generic fallback). Owner-dictated task; confirmed the plan (including
  the one real behavioral change, `live_stages.py`'s new
  `model_pipeline` dispatch branch) before touching any existing file.
- **`src/extraction/naming/if_vs_vsd.py`** (new): `count_expected_curves`
  parses "25°C"/"175°C" (bare) and "TJ = 25°C" (prefixed) labels,
  normalizing degree-sign/spacing variants; returns `None` on no labels,
  any duplicate normalized value (same unconditional DUPLICATE-VALUE RULE
  as vgsth_vs_tj), or a COMPOUND label (temperature + percentile, e.g.
  "150°C, 98%" — owner rule: never parsed, always ambiguous).
  `name_curves_by_labels`: 1 curve → `["if"]` always; multi-curve named
  `if_25C`/`if_175C` etc., every curve independently resolved (no
  elimination-completion) or `None`. **Core design point**: label
  anchoring restricts each curve's candidate points to its own
  LOW-V_SD-region subset (`LOW_VSD_REGION_FRAC = 0.25` of its own
  pixel-column span) before nearest-point matching — never a whole-curve
  average or whole-curve search — because every real if_vs_vsd chart
  converges/crosses at high current, and a curve's high-V_SD points can
  end up geometrically closer to a label anchored to a DIFFERENT curve's
  low-V_SD segment than that curve's own low-V_SD points are. Proved with
  a hand-computed, exact (no floating-point ambiguity) adversarial fixture
  in `TestNameCurvesLowVsdRegionRestriction`: two curves crossing between
  their own cols 60→348, where an UNRESTRICTED whole-curve nearest-point
  search would pick the wrong curve (distance ≈2.8 vs ≈13.15) but the
  RESTRICTED (low-region-only) search correctly picks the right one
  (≈19.3 vs ≈22.2) — verified by hand before writing any implementation,
  then confirmed by the passing test. 31 tests, all green first run.
- **`src/extraction/model_if_vsd.py`** (new): the model-path analogue of
  `classical_vgsth.py` — built on `run_inference` (the same model-inference
  primitive `run_pipeline` itself calls) + the frozen `process_detections`,
  never reimplementing either. Same D-vs-N quarantine ladder already
  proven for vgsth_vs_tj (D==0 → "no curves found"; N given and D<N →
  "crossing"; D>N → "stray/missed label"; N is None and D>1 → "no usable
  labels"; matched but naming ties → "ambiguous naming"). Every quarantine
  path still calls `process_detections(..., expected_curve_count=len(detections))`
  first so calibration/curves/units are always real, never an empty shell
  (the same gap caught and fixed for vgsth_vs_tj, not reintroduced here).
  On any resolved path, curve names are unconditionally overridden with
  `name_curves_by_labels`'s real answer. 25 tests, all green first run.
- **`extraction_registry.py`**: added `model_pipeline: Optional[Callable] = None`
  to `ExtractionSpec` (the "model"-dispatch analogue of `classical_pipeline`),
  and the `if_vs_vsd` entry (`method="model"`, real checkpoint/config,
  `expected_curve_count=(2, 4)` — informational only, like vgsth_vs_tj's
  own `None`: the wrapper derives its real expected count dynamically via
  `count_expected_curves`, never reads this static field). Retired
  `test_if_vs_vsd_not_yet_registered_raises_keyerror` (now factually
  wrong), replaced with 13 new tests. 37/37 green.
- **`naming/__init__.py`**: added `_if_vs_vsd_placeholder_names`, byte-for-byte
  the same disposable "curve_N" pattern already built for vgsth_vs_tj — no
  `_EXPECTED_NAMES` entry (names aren't fixed, same as vgsth_vs_tj). 7 new
  tests in `test_naming.py`, all green.
- **`curve_registry.py`**: new `if_vs_vsd` entry using the owner-confirmed
  phrase set ("forward characteristics"/"reverse diode" captions, I_F/I_SD
  y-axis, V_SD x-axis). **Flagged honestly**: unlike rdson_vs_tj's
  50-device survey or even vgsth_vs_tj's one real embedded fixture, no
  real OCR text was available in this session to build an end-to-end
  fixture from — the test fixture is synthetic-but-confirmed-phrasing,
  flagged in both the registry comment and the test file's own docstring
  for a sanity-check against a real figure once available. Explicit
  cross-contamination check: `"vsd"` (this entry's x-axis token) vs
  `"vds"` (capacitance_vs_vds's own) are the same three letters reversed —
  confirmed as genuinely distinct literal strings, not assumed. 15/15
  green first run, including both-direction checks against every other
  registered curve type (capacitance, id_vs_vgs, rdson_vs_tj, vgsth_vs_tj,
  plus a gate-charge reference figure).
- **`live_stages.py`**: the one real behavioral change, confirmed with the
  owner before editing. `run_extraction`'s `method == "model"` branch now
  checks `spec.model_pipeline` after lazy-loading the model: if set, calls
  it with the same kwargs `run_pipeline` would get; otherwise falls
  through to `run_pipeline` exactly as before (byte-identical `else`
  branch — confirmed via `git diff`, zero effect on capacitance_vs_vds/
  id_vs_vgs, whose `model_pipeline` stays `None`). Mirrors the
  `classical_pipeline` routing fix from the prior session, one dispatch
  branch over. `build_adapter`'s test helper extended symmetrically
  (injects a mock over whatever registry entry is active, only when
  `method=="model"` and `model_pipeline` is already set — existing
  capacitance/id_vs_vgs tests provably unaffected). 6 new tests
  (`TestModelDispatchRoutingAddition`), including a data-driven-dispatch
  proof (fake curve type, zero curve-type-specific adapter code) and a
  static-source check that neither `"if_vs_vsd"` nor `"model_if_vsd"`
  appears in `run_extraction`'s source. 69/69 green in `test_live_stages.py`.
- **Full suite: 1036 passing** (was 939), zero regressions. Confirmed
  directly: `git diff --stat` on all 4 touched existing files reviewed by
  hand — `naming/__init__.py`/`curve_registry.py` are pure insertions,
  zero lines removed; `extraction_registry.py`'s only removals are
  docstring rewording (data entries untouched, confirmed byte-identical);
  `live_stages.py`'s only removal is the old single-branch model call,
  replaced by an `if/else` whose `else` body is byte-identical to it.
  Placeholder-leak safety net holds (`TestPlaceholderNeverLeaksIntoTrustedResult`
  in both `test_model_if_vsd.py` and `test_naming.py`); every quarantine
  path in `model_if_vsd.py` carries real calibration (explicit assertions
  across all quarantine test classes); if_vs_vsd is now genuinely
  reachable end-to-end — classification matches it via its own registry
  entry, extraction routes to `model_if_vsd.run_model_pipeline` by
  identity (not `run_pipeline`), confirmed both via unit tests and a
  direct `get_extraction_spec("if_vs_vsd").model_pipeline is
  model_if_vsd.run_model_pipeline` check.

- **Same-session correction, caught by owner review**: the
  `expected_curve_count=(2, 4)` set above was vestigial —
  `model_if_vsd.run_model_pipeline` accepts the parameter for
  call-signature symmetry only and never reads it (confirmed by grep:
  the bare `expected_curve_count` identifier appears only at its own
  parameter declaration; every `process_detections(...)` call passes the
  locally computed detected count instead). A static tuple sitting in the
  registry when nothing checks it against looks like an enforced rule
  when it isn't — same lesson already applied to vgsth_vs_tj's own
  `expected_curve_count=None`. Changed to `None`, comment updated to
  document the grep-confirmed fact directly rather than merely assert it.
  Retired `test_if_vs_vsd_expected_curve_count_allows_two_or_four`
  (tested the now-wrong tuple), replaced with
  `test_if_vs_vsd_curve_count_is_not_a_fixed_set` (mirrors vgsth_vs_tj's
  own `test_vgsth_vs_tj_curve_count_is_not_a_fixed_set` exactly). Full
  suite re-run: still 1036 passing, zero regressions.

### 2026-07-23 — Session: T33 — dual-side axis-tick calibration (TDD, resolves ticks.py's own caveat #1, additively)

- **Scope**: `src/calibration/ticks.py`'s documented caveat #1 ("Tick
  zoning... assumes a bottom x-axis and left y-axis — right-hand or dual
  axes will mis-bucket") is now partially resolved: the opposite side (top
  30% for x, right 30% for y) is collected and independently fit; whichever
  side (default or opposite) produces more RANSAC inliers wins, ties favor
  default. Design approved by owner beforehand; full docstring/history of
  `ticks.py` read first (CLAUDE.md §2/§4), 22 scenarios (A1–H22) written
  RED FIRST in a dedicated new file, confirmed failing for the right reason
  (missing capability, not fixture bugs) before any implementation, per
  explicit instruction — see the prior session turn's red-phase report.
- **Non-negotiable bar met**: `parse_numeric_ticks` and `fit_axis` are
  **byte-identical** to before this task — confirmed by extracting both
  function bodies from `git show HEAD:...` and the current file and diffing
  them directly (not just eyeballing `git diff`), zero difference either.
  This is what makes the change genuinely additive rather than a rewrite of
  two frozen, heavily-tested (58-test) functions.
- **New code** (both new sibling functions, called only from
  `derive_calibration`'s now-updated internals):
  - `_parse_opposite_side_ticks` (private) + `parse_numeric_ticks_dual_side`
    (public): mirrors `parse_numeric_ticks`'s per-line parsing/exponent-
    repair/stray-zero-drop pipeline to the top/right zones. Deliberately a
    SEPARATE implementation, not a refactor extracting shared logic out of
    `parse_numeric_ticks` — the byte-identical requirement on the original
    forces some structural duplication, documented explicitly in the new
    function's own docstring so it doesn't read as an oversight. No
    tight-corner mirroring for the opposite side (out of scope, no
    analogous "meaningful corner" at top-right the way bottom-left has
    one as the plot origin) — deliberately noted, not silently omitted.
  - `fit_axis_dual_side`: runs the unchanged `fit_axis` on each side
    independently, selects by inlier count (`len(used)`), ties favor
    default, logs distinctly (`"non-default (opposite) side won"`) only
    when a non-default side actually wins — confirmed silent otherwise
    (H22).
  - `derive_calibration`: internals only — signature/return shape
    unchanged; now calls the two new functions instead of
    `parse_numeric_ticks`/`fit_axis` directly.
- **Docstring gap added as instructed**: `detect_y_axis_units` only scans
  the left y-zone even though a right-side y-axis can now win — written
  into both the module docstring (new caveat #7) and the function's own
  docstring, explicitly flagged as NOT fixed in this task (no real chart
  observed needs it yet).
- **Red-to-green bugs found were all in the NEW test file, not the
  implementation**: 4 of the 22 red tests initially failed post-
  implementation because their own "expected value" was computed from the
  raw, unshifted fixture tuples instead of the actual placed/offset OCR
  coordinates (a uniform pixel shift changes intercept, not slope — the
  tests were comparing intercepts against the wrong reference). Fixed by
  deriving expected values from `parse_numeric_ticks_dual_side` on the
  REAL constructed lines (mirroring how the A1 backward-compat test was
  already correctly written) rather than the bare fixture. One test (F18)
  also had a "grounding" assertion pinning PRE-implementation behavior
  (`derive_calibration(...) is None`) that necessarily became false once
  the fix landed — that's the fix working as intended, not a regression;
  removed the now-obsolete assertion, kept the historical framing as a
  comment.
- **Full suite: 1061 passing** (was 1039 in red phase: 1036 + 3 pin tests
  correctly green already; +22 flipped from red to green, 0 newly broken).
  `test_ticks.py` alone: 58/58, unchanged. Every downstream consumer
  (`test_classical.py`, `test_classical_vgsth.py`, `test_model_if_vsd.py`,
  `test_pipeline.py`, `test_live_stages.py`, all three curve-registry real-
  device suites, `test_orchestrator.py`): 264/264, zero edits needed to any
  of them — confirms `derive_calibration`'s unchanged signature/shape was
  the right backward-compat boundary to hold the line at.
