# SETUP.md — Environment & Technology

## Purpose

This repo rebuilds the datasheet **curve-extractor pipeline**: given a CSV of devices, it
downloads datasheet PDFs, runs Azure OCR to crop figure PNGs and read axis-tick text,
classifies the curves in each figure, extracts the curve traces (a classical OpenCV path and a
LineFormer deep-learning path), renders overlays for visual review, validates the results, and
writes the final per-device curve data to MongoDB. Stages 1–3 are imported from the legacy repo
as-is; stages 4–7 are rebuilt here under strict TDD and the rules in `CLAUDE.md`.

## Environment

- **Editor:** VS Code + Claude Code extension; **Remote-SSH to an AWS GPU instance** for
  training and GPU inference work.
- **Pipeline:** Python **3.10+** in a dedicated virtualenv (`.venv` at the repo root).
- **Training (LineFormer/MMDetection):** a **separate conda env**. The legacy training stack
  used Python **3.8**, **pytorch 1.13.1 cu117**, **mmcv-full installed via openmim**, and
  **scipy 1.9.3**.
  ⚠ **The pipeline venv and the training env CANNOT be shared** — the training stack pins
  scipy 1.9.3, which conflicts with the pipeline's scipy. Keep them isolated; never install
  MMDetection into the pipeline venv.

## Technologies (and WHY each is used)

| Technology | Why |
|---|---|
| **OpenCV** | Classical curve tracing and mask operations (stage 5 classical path). |
| **LineFormer / Mask2Former via MMDetection** | Instance segmentation for the hard cases the classical path can't do: overlapping curves, same-color curves, banded (thick) curves. |
| **shapely** | Polyline buffering — turns annotated polylines into polygon masks for training-data generation (T2 converter). |
| **Azure Document Intelligence** | OCR for axis tick labels — stage 3, imported from legacy. |
| **CVAT.ai cloud** | Annotation tool. Convention: **polyline** for thin curves, **polygon** for thick bands; label `line` with attribute `curve_name`. Export **ONLY as "CVAT for images 1.1" XML** — CVAT's own COCO exporter **silently drops polylines**, so we convert XML→COCO ourselves (see `src/dataset_tools/cvat_to_coco.py`). |
| **MongoDB** | Final storage of validated curve data (stage 7). |
| **pytest** | Strict TDD (see `CLAUDE.md` §2). |
| **numpy / scipy / matplotlib** | Numeric work, calibration fitting, overlay rendering. |
| **scikit-image** | Mask skeletonization (stage 5: predicted curve masks -> single-pixel-wide centerlines before pixel-to-data conversion). |
| **pycocotools** | Mask format decoding (RLE/ndarray -> boolean mask) shared by eval, CVAT pre-annotation, and stage-5 inference. |

## Setup steps

```bash
git clone <repo-url>
cd <repo>
python -m venv .venv
# Windows: .venv\Scripts\activate     POSIX: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill in real values — NEVER commit .env
```

Run the tests:

```bash
pytest -v
```

Run tools as they appear (each tool documents its own CLI; current tools):

```bash
# CVAT XML → COCO converter (T2); multiple inputs are merged (T3):
# ids renumbered, unannotated images dropped, duplicate file_names hard-error.
python -m src.dataset_tools.cvat_to_coco <cvat_export.xml> [<more.xml> ...] <output_coco.json> [--buffer-px N]
```

⚠ The pilot export (`tests/fixtures/cvat/pilot7_job4199936.xml`) is a **converter test
fixture only** — it duplicates 7 images of task 4200303 and must NEVER be merged into
training data. The merge's duplicate-file_name hard error enforces this.

**Buffer radius decision (owner, 2026-07-07, T4 review):** the polyline buffer
radius is **4.5 px** (mask ≈ 9 px thick) and is the converter default.
Rationale: annotation polylines jitter 1–2 px off the stroke center, and at
2.0–3.0 px the mask visibly under-covered the stroke in jittery sections;
under-coverage hurts training more than a slightly wide mask. `--buffer-px`
remains overridable for experiments, but the canonical training COCO
(`data/coco/batch_merged.json`) is built at 4.5 px.

```bash
# Overlay visual check of buffered masks (T4) — renders sampled annotations
# over source figures for buffer-width review:
python -m src.dataset_tools.overlay_check <coco.json> <output_dir> \
    --images-dir <dir> [--images-dir <dir2> ...] [--n 6] [--seed 42]
# Source figures are NOT in the repo — collect them first (see below); then:
#   --images-dir data/images
```

```bash
# Collect annotated source images from legacy trees into data/images/ (T4a).
# Read-only search; hash-verifies duplicate finds; exits non-zero unless the
# COCO's full image set was recovered:
python -m src.dataset_tools.collect_images data/coco/batch_merged.json data/images \
    "D:/Extractor/data" "D:/LineFormerDataset_v2"
```

## Training environment (T6 — AWS g4dn.xlarge, T4 16 GB)

Fully scripted; never hand-run ad-hoc installs. On the GPU box, from the repo root:

```bash
bash scripts/setup_training_env.sh      # idempotent env creation + clone + weights
bash scripts/verify_training_env.sh    # smoke tests 1-3 must pass; see fp32 note below
```

### Data volume (`/mnt/data`) — required on this box

The default 25 GB root EBS volume cannot hold `python + torch + cuda + mmcv-full
+ mmdet` (setup hit `ENOSPC` mid-install at 94% root usage the first time). A
second **116 GB EBS volume** is attached, formatted XFS, and mounted at
`/mnt/data` (owner decision, 2026-07-08) — **root is left untouched**, only the
heavy stuff moves:

| What | Where |
|---|---|
| Root filesystem | `/dev/nvme0n1p1`, 25 GB — code/scripts only |
| Data volume | `/dev/nvme1n1`, 116 GB, XFS, mounted at `/mnt/data`, persisted via UUID in `/etc/fstab` (`nofail` so a missing volume never blocks boot) |
| conda envs | `/mnt/data/conda-envs` (`conda config --add envs_dirs` + `CONDA_ENVS_PATH`, both set) |
| conda package cache | `/mnt/data/conda-pkgs` (`conda config --add pkgs_dirs` + `CONDA_PKGS_DIRS`) |
| Pretrained checkpoint | actually stored at `/mnt/data/my-datasheet/weights/`; `data/weights/` in the repo is a **symlink** to it (transparent to every script/path in this repo) |
| Future training checkpoint output | `/mnt/data/my-datasheet/checkpoints/` (created, currently empty — point any `work_dir` here, never at a path under the repo/root) |

`setup_training_env.sh` sets `CONDA_ENVS_PATH`/`CONDA_PKGS_DIRS` automatically
when `/mnt/data` exists (no-op otherwise, e.g. a fresh box before the volume is
attached). If this box is ever rebuilt: attach a volume of similar size,
`mkfs.xfs` it, mount at `/mnt/data`, add the same `/etc/fstab` line (by UUID),
recreate `/mnt/data/conda-envs`, `/mnt/data/conda-pkgs`,
`/mnt/data/my-datasheet/{weights,checkpoints}`, then re-point the
`data/weights` symlink — then the scripts work unchanged.

### fp32-only training decision (owner, 2026-07-08)

Smoke test 4 (fp16/AMP forward pass) fails with:
```
RuntimeError: "ms_deform_attn_forward_cuda" not implemented for 'Half'
```
`mmcv-full==1.7.1`'s deformable-attention CUDA op (`MultiScaleDeformableAttnFunction`,
used by LineFormer's Mask2Former-style pixel decoder) only has a float32 CUDA
kernel — no half-precision variant exists to install or pin around. This is an
architecture/op-build limitation, not an environment misconfiguration.

**Decision: skip AMP entirely, train fp32-only.** Justification: smoke test 3
(real inference, batch size 1, fp32) measured only **~1 GiB peak GPU memory**
on the T4's 15 GiB — fp32 has ample headroom even at larger batch sizes, so
there is no memory-pressure reason to chase fp16. **Do not wrap the model or
training loop in `torch.cuda.amp.autocast()` or use a `GradScaler`.**
`scripts/verify_training_env.sh` reports this test as N/A rather than a
failing gate.

Conda env `lineformer` — pins and **why each exists** (owner-approved
legacy-compatible stack; do not change without approval — a wrong mmcv/torch
pairing produces silent garbage, not errors):

| Pin | Why |
|---|---|
| `python=3.8` | LineFormer/MMDetection 2.x era; newer pythons break mmcv-full 1.7 wheels |
| `pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7` | last torch 1.x line with prebuilt mmcv-full CUDA ops; cu117 matches the T4 driver on the box |
| `mmcv-full==1.7.1` via **openmim** | openmim selects the wheel built for exactly torch 1.13/cu117; plain pip installs a mismatched/CPU build whose CUDA ops fail (`from mmcv.ops import RoIAlign` is the canary) |
| `mmdet==2.28.2` | last MMDetection 2.x compatible with mmcv-full 1.7.x and the LineFormer configs |
| `scipy==1.9.3` | known-good legacy pairing; **conflicts with the pipeline venv's scipy — the two environments must NEVER be shared** |

Reproducibility artifacts (committed under `envs/`):
- `envs/lineformer.lock.yml` — `conda env export` of the working env
- `envs/lineformer.commit` — exact LineFormer commit hash
- `envs/lineformer_checkpoint.sha256` — pretrained-weights hash

LineFormer source: cloned by the setup script into `third_party/lineformer`
(git-ignored). **Never copy from `D:\LineFormerModel`** — the legacy tree is
reference-only.

**Pinned provenance (verified 2026-07-08):**
- LineFormer commit: `7952e27b4653dea025394618fbd655f41d82ab6b`
  (github.com/TheJaeLal/LineFormer, also in `envs/lineformer.commit`; the
  setup script checks out this exact commit and aborts if it can't).
- Pretrained checkpoint → `data/weights/lineformer_pretrained_official_iter3000.pth`
  (570 MB, git-ignored). Source: **README section "Inference", step 1** —
  Google Drive folder
  `https://drive.google.com/drive/folders/1K_zLZwgoUIAJtfjwfCU5Nv33k17R0O5T`
  (file id `1cIWM7lTisd1GajDR98IymDssvvLAKH1n`, fetched via gdown). The file
  is named `iter_3000.pth` at the source; we rename it on arrival — see the
  disambiguation warning below.
- Checkpoint sha256:
  `ac03d7d52a11ce253350bf4bc73416e42ac68021c00bcce14d47fcc28ec65eb0`
  (also in `envs/lineformer_checkpoint.sha256`; the setup script verifies the
  hash after every download/copy and aborts on mismatch — this also catches
  transit corruption when the file is scp'd rather than freshly downloaded).

> ⚠ **Do not confuse this with the project's own old checkpoint.** The legacy
> tree (`D:\LineFormerModel\LineFormer\models\iter_3000.pth`) contains a
> *different*, project-specific fine-tuned checkpoint that **happens to share
> the exact same filename** `iter_3000.pth`. That legacy checkpoint is
> reference-only per `CLAUDE.md` §6 (untrusted, never copied blindly) and its
> quality has not been assessed here. The file in this repo,
> `lineformer_pretrained_official_iter3000.pth`, is always the **official
> upstream pretrained weights** from the README link above — the descriptive
> filename and the sha256 pin are the guard against the two ever being mixed
> up.

## Data layout

| What | Where |
|---|---|
| Figure PNGs (collected annotated set, flat) | `data/images/` (git-ignored; rebuilt by `collect_images`) |
| CVAT XML exports | `data/cvat_exports/` (git-ignored) |
| Converted COCO files | `data/coco/` (git-ignored) |
| Test fixtures (small, committed) | `tests/fixtures/` (CVAT samples in `tests/fixtures/cvat/`) |
| Logs | `logs/` (git-ignored) |

## Standing rule

**Keep this file updated whenever a new dependency or tool is added.**
