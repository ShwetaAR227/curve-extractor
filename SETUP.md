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
| **CVAT.ai cloud** | Annotation tool. Convention: **polyline** for thin curves, **polygon** for thick bands; label `line` with attribute `curve_name`. Export **ONLY as "CVAT for images 1.1" XML** — CVAT's own COCO exporter **silently drops polylines**, so we convert XML→COCO ourselves (see `src/cvat_to_coco.py`). |
| **MongoDB** | Final storage of validated curve data (stage 7). |
| **pytest** | Strict TDD (see `CLAUDE.md` §2). |
| **numpy / scipy / matplotlib** | Numeric work, calibration fitting, overlay rendering. |

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
python -m src.cvat_to_coco <cvat_export.xml> [<more.xml> ...] <output_coco.json> [--buffer-px N]
```

⚠ The pilot export (`tests/fixtures/cvat/pilot7_job4199936.xml`) is a **converter test
fixture only** — it duplicates 7 images of task 4200303 and must NEVER be merged into
training data. The merge's duplicate-file_name hard error enforces this.

```bash
# Overlay visual check of buffered masks (T4) — renders sampled annotations
# over source figures for buffer-width review:
python -m src.dataset_tools.overlay_check <coco.json> <output_dir> \
    --images-dir <dir> [--images-dir <dir2> ...] [--n 6] [--seed 42]
# Source figures are NOT in the repo. Known local locations:
#   D:/datasheet/datasheet-studio-v2/data/cache/<DEVICE>/figures/  (160/164 covered)
#   D:/images/  (flat, single-underscore names; 35/164)
```

## Data layout

| What | Where |
|---|---|
| Figure PNGs (OCR crops) | `data/figures/` (git-ignored) |
| CVAT XML exports | `data/cvat_exports/` (git-ignored) |
| Converted COCO files | `data/coco/` (git-ignored) |
| Test fixtures (small, committed) | `tests/fixtures/` (CVAT samples in `tests/fixtures/cvat/`) |
| Logs | `logs/` (git-ignored) |

## Standing rule

**Keep this file updated whenever a new dependency or tool is added.**
