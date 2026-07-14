# CLAUDE.md — Curve Extractor Rebuild: Project Constitution

> **Claude Code MUST read this file at the start of every session and follow every rule.
> If a rule must be broken, STOP and ask the project owner for explicit permission first.**

## 1. Project Overview

Pipeline:

```
CSV → [2] PDF download → [3] Azure OCR (figure PNGs + tick text) → [4] classify_curves
    → [5] curve extraction (classical OpenCV path + LineFormer deep-learning path)
    → [6] visual review (overlay + HTML gallery) → [7] orchestrator/validation → MongoDB
```

- **Stages 1–3** are imported from the legacy repo **as-is**.
- **Stages 4–7** are being **REBUILT** in this repo.

**Stage-5 hard rules:**
- Calibration **reuses** `parse_numeric_ticks()` + `fit_axis()` from legacy `cv_curve_extract.py`
  (4-tuple return contract: `slope, intercept, used, is_log`) — **never reinvent calibration**.
- The LineFormer path **must** have NMS/dedup + a confidence threshold.
- Output schema is **always keyed by `curve_type`**; an empty `curve_type` is a **write-time error**.

**In-scope curve types (owner-approved, 2026-07-08) — the full scope of stage 4's
classifier and stages 5–7's extraction/review/orchestration logic:**

| # | curve_type | Description | Status |
|---|---|---|---|
| 1 | `capacitance_vs_vds` | Ciss/Coss/Crss vs. drain-source voltage | ✅ **DONE** — Run A, production checkpoint, mAP@50 0.88 |
| 2 | `rdson_vs_tj` | on-resistance vs. junction temperature | 🔨 classical extraction built, pending real-data testing (T24/T25) |
| 3 | `if_vs_vsd` (body_diode) | forward current vs. source-drain voltage | not started |
| 4 | `id_vs_vgs` (transfer_char) | drain current vs. gate-source voltage | 🔨 in progress — Run 3 production checkpoint set, mAP@50 0.74 |
| 5 | `vgs_vs_qg` (gate_charge) | gate-source voltage vs. gate charge | not started |
| 6 | `vgsth_vs_tj` | gate threshold voltage vs. junction temperature | not started |
| 7 | `zth_vs_time` (thermal_impedance) | thermal impedance vs. time | not started |

Other curve-type folders exist in the legacy/local corpus (`avalanche_energy`,
`breakdown_voltage`, `derating`, `output_char`, `soa`, `irrm_vs_didt`,
`qrr_vs_didt`, `switching_energy`) — these are **OUT OF SCOPE** for this
project; do not build classifier/extraction/training support for them
without explicit owner approval first (this is a scope change, same rule as
§4 Pipeline Integrity).

## 2. STRICT TDD

- **Red → Green → Refactor, no exceptions.** Tests are written **BEFORE** implementation.
- Minimum **10–15 test cases per module**.
- The full suite (`pytest -v`) must pass before **any** commit or task completion.
- **Never** skip, weaken, or delete a test to make it pass.
- Every later bug gets a **regression test first**, then the fix.
- Tests live in `tests/` mirroring `src/`; fixtures in `tests/fixtures/`.
- Unit tests need **no GPU and no network** (mock heavy dependencies).

## 3. Architecture & Code Quality

- Follow `ARCHITECTURE.md` once it exists; **do not restructure without permission**.
- **Zero duplication** — call existing functions; shared code lives in `src/common/`.
- One module = one concern.
- All config in one place. **No magic numbers, no hardcoded paths**
  (LEGACY LESSON: the old repo died partly from `/mnt/c/...` and `D:\` paths everywhere).
- Type hints + docstrings on **all public functions**.
- Lightweight, but never at the cost of correctness.
- **Security:** no secrets/keys in the repo, ever
  (LEGACY LESSON: a private AWS `.pem` was found committed).
  Use `.env` + `.gitignore`; **pin all dependency versions**.

## 4. Pipeline Integrity

A stage that is written, tested, and approved is **FROZEN**.
Changing a frozen stage or any inter-stage contract (file formats, JSON schemas, function
signatures) requires the owner's **explicit permission BEFORE editing**.
State the change, the reason, and the impact; **wait for approval**.

## 5. Brainstorm-First

Before implementing any new stage/module: **no code** until the design has been discussed and
approved by the owner. (Design discussion happens in Claude chat; Claude Code implements the
approved plan.)

## 6. Legacy Code Policy

The legacy repo is **reference material only** — read it, never trust it, never copy blindly.

- **Known-good exceptions to lift:** `parse_numeric_ticks()` and `fit_axis()`
  (with their 8 documented caveats in `LEGACY_REVIEW.md`).
- **Known legacy bugs to never reproduce:**
  - `cv_curves.json` schema fork (two writers, flat vs keyed)
  - `""` (empty) `curve_type` keys
  - silent `except:` blocks that reset state
  - disabled quality gates (`BYPASS_WEAK_CURVE_FILTER`)
  - single-valued x tracing
  - hardcoded machine paths
  - committed secrets/artifacts

## 7. Logging

- Python `logging` module only — **no bare `print` in library code**.
- Shared config: timestamp, level, module; console + rotating file under `logs/`.
- Every pipeline stage logs: inputs received, key decisions, counts
  (found / kept / dropped + **why**), outputs written.
- Training runs log: hyperparameters, dataset version, per-epoch metrics, checkpoints.
- **Errors are never silently swallowed.**

## 8. Progress Tracking

- `PROGRESS.md` at the repo root is the **single source of truth**.
- Update it at the **start and end of every working session**.
- A task is **Done** only when its tests pass **and** the owner approved.
- **Never delete history.**

## 9. Definition of Done (every task)

- [ ] design approved
- [ ] tests first (10–15+)
- [ ] all green (`pytest -v`)
- [ ] no duplication
- [ ] logging present
- [ ] `PROGRESS.md` updated
- [ ] owner reviewed
