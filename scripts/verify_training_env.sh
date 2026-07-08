#!/usr/bin/env bash
# T6 — training-environment smoke tests. Tests 1-3 must pass. Run ON the GPU box:
#   bash scripts/verify_training_env.sh [path/to/test_image.png]
# Uses the first PNG in data/images/ when no image argument is given.
#
# fp16/AMP is NOT exercised here (owner decision, 2026-07-08): mmcv-full
# 1.7.1's deformable-attention CUDA op (used by LineFormer's Mask2Former-style
# pixel decoder) has no half-precision kernel —
# `RuntimeError: "ms_deform_attn_forward_cuda" not implemented for 'Half'`.
# Test 3 measured ~1 GiB peak inference memory on the 15 GiB T4, so fp32-only
# training has ample headroom at batch size 1; see SETUP.md for the full
# rationale. Training must run fp32-only — do not wrap the model in
# torch.cuda.amp.autocast().
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="lineformer"
CONFIG="${LINEFORMER_CONFIG:-$REPO_ROOT/third_party/lineformer/lineformer_swin_t_config.py}"
CKPT="${LINEFORMER_CKPT:-$REPO_ROOT/data/weights/lineformer_pretrained_official_iter3000.pth}"
IMAGE="${1:-$(ls "$REPO_ROOT"/data/images/*.png 2>/dev/null | head -1)}"
OUT_DIR="$REPO_ROOT/data/smoke_test"
mkdir -p "$OUT_DIR"

# shellcheck disable=SC1091
# `conda info --base` needs `conda` already on PATH, which a non-interactive
# SSH shell doesn't have even when miniconda is installed — fall back to the
# default install location, same as setup_training_env.sh.
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# A pre-existing `pip install --user` torch (from unrelated prior work on this
# box, ~/.local/lib/python3.8/site-packages) shadows the conda env's own
# correctly-pinned torch via Python's user-site mechanism. Disable it so the
# env's pinned packages are actually the ones imported — this does not touch
# any files or override any version pin, it only fixes import resolution.
export PYTHONNOUSERSITE=1

echo "=== [1/4] GPU visibility ==="
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
name = torch.cuda.get_device_name(0)
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| device:", name)
assert "T4" in name, f"expected a T4, got {name!r}"
PY

echo "=== [2/4] mmdet + mmcv-full CUDA ops ==="
python - <<'PY'
import mmdet, mmcv
from mmcv.ops import RoIAlign  # fails on CPU-only / mismatched mmcv builds
print("mmdet:", mmdet.__version__, "| mmcv-full:", mmcv.__version__,
      "| RoIAlign import: OK")
PY

echo "=== [3/4] pretrained inference on real figure: $IMAGE ==="
[ -f "$CONFIG" ] || { echo "FAIL: config not found: $CONFIG"; exit 1; }
[ -f "$CKPT" ]   || { echo "FAIL: checkpoint not found: $CKPT"; exit 1; }
[ -n "$IMAGE" ] && [ -f "$IMAGE" ] || { echo "FAIL: no test image (copy PNGs into data/images/)"; exit 1; }
CONFIG="$CONFIG" CKPT="$CKPT" IMAGE="$IMAGE" OUT_DIR="$OUT_DIR" python - <<'PY'
import os, torch
from mmdet.apis import init_detector, inference_detector

config, ckpt = os.environ["CONFIG"], os.environ["CKPT"]
image, out_dir = os.environ["IMAGE"], os.environ["OUT_DIR"]
model = init_detector(config, ckpt, device="cuda:0")
torch.cuda.reset_peak_memory_stats()
result = inference_detector(model, image)
# mmdet 2.x instance-seg result: (bbox_results, mask_results)
assert isinstance(result, tuple) and len(result) == 2, f"unexpected result type {type(result)}"
n_masks = sum(len(m) for m in result[1])
assert n_masks > 0, "inference returned zero masks"
peak = torch.cuda.max_memory_allocated() / 1024**2
out = os.path.join(out_dir, "smoke_" + os.path.basename(image))
model.show_result(image, result, out_file=out, score_thr=0.3)
print(f"masks: {n_masks} | peak GPU memory: {peak:.0f} MiB | visualization: {out}")
PY

echo "=== [4/4] fp16 autocast: N/A — architecture limitation, fp32-only training decided ==="
echo "    (mmcv-full 1.7.1 deformable-attn CUDA op has no half-precision kernel;"
echo "     see SETUP.md 'fp32-only training decision' for the full rationale)"

echo "=== ALL APPLICABLE SMOKE TESTS PASSED (fp32-only) ==="
