#!/usr/bin/env bash
# T6 — training-environment smoke tests. ALL must pass. Run ON the GPU box:
#   bash scripts/verify_training_env.sh [path/to/test_image.png]
# Uses the first PNG in data/images/ when no image argument is given.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="lineformer"
CONFIG="${LINEFORMER_CONFIG:-$REPO_ROOT/third_party/lineformer/lineformer_swin_t_config.py}"
CKPT="${LINEFORMER_CKPT:-$REPO_ROOT/data/weights/iter_3000.pth}"
IMAGE="${1:-$(ls "$REPO_ROOT"/data/images/*.png 2>/dev/null | head -1)}"
OUT_DIR="$REPO_ROOT/data/smoke_test"
mkdir -p "$OUT_DIR"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

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

print("=== [4/4] fp16 autocast forward ===")
torch.cuda.reset_peak_memory_stats()
with torch.cuda.amp.autocast():
    _ = inference_detector(model, image)
print(f"autocast forward: OK | peak GPU memory: "
      f"{torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")
PY

echo "=== ALL SMOKE TESTS PASSED ==="
