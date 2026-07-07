#!/usr/bin/env bash
# T6 — LineFormer training environment setup (AWS g4dn.xlarge, T4 16GB).
# Idempotent; run ON the GPU box from the repo root:
#   bash scripts/setup_training_env.sh
# Optional env vars:
#   CHECKPOINT_URL   Override for the pretrained-checkpoint source. Default is
#                    the Google-Drive file id of iter_3000.pth from the
#                    official README ("Inference" section, step 1); expected
#                    sha256 recorded in envs/lineformer_checkpoint.sha256.
#   LINEFORMER_REPO  Git URL of the official LineFormer repo
#                    (default: https://github.com/TheJaeLal/LineFormer).
#
# Version pins are OWNER-APPROVED (legacy-compatible stack known to work with
# LineFormer). Do NOT change them without approval — a wrong mmcv/torch
# pairing produces silent garbage, not errors.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="lineformer"
LINEFORMER_REPO="${LINEFORMER_REPO:-https://github.com/TheJaeLal/LineFormer}"
THIRD_PARTY="$REPO_ROOT/third_party"
WEIGHTS_DIR="$REPO_ROOT/data/weights"
LOCK_DIR="$REPO_ROOT/envs"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$THIRD_PARTY" "$WEIGHTS_DIR" "$LOCK_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup_training_env.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== setup_training_env $(date -u +%FT%TZ) ==="

# --- 1. miniconda ------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found — installing miniconda to \$HOME/miniconda3"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
fi
# shellcheck disable=SC1091
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"

# --- 2. conda env with the pinned legacy-compatible stack ---------------------
if ! conda env list | grep -qE "^$ENV_NAME\s"; then
    conda create -n "$ENV_NAME" python=3.8 -y
fi
conda activate "$ENV_NAME"
python -V

# torch first (official channels; CUDA 11.7 runtime matches the T4 driver).
conda install -n "$ENV_NAME" -y \
    pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 \
    -c pytorch -c nvidia

# mmcv-full MUST come via openmim so the prebuilt CUDA-op wheel matching
# torch 1.13/cu117 is selected; pip alone picks a CPU/mismatched build.
pip install -U openmim
mim install mmcv-full==1.7.1
pip install mmdet==2.28.2
# scipy pinned 1.9.3: the known-good legacy pairing; it CONFLICTS with the
# pipeline venv's scipy — the two environments must never be shared.
pip install scipy==1.9.3 scikit-image opencv-python pillow matplotlib \
    bresenham tqdm

# --- 3. lock file (committed to the repo) -------------------------------------
conda env export -n "$ENV_NAME" > "$LOCK_DIR/lineformer.lock.yml"
echo "lock written: $LOCK_DIR/lineformer.lock.yml"

# --- 4. LineFormer source (reference-only clone; NOT from the legacy tree) ----
if [ ! -d "$THIRD_PARTY/lineformer/.git" ]; then
    git clone "$LINEFORMER_REPO" "$THIRD_PARTY/lineformer"
fi
COMMIT="$(git -C "$THIRD_PARTY/lineformer" rev-parse HEAD)"
echo "$COMMIT" > "$LOCK_DIR/lineformer.commit"
echo "LineFormer commit: $COMMIT (recorded in envs/lineformer.commit; copy into SETUP.md)"

# --- 5. pretrained checkpoint --------------------------------------------------
# Source: official README "Inference" step 1 -> Drive folder
# https://drive.google.com/drive/folders/1K_zLZwgoUIAJtfjwfCU5Nv33k17R0O5T
# containing iter_3000.pth (file id below). Verified sha256 in
# envs/lineformer_checkpoint.sha256 — a mismatch aborts.
CHECKPOINT_URL="${CHECKPOINT_URL:-1cIWM7lTisd1GajDR98IymDssvvLAKH1n}"
CKPT="$WEIGHTS_DIR/iter_3000.pth"
if [ ! -f "$CKPT" ]; then
    case "$CHECKPOINT_URL" in
        http*) curl -fL "$CHECKPOINT_URL" -o "$CKPT" ;;
        *)     pip install -q gdown && gdown "$CHECKPOINT_URL" -O "$CKPT" ;;
    esac
fi
EXPECTED_SHA="$(awk '{print $1}' "$LOCK_DIR/lineformer_checkpoint.sha256")"
ACTUAL_SHA="$(sha256sum "$CKPT" | awk '{print $1}')"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
    echo "FAIL: checkpoint sha256 mismatch: got $ACTUAL_SHA want $EXPECTED_SHA"
    exit 1
fi
echo "checkpoint OK: $CKPT (sha256 verified)"

echo "=== setup complete — now run: bash scripts/verify_training_env.sh ==="
