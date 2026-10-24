#!/usr/bin/env bash
set -euo pipefail
KIMODO_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CHECKPOINT_DIR="${KIMODO_ASSET_ROOT:-$HOME/checkpoints/kimodo}"
if [[ ! -f "$CHECKPOINT_DIR/asset_manifest.json" ]]; then
    echo "Assets are not ready; run scripts/prepare_inference_assets.py first." >&2
    exit 1
fi
export TEXT_ENCODERS_DIR="$CHECKPOINT_DIR/text_encoders"
export TEXT_ENCODER_MODE=local
export TEXT_ENCODER_DEVICE="${TEXT_ENCODER_DEVICE:-cuda}"
export LOCAL_CACHE=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$KIMODO_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${KIMODO_PYTHON:-python}" -m kimodo.scripts.generate \
    --model Kimodo-SOMA-RP-v1.2 "$@"
