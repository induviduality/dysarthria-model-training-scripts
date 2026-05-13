#!/bin/bash
# setup_runpod.sh — One-click setup and training launcher for any GPU cloud.
#
# Manages Python dependencies with uv (https://docs.astral.sh/uv).
# Works on RunPod, Lambda Labs, Vast.ai, Paperspace, or any Linux GPU instance.
#
# Assumes:
#   - A CUDA-capable GPU is present.
#   - HF_TOKEN env var is set if your dataset is on HF Hub or private.
#   - The prepared dataset is on a mounted network volume (or set in config.yaml).
#
# Usage:
#   bash setup_runpod.sh
#   bash setup_runpod.sh --dataset-path /runpod-volume/prepared_dataset
#   bash setup_runpod.sh --baseline-wer 45.3
#   bash setup_runpod.sh --resume                  # resume from latest checkpoint
#   bash setup_runpod.sh --cuda cu128              # override CUDA version (default: cu121)
#   bash setup_runpod.sh --model turbo             # whisper-large-v3-turbo (default)
#   bash setup_runpod.sh --model large-v3          # whisper-large-v3 (full model)
#   bash setup_runpod.sh --model openai/whisper-large-v3-turbo   # explicit HF model id
#   bash setup_runpod.sh --baseline-wer 45.3 --resume
#
# Dataset path resolution order:
#   1. --dataset-path CLI arg
#   2. $DATASET_PATH environment variable (set in cloud provider UI)
#   3. dataset.local_path in config.yaml (default: /workspace/prepared_dataset)
#
# CUDA version guide (run nvidia-smi to check your driver version):
#   cu121 (default) — CUDA 12.1 — covers A100, A40, L4, most cloud GPUs
#   cu128           — CUDA 12.8 — newer H100, RTX 50xx instances
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASELINE_WER=""
RESUME_FLAG=""
DATASET_PATH="${DATASET_PATH:-}"
CUDA_VER="cu121"
MODEL_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline-wer)    BASELINE_WER="$2";  shift 2 ;;
        --baseline-wer=*)  BASELINE_WER="${1#*=}"; shift ;;
        --dataset-path)    DATASET_PATH="$2";  shift 2 ;;
        --dataset-path=*)  DATASET_PATH="${1#*=}"; shift ;;
        --cuda)            CUDA_VER="$2";  shift 2 ;;
        --cuda=*)          CUDA_VER="${1#*=}"; shift ;;
        --model)           MODEL_NAME="$2"; shift 2 ;;
        --model=*)         MODEL_NAME="${1#*=}"; shift ;;
        --resume)          RESUME_FLAG="--resume_from_checkpoint auto"; shift ;;
        *) shift ;;
    esac
done

# Resolve model shorthand aliases to full HF model IDs.
case "$MODEL_NAME" in
    turbo|"")         MODEL_NAME="openai/whisper-large-v3-turbo" ;;
    large-v3|large)   MODEL_NAME="openai/whisper-large-v3" ;;
    # Anything else is treated as a literal HF model ID (e.g. openai/whisper-medium).
esac

echo "========================================================"
echo "  Whisper Fine-tuning Setup — $MODEL_NAME"
echo "========================================================"

# ── 1. Install uv ─────────────────────────────────────────────────────────────
echo ""
echo "[1/5] Setting up uv..."
if ! command -v uv &>/dev/null; then
    echo "  uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "  uv $(uv --version)"

# ── 2. Install dependencies ────────────────────────────────────────────────────
echo ""
echo "[2/5] Installing dependencies..."
# uv sync reads pyproject.toml and installs all deps into .venv.
# torch/torchaudio are pulled from the pytorch-cu121 index automatically on Linux.
uv sync

# Override torch for a non-default CUDA version if requested.
if [ "$CUDA_VER" != "cu121" ]; then
    echo "  Reinstalling torch/torchaudio for CUDA $CUDA_VER..."
    uv pip install torch torchaudio \
        --index-url "https://download.pytorch.org/whl/$CUDA_VER" \
        --reinstall
fi

# Activate the virtual environment — all subsequent python/huggingface-cli
# calls will use the deps installed above.
source .venv/bin/activate
echo "  Python: $(python --version)"

# ── 3. HuggingFace authentication ─────────────────────────────────────────────
echo ""
echo "[3/5] HuggingFace authentication..."
if [ -z "$HF_TOKEN" ]; then
    echo "  WARNING: HF_TOKEN not set. Set it in your cloud provider's environment variables."
    echo "           Required if your dataset is private or source=huggingface in config.yaml."
else
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    echo "  HF auth: OK"
fi

# ── 4. Verify GPU and dataset ─────────────────────────────────────────────────
echo ""
echo "[4/5] Checking GPU..."
python - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    print("ERROR: No CUDA GPU found. Verify the CUDA-enabled torch build was installed.")
    sys.exit(1)
name   = torch.cuda.get_device_name(0)
vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  GPU  : {name}")
print(f"  VRAM : {vram:.1f} GB")
print(f"  CUDA : {torch.version.cuda}")
EOF

echo ""
echo "  Checking dataset..."
if [ -n "$DATASET_PATH" ]; then
    if [ ! -d "$DATASET_PATH" ]; then
        echo "  ERROR: Dataset path not found: $DATASET_PATH"
        echo "         Mount your network volume and verify the path, or set DATASET_PATH."
        exit 1
    fi
    echo "  Dataset path : $DATASET_PATH (OK)"
else
    python - <<'EOF'
import yaml, os
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
source = cfg["dataset"]["source"]
if source == "local":
    path = cfg["dataset"].get("local_path", "")
    if not os.path.isdir(path):
        print(f"  WARNING: dataset.local_path in config.yaml does not exist: {path}")
        print(f"           Pass --dataset-path /your/mount/path or set DATASET_PATH env var.")
    else:
        print(f"  Dataset path : {path} (OK)")
else:
    print(f"  Dataset source: {source} (HuggingFace Hub)")
EOF
fi

# ── 5. Auto-tune batch size and launch training ────────────────────────────────
echo ""
echo "[5/5] Auto-tuning batch size for detected VRAM..."
python - <<'EOF'
import torch, yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

if vram_gb >= 45:
    batch, accum = 8, 2
    print(f"  Detected ≥45 GB VRAM ({vram_gb:.0f} GB) → batch=8, grad_accum=2 (eff. 16)")
elif vram_gb >= 22:
    batch, accum = 4, 4
    print(f"  Detected ~24 GB VRAM ({vram_gb:.0f} GB) → batch=4, grad_accum=4 (eff. 16)")
else:
    batch, accum = 2, 8
    print(f"  Detected {vram_gb:.0f} GB VRAM → batch=2, grad_accum=8 (eff. 16)")

cfg["training"]["per_device_train_batch_size"] = batch
cfg["training"]["gradient_accumulation_steps"] = accum

with open("config.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print("  config.yaml updated.")
EOF

echo ""
echo "  Starting fine-tuning..."
echo ""

CMD="python train/finetune.py --config config.yaml --model_name $MODEL_NAME"

if [ -n "$DATASET_PATH" ]; then
    CMD="$CMD --local_dataset_path $DATASET_PATH"
fi

if [ -n "$BASELINE_WER" ]; then
    CMD="$CMD --baseline_wer $BASELINE_WER"
fi

if [ -n "$RESUME_FLAG" ]; then
    CMD="$CMD $RESUME_FLAG"
fi

if [ -n "$HF_TOKEN" ]; then
    CMD="$CMD --hf_token $HF_TOKEN"
fi

if [ -n "$WANDB_API_KEY" ]; then
    CMD="$CMD --wandb_project whisper-turbo-dysarthria"
else
    CMD="$CMD --no_wandb"
fi

echo "Running: $CMD"
echo ""
eval "$CMD"

echo ""
echo "========================================================"
echo "  Training complete!"
echo "  Next steps:"
echo "    python eval/eval_wer.py --model_path ./model-outputs/model-hf --baseline_wer <N>"
echo "    # HF model is at ./model-outputs/model-hf — ready for inference or CT2 conversion."
echo "========================================================"
