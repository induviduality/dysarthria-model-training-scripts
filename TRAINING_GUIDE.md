# Whisper Large v3 Turbo — Fine-tuning & Evaluation Guide

End-to-end reference for training on any GPU cloud (RunPod, Lambda Labs, Vast.ai,
Paperspace, etc.) or locally. Everything that needs a decision is called out explicitly.

---

## Table of Contents

1. [GPU Selection & Config](#1-gpu-selection--config)
2. [Infrastructure Setup](#2-infrastructure-setup)
3. [Dataset Setup](#3-dataset-setup)
4. [config.yaml Reference](#4-configyaml-reference)
5. [Running Training](#5-running-training)
6. [Monitoring & Checkpoints](#6-monitoring--checkpoints)
7. [Resuming After Interruption](#7-resuming-after-interruption)
8. [Evaluation](#8-evaluation)
9. [CT2 Conversion (faster-whisper)](#9-ct2-conversion-faster-whisper)
10. [Analysis Scripts](#10-analysis-scripts)
11. [Local Smoke Test](#11-local-smoke-test)

---

## 1. GPU Selection & Config

### Supported GPUs and required config changes

| GPU | VRAM | `bf16` | `fp16` | batch | grad_accum | `gradient_checkpointing` | Notes |
|-----|------|--------|--------|-------|------------|--------------------------|-------|
| A100 (SXM/PCIe) | 40–80 GB | `true` | `false` | 8 | 2 | optional | Best option. Flash-attn supported. |
| A40 | 48 GB | `true` | `false` | 8 | 2 | optional | Same settings as A100. |
| L4 / RTX 4090 | 24 GB | `true` | `false` | 4 | 4 | **required** | Default config targets this. |
| V100 | 16–32 GB | **`false`** | **`true`** | 2–4 | 4–8 | **required** | No bf16 support (Volta arch). |
| T4 | 16 GB | **`false`** | **`true`** | 2 | 8 | **required** | No bf16 support. Slowest option. |

`setup_runpod.sh` auto-sets `batch` and `grad_accum` from VRAM. **You must manually
set `bf16`/`fp16` for V100/T4** — the script does not detect architecture.

For V100 or T4, edit `config.yaml` before running:
```yaml
training:
  bf16: false
  fp16: true
```

### Model selection

| Model | Params | Min VRAM | Notes |
|-------|--------|----------|-------|
| `whisper-large-v3-turbo` | 809M | 16 GB | Default. Faster training, good accuracy. L4/T4/V100 viable. |
| `whisper-large-v3` | 1.5B | ~40 GB | Full model. Better ceiling, needs A40/A100. |

Pass `--model` to `setup_runpod.sh` at launch — no config edit needed:

```bash
bash setup_runpod.sh --model turbo       # openai/whisper-large-v3-turbo (default)
bash setup_runpod.sh --model large-v3    # openai/whisper-large-v3
```

Or override directly via `finetune.py`:

```bash
python train/finetune.py --model_name openai/whisper-large-v3
```

### Flash Attention (A100 / H100 / L4 — optional, ~20% speedup)

```bash
pip install flash-attn --no-build-isolation
```

Then add to your training launch command or uncomment in `requirements.txt`.

---

## 2. Infrastructure Setup

### Step 1 — Get the code onto the machine

The simplest way on any cloud instance:

```bash
git clone https://github.com/YOUR_ORG/YOUR_REPO.git /workspace/whisper_fine_tune
cd /workspace/whisper_fine_tune
```

Or upload via `scp` / rsync if you don't have a remote repo.

### Step 2 — Install dependencies

This project uses [uv](https://docs.astral.sh/uv) for dependency management.

**Option A — Use `setup_runpod.sh` (recommended for any provider)**

```bash
bash setup_runpod.sh --dataset-path /path/to/prepared_dataset
```

The script installs uv if needed, runs `uv sync`, activates the venv, checks GPU and
dataset, auto-tunes batch size, and launches training in one shot.
See [Section 5](#5-running-training) for all flags.

**Option B — Manual setup with uv**

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (torch pulled from pytorch-cu121 index automatically on Linux)
uv sync

# Optional extras:
uv sync --extra ct2     # adds ctranslate2 + faster-whisper (for convert/eval CT2)
uv sync --extra stream  # adds sounddevice (for stream_transcribe.py)

# Activate the virtual environment
source .venv/bin/activate
```

**Override the CUDA version** (if your driver is CUDA 12.8 or you need a different build):

```bash
# After uv sync, reinstall torch with the correct index:
uv pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 --reinstall

# Or use the --cuda flag in setup_runpod.sh:
bash setup_runpod.sh --cuda cu128 --dataset-path /path/to/prepared_dataset
```

> **How to find your CUDA version:** run `nvidia-smi` — the version shown top-right is
> your maximum supported CUDA. cu121 is backward-compatible with cu12.x drivers, so if
> unsure, the default cu121 is the safe choice.

**Fallback — plain pip (no uv)**

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Windows local dev**

```powershell
# 1. Install uv (PowerShell — restart terminal after)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Install deps (pulls CPU torch by default on Windows)
uv sync

# 3. If you have a local GPU, reinstall torch with CUDA:
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 --reinstall

# 4. Run scripts via uv run (no manual venv activation needed)
uv run python train/finetune.py --config config_smoke.yaml
```

### Step 3 — Set environment variables (optional but useful)

Set these in your cloud provider's UI or export before running:

```bash
export HF_TOKEN=hf_xxx          # Required if loading from HF Hub
export DATASET_PATH=/workspace/prepared_dataset   # Dataset location
export WANDB_API_KEY=xxx        # Optional: Weights & Biases logging
```

### Provider-specific dataset mount paths

| Provider | Network storage default path |
|----------|------------------------------|
| RunPod | `/workspace/` or `/runpod-volume/` |
| Lambda Labs | `/home/ubuntu/` |
| Vast.ai | `/workspace/` |
| Paperspace | `/storage/` (persistent) or `/notebooks/` |
| Google Colab | `/content/drive/MyDrive/` (after mounting) |

---

## 3. Dataset Setup

The prepared dataset (`prepared_dataset/`) must be on the training machine.
You have two options:

### Option A — Network volume (recommended for cloud)

Upload `prepared_dataset/` to your cloud provider's persistent/network storage
before spinning up the GPU instance. Then point to it at launch:

```bash
bash setup_runpod.sh --dataset-path /workspace/prepared_dataset
```

Or set in `config.yaml`:
```yaml
dataset:
  source: "local"
  local_path: "/workspace/prepared_dataset"
```

### Option B — HuggingFace Hub

If you've already pushed the dataset with `prepare_hf_dataset.py`:

```bash
python data/prepare_hf_dataset.py \
    --dataset_root /path/to/raw-dysarthric-speech \
    --output_repo  your-username/dysarthria-prepared \
    --hf_token     $HF_TOKEN
```

Then update `config.yaml`:
```yaml
dataset:
  source: "huggingface"
  hf_dataset_id: "your-username/dysarthria-prepared"
```

---

## 4. config.yaml Reference

All settings that matter for a training run:

### Settings you may want to change before training

```yaml
# ── Model — override at launch with --model turbo|large-v3 (setup_runpod.sh)
#            or --model_name openai/whisper-large-v3 (finetune.py directly) ──
model:
  name: "openai/whisper-large-v3-turbo"   # swap to openai/whisper-large-v3 for full model (needs ≥40 GB)

# ── GPU settings ──────────────────────────────────────────────────────────────
training:
  bf16: true          # true for A100/A40/L4/RTX 4090. MUST be false for V100/T4.
  fp16: false         # true for V100/T4. Leave false otherwise.
  gradient_checkpointing: true   # Required on 24 GB. Safe to keep on any GPU.

  # Batch size — setup_runpod.sh auto-sets these from VRAM.
  # Override manually if needed:
  per_device_train_batch_size: 4   # L4: 4, A40/A100: 8, T4/V100: 2
  gradient_accumulation_steps: 4   # L4: 4, A40/A100: 2, T4/V100: 8

  max_steps: 5000     # Primary stopping criterion. ~5–10 passes through dataset.
  eval_steps: 500     # Evaluate every N steps.
  save_steps: 500     # Save checkpoint every N steps.
  safety_checkpoint_steps: 50   # Extra checkpoint for preemption safety.

# ── Dataset ────────────────────────────────────────────────────────────────────
dataset:
  source: "local"     # "local" for network volume, "huggingface" for HF Hub
  local_path: "/workspace/prepared_dataset"   # adjust to your mount path

# ── Local dev — keep false for real training runs ─────────────────────────────
local_dev:
  enabled: false      # true = whisper-small, 200 steps, no bf16 (for local testing only)

# ── Baseline WER (optional but useful for tracking improvement) ───────────────
evaluation:
  baseline_wer: null  # Set to your pre-measured baseline, e.g. 47.3
```

### Settings you generally don't change

```yaml
model:
  name: "openai/whisper-large-v3-turbo"

lora:
  r: 16               # Increase to 32 if underfitting
  alpha: 32
  target_modules: [q_proj, v_proj, k_proj, out_proj, fc1, fc2]

early_stopping:
  enabled: true
  patience: 5         # Stop after 5 eval checkpoints (~2,500 steps) with no WER gain

sampling:
  enabled: true
  ua_frac: 0.60       # 60% UASpeech : 40% TORGO per step
  torgo_frac: 0.40
```

---

## 5. Running Training

### Recommended: one-command launch via `setup_runpod.sh`

Works on any Linux cloud instance, not just RunPod.

```bash
# Basic launch — defaults to whisper-large-v3-turbo
bash setup_runpod.sh

# Choose model (two shorthands supported):
bash setup_runpod.sh --model turbo       # openai/whisper-large-v3-turbo (default)
bash setup_runpod.sh --model large-v3    # openai/whisper-large-v3 (full model, needs ~48 GB)

# Or pass a full HF model ID:
bash setup_runpod.sh --model openai/whisper-medium

# Specify dataset path explicitly
bash setup_runpod.sh --dataset-path /workspace/prepared_dataset

# With baseline WER for improvement tracking
bash setup_runpod.sh --dataset-path /workspace/prepared_dataset --baseline-wer 47.3

# Resume from latest checkpoint
bash setup_runpod.sh --dataset-path /workspace/prepared_dataset --resume

# All flags together
bash setup_runpod.sh \
    --model large-v3 \
    --dataset-path /workspace/prepared_dataset \
    --baseline-wer 47.3 \
    --resume
```

What the script does:
1. Verifies a GPU is present and prints VRAM
2. Installs deps (PyTorch cu121 + requirements.txt)
3. Logs into HuggingFace if `$HF_TOKEN` is set
4. Checks the dataset path exists
5. Auto-tunes batch size for detected VRAM (writes back to config.yaml)
6. Launches `finetune.py`

### Alternative: run `finetune.py` directly

Useful when deps are already installed (e.g., pre-built Docker image) or you want
fine-grained control without the setup overhead.

```bash
# Standard run
python train/finetune.py --config config.yaml

# Override dataset path without editing config.yaml
python train/finetune.py --config config.yaml \
    --dataset_source local \
    --local_dataset_path /workspace/prepared_dataset

# Override specific hyperparameters
python train/finetune.py --config config.yaml \
    --max_steps 3000 \
    --learning_rate 5e-5 \
    --lora_rank 32

# Force fresh start (deletes existing checkpoints)
python train/finetune.py --config config.yaml --fresh_start

# Resume from latest checkpoint
python train/finetune.py --config config.yaml --resume_from_checkpoint auto

# Resume from a specific checkpoint
python train/finetune.py --config config.yaml \
    --resume_from_checkpoint ./model-outputs/finetuned/checkpoint-2500
```

### Outputs produced by training

```
model-outputs/
  finetuned/
    checkpoint-500/        # Checkpoint every eval_steps
    checkpoint-1000/
    ...
    training_summary.json  # WER history, best WER, improvement vs baseline
    train_results.json
    eval_results.json

  model-hf/    # LoRA merged into base — use this for inference/eval
    config.json
    model.safetensors
    tokenizer*
```

---

## 6. Monitoring & Checkpoints

### TensorBoard (built-in)

```bash
tensorboard --logdir ./model-outputs/finetuned/runs --port 6006
```

Then open `http://localhost:6006` (or forward the port from your cloud instance).
Tracks: training loss, eval WER, learning rate.

### Weights & Biases (optional)

Set `WANDB_API_KEY` as an env var and the script enables it automatically.
Or pass `--no_wandb` to disable regardless.

### What to watch

- **eval_wer** should trend down over the first ~2,000 steps then plateau or trigger early stopping.
- **train_loss** should drop from ~3.5 to ~0.5–1.0 range.
- If eval WER goes up after step ~1,000, you may be overfitting — consider reducing `max_steps`.

---

## 7. Resuming After Interruption

Checkpoints are saved every `save_steps` (500) and every `safety_checkpoint_steps` (50).
If the pod is preempted or you stop training:

```bash
# Auto-resumes from the latest checkpoint
python train/finetune.py --config config.yaml --resume_from_checkpoint auto

# Or via setup_runpod.sh
bash setup_runpod.sh --dataset-path /workspace/prepared_dataset --resume
```

Ctrl+C during training triggers a graceful stop: the current step finishes, a checkpoint
is saved, then training exits. Running again with `--resume` continues from that point.

---

## 8. Evaluation

All evaluation reads `config.yaml` for dataset settings, so you don't need to
re-specify paths if config is correct.

### 8a. Evaluate the HF merged model (after training)

```bash
# Validation split (default)
python eval/eval_wer.py \
    --model_path ./model-outputs/model-hf \
    --model_type hf

# With baseline comparison
python eval/eval_wer.py \
    --model_path ./model-outputs/model-hf \
    --model_type hf \
    --baseline_wer 47.3

# Evaluate on test split
python eval/eval_wer.py \
    --model_path ./model-outputs/model-hf \
    --model_type hf \
    --split test

# Quick check on 50 samples
python eval/eval_wer.py \
    --model_path ./model-outputs/model-hf \
    --model_type hf \
    --max_samples 50

# Save results to a named file (required before running analysis scripts)
python eval/eval_wer.py \
    --model_path ./model-outputs/model-hf \
    --model_type hf \
    --output_json inference-outputs/evaluation_results.json

# Evaluate the base model (for comparison)
python eval/eval_wer.py \
    --model_path openai/whisper-large-v3-turbo \
    --model_type hf \
    --output_json inference-outputs/evaluation_results_base.json
```

### 8b. Evaluate the CT2 model (after conversion)

```bash
python eval/eval_wer.py \
    --model_path ./model-outputs/model-ct2 \
    --model_type ct2
```

### What `eval_wer.py` outputs

- Overall WER and CER
- Per-speaker WER breakdown (M05, F02, etc.)
- First 5 sample predictions side-by-side with references
- `inference-outputs/evaluation_results.json` with per-sample `reference`, `hypothesis`, `top3` candidates

---

## 9. CT2 Conversion (faster-whisper)

> For full inference docs — batch files, real-time mic streaming, VAD tuning,
> model format comparison — see **[INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)**.

Requires the `ct2` optional dependency group:

```bash
uv sync --extra ct2
```

Converts the merged HF model to CTranslate2 format for fast production inference.

```bash
# Default: reads paths from config.yaml, int8 quantization
python inference/convert_to_ct2.py

# GPU inference — use float16 (better accuracy than int8 on GPU)
python inference/convert_to_ct2.py --quantization float16

# Override paths explicitly
python inference/convert_to_ct2.py \
    --model_dir  ./model-outputs/model-hf \
    --output_dir ./model-outputs/model-ct2 \
    --quantization int8
```

Quantization guide:

| Setting | Use case | Size | Speed |
|---------|----------|------|-------|
| `int8` | CPU or GPU inference | ~600 MB | Fast on CPU |
| `float16` | GPU inference | ~1.5 GB | Fastest on GPU |
| `int8_float16` | GPU inference, memory constrained | ~600 MB | Good balance |

---

## 10. Analysis Scripts

These scripts expect two JSON files produced by `eval_wer.py`:
- `inference-outputs/evaluation_results_base.json` — base model results
- `inference-outputs/evaluation_results.json` — fine-tuned model results

Run `eval_wer.py` twice (once for base, once for fine-tuned) and save to these filenames
before running the analysis scripts. **Run these scripts from the project root.**

### Standard WER breakdown by dataset source

```bash
# Reads inference-outputs/evaluation_results_base.json and inference-outputs/evaluation_results.json
# Shows TORGO vs UASpeech WER, per-speaker table, improvement comparison
python eval/analyze_by_dataset.py
```

### Phonetic word-accuracy analysis (more lenient than strict WER)

Uses Soundex phonetic matching + number normalization to count "close enough" words
as correct. Gives a more generous accuracy view for dysarthric speech.

```bash
# Same inputs as above, uses phonetic matching rules
python eval/analyze_by_dataset_accurate.py
```

### Per-sample transcript listing

```bash
# Reads inference-outputs/evaluation_results.json only
# Prints every sample: reference vs hypothesis, per-word accuracy
python eval/analyze_transcripts.py
```

### Comprehensive evaluation report

```bash
# Full breakdown: holistic accuracy, dataset-wise, per-speaker, speaker ranking
python eval/evaluate_finetuned_comprehensive.py
```

> **Note:** All analysis scripts have hardcoded paths to `inference-outputs/evaluation_results*.json`.
> Always run them from the project root so relative paths resolve correctly.

---

## 11. Local Smoke Test

Two options depending on what you want to validate:

### Option A — `config_smoke.yaml` (recommended for pipeline validation)

Runs the full `finetune.py` training loop with `whisper-tiny`, 20 steps, and your
actual `prepared_dataset`. Exercises every code path — data loading, speaker split,
LoRA setup, weighted sampler, training loop, eval, WER, checkpoint save.
Completes in **~2–5 minutes on GPU** or ~15 minutes on CPU.

```bash
# Linux / GPU cloud
uv run python train/finetune.py --config config_smoke.yaml

# Windows (no GPU)
uv run python train/finetune.py --config config_smoke.yaml --allow_cpu

# Output goes to ./model-outputs/smoke-finetuned/ — won't collide with real training outputs.
```

Use this when you want to confirm the full finetune pipeline works end-to-end
before launching a real run on a cloud GPU.

### Option B — `smoke_test.py` (unit-style checks, requires raw dataset)

Validates individual pipeline components using 6 real samples from the raw
dysarthric-speech dataset (needs `manifest.jsonl`). Runs in ~2 minutes on CPU.

```bash
# Requires local dysarthric-speech dataset with manifest.jsonl
python train/smoke_test.py --dataset_root D:/Datasets/dysarthric-speech

# More samples, specific model
python train/smoke_test.py \
    --dataset_root D:/Datasets/dysarthric-speech \
    --n_samples 10 \
    --model openai/whisper-small
```

9 checks are run: data loading, preprocessing, LoRA application, data collation,
WER metric, 2-step training, checkpoint save, LoRA merge, and HF pipeline inference.
All 9 must pass before the real training run is safe.

---

## Quick Reference: Full Training Workflow

```bash
# 0. Install uv (once)
#    Linux:   curl -LsSf https://astral.sh/uv/install.sh | sh
#    Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 1. (Once, locally) Install deps and prepare dataset
uv sync
# Linux GPU — uv sync pulls CUDA torch automatically via pytorch-cu121 index.
# Windows GPU — reinstall torch with CUDA after sync:
#   uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 --reinstall

source .venv/bin/activate   # Linux; Windows: .venv\Scripts\activate (or use uv run)
python data/prepare_hf_dataset.py \
    --dataset_root /path/to/raw-data \
    --output_dir   /path/to/prepared_dataset

# 2. (Once, locally) Validate pipeline with smoke config (~2–5 min on GPU)
uv run python train/finetune.py --config config_smoke.yaml

# 3. Upload prepared_dataset/ to cloud network volume

# 4. On GPU instance: train (setup_runpod.sh installs uv + runs uv sync automatically)
bash setup_runpod.sh \
    --dataset-path /workspace/prepared_dataset \
    --baseline-wer 47.3

# 5. Evaluate fine-tuned model
python eval/eval_wer.py --model_path ./model-outputs/model-hf --model_type hf \
    --output_json inference-outputs/evaluation_results.json --baseline_wer 47.3

# 6. Evaluate base model for comparison
python eval/eval_wer.py --model_path openai/whisper-large-v3-turbo --model_type hf \
    --output_json inference-outputs/evaluation_results_base.json

# 7. Run analysis
python eval/analyze_by_dataset.py
python eval/analyze_by_dataset_accurate.py

# 8. (Optional) Install CT2 extras and convert to faster-whisper
uv sync --extra ct2
python inference/convert_to_ct2.py --quantization float16
```
