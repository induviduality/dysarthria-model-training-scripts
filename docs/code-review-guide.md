# Code Review Guide — Dysarthric ASR LoRA Fine-tuning

This document maps the training strategy (training-strategy-lora.md) to exact file locations,
explains what each script does, and shows how to run the full pipeline with a single command.

---

## File Overview

| File | Purpose |
|---|---|
| `config.yaml` | Single source of truth for all hyperparameters. Edit this before every run. |
| `data/prepare_hf_dataset.py` | One-time local step: filter manifest → build HF Dataset → push to Hub. |
| `train/finetune.py` | Full training loop: load → sample → train → eval → merge LoRA. |
| `eval/eval_wer.py` | Post-training WER analysis on a held-out split (per-speaker + overall). |
| `train/smoke_test.py` | Pipeline validation using whisper-tiny and 6 real audio clips (~3 min, CPU). |
| `inference/stream_transcribe.py` | Real-time microphone transcription using the CT2 deployed model. |

---

## Strategy → Implementation Map

### Fix 1 — Remove `transcript == 'xxx'` (strategy §1)

**File:** `data/prepare_hf_dataset.py`, function `load_manifest()`, lines ~62–70

```python
before  = len(entries)
entries = [e for e in entries if e.get("transcript", "").strip().lower() != "xxx"]
removed = before - len(entries)
if removed:
    print(f"  Removed {removed} entries with transcript='xxx' (no linguistic signal)")
```

34 TORGO utterances are flagged `xxx` (unintelligible, no linguistic content). They're removed
here, once, before the dataset is pushed to HF Hub — so training never sees them.

---

### Fix 2 — Remove control speakers (strategy §1)

**File:** `data/prepare_hf_dataset.py`, function `load_manifest()`, line ~73

```python
if group != "all":
    entries = [e for e in entries if e.get("group") == group]
```

`--group dysarthric` (the default) keeps only entries where `group == "dysarthric"`, dropping
all 5,058 TORGO control and 9,945 UASpeech control utterances. LoRA's frozen backbone already
encodes normal speech; controls waste adaptation budget.

---

### Fix 3 — 60/40 UASpeech:TORGO weighted sampling (strategy §2)

**File:** `train/finetune.py`

- **Weight computation:** function `compute_sample_weights()`, ~lines 392–427
- **Custom Trainer:** class `DysarthriaTrainer`, ~lines 670–716
- **Activated from:** `main()`, reads `config.yaml → sampling.enabled`

How it works:
1. `compute_sample_weights()` computes a per-sample weight so that when
   `WeightedRandomSampler` draws samples, 60% come from UASpeech and 40% from TORGO —
   regardless of the raw 4:1 size imbalance.
2. `DysarthriaTrainer.get_train_dataloader()` overrides the standard HF dataloader
   and injects `WeightedRandomSampler(num_samples = max_steps × effective_batch_size)`.
   This is "not len(dataset)" as the strategy requires — the sampler length is tied to the
   step budget so it is exhausted exactly when training ends.
3. Weights are computed on the **raw** (unpreprocessed) train split before `preprocess_split()`
   removes the `dataset` column.

To turn off (uniform sampling): set `sampling.enabled: false` in `config.yaml`.

---

### Fix 4 — Step-based training, not epoch-based (strategy §3)

**File:** `config.yaml`, `training` section

```yaml
max_steps: 5000       # Primary stopping criterion
eval_strategy: "steps"
eval_steps: 500
save_strategy: "steps"
save_steps: 500
warmup_steps: 250     # 5% of 5,000 steps
```

**File:** `train/finetune.py`, function `build_training_args()`, ~lines 797–841

`max_steps` is passed to `Seq2SeqTrainingArguments` and takes priority over `num_train_epochs`
in HF Trainer. The strategy prohibits epoch-based scheduling because "epoch" is undefined under
weighted sampling (the dataset length is artificial).

**Intervention:** if you want to shift to 55/45 (strategy §4 intervention table), change
`sampling.ua_frac: 0.55` and `sampling.torgo_frac: 0.45` in `config.yaml` — no code change needed.

---

### Fix 5 — Per-speaker WER at every checkpoint (strategy §4)

**File:** `train/finetune.py`, class `PerSpeakerWERCallback`, ~lines 603–666

The callback:
1. Holds raw audio + transcripts for each validation speaker (built in `build_speaker_val_data()`
   before preprocessing strips the `speaker_id` column).
2. After each standard `on_evaluate` event, runs inference directly on these audio clips
   grouped by speaker ID.
3. Logs a one-liner like `[Per-speaker WER @ step 500] F02: 38.1% | M05: 52.4%`.

Capped at `MAX_SAMPLES_PER_SPEAKER = 30` clips per speaker to keep eval overhead < 1 min on GPU.

**What to watch for (strategy §4):**
- TORGO val WER diverging up while UASpeech improves → shift ratio to 55/45
- All val WER plateauing < 1,000 steps → add SpecAugment, reduce LoRA rank

---

### Fix 6 — Local dev mode (single config change) (strategy — pipeline validation)

**File:** `config.yaml`, `local_dev` section (bottom of file)

```yaml
local_dev:
  enabled: false    # ← change this to true for local run
  model_name: "openai/whisper-small"
  max_steps: 200
  ...
```

**One-command local run:**
```bash
# Option A — flip the config flag (persistent):
#   Edit config.yaml: local_dev.enabled: true
python train/finetune.py

# Option B — CLI flag (no config edit needed):
python train/finetune.py --local_dev
```

`apply_local_dev_overrides()` in `train/finetune.py` (~lines 184–208) applies all local_dev settings
before CLI args so explicit overrides (`--model_name`, `--max_steps`) still win. The GPU check
becomes a warning instead of a hard exit when `local_dev` or `--allow_cpu` is active.

---

## Checkpointing & Resume

### How checkpoints are saved

| Trigger | Frequency | What's saved |
|---|---|---|
| Regular eval checkpoint | Every 500 steps (`save_steps`) | Model, optimizer, LR scheduler, RNG state, trainer state |
| Safety checkpoint | Every 50 steps (`safety_checkpoint_steps`) | Same as above — full trainer state |
| Graceful interrupt | On the step after Ctrl+C | Same as above |

Only 5 checkpoints are kept at a time (`save_total_limit: 5`), plus the best-by-WER checkpoint is always protected from deletion.

### Resuming after an interrupt

```bash
# Default — auto-resumes from the latest checkpoint every time:
python train/finetune.py

# Resume from a specific checkpoint:
python train/finetune.py --resume_from_checkpoint ./finetuned/checkpoint-1500

# Ignore checkpoints and start from where you are, without deleting them:
python train/finetune.py --resume_from_checkpoint none

# Delete all checkpoints and start from scratch:
python train/finetune.py --fresh_start
```

HF Trainer restores model weights, optimizer state, LR scheduler, and RNG state from the checkpoint. It then skips the already-seen batches in the dataloader so training continues at exactly the right step count and learning rate.

### How Ctrl+C works

- **First Ctrl+C**: `GracefulInterruptCallback` (`train/finetune.py:590`) sets `control.should_training_stop = True` and `control.should_save = True` at the end of the current step. Training exits cleanly and a full checkpoint is written.
- **Second Ctrl+C**: Forces immediate exit (checkpoint may be incomplete).

Re-running `python train/finetune.py` after either kind of interrupt will auto-resume from the saved checkpoint.

---

## How to Run the Full Pipeline

### Step 1 — Prepare dataset (local, once)

```bash
python data/prepare_hf_dataset.py \
    --dataset_root D:/Datasets/dysarthric-speech \
    --output_repo  your-hf-username/dysarthria-prepared \
    --hf_token     hf_xxx
```

What it does: filters `xxx` transcripts, drops control speakers, carves val split, pushes to Hub.
Then update `config.yaml → dataset.hf_dataset_id` with the output repo name.

### Step 2 — Validate pipeline locally (whisper-small, ~5 min)

```bash
# Full training loop validation (200 steps, CPU):
python train/finetune.py --local_dev

# Or the faster component-level smoke test (whisper-tiny, ~3 min, no GPU needed):
python train/smoke_test.py --dataset_root D:/Datasets/dysarthric-speech
```

### Step 3 — Train on RunPod (whisper-large-v3-turbo)

```bash
# On RunPod, after setup_runpod.sh:
python train/finetune.py --hf_token hf_xxx --baseline_wer 45.3
```

Checkpoints saved every 500 steps + safety checkpoint every 50 steps.
Training stops when val WER doesn't improve for 5 consecutive checkpoints (= 2,500 steps).

### Step 4 — Evaluate WER on test split

```bash
python eval/eval_wer.py \
    --model_path ./model-hf \
    --split test \
    --baseline_wer 45.3
```

Outputs: overall WER, per-speaker WER table, per-sample predictions JSON.

---

## Key Metrics to Watch

| Metric | Where logged | What to check |
|---|---|---|
| `eval_wer` | Console + TensorBoard | Primary stopping signal |
| Per-speaker WER | `[Per-speaker WER @ step N]` in console | TORGO val vs UASpeech val balance |
| `train_loss` | Console every 10 steps | Should decrease first ~1,000 steps then flatten |
| Control WER | Run `eval/eval_wer.py --split validation` on normal speech set | Should stay flat (LoRA forgetting check) |

---

## Intervention Knobs (without code changes)

| Scenario | Config change |
|---|---|
| TORGO val WER diverging up | `sampling.ua_frac: 0.55`, `sampling.torgo_frac: 0.45` |
| M08/M09/M10 causing loss instability | Add per-sample severity weights (needs code; see strategy §4) |
| Overfitting early | Reduce `lora.r: 8`, add `lora.dropout: 0.1` |
| Training collapse | Reduce `training.learning_rate: 5e-5` |
| Out of VRAM on L4 | Set `lora.decoder_only: true` (~30% VRAM saving) |
