#!/usr/bin/env python3
"""
finetune.py — Fine-tune openai/whisper-large-v3-turbo with LoRA for dysarthric speech.
Plug-and-play for RunPod (L4 / A40 / A100) and local dev (whisper-small, CPU).

Features:
  - YAML config + CLI overrides
  - HuggingFace Hub or local dataset loading
  - LoRA on all attention + FFN modules (configurable)
  - Gradient checkpointing for VRAM efficiency on 24 GB GPUs
  - 60/40 UASpeech:TORGO weighted sampling via WeightedRandomSampler (strategy §2)
  - Step-based training with max_steps as the primary stopping criterion (strategy §3)
  - Per-speaker WER logged at every evaluation checkpoint (strategy §4)
  - Early stopping when held-out speaker WER plateaus
  - Safety checkpoint every N steps (survives RunPod preemption)
  - Per-epoch WER logging vs your pre-measured baseline
  - Auto-resume from the latest checkpoint
  - TensorBoard + optional W&B
  - local_dev mode: single config flag → whisper-small on CPU for pipeline validation

Usage:
    python finetune.py                                    # defaults from config.yaml
    python finetune.py --local_dev                       # whisper-small, CPU, fast smoke run
    python finetune.py --model_name openai/whisper-small # override model only
    python finetune.py --max_steps 3000                  # override step budget
    python finetune.py --baseline_wer 45.3               # compare against baseline
    python finetune.py --resume_from_checkpoint auto     # resume after interruption
    python finetune.py --dataset_source local            # use local dataset instead
    python finetune.py --lora_rank 32 --epochs 40        # override specific values
"""

import argparse
import json
import logging
import math
import os
import re
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
import yaml
from datasets import Audio, DatasetDict, load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Transcript normalization (strategy §1) ───────────────────────────────────

_ACRONYM_MAP = {
    "tv": "t v", "pc": "p c", "usa": "u s a", "uk": "u k",
    "id": "i d", "ok": "okay",
}

# Per-dataset initial prompts — passed as prompt_ids to model.generate() so the
# model sees the context but never outputs it (WER computation stays clean).
_DATASET_PROMPTS = {
    "UASpeech": "Single isolated word.",
    "TORGO":    "Complete sentence.",
}

try:
    from num2words import num2words as _num2words
    def _num_to_word(m):
        try:
            return _num2words(int(m.group()))
        except (ValueError, TypeError):
            return m.group()
except ImportError:
    def _num_to_word(m):
        return m.group()


def _get_prompt_ids(processor, prompt_text: str, device):
    """Return 1-D prompt token tensor for model.generate(), or None on failure."""
    try:
        return processor.get_prompt_ids(prompt_text, return_tensors="pt").to(device)
    except Exception:
        return None


def normalize_transcript(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\.(?=[a-z])', '', text)          # strip dots inside acronyms
    text = re.sub(r'\b\d+\b', _num_to_word, text)   # numerals → words
    for acr, exp in _ACRONYM_MAP.items():
        text = re.sub(rf'\b{acr}\b', exp, text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.replace('-', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─── Graceful interrupt (Ctrl+C / SIGINT) ────────────────────────────────────
# Set by the signal handler; read by GracefulInterruptCallback on every step end.
_interrupt_training = False


def _install_interrupt_handler() -> None:
    """Replace default SIGINT so Ctrl+C finishes the current step, saves a
    checkpoint, then exits cleanly.  A second Ctrl+C forces an immediate exit."""
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(sig, frame):
        global _interrupt_training
        if _interrupt_training:
            log.warning("Second interrupt — forcing exit (checkpoint may be incomplete).")
            signal.signal(signal.SIGINT, original_handler)
            raise KeyboardInterrupt
        _interrupt_training = True
        log.warning(
            "\nInterrupt received — finishing current step, saving checkpoint, then stopping.\n"
            "  Press Ctrl+C again to force-quit without saving."
        )

    signal.signal(signal.SIGINT, handler)


# ─── CLI & Config ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA for dysarthric speech")
    p.add_argument("--config", default="config.yaml")
    # Dataset overrides
    p.add_argument("--dataset_source", choices=["huggingface", "local"])
    p.add_argument("--hf_dataset_id", type=str)
    p.add_argument("--local_dataset_path", type=str)
    # Model override
    p.add_argument("--model_name", type=str,
                   help="Override model (e.g., openai/whisper-small for local testing).")
    # Training overrides
    p.add_argument("--output_dir",    type=str)
    p.add_argument("--epochs",        type=int)
    p.add_argument("--max_steps",     type=int,
                   help="Total training steps (takes priority over --epochs).")
    p.add_argument("--batch_size",    type=int)
    p.add_argument("--grad_accum",    type=int)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--lora_rank",     type=int)
    p.add_argument("--lora_alpha",    type=int)
    # Local dev / CPU
    p.add_argument("--local_dev",  action="store_true",
                   help="Apply local_dev config overrides (whisper-small, small batch, fewer steps).")
    p.add_argument("--allow_cpu",  action="store_true",
                   help="Allow training on CPU when no GPU is available (very slow).")
    # Tracking
    p.add_argument("--baseline_wer", type=float,
                   help="Your pre-measured baseline WER%% to track improvement against.")
    # Auth & resume
    p.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Checkpoint path to resume from. "
                        "Omit to auto-resume from the latest checkpoint if one exists. "
                        "Pass 'none' to force a fresh start without deleting checkpoints.")
    p.add_argument("--fresh_start", action="store_true",
                   help="Delete all existing checkpoints and start training from scratch.")
    # W&B
    p.add_argument("--wandb_project", type=str)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def apply_local_dev_overrides(cfg: dict, force_local_dev: bool = False) -> dict:
    """
    Apply local_dev section when enabled (or --local_dev flag).
    Allows a single config change (local_dev.enabled: true) to switch the full
    pipeline to whisper-small on CPU for local validation.
    """
    ld = cfg.get("local_dev", {})
    if not (ld.get("enabled", False) or force_local_dev):
        return cfg

    log.info("LOCAL DEV MODE — whisper-small, CPU-friendly settings active")
    cfg["model"]["name"]                              = ld.get("model_name", "openai/whisper-small")
    cfg["training"]["per_device_train_batch_size"]    = ld.get("per_device_train_batch_size", 1)
    cfg["training"]["gradient_accumulation_steps"]    = ld.get("gradient_accumulation_steps", 1)
    cfg["training"]["max_steps"]                      = ld.get("max_steps", 200)
    cfg["training"]["bf16"]                           = ld.get("bf16", False)
    cfg["training"]["fp16"]                           = ld.get("fp16", False)
    cfg["training"]["warmup_steps"]                   = ld.get("warmup_steps", 10)
    cfg["training"]["eval_steps"]                     = ld.get("eval_steps", 50)
    cfg["training"]["save_steps"]                     = ld.get("save_steps", 50)
    cfg["training"]["gradient_checkpointing"]         = ld.get("gradient_checkpointing", False)
    cfg["training"]["dataloader_num_workers"]         = ld.get("dataloader_num_workers", 0)
    if "dataset_source" in ld:
        cfg["dataset"]["source"]                      = ld["dataset_source"]
    return cfg


def load_config(config_path: str, args) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # local_dev overrides first so explicit CLI args can still win below.
    cfg = apply_local_dev_overrides(cfg, force_local_dev=getattr(args, "local_dev", False))

    cli_map = [
        (("dataset",    "source"),                       args.dataset_source),
        (("dataset",    "hf_dataset_id"),                args.hf_dataset_id),
        (("dataset",    "local_path"),                   args.local_dataset_path),
        (("output",     "dir"),                          args.output_dir),
        (("model",      "name"),                         args.model_name),
        (("training",   "epochs"),                       args.epochs),
        (("training",   "max_steps"),                    args.max_steps),
        (("training",   "per_device_train_batch_size"),  args.batch_size),
        (("training",   "gradient_accumulation_steps"),  args.grad_accum),
        (("training",   "learning_rate"),                args.learning_rate),
        (("lora",       "r"),                            args.lora_rank),
        (("lora",       "alpha"),                        args.lora_alpha),
        (("evaluation", "baseline_wer"),                 args.baseline_wer),
    ]
    for (section, key), value in cli_map:
        if value is not None:
            cfg[section][key] = value

    # allow_cpu flag sets the internal sentinel independently of local_dev.
    if getattr(args, "allow_cpu", False):
        cfg["_allow_cpu"] = True

    return cfg


# ─── Dataset ──────────────────────────────────────────────────────────────────

def load_data(cfg: dict, hf_token: Optional[str]) -> DatasetDict:
    source    = cfg["dataset"]["source"]
    audio_col = cfg["dataset"]["audio_column"]
    sr        = cfg["dataset"]["sampling_rate"]

    if source == "huggingface":
        repo = cfg["dataset"]["hf_dataset_id"]
        log.info(f"Loading dataset from HF Hub: {repo}")
        ds = load_dataset(repo, token=hf_token)
    elif source == "local":
        path = cfg["dataset"]["local_path"]
        log.info(f"Loading dataset from local disk: {path}")
        ds = load_from_disk(path)
    else:
        raise ValueError(f"dataset.source must be 'huggingface' or 'local', got: {source!r}")

    # DO NOT cast Audio feature yet — this triggers torchcodec on Windows when
    # reading speaker_id strings. We'll cast after speaker-based split.

    val_split_name   = cfg["dataset"]["validation_split"]
    train_split_name = cfg["dataset"]["train_split"]
    val_speakers     = cfg["dataset"].get("validation_speakers") or []

    if val_speakers:
        speaker_col = cfg["dataset"].get("speaker_column", "speaker_id")
        val_set     = set(val_speakers)

        all_train_rows = ds[train_split_name]

        if speaker_col not in all_train_rows.column_names:
            raise ValueError(
                f"Column '{speaker_col}' not found in dataset. "
                f"Available columns: {all_train_rows.column_names}. "
                f"Set dataset.speaker_column in config.yaml."
            )

        log.info(f"Speaker-based split — validation speakers: {val_speakers}")
        # Use select() with pre-computed indices instead of filter() so we only
        # read the speaker_id string column — filter() decodes the full example
        # including audio, which triggers the audio backend (torchcodec on Windows).
        speaker_ids = all_train_rows[speaker_col]
        val_idx   = [i for i, s in enumerate(speaker_ids) if s in val_set]
        train_idx = [i for i, s in enumerate(speaker_ids) if s not in val_set]
        val_ds_raw   = all_train_rows.select(val_idx)
        train_ds_raw = all_train_rows.select(train_idx)
        log.info(
            f"  Train: {len(train_ds_raw)} samples "
            f"({len(set(train_ds_raw[speaker_col]))} speakers) | "
            f"Val: {len(val_ds_raw)} samples "
            f"({len(set(val_ds_raw[speaker_col]))} speakers)"
        )
        ds = DatasetDict({
            train_split_name: train_ds_raw,
            val_split_name:   val_ds_raw,
            **{k: v for k, v in ds.items() if k not in (train_split_name, val_split_name, "test")},
        })

    elif val_split_name not in ds:
        val_pct = cfg["dataset"].get("validation_pct", 0.1)
        seed    = cfg["dataset"].get("seed", 42)
        log.info(
            f"No '{val_split_name}' split and no validation_speakers set — "
            f"carving {val_pct*100:.0f}% from '{train_split_name}' (seed={seed})"
        )
        split_result = ds[train_split_name].train_test_split(test_size=val_pct, seed=seed)
        ds = DatasetDict({
            train_split_name: split_result["train"],
            val_split_name:   split_result["test"],
            **{k: v for k, v in ds.items() if k != train_split_name},
        })

    log.info(f"Dataset: {ds}")
    return ds


def make_preprocess_fn(processor: WhisperProcessor, cfg: dict):
    audio_col = cfg["dataset"]["audio_column"]
    text_col  = cfg["dataset"]["text_column"]
    sr        = cfg["dataset"]["sampling_rate"]

    def preprocess(example):
        import io
        import soundfile as sf

        audio_data = example[audio_col]

        if isinstance(audio_data, dict) and "array" in audio_data:
            # Already decoded (datasets<4.x or decode=True path)
            array     = np.array(audio_data["array"], dtype=np.float32)
            actual_sr = audio_data.get("sampling_rate", sr)
        elif isinstance(audio_data, dict):
            # decode=False: {"path": ..., "bytes": ...}
            if audio_data.get("bytes"):
                array, actual_sr = sf.read(io.BytesIO(audio_data["bytes"]), dtype="float32")
            else:
                array, actual_sr = sf.read(audio_data["path"], dtype="float32")
        elif isinstance(audio_data, str):
            array, actual_sr = sf.read(audio_data, dtype="float32")
        else:
            array     = np.array(audio_data, dtype=np.float32)
            actual_sr = sr

        if array.ndim > 1:
            array = array.mean(axis=1)

        if actual_sr != sr:
            import librosa
            array = librosa.resample(array, orig_sr=actual_sr, target_sr=sr)

        input_features = processor.feature_extractor(
            array, sampling_rate=sr
        ).input_features[0]

        labels = processor.tokenizer(normalize_transcript(str(example[text_col]))).input_ids

        return {"input_features": input_features, "labels": labels}

    return preprocess


def preprocess_split(ds_split, processor, cfg, desc: str):
    # Use num_proc=0 to avoid multiprocessing audio decoding issues on Windows.
    # For RunPod (Linux), set num_proc from config or default to 4.
    num_proc = 0 if cfg.get("_allow_cpu") else cfg["training"].get("dataloader_num_workers", 4)

    # Cast to Audio(decode=False) so datasets returns raw {path, bytes} dicts
    # instead of routing through its audio backend (torchcodec in datasets>=4.x).
    # Our preprocess function decodes via soundfile directly.
    audio_col = cfg["dataset"]["audio_column"]
    if audio_col in ds_split.column_names:
        from datasets import Audio as _Audio
        ds_split = ds_split.cast_column(audio_col, _Audio(decode=False))

    return ds_split.map(
        make_preprocess_fn(processor, cfg),
        remove_columns=ds_split.column_names,
        num_proc=num_proc,
        desc=desc,
    )


# ─── Weighted Sampling (strategy §2) ─────────────────────────────────────────

def compute_sample_weights(raw_ds, ua_frac: float = 0.60, torgo_frac: float = 0.40,
                           dataset_col: str = "dataset") -> Optional[List[float]]:
    """
    Per-sample weights so that a WeightedRandomSampler draws 60% UASpeech / 40% TORGO
    at every step regardless of the raw corpus size imbalance.

    Returns None if the dataset column is missing, causing DysarthriaTrainer to fall
    back to uniform sampling with a warning.
    """
    if dataset_col not in raw_ds.column_names:
        log.warning(
            f"Column '{dataset_col}' not found in train split — falling back to uniform "
            f"sampling (no 60/40 ratio). Available: {raw_ds.column_names}"
        )
        return None

    target: Dict[str, float] = {"UASpeech": ua_frac, "TORGO": torgo_frac}
    sources = raw_ds[dataset_col]

    counts: Dict[str, int] = {}
    for s in sources:
        counts[s] = counts.get(s, 0) + 1

    log.info(
        f"Weighted sampling — corpus counts: {counts} | "
        f"target fractions: {target} | "
        f"TORGO repeat factor vs UA: "
        f"{(counts.get('UASpeech',1)/max(counts.get('TORGO',1),1)) * (torgo_frac/ua_frac):.2f}×"
    )

    weights: List[float] = []
    for s in sources:
        frac = target.get(s, 1.0 / max(len(counts), 1))
        weights.append(frac / counts[s])

    return weights


# ─── Per-Speaker Val Data (strategy §4) ──────────────────────────────────────

def build_speaker_val_data(raw_val_ds, audio_col: str, text_col: str,
                           speaker_col: str, sr: int) -> Dict[str, list]:
    """
    Extract per-speaker audio+transcript lists from the raw (unpreprocessed) val dataset.
    Used by PerSpeakerWERCallback to run per-speaker WER evaluation during training
    without re-preprocessing or interfering with the main evaluation loop.
    """
    if speaker_col not in raw_val_ds.column_names:
        log.warning(
            f"Column '{speaker_col}' not in val split — per-speaker WER callback disabled. "
            f"Available: {raw_val_ds.column_names}"
        )
        return {}

    data: Dict[str, list] = {}
    for sample in raw_val_ds:
        spk   = sample.get(speaker_col, "unknown")
        audio = sample[audio_col]
        try:
            if isinstance(audio, dict):
                array     = np.array(audio["array"], dtype=np.float32)
                actual_sr = audio.get("sampling_rate", sr)
            else:
                import soundfile as sf
                array, actual_sr = sf.read(str(audio), dtype="float32")
        except Exception:
            continue

        if array.ndim > 1:
            array = array.mean(axis=1)
        if actual_sr != sr:
            import librosa
            array = librosa.resample(array, orig_sr=actual_sr, target_sr=sr)

        if spk not in data:
            data[spk] = []
        data[spk].append({
            "array":      array,
            "transcript": normalize_transcript(str(sample[text_col])),
            "dataset":    sample.get("dataset", ""),
        })

    total = sum(len(v) for v in data.values())
    log.info(
        f"Per-speaker val data: {total} samples across {len(data)} speakers — "
        f"{sorted(data.keys())}"
    )
    return data


# ─── Data Collator ────────────────────────────────────────────────────────────

@dataclass
class SpeechCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, torch.Tensor]:
        input_feats  = [{"input_features": f["input_features"]} for f in features]
        label_feats  = [{"input_ids": f["labels"]}              for f in features]

        batch        = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        decoder_attn_mask = labels_batch.attention_mask
        # Strip the BOS token that WhisperTokenizer prepends — the model generates it.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
            decoder_attn_mask = decoder_attn_mask[:, 1:]

        batch["labels"] = labels
        batch["decoder_attention_mask"] = decoder_attn_mask
        return batch


# ─── Metrics ──────────────────────────────────────────────────────────────────

def build_compute_metrics(processor: WhisperProcessor):
    wer_metric = evaluate.load("wer")
    tokenizer  = processor.tokenizer

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True, clean_up_tokenization_spaces=False)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        pred_str  = [normalize_transcript(s) for s in pred_str]
        label_str = [normalize_transcript(s) for s in label_str]

        pairs = [(p, l) for p, l in zip(pred_str, label_str) if l]
        if not pairs:
            return {"wer": 100.0}
        preds, labels = zip(*pairs)

        wer = wer_metric.compute(predictions=list(preds), references=list(labels))
        return {"wer": round(100.0 * wer, 3)}

    return compute_metrics


# ─── Callbacks ────────────────────────────────────────────────────────────────

class BaselineWERLogger(TrainerCallback):
    """Log WER vs baseline and build a history for the final summary."""

    def __init__(self, baseline_wer: Optional[float]):
        self.baseline_wer = baseline_wer
        self.history: List[tuple] = []
        self.best_wer   = float("inf")
        self.best_epoch = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        wer = metrics.get("eval_wer")
        if wer is None:
            return

        epoch = round(state.epoch or 0, 1)
        self.history.append((epoch, wer))

        if wer < self.best_wer:
            self.best_wer   = wer
            self.best_epoch = epoch

        if self.baseline_wer is not None:
            delta_pp  = self.baseline_wer - wer
            delta_rel = delta_pp / self.baseline_wer * 100
            arrow     = "↓" if delta_pp >= 0 else "↑"
            log.info(
                f"[WER] Step {state.global_step} | {wer:.2f}% "
                f"(baseline {self.baseline_wer:.2f}% | {arrow}{abs(delta_pp):.2f}pp | {delta_rel:+.1f}% rel)"
            )
        else:
            log.info(f"[WER] Step {state.global_step} | {wer:.2f}%")


class SafetyCheckpointCallback(TrainerCallback):
    """Save a checkpoint every N steps regardless of eval boundaries.
    Protects against unexpected RunPod pod termination mid-epoch."""

    def __init__(self, every_n_steps: int):
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.every_n_steps > 0 and state.global_step % self.every_n_steps == 0:
            control.should_save = True
        return control


class GracefulInterruptCallback(TrainerCallback):
    """Catch Ctrl+C / SIGINT mid-training: finish the current step, save a full
    checkpoint (model + optimizer + scheduler + RNG state), then stop cleanly.
    Re-running `python finetune.py` will auto-resume from this checkpoint."""

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if _interrupt_training:
            log.info(f"Graceful stop — saving checkpoint at step {state.global_step}.")
            control.should_save = True
            control.should_training_stop = True
        return control


class PerSpeakerWERCallback(TrainerCallback):
    """
    Log per-speaker WER after each evaluation checkpoint (strategy §4).

    Runs inference directly on the raw (pre-preprocessed) validation audio so that
    speaker grouping is preserved without modifying the main Trainer eval loop.
    Limited to MAX_SAMPLES_PER_SPEAKER per speaker to keep evaluation overhead manageable.
    """

    MAX_SAMPLES_PER_SPEAKER = 30

    def __init__(self, speaker_val_data: Dict[str, list],
                 processor: WhisperProcessor, sr: int = 16000):
        self.speaker_val_data = speaker_val_data
        self.processor        = processor
        self.sr               = sr
        self._wer_metric      = evaluate.load("wer")

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl,
                    model=None, **kwargs):
        if model is None or not self.speaker_val_data:
            return

        device = next(model.parameters()).device
        model.eval()
        results: Dict[str, float] = {}

        for spk, samples in sorted(self.speaker_val_data.items()):
            refs: List[str] = []
            hyps: List[str] = []
            for s in samples[:self.MAX_SAMPLES_PER_SPEAKER]:
                feat_out = self.processor.feature_extractor(
                    s["array"], sampling_rate=self.sr,
                    return_tensors="pt", return_attention_mask=True,
                )
                feats    = feat_out.input_features.to(device)
                attn_mask = feat_out.attention_mask.to(device)
                prompt_text = _DATASET_PROMPTS.get(s.get("dataset", ""))
                prompt_ids  = (_get_prompt_ids(self.processor, prompt_text, device)
                               if prompt_text else None)
                with torch.no_grad():
                    ids = model.generate(
                        feats,
                        attention_mask=attn_mask,
                        forced_decoder_ids=None,
                        prompt_ids=prompt_ids,
                    )
                hyp = self.processor.tokenizer.batch_decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                refs.append(s["transcript"])
                hyps.append(normalize_transcript(hyp))

            if refs:
                results[spk] = round(
                    self._wer_metric.compute(predictions=hyps, references=refs) * 100, 2
                )

        if results:
            summary = " | ".join(f"{k}: {v:.1f}%" for k, v in sorted(results.items()))
            log.info(f"[Per-speaker WER @ step {state.global_step}] {summary}")

        model.train()


# ─── Weighted Sampler Trainer (strategy §2) ──────────────────────────────────

class DysarthriaTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer with 60/40 UASpeech:TORGO weighted sampling.

    Overrides get_train_dataloader to inject a WeightedRandomSampler whose
    num_samples = max_steps × effective_batch_size, matching the step budget
    rather than the raw dataset size (strategy §2: "not len(dataset)").
    Falls back to standard uniform sampling when sample_weights is None.
    """

    def __init__(self, *args, sample_weights: Optional[List[float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_weights = sample_weights

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # decoder_attention_mask is valid during the training forward pass (model(**inputs))
        # but must be stripped before model.generate() in eval: Whisper's no-speech logits
        # processor captures model_kwargs and re-runs the model, causing a batch-size
        # mismatch (ValueError) when decoder_attention_mask is present in those kwargs.
        inputs = {k: v for k, v in inputs.items() if k != "decoder_attention_mask"}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def get_train_dataloader(self):
        if self._sample_weights is None:
            return super().get_train_dataloader()

        from torch.utils.data import DataLoader, WeightedRandomSampler

        max_steps = self.args.max_steps if self.args.max_steps > 0 else 5000
        eff_batch = (self.args.per_device_train_batch_size
                     * self.args.gradient_accumulation_steps)
        num_samples = max_steps * eff_batch

        sampler = WeightedRandomSampler(
            weights=self._sample_weights,
            num_samples=num_samples,
            replacement=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=getattr(self.args, "dataloader_pin_memory", False),
        )


# ─── Model Setup ──────────────────────────────────────────────────────────────

def load_model_with_lora(cfg: dict, hf_token: Optional[str]):
    model_name = cfg["model"]["name"]
    language   = cfg["model"]["language"]
    task       = cfg["model"]["task"]
    lcfg       = cfg["lora"]

    log.info(f"Loading processor + model: {model_name}")
    processor = WhisperProcessor.from_pretrained(
        model_name, language=language, task=task, token=hf_token
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16,
    )

    model.generation_config.language           = language
    model.generation_config.task               = task
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens               = []
    model.config.use_cache                     = False
    # Explicitly set pad_token_id to prevent attention mask inference warning
    # (Whisper has pad_token_id == eos_token_id, which confuses transformers)
    model.config.pad_token_id                  = processor.tokenizer.pad_token_id
    model.generation_config.pad_token_id       = processor.tokenizer.pad_token_id
    # suppress_tokens: set ONLY on model.config — Whisper's generate() reads it from
    # there internally.  Setting on generation_config AS WELL creates a duplicate
    # SuppressTokensLogitsProcessor, triggering the "created twice" warning.
    # (model.config.suppress_tokens = [] is already set above)
    #
    # Dysarthric-speech generation settings (strategy §7)
    model.generation_config.no_repeat_ngram_size       = 3
    model.generation_config.repetition_penalty         = 1.2
    # None = never suppress segment as "no speech" — better for dysarthric audio
    # (non-None values activate WhisperNoSpeechDetection, a model-calling logits
    # processor that crashes with a batch-size mismatch during trainer eval)
    model.generation_config.no_speech_threshold        = None
    # None = disable fallback mechanism — non-None triggers _need_fallback which
    # inspects seek_outputs[index]["scores"] expecting a dict, but trainer's
    # generate() returns a plain tensor → IndexError crash
    model.generation_config.logprob_threshold          = None
    model.generation_config.compression_ratio_threshold = None
    model.generation_config.condition_on_prev_tokens   = False
    # Explicitly None-out suppress_tokens on generation_config — the pretrained
    # defaults may already have them set, which creates a duplicate
    # SuppressTokensLogitsProcessor alongside the one Whisper creates internally.
    model.generation_config.suppress_tokens            = None
    model.generation_config.begin_suppress_tokens      = None

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Base model parameters: {total_params:,}")

    if lcfg.get("decoder_only", False):
        log.info("Freezing encoder weights (decoder_only=true in config).")
        for _, param in model.model.encoder.named_parameters():
            param.requires_grad = False

    target_modules = lcfg["target_modules"]
    log.info(f"Applying LoRA: r={lcfg['r']}, alpha={lcfg['alpha']}, "
             f"target_modules={target_modules}")

    lora_config = LoraConfig(
        r              = lcfg["r"],
        lora_alpha     = lcfg["alpha"],
        lora_dropout   = lcfg["dropout"],
        bias           = lcfg["bias"],
        target_modules = target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


# ─── Training Arg Builder ─────────────────────────────────────────────────────

def build_training_args(cfg: dict, no_wandb: bool, pin_memory: bool = True) -> Seq2SeqTrainingArguments:
    tcfg       = cfg["training"]
    output_dir = cfg["output"]["dir"]

    report_to = []
    if cfg["logging"].get("tensorboard", True):
        report_to.append("tensorboard")
    if cfg["logging"].get("wandb", False) and not no_wandb:
        report_to.append("wandb")
    if not report_to:
        report_to = ["none"]

    return Seq2SeqTrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = tcfg["epochs"],
        max_steps                   = tcfg.get("max_steps", -1),
        per_device_train_batch_size = tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size  = tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps = tcfg["gradient_accumulation_steps"],
        learning_rate               = tcfg["learning_rate"],
        lr_scheduler_type           = tcfg["lr_scheduler_type"],
        warmup_steps                = tcfg["warmup_steps"],
        weight_decay                = tcfg["weight_decay"],
        bf16                        = tcfg.get("bf16", False),
        fp16                        = tcfg.get("fp16", False),
        gradient_checkpointing      = tcfg.get("gradient_checkpointing", True),
        eval_strategy               = tcfg["eval_strategy"],
        eval_steps                  = tcfg.get("eval_steps", 500),
        save_strategy               = tcfg["save_strategy"],
        save_steps                  = tcfg.get("save_steps", 500),
        logging_steps               = tcfg["logging_steps"],
        save_total_limit            = tcfg["save_total_limit"],
        load_best_model_at_end      = tcfg["load_best_model_at_end"],
        metric_for_best_model       = tcfg["metric_for_best_model"],
        greater_is_better           = tcfg["greater_is_better"],
        predict_with_generate       = tcfg["predict_with_generate"],
        generation_max_length       = tcfg.get("generation_max_length", 225),
        dataloader_num_workers      = tcfg.get("dataloader_num_workers", 12),
        dataloader_pin_memory       = pin_memory,
        logging_dir                 = str(Path(output_dir) / "logs"),
        report_to                   = report_to,
        remove_unused_columns       = False,
        label_names                 = ["labels"],
    )


def resolve_checkpoint(resume_arg: Optional[str], output_dir: str) -> Optional[str]:
    """Return the checkpoint path to pass to trainer.train(), or None to start fresh.

    Default (resume_arg=None): auto-resume from the latest checkpoint in output_dir.
    'auto': same as None.
    'none': explicit fresh start — ignore any existing checkpoints.
    Any other string: treat as a literal checkpoint directory path.
    Use --fresh_start to delete checkpoints before calling this.
    """
    if resume_arg == "none":
        log.info("Fresh start — ignoring any existing checkpoints.")
        return None

    ckpts = sorted(
        (p for p in Path(output_dir).glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )

    if not ckpts:
        log.info(f"No checkpoints found in {output_dir} — starting fresh.")
        return None

    if resume_arg is None or resume_arg == "auto":
        log.info(f"Auto-resuming from latest checkpoint: {ckpts[-1].name}")
        return str(ckpts[-1])

    return resume_arg  # explicit path provided by user


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config, args)

    hf_token  = args.hf_token or os.environ.get("HF_TOKEN")
    allow_cpu = cfg.get("_allow_cpu", False)

    # ── GPU check ──────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        if allow_cpu:
            log.warning(
                "No CUDA GPU detected — running on CPU. "
                "Training will be very slow; this is only suitable for local pipeline validation."
            )
        else:
            log.error(
                "No CUDA GPU detected.\n"
                "  → Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
                "  → To run on CPU anyway (very slow): pass --allow_cpu"
            )
            sys.exit(1)
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log.info(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB | PyTorch CUDA: {torch.version.cuda}")

    # Print run summary.
    eff_batch = (cfg["training"]["per_device_train_batch_size"]
                 * cfg["training"]["gradient_accumulation_steps"])
    max_steps = cfg["training"].get("max_steps", -1)

    print("\n" + "=" * 62)
    print("  Whisper LoRA Fine-tuning — Dysarthric ASR")
    print("=" * 62)
    print(f"  Model        : {cfg['model']['name']}")
    print(f"  Dataset      : {cfg['dataset']['source']} "
          f"({cfg['dataset'].get('hf_dataset_id', cfg['dataset']['local_path'])})")
    print(f"  LoRA rank    : {cfg['lora']['r']} | alpha: {cfg['lora']['alpha']}")
    print(f"  Batch        : {cfg['training']['per_device_train_batch_size']} × "
          f"{cfg['training']['gradient_accumulation_steps']} accum = {eff_batch} effective")
    print(f"  LR           : {cfg['training']['learning_rate']}")
    print(f"  Max steps    : {max_steps if max_steps > 0 else 'unlimited (epoch-based)'}")
    print(f"  Eval every   : {cfg['training'].get('eval_steps', 500)} steps")
    print(f"  bf16         : {cfg['training'].get('bf16', False)}")
    print(f"  Grad ckpt    : {cfg['training'].get('gradient_checkpointing', True)}")
    print(f"  Sampling     : {'60/40 UASpeech:TORGO' if cfg.get('sampling', {}).get('enabled') else 'uniform'}")
    print(f"  Early stop   : patience={cfg['early_stopping']['patience']} eval checkpoints")
    baseline = cfg["evaluation"].get("baseline_wer")
    if baseline:
        print(f"  Baseline WER : {baseline:.2f}%")
    print("=" * 62 + "\n")

    # ── W&B ────────────────────────────────────────────────────────────────
    if args.wandb_project and not args.no_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    elif args.no_wandb or not cfg["logging"].get("wandb", False):
        os.environ["WANDB_DISABLED"] = "true"

    # ── Dataset ────────────────────────────────────────────────────────────
    ds          = load_data(cfg, hf_token)
    train_split = cfg["dataset"]["train_split"]
    val_split   = cfg["dataset"]["validation_split"]

    # ── Weighted sampling weights (computed on raw train before preprocessing) ──
    sample_weights: Optional[List[float]] = None
    scfg = cfg.get("sampling", {})
    if scfg.get("enabled", False):
        sample_weights = compute_sample_weights(
            ds[train_split],
            ua_frac     = scfg.get("ua_frac",     0.60),
            torgo_frac  = scfg.get("torgo_frac",  0.40),
            dataset_col = scfg.get("dataset_column", "dataset"),
        )

    # ── Per-speaker val data (built from raw val before preprocessing) ─────
    speaker_col      = cfg["dataset"].get("speaker_column", "speaker_id")
    try:
        speaker_val_data = build_speaker_val_data(
            ds[val_split],
            audio_col   = cfg["dataset"]["audio_column"],
            text_col    = cfg["dataset"]["text_column"],
            speaker_col = speaker_col,
            sr          = cfg["dataset"]["sampling_rate"],
        )
    except Exception as e:
        log.warning(
            f"Per-speaker WER evaluation disabled: {type(e).__name__}: {e}\n"
            f"(This is expected on local dev if FFmpeg/torchcodec is unavailable.)"
        )
        speaker_val_data = {}

    # ── Model + LoRA ───────────────────────────────────────────────────────
    model, processor = load_model_with_lora(cfg, hf_token)

    if cfg["training"].get("gradient_checkpointing", True):
        model.enable_input_require_grads()

    # ── Preprocess ─────────────────────────────────────────────────────────
    log.info("Preprocessing train split...")
    train_ds = preprocess_split(ds[train_split], processor, cfg, "train")

    log.info("Preprocessing validation split...")
    val_ds = preprocess_split(ds[val_split], processor, cfg, "validation")

    log.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── Collator ───────────────────────────────────────────────────────────
    collator = SpeechCollator(
        processor              = processor,
        decoder_start_token_id = model.config.decoder_start_token_id,
    )

    # ── Training args ──────────────────────────────────────────────────────
    training_args = build_training_args(cfg, args.no_wandb, pin_memory=torch.cuda.is_available())

    # ── Callbacks ──────────────────────────────────────────────────────────
    wer_logger = BaselineWERLogger(baseline_wer=baseline)
    callbacks  = [wer_logger]

    if cfg["early_stopping"]["enabled"]:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience  = cfg["early_stopping"]["patience"],
            early_stopping_threshold = cfg["early_stopping"]["min_delta"],
        ))

    safety_steps = cfg["training"].get("safety_checkpoint_steps", 0)
    if safety_steps > 0:
        callbacks.append(SafetyCheckpointCallback(every_n_steps=safety_steps))

    if speaker_val_data:
        callbacks.append(PerSpeakerWERCallback(
            speaker_val_data = speaker_val_data,
            processor        = processor,
            sr               = cfg["dataset"]["sampling_rate"],
        ))

    # GracefulInterruptCallback must be last so Ctrl+C can override any other
    # callback that sets should_training_stop = False.
    _install_interrupt_handler()
    callbacks.append(GracefulInterruptCallback())

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = DysarthriaTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        compute_metrics = build_compute_metrics(processor),
        callbacks       = callbacks,
        sample_weights  = sample_weights,
    )

    # ── Checkpoint resolution ──────────────────────────────────────────────
    if args.fresh_start:
        import shutil
        out_path   = Path(cfg["output"]["dir"])
        to_remove  = sorted(out_path.glob("checkpoint-*")) if out_path.exists() else []
        for ckpt in to_remove:
            shutil.rmtree(ckpt)
            log.info(f"Removed checkpoint: {ckpt.name}")
        if to_remove:
            log.info(f"Fresh start: removed {len(to_remove)} checkpoint(s).")
        checkpoint = None
    else:
        checkpoint = resolve_checkpoint(args.resume_from_checkpoint, cfg["output"]["dir"])

    log.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    # ── Final eval ─────────────────────────────────────────────────────────
    log.info("Running final evaluation on validation set...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    final_wer = eval_metrics.get("eval_wer", float("inf"))

    # ── Merge LoRA and save ────────────────────────────────────────────────
    merged_dir = cfg["output"]["merged_model_dir"]
    log.info(f"Merging LoRA adapters and saving to: {merged_dir}")
    Path(merged_dir).mkdir(parents=True, exist_ok=True)

    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)

    # ── Save training summary JSON ─────────────────────────────────────────
    summary = {
        "model_name":            cfg["model"]["name"],
        "lora_r":                cfg["lora"]["r"],
        "lora_alpha":            cfg["lora"]["alpha"],
        "lora_target_modules":   cfg["lora"]["target_modules"],
        "total_steps_trained":   train_metrics.get("train_steps", "N/A"),
        "effective_batch_size":  eff_batch,
        "learning_rate":         cfg["training"]["learning_rate"],
        "train_loss":            round(train_metrics.get("train_loss", 0), 4),
        "final_eval_wer":        round(final_wer, 3),
        "best_eval_wer":         round(wer_logger.best_wer, 3),
        "best_wer_at_step":      wer_logger.best_epoch,
        "baseline_wer":          baseline,
        "wer_improvement_pp":    round(baseline - wer_logger.best_wer, 3) if baseline else None,
        "wer_improvement_pct":   round((baseline - wer_logger.best_wer) / baseline * 100, 1) if baseline else None,
        "wer_history":           [(e, w) for e, w in wer_logger.history],
        "weighted_sampling":     scfg.get("enabled", False),
        "sampling_ua_frac":      scfg.get("ua_frac", None),
    }
    summary_path = Path(cfg["output"]["dir"]) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    # ── Final report ───────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Training Complete!")
    print("=" * 62)
    print(f"  Best Val WER  : {wer_logger.best_wer:.2f}% (step {wer_logger.best_epoch})")
    if baseline:
        improvement = baseline - wer_logger.best_wer
        print(f"  Baseline WER  : {baseline:.2f}%")
        print(f"  Improvement   : {improvement:.2f}pp ({improvement/baseline*100:.1f}% relative)")
    print(f"  Merged model  : {merged_dir}")
    print(f"  Summary JSON  : {summary_path}")
    print(f"\n  Next step: python eval/eval_wer.py --model_path {merged_dir} --baseline_wer <N>"
          f"\n             (outputs saved to inference-outputs/evaluation_results.json)")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
