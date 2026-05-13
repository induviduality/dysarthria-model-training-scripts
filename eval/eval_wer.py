#!/usr/bin/env python3
"""
evaluate.py — Evaluate a fine-tuned Whisper model and compare WER to baseline.

Supports two model types:
  --model_type hf   : Use the merged HuggingFace model (whisper-turbo-merged/).
  --model_type ct2  : Use the CTranslate2 faster-whisper model (whisper-turbo-ct2/).

Usage:
    # After fine-tuning, evaluate the HF merged model:
    python evaluate.py --model_path ./whisper-turbo-merged --model_type hf

    # After CT2 conversion, evaluate the faster-whisper model:
    python evaluate.py --model_path ./whisper-turbo-ct2 --model_type ct2

    # Compare to a known baseline WER:
    python evaluate.py --model_path ./whisper-turbo-merged --baseline_wer 45.3

    # Evaluate on a specific HF dataset split:
    python evaluate.py --model_path ./whisper-turbo-merged --split test
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_ACRONYM_MAP = {
    "tv": "t v", "pc": "p c", "usa": "u s a", "uk": "u k",
    "id": "i d", "ok": "okay",
}

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


def normalize_transcript(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\.(?=[a-z])', '', text)
    text = re.sub(r'\b\d+\b', _num_to_word, text)
    for acr, exp in _ACRONYM_MAP.items():
        text = re.sub(rf'\b{acr}\b', exp, text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.replace('-', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

import numpy as np
import yaml
from datasets import Audio, load_dataset, load_from_disk
from jiwer import cer, wer


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Whisper model WER vs baseline")
    p.add_argument("--model_path", required=True,
                   help="Path to the merged HF model or CT2 model directory.")
    p.add_argument("--model_type", choices=["hf", "ct2"], default="hf",
                   help="'hf' for HuggingFace merged model, 'ct2' for CTranslate2 faster-whisper.")
    p.add_argument("--config",      default="config.yaml")
    p.add_argument("--split",       default=None,
                   help="Dataset split to evaluate (overrides config validation_split).")
    p.add_argument("--baseline_wer", type=float, default=None,
                   help="Baseline WER%% for comparison.")
    p.add_argument("--output_json", default="./inference-outputs/evaluation_results.json")
    p.add_argument("--hf_token",    default=os.environ.get("HF_TOKEN"))
    p.add_argument("--beam_size",   type=int, default=5)
    p.add_argument("--language",    default="en")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit evaluation to N samples (useful for quick checks).")
    p.add_argument("--initial_prompt", type=str, default=None,
                   help="Initial prompt for Whisper (e.g. 'Complete sentence.'). "
                        "Auto-selected per sample when 'dataset' column is present.")
    return p.parse_args()


def load_cfg(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_audio(audio_data, target_sr: int) -> tuple:
    """Return (array, sr) from any audio_data form datasets may return."""
    import io
    import soundfile as sf

    if isinstance(audio_data, dict) and "array" in audio_data:
        array = np.array(audio_data["array"], dtype=np.float32)
        sr    = audio_data.get("sampling_rate", target_sr)
    elif isinstance(audio_data, dict):
        # decode=False: {"path": ..., "bytes": ...}
        if audio_data.get("bytes"):
            array, sr = sf.read(io.BytesIO(audio_data["bytes"]), dtype="float32")
        else:
            array, sr = sf.read(audio_data["path"], dtype="float32")
    else:
        array, sr = sf.read(str(audio_data), dtype="float32")

    if array.ndim > 1:
        array = array.mean(axis=1)
    return array, sr


# ─── HF model evaluator ───────────────────────────────────────────────────────

def _get_prompt_ids(processor, prompt_text: str, device):
    try:
        return processor.get_prompt_ids(prompt_text, return_tensors="pt").to(device)
    except Exception:
        return None


def evaluate_hf(model_path: str, samples: List[dict], audio_col: str, text_col: str,
                language: str, beam_size: int, initial_prompt: Optional[str] = None):
    """
    Transcribe audio using a HuggingFace merged Whisper model.
    Returns (best_hypotheses, per_sample_top3, per_sample_token_data).
      best_hypotheses  — List[str], rank-1 transcript per sample (used for WER).
      per_sample_top3  — List[List[dict]], top-3 beam candidates per sample.
      per_sample_tokens— List[List[dict]], per-token confidence for best hypothesis.
    """
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device       = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading HF model from: {model_path}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path, dtype=compute_type, low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_path)

    model.generation_config.language = language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = None
    model.generation_config.begin_suppress_tokens = None
    model.generation_config.no_speech_threshold = None
    model.generation_config.logprob_threshold = None
    model.generation_config.compression_ratio_threshold = None
    model.generation_config.condition_on_prev_tokens = False
    # Explicitly set pad_token_id to prevent attention mask inference warning
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    # Clear saved max_length so max_new_tokens (passed per-call) doesn't warn.
    model.generation_config.max_length = None

    num_ret = min(3, beam_size)
    best_hypotheses: List[str] = []
    per_sample_top3: List[List[dict]] = []
    per_sample_tokens: List[List[dict]] = []

    for i, sample in enumerate(samples):
        array, sr = _load_audio(sample[audio_col], 16000)
        if sr != 16000:
            import librosa
            array = librosa.resample(array, orig_sr=sr, target_sr=16000)

        feat_out  = processor.feature_extractor(
            array, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        features  = feat_out.input_features.to(device=device, dtype=compute_type)
        attn_mask = feat_out.attention_mask.to(device)

        # Resolve prompt: per-sample dataset column wins, CLI arg is fallback
        dataset_name = sample.get("dataset", "")
        prompt_text  = _DATASET_PROMPTS.get(dataset_name) or initial_prompt
        prompt_ids   = _get_prompt_ids(processor, prompt_text, device) if prompt_text else None

        with torch.no_grad():
            # Plain tensor output — return_dict_in_generate=True (and output_scores=True)
            # on generation_config propagates into Whisper's generate_with_fallback()
            # → super().generate() returning a dict that generate_with_fallback() then
            # tries to index as a tensor → CUDA scatter/gather assertion OOB.
            # Plain tensor output sidesteps this entirely.
            generated_ids = model.generate(
                input_features=features,
                attention_mask=attn_mask,
                num_beams=beam_size,
                num_return_sequences=num_ret,
                max_new_tokens=225,
                prompt_ids=prompt_ids,
            )
        # generated_ids: [num_ret, seq_len] — one row per beam hypothesis

        top3 = []
        for seq_idx in range(generated_ids.shape[0]):
            text = processor.tokenizer.decode(
                generated_ids[seq_idx],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            text = normalize_transcript(text)
            top3.append({"rank": seq_idx + 1, "transcript": text, "score": None})
        per_sample_top3.append(top3)
        best_hypotheses.append(top3[0]["transcript"] if top3 else "")

        # Per-token confidence omitted: requires output_scores=True which propagates
        # into Whisper's internal generate_with_fallback() and causes CUDA assertion.
        per_sample_tokens.append([])

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"  Transcribed {i+1}/{len(samples)}", end="\r")

    print()
    return best_hypotheses, per_sample_top3, per_sample_tokens


# ─── CT2 (faster-whisper) evaluator ──────────────────────────────────────────

def evaluate_ct2(model_path: str, samples: List[dict], audio_col: str,
                 language: str, beam_size: int, vad_filter: bool) -> List[str]:
    """Transcribe audio using faster-whisper (CTranslate2)."""
    from faster_whisper import WhisperModel
    import torch

    device       = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8_float16" if device == "cuda" else "int8"

    print(f"Loading CT2 model from: {model_path} (compute_type={compute_type})")
    model = WhisperModel(model_path, device=device, compute_type=compute_type)

    predictions = []
    for i, sample in enumerate(samples):
        array, _ = _load_audio(sample[audio_col], 16000)
        segments, _ = model.transcribe(
            array,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        text = " ".join(seg.text for seg in segments).strip().lower()
        predictions.append(text)

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"  Transcribed {i+1}/{len(samples)}", end="\r")

    print()
    return predictions


# ─── WER computation helpers ──────────────────────────────────────────────────

def compute_wer_stats(references: List[str], hypotheses: List[str]) -> dict:
    return {
        "wer": round(wer(references, hypotheses) * 100, 3),
        "cer": round(cer(references, hypotheses) * 100, 3),
        "n_samples": len(references),
    }


def per_speaker_wer(samples: List[dict], references: List[str],
                    hypotheses: List[str], speaker_col: Optional[str]) -> Dict[str, dict]:
    if not speaker_col:
        return {}

    speakers: Dict[str, dict] = {}
    for sample, ref, hyp in zip(samples, references, hypotheses):
        spk = sample.get(speaker_col, "unknown")
        if spk not in speakers:
            speakers[spk] = {"refs": [], "hyps": []}
        speakers[spk]["refs"].append(ref)
        speakers[spk]["hyps"].append(hyp)

    return {
        spk: compute_wer_stats(data["refs"], data["hyps"])
        for spk, data in speakers.items()
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_cfg(args.config)

    hf_token  = args.hf_token or os.environ.get("HF_TOKEN")
    audio_col = cfg["dataset"]["audio_column"]
    text_col  = cfg["dataset"]["text_column"]
    sr        = cfg["dataset"]["sampling_rate"]
    split     = args.split or cfg["dataset"]["validation_split"]
    baseline  = args.baseline_wer or cfg["evaluation"].get("baseline_wer")
    vad       = cfg["evaluation"].get("vad_filter", True)

    # ── Load dataset ───────────────────────────────────────────────────────
    if cfg["dataset"]["source"] == "huggingface":
        print(f"Loading {cfg['dataset']['hf_dataset_id']} ({split} split) from HF Hub...")
        ds = load_dataset(cfg["dataset"]["hf_dataset_id"], split=split, token=hf_token)
    else:
        print(f"Loading local dataset ({split} split)...")
        full_ds = load_from_disk(cfg["dataset"]["local_path"])
        ds = full_ds[split]

    if audio_col in ds.column_names:
        ds = ds.cast_column(audio_col, Audio(decode=False))

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print(f"Evaluating on {len(ds)} samples from split '{split}'.")

    samples    = list(ds)
    references = [normalize_transcript(str(s[text_col])) for s in samples]

    # ── Transcribe ─────────────────────────────────────────────────────────
    t0 = time.time()

    per_sample_top3   = []
    per_sample_tokens = []
    if args.model_type == "hf":
        hypotheses, per_sample_top3, per_sample_tokens = evaluate_hf(
            args.model_path, samples, audio_col, text_col,
            args.language, args.beam_size,
            initial_prompt=args.initial_prompt,
        )
    else:
        hypotheses = evaluate_ct2(
            args.model_path, samples, audio_col,
            args.language, args.beam_size, vad,
        )
        hypotheses = [normalize_transcript(h) for h in hypotheses]
        per_sample_top3 = [[{"rank": 1, "transcript": h, "score": None}] for h in hypotheses]

    elapsed = time.time() - t0

    # ── Compute metrics ────────────────────────────────────────────────────
    overall   = compute_wer_stats(references, hypotheses)
    spk_col   = "speaker_id" if "speaker_id" in (samples[0] if samples else {}) else None
    by_speaker = per_speaker_wer(samples, references, hypotheses, spk_col)

    # ── Print results ──────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  Evaluation Results")
    print("=" * 58)
    print(f"  Model          : {args.model_path}")
    print(f"  Model type     : {args.model_type}")
    print(f"  Split          : {split} ({len(ds)} samples)")
    print(f"  Language       : {args.language}")
    print(f"  Elapsed        : {elapsed:.1f}s ({elapsed/len(samples):.2f}s per sample)")
    print(f"\n  Overall WER    : {overall['wer']:.2f}%")
    print(f"  Overall CER    : {overall['cer']:.2f}%")

    if baseline:
        delta_pp  = baseline - overall["wer"]
        delta_rel = delta_pp / baseline * 100
        arrow     = "↓" if delta_pp >= 0 else "↑"
        print(f"\n  Baseline WER   : {baseline:.2f}%")
        print(f"  Improvement    : {arrow}{abs(delta_pp):.2f}pp ({delta_rel:+.1f}% relative)")

    if by_speaker:
        print("\n  Per-speaker WER:")
        for spk, stats in sorted(by_speaker.items()):
            print(f"    {spk}: {stats['wer']:.2f}%  (n={stats['n_samples']})")

    # ── Sample predictions ────────────────────────────────────────────────
    print("\n  Sample predictions (first 5):")
    for i in range(min(5, len(samples))):
        print(f"    [{i+1}] REF  : {references[i]}")
        top3 = per_sample_top3[i] if i < len(per_sample_top3) else []
        for cand in top3:
            score_str = f"  (score {cand['score']:.3f})" if cand["score"] is not None else ""
            marker = ">>>" if cand["rank"] == 1 else "   "
            print(f"         {marker} [{cand['rank']}] {cand['transcript']}{score_str}")
        tokens = per_sample_tokens[i] if i < len(per_sample_tokens) else []
        if tokens:
            avg = sum(t["confidence"] for t in tokens) / len(tokens)
            low = [t for t in tokens if t["confidence"] < 0.5]
            print(f"          tokens: avg_conf={avg:.2f}  "
                  f"low_conf={[t['token'] for t in low[:5]]}")
        print()

    # ── Save JSON ──────────────────────────────────────────────────────────
    results = {
        "model_path":   args.model_path,
        "model_type":   args.model_type,
        "split":        split,
        "n_samples":    len(samples),
        "elapsed_s":    round(elapsed, 2),
        "overall_wer":  overall["wer"],
        "overall_cer":  overall["cer"],
        "baseline_wer": baseline,
        "wer_improvement_pp":  round(baseline - overall["wer"], 3) if baseline else None,
        "wer_improvement_pct": round((baseline - overall["wer"]) / baseline * 100, 1) if baseline else None,
        "per_speaker_wer": by_speaker,
        "per_sample": [
            {
                "reference": r,
                "hypothesis": h,
                "top3": top3,
                "token_confidence": toks,
            }
            for r, h, top3, toks in zip(
                references, hypotheses,
                per_sample_top3 or [[]] * len(references),
                per_sample_tokens or [[]] * len(references),
            )
        ],
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to: {out_path}")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    main()
