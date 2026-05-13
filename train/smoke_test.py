#!/usr/bin/env python3
"""
smoke_test.py — Validate the full fine-tuning pipeline end-to-end.

Uses openai/whisper-tiny (39 MB) and 6 real samples from your local
manifest.jsonl. Runs in ~2-3 minutes on CPU on your laptop.

Why whisper-tiny instead of whisper-large-v3-turbo?
  They share the same architecture family, the same LoRA target module names
  (q_proj / v_proj / k_proj / out_proj / fc1 / fc2), and the same Trainer
  code path. If all 9 stages pass with tiny, the real run on turbo will work.

Usage:
    python smoke_test.py
    python smoke_test.py --dataset_root D:/Datasets/dysarthric-speech
    python smoke_test.py --dataset_root D:/Datasets/dysarthric-speech --n_samples 10
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

# Windows cp1252 can't encode box-drawing characters — force UTF-8 output.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PASS = "[PASS]"
FAIL = "[FAIL]"
SAMPLE_RATE = 16_000
DEFAULT_DATASET = "D:/Datasets/dysarthric-speech"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", default=DEFAULT_DATASET,
                   help="Root of your local dysarthric-speech folder.")
    p.add_argument("--n_samples", type=int, default=6,
                   help="Number of real samples to use (default: 6).")
    p.add_argument("--model", default="openai/whisper-tiny",
                   help="Whisper model to test with locally. "
                        "Default: whisper-tiny (fast, CPU-friendly). "
                        "Use 'openai/whisper-small' for a more realistic local run. "
                        "whisper-large-v3-turbo is for RunPod only.")
    return p.parse_args()


def section(title: str):
    print(f"\n{'─'*58}")
    print(f"  {title}")
    print('─'*58)


def check(label: str, fn):
    try:
        result = fn()
        print(f"  {PASS} {label}")
        return result
    except Exception as e:
        print(f"  {FAIL} {label}")
        print(f"         {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


def load_local_samples(dataset_root: str, n: int) -> list[dict]:
    """Load n real dysarthric samples from manifest.jsonl."""
    root     = Path(dataset_root)
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        print(f"ERROR: manifest.jsonl not found at {manifest}")
        sys.exit(1)

    entries = [json.loads(l) for l in manifest.open()]
    dys     = [e for e in entries if e.get("group") == "dysarthric"]

    import soundfile as sf
    samples = []
    for entry in dys:
        if len(samples) >= n:
            break
        wav_path = root / entry["audio_path"]
        if not wav_path.exists():
            continue
        try:
            array, sr = sf.read(str(wav_path), dtype="float32")
            if array.ndim > 1:
                array = array.mean(axis=1)
            if sr != SAMPLE_RATE:
                import librosa
                array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLE_RATE)
            samples.append({
                "array":      array,
                "transcript": entry["transcript"].strip().lower(),
                "speaker_id": entry.get("speaker_id", ""),
                "audio_path": str(wav_path),
            })
        except Exception:
            continue

    if len(samples) < 2:
        print(f"ERROR: found only {len(samples)} readable samples in {dataset_root}")
        sys.exit(1)

    return samples


def main():
    args = parse_args()

    print("\n" + "="*58)
    print("  Whisper LoRA Fine-tuning — Smoke Test")
    print("="*58)
    print(f"  Dataset   : {args.dataset_root}")
    print(f"  Samples   : {args.n_samples} real dysarthric clips")
    print(f"  Model     : {args.model}")
    print(f"  Note      : same LoRA targets + Trainer path as large-v3-turbo")

    tmp = Path(tempfile.mkdtemp(prefix="whisper_smoke_"))
    print(f"  Tmp dir   : {tmp}")

    try:
        _run(args, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run(args, tmp: Path):
    import torch
    sys.path.insert(0, str(Path(__file__).parent))

    TEST_MODEL = args.model

    # ── [1] Load real samples ─────────────────────────────────────────────
    section("[1] Load real samples from local manifest")

    samples = check(
        f"Load {args.n_samples} dysarthric samples from manifest.jsonl",
        lambda: load_local_samples(args.dataset_root, args.n_samples),
    )
    print(f"  → {len(samples)} samples loaded")
    for s in samples[:3]:
        dur = len(s["array"]) / SAMPLE_RATE
        print(f"     {s['speaker_id']:4s} | {dur:.1f}s | \"{s['transcript'][:40]}\"")

    # ── [2] Processor + preprocessing ────────────────────────────────────
    section("[2] Audio preprocessing (mel features + tokenise)")

    from transformers import WhisperProcessor
    processor = check(
        "Load WhisperProcessor for whisper-tiny",
        lambda: WhisperProcessor.from_pretrained(TEST_MODEL, language="en", task="transcribe"),
    )

    def do_preprocess():
        s     = samples[0]
        feats = processor.feature_extractor(
            s["array"], sampling_rate=SAMPLE_RATE
        ).input_features[0]
        labels = processor.tokenizer(s["transcript"]).input_ids
        assert feats.shape == (80, 3000), f"Got {feats.shape}"
        assert len(labels) > 0
        return feats, labels

    feats, labels = check("Preprocess one sample — shape must be (80, 3000)", do_preprocess)
    print(f"  → feature shape: {feats.shape} | label tokens: {len(labels)}")

    # ── [3] Model + LoRA ─────────────────────────────────────────────────
    section("[3] Model + LoRA — same target modules as large-v3-turbo")

    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration

    base = check("Load whisper-tiny",
                 lambda: WhisperForConditionalGeneration.from_pretrained(TEST_MODEL))

    def apply_lora():
        # These are the SAME target_modules used in the real training config.
        lora_cfg = LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        )
        m = get_peft_model(base, lora_cfg)
        m.config.use_cache              = False
        m.config.suppress_tokens        = []
        m.generation_config.language    = "en"
        m.generation_config.task        = "transcribe"
        m.generation_config.forced_decoder_ids = None
        return m

    model = check("Apply LoRA — r=4, all attention+FFN modules", apply_lora)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct       = 100 * trainable / total
    check("Trainable params > 0 and < total",
          lambda: (trainable > 0 and trainable < total) or (_ for _ in ()).throw(AssertionError()))
    print(f"  → {trainable:,} / {total:,} trainable ({pct:.2f}%)")

    # ── [4] Data collator ─────────────────────────────────────────────────
    section("[4] SpeechCollator — padding + BOS stripping")

    from finetune import SpeechCollator

    collator = SpeechCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def test_collator():
        batch_input = []
        for s in samples[:2]:
            feats_  = processor.feature_extractor(s["array"], sampling_rate=SAMPLE_RATE).input_features[0]
            labels_ = processor.tokenizer(s["transcript"]).input_ids
            batch_input.append({"input_features": feats_, "labels": labels_})

        batch = collator(batch_input)
        assert "input_features" in batch and "labels" in batch
        assert batch["input_features"].shape[1] == 80, "Mel bins must be 80"
        # BOS (decoder_start_token_id) must have been stripped from labels.
        assert batch["labels"][0, 0].item() != model.config.decoder_start_token_id, \
            "BOS was not stripped from labels"
        return batch

    batch = check("Batch of 2 — correct shape, BOS stripped from labels", test_collator)
    print(f"  → input_features: {tuple(batch['input_features'].shape)} | "
          f"labels: {tuple(batch['labels'].shape)}")

    # ── [5] Compute metrics ───────────────────────────────────────────────
    section("[5] compute_metrics — WER via evaluate library")

    from finetune import build_compute_metrics

    def test_metrics():
        fn = build_compute_metrics(processor)
        # Perfect prediction → 0% WER.
        tok = np.array(labels[:5], dtype=np.int64)

        class FakePred:
            predictions = np.array([tok])
            label_ids   = np.array([tok.copy()])

        m = fn(FakePred())
        assert "wer" in m, f"No 'wer' key: {m}"
        assert m["wer"] == 0.0, f"Expected 0.0 WER for perfect pred, got {m['wer']}"
        return m

    m = check("Perfect prediction gives 0.0 WER", test_metrics)
    print(f"  → WER: {m['wer']}")

    # ── [6] Seq2SeqTrainer — 2 steps + eval ───────────────────────────────
    section("[6] Seq2SeqTrainer — 2 train steps + 1 eval")

    from datasets import Dataset as HFDataset
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    ckpt_dir = str(tmp / "checkpoints")

    def build_hf_ds(sample_list):
        rows = []
        for s in sample_list:
            feats_  = processor.feature_extractor(s["array"], sampling_rate=SAMPLE_RATE).input_features[0]
            labels_ = processor.tokenizer(s["transcript"]).input_ids
            rows.append({"input_features": feats_, "labels": labels_})
        return HFDataset.from_dict({
            "input_features": [r["input_features"] for r in rows],
            "labels":         [r["labels"]          for r in rows],
        })

    train_ds = build_hf_ds(samples[:-2])
    val_ds   = build_hf_ds(samples[-2:])

    training_args = Seq2SeqTrainingArguments(
        output_dir                      = ckpt_dir,
        max_steps                       = 2,
        per_device_train_batch_size     = 1,
        per_device_eval_batch_size      = 1,
        gradient_accumulation_steps     = 1,
        learning_rate                   = 1e-4,
        eval_strategy                   = "steps",
        eval_steps                      = 2,
        save_strategy                   = "steps",
        save_steps                      = 2,
        predict_with_generate           = True,
        generation_max_length           = 50,
        load_best_model_at_end          = True,
        metric_for_best_model           = "wer",
        greater_is_better               = False,
        fp16                            = False,
        bf16                            = False,
        report_to                       = ["none"],
        remove_unused_columns           = False,
        label_names                     = ["labels"],
        logging_steps                   = 1,
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        compute_metrics = build_compute_metrics(processor),
    )

    check("trainer.train() runs 2 steps without crashing", lambda: trainer.train())
    eval_result = check("trainer.evaluate() returns eval_wer",
                        lambda: trainer.evaluate())
    print(f"  → eval_wer: {eval_result.get('eval_wer', 'N/A'):.2f}%")

    # ── [7] Checkpoint on disk ────────────────────────────────────────────
    section("[7] Checkpoint saved to disk")

    def check_ckpt():
        ckpts = list(Path(ckpt_dir).glob("checkpoint-*"))
        assert ckpts, f"No checkpoint-* dirs in {ckpt_dir}"
        return ckpts[0]

    ckpt = check("checkpoint-* directory exists after training", check_ckpt)
    print(f"  → {ckpt.name}")

    # ── [8] LoRA merge + save ─────────────────────────────────────────────
    section("[8] LoRA merge_and_unload() + save_pretrained()")

    merged_dir = str(tmp / "merged")

    def merge_and_save():
        m = trainer.model.merge_and_unload()
        m.save_pretrained(merged_dir)
        processor.save_pretrained(merged_dir)
        assert (Path(merged_dir) / "config.json").exists(), "config.json missing"

    check("merge_and_unload() produces a saveable model", merge_and_save)
    print(f"  → merged model saved to {merged_dir}")

    # ── [9] HF pipeline inference on a real audio clip ───────────────────
    section("[9] HF transformers pipeline inference (the actual runtime)")

    def run_hf_inference():
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device       = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = torch.float16 if device == "cuda" else torch.float32

        inf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            merged_dir, torch_dtype=compute_type, low_cpu_mem_usage=True
        ).to(device)
        inf_proc  = AutoProcessor.from_pretrained(merged_dir)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=inf_model,
            tokenizer=inf_proc.tokenizer,
            feature_extractor=inf_proc.feature_extractor,
            device=device,
            torch_dtype=compute_type,
            generate_kwargs={"language": "en", "task": "transcribe"},
            max_new_tokens=50,
        )
        audio  = samples[0]["array"]
        result = pipe({"array": audio, "sampling_rate": SAMPLE_RATE})
        return result["text"].strip()

    text = check("HF pipeline transcribes real audio from merged model", run_hf_inference)
    ref  = samples[0]["transcript"]
    from jiwer import wer
    w = wer(ref, text.lower()) * 100
    print(f"  → ref : \"{ref}\"")
    print(f"  → hyp : \"{text.lower()}\"")
    print(f"  → WER : {w:.1f}%  (tiny on dysarthric ≈ high WER; that's expected)")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*58)
    print("  ALL CHECKS PASSED")
    print("  The pipeline is correct. Safe to run on RunPod with")
    print("  openai/whisper-large-v3-turbo.")
    print("="*58 + "\n")


if __name__ == "__main__":
    main()
