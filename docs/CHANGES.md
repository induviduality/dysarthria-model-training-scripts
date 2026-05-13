# Applied Fixes — fixing-plan.md

Each fix listed with the exact file and function where it lands.

---

## §1 — Transcript Normalization

| What | File | Function |
|---|---|---|
| `normalize_transcript()` defined | `train/finetune.py` | module-level (after log setup) |
| Applied to training labels | `train/finetune.py` | `make_preprocess_fn()` → `preprocess()` inner |
| Applied to eval predictions | `train/finetune.py` | `compute_metrics()` inside `build_compute_metrics()` |
| Applied to per-speaker eval hyps | `train/finetune.py` | `PerSpeakerWERCallback.on_evaluate()` |
| Applied to per-speaker val refs | `train/finetune.py` | `build_speaker_val_data()` |
| `normalize_transcript()` defined | `data/prepare_hf_dataset.py` | module-level |
| Applied to sentences at prep time | `data/prepare_hf_dataset.py` | `entries_to_dataset()` |
| `normalize_transcript()` defined | `eval/eval_wer.py` | module-level |
| Applied to eval references | `eval/eval_wer.py` | `main()` |
| Applied to eval hypotheses | `eval/eval_wer.py` | `evaluate_hf()` |
| Applied to CT2 hypotheses | `eval/eval_wer.py` | `main()` |

---

## §2 — Weighted Batch Sampling

**Already implemented** before this session.

| What | File | Function |
|---|---|---|
| Weight computation | `train/finetune.py` | `compute_sample_weights()` |
| WeightedRandomSampler injection | `train/finetune.py` | `DysarthriaTrainer.get_train_dataloader()` |

---

## §3 — Initial Prompt Conditioning (via `prompt_ids`)

Uses Whisper's `<|startofprev|>` mechanism — prompt seen by decoder but never output. WER unaffected.

| What | File | Function |
|---|---|---|
| `_DATASET_PROMPTS` dict | `train/finetune.py` | module-level |
| `_get_prompt_ids()` helper | `train/finetune.py` | module-level |
| `dataset` field preserved in val data | `train/finetune.py` | `build_speaker_val_data()` |
| Per-sample prompt passed to generate | `train/finetune.py` | `PerSpeakerWERCallback.on_evaluate()` |
| `_DATASET_PROMPTS` dict | `eval/eval_wer.py` | module-level |
| `_get_prompt_ids()` helper | `eval/eval_wer.py` | module-level |
| `--initial_prompt` CLI arg | `eval/eval_wer.py` | `parse_args()` |
| prompt_ids passed to generate | `eval/eval_wer.py` | `evaluate_hf()` |
| `--initial_prompt` CLI arg | `inference/stream_transcribe.py` | `parse_args()` |
| `self.initial_prompt` stored | `inference/stream_transcribe.py` | `StreamingTranscriber.__init__()` |
| prompt_ids passed to generate | `inference/stream_transcribe.py` | `StreamingTranscriber._inference_worker()` |

---

## §4 — Attention Mask Fix (`decoder_attention_mask`)

| What | File | Function |
|---|---|---|
| `decoder_attention_mask` emitted | `train/finetune.py` | `SpeechCollator.__call__()` |
| Mask trimmed in sync with BOS strip | `train/finetune.py` | `SpeechCollator.__call__()` |
| Stripped before `generate()` call | `train/finetune.py` | `DysarthriaTrainer.prediction_step()` |

> **Why the strip in prediction_step**: Whisper's `WhisperNoSpeechDetection` logits processor captures model kwargs at init and re-runs the model during sampling. If `decoder_attention_mask` is present, the re-run triggers a batch-size mismatch crash. The mask is valid and used during training forward passes (`model(**inputs)`) but excluded from eval generation calls.

---

## §5 — Error Fixes

### Error A — `pin_memory=True` on CPU

| What | File | Function |
|---|---|---|
| `pin_memory` parameter added | `train/finetune.py` | `build_training_args()` |
| `dataloader_pin_memory` set | `train/finetune.py` | `build_training_args()` → `Seq2SeqTrainingArguments(...)` |
| `torch.cuda.is_available()` passed | `train/finetune.py` | `main()` |

### Error B — Duplicate `SuppressTokensLogitsProcessor` warning

| What | File | Function |
|---|---|---|
| `suppress_tokens=[]` on `model.config` only | `train/finetune.py` | `load_model_with_lora()` |
| NOT set on `generation_config` (prevents duplicate) | `train/finetune.py` | `load_model_with_lora()` |

> Whisper's `generate()` reads `suppress_tokens` from `model.config` internally and creates the processor once. Setting it on `generation_config` as well creates a second processor → duplicate warning. Fix: leave `generation_config.suppress_tokens` unset.

### Error E — `clean_up_tokenization_spaces` warning

| What | File | Function |
|---|---|---|
| `clean_up_tokenization_spaces=False` | `train/finetune.py` | `compute_metrics()` inside `build_compute_metrics()` |
| `clean_up_tokenization_spaces=False` | `train/finetune.py` | `PerSpeakerWERCallback.on_evaluate()` |
| `clean_up_tokenization_spaces=False` | `eval/eval_wer.py` | `evaluate_hf()` |
| `clean_up_tokenization_spaces=False` | `inference/stream_transcribe.py` | `StreamingTranscriber._inference_worker()` |

---

## §6 — Per-Token Log-Probabilities

| What | File | Function |
|---|---|---|
| `compute_transition_scores()` call | `eval/eval_wer.py` | `evaluate_hf()` |
| Token confidence list per sample | `eval/eval_wer.py` | `evaluate_hf()` → returned as `per_sample_tokens` |
| Printed in sample output | `eval/eval_wer.py` | `main()` |
| Saved to JSON per sample | `eval/eval_wer.py` | `main()` → `results["per_sample"]` |
| `compute_transition_scores()` call | `inference/stream_transcribe.py` | `StreamingTranscriber._inference_worker()` |
| `avg_conf` printed per segment | `inference/stream_transcribe.py` | `StreamingTranscriber._inference_worker()` |

---

## §7 — Repetition / No-Speech Penalties

All set in `finetune.py` → `load_model_with_lora()` on `model.generation_config`:

| Setting | Value | Note |
|---|---|---|
| `no_repeat_ngram_size` | 3 | Prevents repetition loops |
| `repetition_penalty` | 1.2 | Mild per-token repeat penalty |
| `no_speech_threshold` | `None` | Disables no-speech detection — never suppress dysarthric audio as silence; also avoids model-calling logits processor crash |
| `logprob_threshold` | `None` | Disables fallback mechanism — non-None triggers `_need_fallback` which inspects `seek_outputs[index]["scores"]` expecting a dict but gets a plain tensor → IndexError crash |
| `compression_ratio_threshold` | `None` | Disabled alongside logprob_threshold — both guard the same fallback path |
| `condition_on_prev_tokens` | `False` | Prevents cascading hallucination |
| `suppress_tokens` | `None` | Clears pretrained default — prevents duplicate `SuppressTokensLogitsProcessor` (Whisper creates one internally; a second from generation_config triggers a warning) |
| `begin_suppress_tokens` | `None` | Same reason as `suppress_tokens` — clears pretrained default to avoid duplicate processor |

Same settings applied in `stream_transcribe.py` → `load_whisper()` on `model.generation_config`.

---

## Top-3 Outputs (user requirement, not in fixing-plan.md)

| What | File | Function |
|---|---|---|
| `num_return_sequences=min(3, beam_size)` | `eval/eval_wer.py` | `evaluate_hf()` |
| `top3` list returned per sample | `eval/eval_wer.py` | `evaluate_hf()` |
| Top-3 printed in sample output | `eval/eval_wer.py` | `main()` |
| `top3` saved to JSON per sample | `eval/eval_wer.py` | `main()` → `results["per_sample"]` |
| `num_return_sequences=min(3, beam_size)` | `inference/stream_transcribe.py` | `StreamingTranscriber._inference_worker()` |
| Top-3 printed per VAD segment | `inference/stream_transcribe.py` | `StreamingTranscriber._inference_worker()` |
