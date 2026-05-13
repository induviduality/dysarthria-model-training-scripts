# Inference Guide — Fine-tuned Whisper

How to run inference with your fine-tuned model, from a single audio file through
to real-time microphone transcription.

---

## Table of Contents

1. [Model formats](#1-model-formats)
2. [Install inference dependencies](#2-install-inference-dependencies)
3. [Convert to CTranslate2 (faster-whisper)](#3-convert-to-ctranslate2-faster-whisper)
4. [Batch inference with faster-whisper](#4-batch-inference-with-faster-whisper)
5. [Batch inference with the HF model](#5-batch-inference-with-the-hf-model)
6. [Real-time streaming (microphone)](#6-real-time-streaming-microphone)
7. [Evaluate WER on the validation set](#7-evaluate-wer-on-the-validation-set)
8. [Choosing a model format](#8-choosing-a-model-format)

---

## 1. Model formats

Training produces two usable model directories:

| Directory | Format | Use with |
|-----------|--------|----------|
| `./model-outputs/model-hf/` | HuggingFace Transformers | custom pipelines, fine-tuning continuation |
| `./model-outputs/model-ct2/` | CTranslate2 | `stream_transcribe.py`, `faster-whisper` library, `eval_wer.py --model_type ct2` |

The merged HF model is created automatically at the end of `finetune.py`.
The CT2 model requires a one-time conversion step (section 3).

---

## 2. Install inference dependencies

**Core inference (HF model)** — already included in `uv sync`:

```bash
uv sync
```

**faster-whisper + ctranslate2** — install the `ct2` optional group:

```bash
uv sync --extra ct2
```

**Live microphone streaming** — install the `stream` optional group:

```bash
uv sync --extra stream
```

Or install both at once:

```bash
uv sync --extra ct2 --extra stream
```

---

## 3. Convert to CTranslate2 (faster-whisper)

Run once after training. Reads paths from `config.yaml` automatically.

```bash
# GPU inference — float16 (best accuracy, fastest on GPU)
uv run python inference/convert_to_ct2.py --quantization float16

# CPU inference — int8 (smallest file, fastest on CPU)
uv run python inference/convert_to_ct2.py --quantization int8

# Override paths explicitly
uv run python inference/convert_to_ct2.py \
    --model_dir  ./model-outputs/model-hf \
    --output_dir ./model-outputs/model-ct2 \
    --quantization float16
```

### Quantization guide

| Setting | Use case | Model size | Notes |
|---------|----------|------------|-------|
| `float16` | GPU inference | ~1.5 GB | Best accuracy. Requires CUDA GPU. |
| `int8` | CPU inference | ~600 MB | Fast on CPU, slight accuracy trade-off. |
| `int8_float16` | GPU, memory-constrained | ~600 MB | Good balance on smaller GPUs. |

---

## 4. Batch inference with faster-whisper

Use this for transcribing audio files — faster and more memory-efficient than the
HF pipeline for batch workloads.

```python
from faster_whisper import WhisperModel

# GPU
model = WhisperModel("./model-outputs/model-ct2", device="cuda", compute_type="float16")

# CPU
model = WhisperModel("./model-outputs/model-ct2", device="cpu", compute_type="int8")

# Transcribe a file
segments, info = model.transcribe("audio.wav", language="en", beam_size=5)

for segment in segments:
    print(f"[{segment.start:.1f}s → {segment.end:.1f}s]  {segment.text}")
```

### Dysarthric speech settings

The fine-tuned model is tuned for dysarthric speech. Use these settings to get the
best results:

```python
segments, info = model.transcribe(
    "audio.wav",
    language="en",
    beam_size=5,
    vad_filter=False,           # Disable VAD — dysarthric speech often triggers false silences
    initial_prompt="Complete sentence.",   # Or "Single isolated word." for UASpeech-style input
    no_repeat_ngram_size=3,
    repetition_penalty=1.2,
)
```

### Transcribe a folder of files

```python
from pathlib import Path
from faster_whisper import WhisperModel

model = WhisperModel("./model-outputs/model-ct2", device="cuda", compute_type="float16")

for wav in Path("./audio_files").glob("*.wav"):
    segments, _ = model.transcribe(str(wav), language="en", beam_size=5)
    transcript = " ".join(s.text for s in segments).strip()
    print(f"{wav.name}: {transcript}")
```

---

## 5. Batch inference with the HF model

Use this when you want to continue fine-tuning, need access to token-level
probabilities, or don't want to do the CT2 conversion step.

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import soundfile as sf

model_path = "./model-outputs/model-hf"
device     = "cuda" if torch.cuda.is_available() else "cpu"
dtype      = torch.float16 if device == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(model_path)
model     = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path, torch_dtype=dtype, low_cpu_mem_usage=True
).to(device)

# Load audio
audio, sr = sf.read("audio.wav", dtype="float32")

# Preprocess
inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)

# Generate
with torch.no_grad():
    ids = model.generate(
        **inputs,
        language="en",
        task="transcribe",
        num_beams=5,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
    )

transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
print(transcript)
```

---

## 6. Real-time streaming (microphone)

`stream_transcribe.py` uses keyboard controls to start and stop recording, then
sends the captured audio to the CT2 model. Transcription runs in a background
thread so you can start the next recording immediately.

Requires the `stream` and `ct2` extras:

```bash
uv sync --extra ct2 --extra stream
```

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Start recording |
| `SPACE` | Stop recording and send to model |
| `Q` | Quit |

### Usage

```bash
# GPU (float16 auto-selected)
uv run python inference/stream_transcribe.py \
    --model_path /workspace/model-outputs/model-ct2

# CPU (int8 auto-selected)
uv run python inference/stream_transcribe.py \
    --model_path /workspace/model-outputs/model-ct2 \
    --device cpu

# Single-word input (UASpeech-style)
uv run python inference/stream_transcribe.py \
    --model_path /workspace/model-outputs/model-ct2 \
    --initial_prompt "Single isolated word."

# All options
uv run python inference/stream_transcribe.py \
    --model_path /workspace/model-outputs/model-ct2 \
    --device cuda \
    --compute_type float16 \
    --beam_size 5 \
    --language en \
    --initial_prompt "Complete sentence."
```

### Output format

```
  [ RECORDING ... ]
  [ STOPPED — 3.2s captured ]
  [transcribing 3.2s ...]
  RTF 0.18x | 3.2s audio → 0.6s inference
  >>> the cat sat on the mat

  SPACE = start/stop recording   Q = quit
```

- **RTF** — real-time factor. `0.18x` = transcription took 18% of the audio duration.
- Clips shorter than 0.2s are automatically discarded as noise.

---

## 7. Evaluate WER on the validation set

`eval_wer.py` runs the model against the held-out validation split and reports
overall + per-speaker WER.

```bash
# Evaluate the HF merged model
uv run python eval/eval_wer.py \
    --model_path ./model-outputs/model-hf \
    --model_type hf \
    --output_json inference-outputs/evaluation_results.json

# Evaluate the CT2 model
uv run python eval/eval_wer.py \
    --model_path ./model-outputs/model-ct2 \
    --model_type ct2 \
    --output_json inference-outputs/evaluation_results.json

# Compare against baseline WER
uv run python eval/eval_wer.py \
    --model_path ./model-outputs/model-hf \
    --model_type hf \
    --baseline_wer 47.3 \
    --output_json inference-outputs/evaluation_results.json

# Quick sanity check — first 50 samples only
uv run python eval/eval_wer.py \
    --model_path ./model-outputs/model-hf \
    --model_type hf \
    --max_samples 50
```

---

## 8. Choosing a model format

| Situation | Use |
|-----------|-----|
| Production inference, speed matters | CT2 (`faster-whisper`) with `float16` on GPU |
| CPU-only deployment | CT2 with `int8` |
| Real-time microphone | `stream_transcribe.py` with CT2 model (`model-outputs/model-ct2`) |
| Debugging / exploring predictions | HF model directly |
| Continuing fine-tuning | HF model (`model-outputs/model-hf`) |
| WER benchmarking | `eval_wer.py` with either format |

The CT2 model is generally **2–4× faster** than the HF pipeline for batch inference
and uses less memory, but requires the conversion step and the `ct2` extra.
The HF model is more flexible (token probabilities, beam alternatives, generation
config tweaks) and works without any conversion.
