# DysarthriaDecoder — Single-Shot Implementation Plan

7 items, ordered by where they go in the codebase (data prep → training → inference).

---

## 1. Transcript Normalization (Data Prep — Apply Before Training)

Critical for single-shot. You want Layer 1 outputting phonetic transcripts, not formatted text. Apply to **all training transcripts, validation transcripts, and evaluation references** — must be consistent across all three.

```python
import re
from num2words import num2words  # pip install num2words

# Acronym expansion dictionary — populate from your dataset inspection
ACRONYM_MAP = {
    "tv": "t v",
    "pc": "p c",
    "usa": "u s a",
    "uk": "u k",
    "id": "i d",
    "ok": "okay",
    # Add others you find in TORGO/UASpeech transcripts
    # TORGO has prompts like "ABC", "XYZ" — expand these too
}

def normalize_transcript(text: str) -> str:
    """
    Normalize transcripts for phonetic-focused training.
    Apply identically to training data AND evaluation references.
    """
    # Lowercase first
    text = text.lower()
    
    # Expand acronyms BEFORE punctuation stripping (so "U.S.A." matches "usa")
    text = re.sub(r'\.(?=[a-z])', '', text)  # remove dots inside acronyms first
    
    # Convert numerals to words ("3" -> "three", "21" -> "twenty-one")
    def num_to_word(match):
        try:
            return num2words(int(match.group()))
        except ValueError:
            return match.group()
    text = re.sub(r'\b\d+\b', num_to_word, text)
    
    # Expand acronyms (word boundaries)
    for acronym, expansion in ACRONYM_MAP.items():
        text = re.sub(rf'\b{acronym}\b', expansion, text)
    
    # Strip all punctuation (after acronym/number handling)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Hyphens from num2words become spaces ("twenty-one" -> "twenty one")
    text = text.replace('-', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
```

**Where to apply:**
- In your manifest CSV preprocessing step, run this on the `transcript` column before saving.
- In your evaluation metric computation, apply this same function to both predictions and references before computing WER.

**Inspect your data first** to populate `ACRONYM_MAP`:

```python
# Run once to find candidates
import pandas as pd
manifest = pd.read_csv('your_manifest.csv')
# Look for short all-caps or dotted patterns
for txt in manifest['transcript'].unique():
    if re.search(r'\b[A-Z]{2,5}\b', txt):
        print(txt)
```

---

## 2. Weighted Batch Sampling (DataLoader)

Single-shot constraint changes the strategy. You can't iterate, so don't over-tune. Use a principled weighting that handles both the dataset imbalance (TORGO sentences vs UASpeech words) AND severity imbalance in one pass.

```python
import pandas as pd
from torch.utils.data import WeightedRandomSampler

def compute_sample_weights(manifest: pd.DataFrame) -> list:
    """
    Combined weighting:
    - Inverse frequency by severity (so severe speakers aren't drowned out)
    - Mild upweighting of TORGO (sentences are scarcer and harder)
    """
    # Severity inverse frequency
    severity_counts = manifest['intelligibility'].value_counts()
    severity_weight = manifest['intelligibility'].map(lambda s: 1.0 / severity_counts[s])
    
    # Dataset weight — TORGO slightly upweighted since sentences are scarcer
    # and your deployment likely uses sentences more than isolated words
    DATASET_WEIGHT = {'torgo': 1.3, 'uaspeech': 1.0}
    dataset_weight = manifest['dataset'].map(DATASET_WEIGHT)
    
    # Combine
    weights = severity_weight * dataset_weight
    weights = weights / weights.sum()
    
    return weights.values.tolist()

# Usage
train_manifest = pd.read_csv('train_manifest.csv')
weights = compute_sample_weights(train_manifest)

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(train_manifest),
    replacement=True
)

# Pass to DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,  # NOTE: use sampler instead of shuffle=True
    collate_fn=data_collator,
    pin_memory=True,
)
```

**Important for HuggingFace Trainer:** If you're using `Seq2SeqTrainer`, you need to override `_get_train_sampler`:

```python
class WeightedTrainer(Seq2SeqTrainer):
    def __init__(self, *args, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights
    
    def _get_train_sampler(self):
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.train_dataset),
            replacement=True
        )

# Use WeightedTrainer instead of Seq2SeqTrainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    sample_weights=weights,
    ...
)
```

---

## 3. Prompt Conditioning (Data Collator)

Adds a small guardrail against long-form hallucinations on single-word inputs. Implemented in the data collator so each sample gets its mode tag.

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class DataCollatorWithPromptConditioning:
    processor: Any
    decoder_start_token_id: int
    
    def __call__(self, features):
        # Standard audio features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        
        # Tokenize labels WITH prompt prefix based on dataset
        label_features = []
        for f in features:
            if f.get("dataset") == "uaspeech":
                prompt = "Single isolated word."
            else:  # torgo
                prompt = "Complete sentence."
            
            # Prepend prompt to the transcript for the decoder
            full_text = prompt + " " + f["transcript"]
            tokenized = self.processor.tokenizer(full_text, return_tensors="pt")
            label_features.append({"input_ids": tokenized.input_ids[0]})
        
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        
        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Remove decoder_start_token if present at beginning
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        return batch
```

**Make sure your dataset `__getitem__` returns the `dataset` field** so the collator can read it.

**At inference time:** Default to "Complete sentence." prompt for all real-world use. The word-mode prompt is mainly to prevent training-time mode confusion.

---

## 4. Attention Mask Fix

The warning `The attention mask is not set and cannot be inferred from input because pad token is same as eos token` happens during eval/generation. Fix by explicitly passing attention masks.

**In your data collator**, after padding labels:

```python
# After labels processing in the collator:
batch["decoder_attention_mask"] = labels_batch.attention_mask
```

**In evaluation/generation calls**, explicitly pass attention mask:

```python
# When calling generate() during eval or inference
generated_ids = model.generate(
    input_features=batch["input_features"],
    attention_mask=batch.get("attention_mask"),  # for encoder input
    # ... other generate kwargs
)
```

**For the tokenizer itself**, set the pad token to something distinct if Whisper allows:

```python
# After loading processor/tokenizer
if processor.tokenizer.pad_token_id == processor.tokenizer.eos_token_id:
    # Whisper uses <|endoftext|> as both — set explicit attention masks instead
    # Don't try to add a new pad token, it breaks Whisper's special token handling
    pass

# When tokenizing, always request attention_mask
tokenized = processor.tokenizer(
    text,
    padding="longest",
    return_attention_mask=True,
    return_tensors="pt"
)
```

---

## 5. Fixing the Pasted Errors / Warnings

### Error A: `pin_memory=True but no accelerator found`

You're training on CPU. This is a **massive red flag** — Whisper Turbo on CPU will take days, not hours. Verify GPU is being used:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Force device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

If CUDA isn't available locally (your RTX 3050 should work but only with Whisper Small), **do not run this training locally**. Move to Kaggle T4×2 or Colab immediately.

If you're forced to test locally to validate the pipeline, set:

```python
training_args = Seq2SeqTrainingArguments(
    ...
    dataloader_pin_memory=False,  # Set to False on CPU
    use_cpu=True,  # explicit, suppresses pin_memory warning
)
```

But again — only for pipeline smoke testing, not real training.

### Error B: Custom logits processor warnings

```
A custom logits processor of type <class '...SuppressTokensLogitsProcessor'> has been passed to .generate(), but it was also created in .generate()...
```

This happens when you pass `suppress_tokens` both via `generation_config` AND directly to `generate()`. Fix by setting them in ONE place only:

```python
# In your model's generation_config (set once at model load)
model.generation_config.suppress_tokens = []  # or your custom list
model.generation_config.begin_suppress_tokens = []

# Then in generate() calls, do NOT pass suppress_tokens again
generated_ids = model.generate(
    input_features=batch["input_features"],
    # Do NOT pass: suppress_tokens=...
    # Do NOT pass: begin_suppress_tokens=...
)
```

### Error C: `logging_dir is deprecated`

Cosmetic. Fix by setting the env var:

```python
import os
os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs"

# Remove logging_dir from training_args
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-turbo-finetuned",
    # logging_dir="./logs",  # REMOVE THIS
    report_to=["tensorboard"],
    ...
)
```

### Error D: `torch_dtype is deprecated`

Cosmetic. Replace with `dtype`:

```python
# Old
model = WhisperForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16
)

# New
model = WhisperForConditionalGeneration.from_pretrained(
    model_name, dtype=torch.float16
)
```

### Error E: `clean_up_tokenization_spaces` warning

Fix by setting it explicitly on decode:

```python
predicted_text = processor.tokenizer.batch_decode(
    predicted_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False  # CRITICAL — set to False for Whisper BPE
)
```

If you don't set this to False, you'll get corrupted output (spaces stripped before punctuation).

### Error F: `Using pipelines sequentially on GPU`

Comes from using `pipeline()` for inference one sample at a time. For evaluation, batch through the model directly instead of via pipeline:

```python
# Instead of pipeline iteration
from torch.utils.data import DataLoader

eval_loader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)

model.eval()
predictions = []
with torch.no_grad():
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        generated_ids = model.generate(
            input_features=batch["input_features"],
            # ... generation config
        )
        decoded = processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        predictions.extend(decoded)
```

---

## 6. Per-Token Log-Probabilities Emission

For Layer 2 handoff. Modify your inference function to return token-level confidence.

```python
import torch

def transcribe_with_confidence(model, processor, audio_features, device):
    """
    Returns transcript + per-token confidence scores.
    Use this for inference at deployment time, not during training.
    """
    with torch.no_grad():
        outputs = model.generate(
            input_features=audio_features.to(device),
            return_dict_in_generate=True,
            output_scores=True,
            num_beams=1,  # greedy — better for confidence interpretation
            max_new_tokens=200,
        )
    
    # outputs.scores is a tuple of length [num_generated_tokens]
    # Each element is a tensor of shape [batch_size, vocab_size]
    
    token_ids = outputs.sequences[0]  # batch_size=1 assumed
    
    token_data = []
    for step_idx, step_scores in enumerate(outputs.scores):
        probs = torch.softmax(step_scores[0], dim=-1)
        chosen_token_id = token_ids[step_idx + 1].item()  # +1 to skip decoder_start
        chosen_prob = probs[chosen_token_id].item()
        token_text = processor.tokenizer.decode([chosen_token_id])
        
        token_data.append({
            "token": token_text,
            "token_id": chosen_token_id,
            "confidence": chosen_prob,
        })
    
    transcript = processor.tokenizer.decode(
        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Compute sentence-level avg confidence (excluding special tokens)
    non_special_confs = [t["confidence"] for t in token_data 
                         if t["token_id"] not in processor.tokenizer.all_special_ids]
    avg_confidence = sum(non_special_confs) / len(non_special_confs) if non_special_confs else 0.0
    
    return {
        "transcript": transcript,
        "tokens": token_data,
        "avg_confidence": avg_confidence,
    }
```

**Output format for Layer 2:**

```json
{
  "transcript": "i want apple",
  "avg_confidence": 0.67,
  "tokens": [
    {"token": "i", "confidence": 0.91},
    {"token": " want", "confidence": 0.85},
    {"token": " apple", "confidence": 0.34}
  ]
}
```

Layer 2 uses low-confidence tokens (e.g., < 0.5) as targets for fingerprint-based correction.

---

## 7. Repetition / No-Speech Penalty (Generation Config)

Set these on the model's generation config or pass to `generate()`. Whisper's known failure modes include loops and empty outputs on atypical audio.

```python
# Set on model.generation_config — applies to all generate() calls
model.generation_config.no_repeat_ngram_size = 3  # prevents trigram repetition loops
model.generation_config.repetition_penalty = 1.2  # mildly penalize repeated tokens
model.generation_config.no_speech_threshold = 0.6  # higher = less aggressive silence detection
model.generation_config.logprob_threshold = -1.0  # lower threshold = more permissive
model.generation_config.compression_ratio_threshold = 2.4  # default, keep
model.generation_config.condition_on_prev_tokens = False  # CRITICAL for dysarthric speech
```

**Why each setting:**

- `no_repeat_ngram_size=3`: Whisper occasionally loops on unclear audio ("the the the the"). This blocks trigram repetition.
- `repetition_penalty=1.2`: Mild penalty for repeating any token. Don't go above 1.3 — over-penalizing hurts legitimate repetition ("very very good").
- `no_speech_threshold=0.6`: Default is 0.6. Increase to 0.7-0.8 if Whisper is incorrectly marking dysarthric speech as silence. Decrease if it's transcribing silence as words.
- `logprob_threshold=-1.0`: Default. If your output is being aggressively filtered out as "low confidence garbage," set to `None` to disable.
- `condition_on_prev_tokens=False`: **Most important setting in this section.** Prevents cascading hallucination from one bad decode to the next. For fragmented dysarthric speech, this alone may reduce hallucination significantly.

**For isolated word mode (UASpeech-style inputs at inference):**

```python
# Override at inference call for word inputs
generated = model.generate(
    input_features=features,
    num_beams=1,
    max_new_tokens=10,           # hard cap
    no_repeat_ngram_size=2,      # more aggressive
    repetition_penalty=1.5,      # stronger
    return_timestamps=False,
)
```

---

## Application Order

Given single-shot constraint:

1. **First** — fix the CPU/GPU issue (Error A). Don't waste your shot on the wrong device.
2. **Second** — implement transcript normalization in data prep, regenerate manifest.
3. **Third** — implement weighted sampler in DataLoader / custom Trainer.
4. **Fourth** — implement prompt conditioning in collator.
5. **Fifth** — fix attention mask + cosmetic warnings.
6. **Sixth** — set generation_config with repetition/no-speech penalties before training (so eval steps use them too).
7. **Seventh** — implement per-token logprobs in your inference module (this is for after training, for Layer 2 handoff — doesn't affect the training run itself).

Items 1–6 must be done before training starts. Item 7 can be done in parallel or after.