# Script Reference — Function Signatures & Functionality

Searchable reference for every public function/class across all scripts.

---

## `train/finetune.py`

### Module-level constants
```python
_ACRONYM_MAP: dict          # acronym → expansion mapping used by normalize_transcript
_DATASET_PROMPTS: dict      # {"UASpeech": "Single isolated word.", "TORGO": "Complete sentence."}
```

### Helpers
```python
def _get_prompt_ids(processor, prompt_text: str, device) -> Optional[torch.Tensor]
    # Returns 1-D token tensor for model.generate(prompt_ids=...), or None on failure.
    # Uses processor.get_prompt_ids() which prepends <|startofprev|>.

def _num_to_word(m: re.Match) -> str
    # Regex substitution helper: converts digit match to English word via num2words.

def normalize_transcript(text: str) -> str
    # Lowercase → strip acronym dots → numerals→words → expand acronyms →
    # strip punctuation → collapse hyphens/spaces. Applied to ALL transcripts.
```

### Config & CLI
```python
def parse_args() -> argparse.Namespace
    # Defines all CLI flags. Key flags:
    #   --local_dev       switches to whisper-small on CPU
    #   --allow_cpu       bypass GPU check
    #   --resume_from_checkpoint auto|none|<path>
    #   --fresh_start     deletes all checkpoints before training
    #   --baseline_wer    float, used for delta reporting

def apply_local_dev_overrides(cfg: dict, force_local_dev: bool = False) -> dict
    # Merges local_dev section of config.yaml over the main config when enabled.

def load_config(config_path: str, args) -> dict
    # Loads config.yaml, applies local_dev overrides, then applies CLI arg overrides.
```

### Dataset
```python
def load_data(cfg: dict, hf_token: Optional[str]) -> DatasetDict
    # Loads from HF Hub or local disk. Performs speaker-based val split if
    # validation_speakers is set in config, otherwise carves val_pct from train.

def make_preprocess_fn(processor: WhisperProcessor, cfg: dict) -> Callable
    # Returns a map-compatible function that:
    #   1. Decodes audio (handles bytes/path/array forms)
    #   2. Resamples to 16kHz if needed
    #   3. Runs feature_extractor → input_features
    #   4. Runs normalize_transcript() then tokenizer → labels

def preprocess_split(ds_split, processor, cfg, desc: str) -> Dataset
    # Casts audio to decode=False (soundfile path, avoids torchcodec on Windows),
    # then calls ds_split.map(make_preprocess_fn, remove_columns=all).
    # Uses num_proc=0 on CPU, config value on GPU.
```

### Sampling
```python
def compute_sample_weights(
    raw_ds,
    ua_frac: float = 0.60,
    torgo_frac: float = 0.40,
    dataset_col: str = "dataset"
) -> Optional[List[float]]
    # Computes per-sample weights so WeightedRandomSampler draws ua_frac UASpeech
    # and torgo_frac TORGO at every step. Returns None if dataset column missing.
```

### Per-speaker validation
```python
def build_speaker_val_data(
    raw_val_ds,
    audio_col: str,
    text_col: str,
    speaker_col: str,
    sr: int
) -> Dict[str, list]
    # Extracts per-speaker {array, transcript, dataset} from raw (unpreprocessed) val split.
    # Used by PerSpeakerWERCallback without going through the main eval loop.
    # Returns {} if speaker column missing.
```

### Data collator
```python
@dataclass
class SpeechCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]
        # Pads input_features and labels.
        # Masks padding positions in labels with -100 (ignored by loss).
        # Strips BOS token from labels (model generates it).
        # Emits decoder_attention_mask (trimmed in sync with BOS strip).
        # decoder_attention_mask is used during training forward pass but
        # stripped before generate() by DysarthriaTrainer.prediction_step().
```

### Metrics
```python
def build_compute_metrics(processor: WhisperProcessor) -> Callable
    # Returns compute_metrics(pred) used by Seq2SeqTrainer.
    # Decodes predictions and labels, applies normalize_transcript to both,
    # filters empty references, computes WER via evaluate library.
    # Returns {"wer": float (percentage)}.
```

### Callbacks
```python
class BaselineWERLogger(TrainerCallback)
    # on_evaluate: logs WER delta vs baseline_wer, tracks best WER + epoch.
    # Attributes: .history [(epoch, wer)], .best_wer, .best_epoch

class SafetyCheckpointCallback(TrainerCallback)
    # on_step_end: sets control.should_save=True every N steps.
    # Protects against RunPod preemption between eval checkpoints.

class GracefulInterruptCallback(TrainerCallback)
    # on_step_end: if _interrupt_training flag set (by SIGINT handler),
    # sets should_save=True and should_training_stop=True.
    # Re-running finetune.py auto-resumes from this checkpoint.

class PerSpeakerWERCallback(TrainerCallback)
    # on_evaluate: runs inference on speaker_val_data (raw audio, not preprocessed).
    # Uses per-sample prompt_ids from _DATASET_PROMPTS.
    # Logs "Speaker X: WER%" for each speaker. Limited to MAX_SAMPLES_PER_SPEAKER=30.
    #
    # __init__(self, speaker_val_data: Dict[str, list],
    #          processor: WhisperProcessor, sr: int = 16000)
```

### Trainer
```python
class DysarthriaTrainer(Seq2SeqTrainer)
    # __init__(self, *args, sample_weights: Optional[List[float]] = None, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None)
        # Strips decoder_attention_mask from inputs before calling super().
        # Prevents crash in Whisper's WhisperNoSpeechDetection logits processor
        # which re-runs the model with captured kwargs during eval generation.

    def get_train_dataloader(self) -> DataLoader
        # If sample_weights set: injects WeightedRandomSampler with
        # num_samples = max_steps * effective_batch_size (matches step budget).
        # Falls back to super() (uniform) if sample_weights is None.
```

### Model setup
```python
def load_model_with_lora(cfg: dict, hf_token: Optional[str]) -> Tuple[PeftModel, WhisperProcessor]
    # Loads processor + base model. Sets generation_config (language, task,
    # suppress_tokens, no_repeat_ngram_size, repetition_penalty,
    # no_speech_threshold=None, condition_on_prev_tokens=False, etc.).
    # Optionally freezes encoder (decoder_only config flag).
    # Applies LoRA via peft.get_peft_model(). Prints trainable param count.

def build_training_args(cfg: dict, no_wandb: bool, pin_memory: bool = True) -> Seq2SeqTrainingArguments
    # Builds Seq2SeqTrainingArguments from config. Sets dataloader_pin_memory=pin_memory.
    # report_to: tensorboard and/or wandb based on config + no_wandb flag.

def resolve_checkpoint(resume_arg: Optional[str], output_dir: str) -> Optional[str]
    # "none" → None (fresh start).
    # None or "auto" → latest checkpoint-* in output_dir.
    # Any other string → literal path.
    # Used to pass resume_from_checkpoint= to trainer.train().
```

### Entry point
```python
def main()
    # 1. Parse args + load config
    # 2. GPU check (exits unless --allow_cpu)
    # 3. Load dataset → compute sample weights → build speaker_val_data
    # 4. load_model_with_lora
    # 5. preprocess_split (train + val)
    # 6. build SpeechCollator, training_args, callbacks, DysarthriaTrainer
    # 7. resolve_checkpoint → trainer.train()
    # 8. Final eval → merge LoRA → save merged model + training_summary.json
```

---

## `data/prepare_hf_dataset.py`

```python
def normalize_transcript(text: str) -> str
    # Same logic as finetune.py version (duplicate by design — scripts are standalone).

def parse_args() -> argparse.Namespace
    # --dataset_root  root folder containing manifest.jsonl
    # --output_dir    save locally (use as dataset.local_path in config.yaml)
    # --output_repo   push to HF Hub (requires --hf_token)
    # --group         "dysarthric" | "control" | "all"
    # --val_ratio     fraction carved from train for validation (default 0.10)

def load_manifest(root: Path, group: str) -> Tuple[list[dict], list[dict]]
    # Reads manifest.jsonl. Filters by group. Removes transcript="xxx" entries
    # (34 TORGO utterances flagged unintelligible — no signal, only noise).
    # Returns (train_entries, test_entries).

def entries_to_dataset(entries: list[dict], root: Path, sr: int) -> Dataset
    # Validates first 5 audio paths exist. Builds HF Dataset with columns:
    # audio | sentence | speaker_id | dataset | severity
    # Applies normalize_transcript() to all sentences.
    # Casts audio column with Audio(sampling_rate=sr) — embeds bytes.

def print_stats(name: str, ds: Dataset)
    # Prints sample count, dataset breakdown, speaker breakdown.

def main()
    # load_manifest → entries_to_dataset (train + test) → train_test_split for val
    # → save_to_disk or push_to_hub
```

---

## `eval/eval_wer.py`

```python
def normalize_transcript(text: str) -> str
    # Standalone copy of the same normalization function.

_DATASET_PROMPTS: dict
    # {"UASpeech": "Single isolated word.", "TORGO": "Complete sentence."}

def _get_prompt_ids(processor, prompt_text: str, device) -> Optional[torch.Tensor]
    # Safe wrapper around processor.get_prompt_ids(). Returns None on failure.

def parse_args() -> argparse.Namespace
    # --model_path      path to merged HF model or CT2 model dir
    # --model_type      "hf" | "ct2"
    # --split           dataset split to evaluate (default: validation_split from config)
    # --baseline_wer    float for delta reporting
    # --beam_size       int, default 5
    # --max_samples     int, limit for quick checks
    # --initial_prompt  string passed as prompt_ids; auto-selected per sample if
    #                   dataset column present, this is the fallback

def load_cfg(config_path: str) -> dict
    # yaml.safe_load of config.yaml.

def _load_audio(audio_data, target_sr: int) -> Tuple[np.ndarray, int]
    # Handles all audio forms: decoded dict (array+sr), raw dict (bytes/path), plain string.
    # Returns (float32 array, sr). Caller is responsible for resampling.

def evaluate_hf(
    model_path: str,
    samples: List[dict],
    audio_col: str,
    text_col: str,
    language: str,
    beam_size: int,
    initial_prompt: Optional[str] = None
) -> Tuple[List[str], List[List[dict]], List[List[dict]]]
    # Returns (best_hypotheses, per_sample_top3, per_sample_token_data).
    # Uses model.generate() directly (not pipeline) with:
    #   num_beams=beam_size, num_return_sequences=min(3, beam_size), max_new_tokens=225
    # Per-sample prompt_ids: dataset column wins over initial_prompt CLI arg.
    # NOTE: output_scores=True and compute_transition_scores() are disabled — they
    # propagate into Whisper's generate_with_fallback() causing CUDA assertion OOB.
    # per_sample_token_data is always an empty list [].

def evaluate_ct2(
    model_path: str,
    samples: List[dict],
    audio_col: str,
    language: str,
    beam_size: int,
    vad_filter: bool
) -> List[str]
    # Uses faster_whisper.WhisperModel. Returns best hypothesis only (CT2 does not
    # expose beam alternatives or per-token scores via this API).

def compute_wer_stats(references: List[str], hypotheses: List[str]) -> dict
    # Returns {"wer": float%, "cer": float%, "n_samples": int} via jiwer.

def per_speaker_wer(
    samples: List[dict],
    references: List[str],
    hypotheses: List[str],
    speaker_col: Optional[str]
) -> Dict[str, dict]
    # Groups refs/hyps by speaker_id, calls compute_wer_stats per group.

def main()
    # Loads dataset → evaluates via evaluate_hf or evaluate_ct2
    # → compute_wer_stats + per_speaker_wer → prints results → saves to --output_json path
    # (default: inference-outputs/evaluation_results.json).
    # JSON per_sample includes: reference, hypothesis, top3.
    # token_confidence key is present but always [] (disabled; see evaluate_hf note).
```

---

## `inference/stream_transcribe.py`

Keyboard-controlled microphone transcription using a CTranslate2 faster-whisper model.
Records audio on SPACE, transcribes in a background thread, prints result + RTF.

```python
SAMPLE_RATE: int = 16_000
MIN_CLIP_S: float = 0.2     # clips shorter than this are silently discarded

def parse_args() -> argparse.Namespace
    # --model_path      path to CT2 model directory (required)
    # --device          "cuda" | "cpu". Auto-detected if omitted.
    # --compute_type    "float16" | "int8". Auto-selected from device if omitted.
    # --language        default "en"
    # --beam_size       int, default 5
    # --initial_prompt  decoder prompt; default "Complete sentence."
    #                   Use "Single isolated word." for UASpeech-style input.
    # --device_index    microphone device index (see --list_devices)
    # --list_devices    print available audio input devices and exit

def getch() -> str
    # Single keypress without Enter. msvcrt.getwch() on Windows, termios on Unix.
    # Falls back to readline() if stdin is not a tty.

class Recorder:
    # Thread-safe audio accumulator.
    # .recording bool — set by start()/stop().

    def start(self)
        # Clears frames buffer, sets recording=True.

    def stop(self) -> np.ndarray
        # Sets recording=False. Returns concatenated float32 audio or empty array.

    def feed(self, chunk: np.ndarray)
        # Appends chunk to frames if recording is active. Called from sounddevice callback.

def main()
    # 1. UTF-8 stdout/stderr reconfigure on Windows.
    # 2. Auto-detect device + compute_type.
    # 3. Load WhisperModel (CT2) from model_path.
    # 4. Start infer_worker daemon thread (reads from infer_q, prints RTF + transcript).
    # 5. Open sounddevice InputStream (16kHz, mono, float32).
    # 6. Keyboard loop:
    #      SPACE → recorder.start() or recorder.stop() + infer_q.put(audio)
    #      Q / Ctrl+C → break
    # 7. infer_q.join() + None sentinel to drain and stop worker thread.
```
