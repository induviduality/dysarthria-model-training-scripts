# Dysarthric ASR — Final LoRA Training Strategy
**TORGO + UASpeech | Whisper LoRA fine-tuning**
*Derived from dataset imbalance analysis — 2026-05-11*

---

## 1. Pre-Training Data Filtering

Do this once, before building any dataloader. Mutate the manifest.

**Remove:**
- All TORGO utterances where `transcript == 'xxx'` — 34 utterances, no linguistic signal
- All control speakers — both TORGO (5,058 utts) and UASpeech (9,945 utts)

**Rationale for dropping controls:** LoRA's frozen backbone already encodes normal speech. You are not at risk of catastrophic forgetting on a small adapter. Thousands of control utterances add gradient steps that steer adaptation away from the dysarthric distribution, which is the opposite of what you want. Drop them in run 1. Add a 50–100 utterance held-out control set to your *validation loop only* so you can detect forgetting if it occurs.

**Keep:**
- All dysarthric speakers including M08 (intel=6%), M09 (intel=2%), M10 (intel=13%) at **full weight**. These represent the hard end of your deployment target. Suppress them only if training collapses on their loss curve — not preemptively.
- All within-speaker TORGO repeats (multi-session re-recordings, max 12× per prompt per speaker) — legitimate acoustic variation.
- All cross-speaker UASpeech repeats — each of the 449 prompts spoken by different speakers is acoustic diversity, not redundancy.

**Post-filter dysarthric training corpus:**

| | TORGO | UASpeech | Total |
|---|---|---|---|
| Utterances | 2,395 | 9,690 | 12,085 |
| Speakers | 6 | 13 | 19 |
| Est. duration | ~2.2 h | ~9.0 h | ~11.2 h |

---

## 2. Sampling Strategy

### Ratio
**60% UASpeech : 40% TORGO**, applied per batch, every batch.

Not per epoch — "epoch" is a meaningless concept once you're oversampling. Think in steps.

### Effective repetition at 60/40

At any step budget S with batch size B:
- TORGO samples drawn: `S × B × 0.40`
- UA samples drawn: `S × B × 0.60`
- TORGO repeat factor vs UA: `(9690 / 2395) × (0.40 / 0.60) ≈ 2.7×`

Each TORGO sample is seen ~2.7× for every time a UASpeech sample is seen. With 2,395 TORGO samples this is manageable, not memorization territory.

### Implementation

Use `WeightedRandomSampler`. Assign per-sample weights:

```python
from torch.utils.data import WeightedRandomSampler

TARGET_RATIO = {"UASpeech": 0.60, "TORGO": 0.40}

dataset_sizes = {"UASpeech": 9690, "TORGO": 2395}

weights = [
    TARGET_RATIO[sample["dataset"]] / dataset_sizes[sample["dataset"]]
    for sample in train_manifest  # list of dicts with "dataset" key
]

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=TOTAL_STEPS * BATCH_SIZE,  # total tokens to draw
    replacement=True,
)
```

Set `num_samples` to your step budget × batch size — not to `len(dataset)`.

### Within-TORGO cycling (optional but cleaner)

`WeightedRandomSampler` with replacement can accidentally draw the same TORGO sample multiple times in quick succession. If you want guaranteed full-coverage cycles within TORGO before repeating, use a custom interleaved sampler:

```python
import itertools, random

def interleaved_sampler(ua_indices, torgo_indices, ua_frac=0.60):
    """Yields index stream at target ratio, cycling each dataset independently."""
    ua_pool = list(ua_indices)
    tg_pool = list(torgo_indices)
    random.shuffle(ua_pool)
    random.shuffle(tg_pool)
    ua_iter = itertools.cycle(ua_pool)   # reshuffles on each cycle
    tg_iter = itertools.cycle(tg_pool)
    while True:
        if random.random() < ua_frac:
            yield next(ua_iter)
        else:
            yield next(tg_iter)
```

This ensures TORGO is fully exhausted (and reshuffled) before heavy repeats accumulate. Worth the 20-line implementation cost; skip it only if you're time-constrained.

---

## 3. Training Loop

### Axes to care about

| Axis | Value |
|---|---|
| Unit | **Steps**, not epochs |
| Batch size | 16 (adjust for VRAM) |
| Rough step budget | 3,000 – 8,000 (see below) |
| Validation cadence | Every 200–500 steps |
| Stopping criterion | Held-out speaker WER plateau |

### Step budget rationale

With 12,085 dysarthric samples and batch size 16, one "true pass" through the unique data ≈ 755 steps. Target 5–10 effective passes before expecting convergence = **3,775 – 7,550 steps**. Use this as your search window, not a hard limit. Let validation WER tell you when to stop.

### Validation set

Your test speakers are TORGO {F01, M01} and UASpeech {F03, M07} — these are fully held out from training. Track:

1. **Per-speaker WER** on all 4 test speakers at every checkpoint — the primary metric
2. **Severity-stratified WER**: group test speakers by intelligibility to check you're not improving easy speakers at the cost of hard ones
3. **Control WER** on a 50–100 utterance normal-speech set — monitor for catastrophic forgetting (expect it to stay flat with LoRA)

Stop when held-out dysarthric WER (averaged across 4 test speakers) stops improving for 3–5 consecutive validation checkpoints.

### Do not use
- Epoch-based LR scheduling tied to `len(dataset)` — it will be wrong under weighted sampling
- Validation loss as stopping criterion — WER is the actual task metric and they can diverge

Use step-based LR schedule (warmup for first 5–10% of steps, then cosine or linear decay to step budget).

---

## 4. What to Monitor and When to Intervene

### Normal behaviour
- Training loss decreases steadily for first ~1,000 steps, then flattens
- Validation WER improves unevenly — expect UASpeech test speakers to improve faster (more training data from same distribution)

### Intervention triggers

| Signal | Diagnosis | Action |
|---|---|---|
| TORGO test speakers (F01, M01) WER diverges up while UASpeech test improves | Corpus dominance — UASpeech overwhelming TORGO adaptation | Shift ratio to 55/45 UA:TORGO |
| M08/M09/M10 training loss stalls or diverges | Very-low intelligibility gradient noise | Downweight to 0.5× — now, not before |
| Control WER degrades >5% relative | Catastrophic forgetting (unlikely with LoRA) | Add small control subset at 0.2× weight |
| All test WER plateaus very early (<1,000 steps) | Overfitting to small train set | Add SpecAugment / speed perturbation on TORGO; reduce LoRA rank |
| Training loss drops but test WER does not improve | Model memorising training prompts (esp. UASpeech word list) | Lower LR; add weight decay to LoRA params |

### Metrics to log every checkpoint
```
- wer_overall (all test speakers)
- wer_torgo_test (F01, M01)
- wer_uaspeech_test (F03, M07)
- wer_high_severity (M01 UASpeech + TORGO M02/M03)
- loss_train (step-level)
- loss_val (checkpoint-level)
```

---

## 5. What This Strategy Does NOT Do (and Why)

| Skipped technique | Reason |
|---|---|
| Transcript deduplication | Would gut UASpeech. Cross-speaker repetition = acoustic diversity. Don't touch it. |
| Speaker-balanced sampling within UASpeech | UASpeech speakers already near-uniform (mostly 765 utts each). Not needed. |
| Prompt-aware batch construction | Low marginal benefit vs. implementation cost. Standard shuffle is sufficient. |
| Control speaker inclusion | LoRA frozen backbone retains normal speech. Controls consume adaptation budget without benefit. |
| Pre-emptive very-low intel downweighting | Deployment target includes hard dysarthria. Suppress only on observed training collapse. |
| 50/50 dataset ratio | TORGO is single-mic, ~2,395 samples — 4× oversampling is real overfitting risk. 60/40 is the right starting point. |

---

## 6. Run Sequence

**Run 1 (baseline):** 60/40 ratio, no controls, all dysarthric speakers at full weight, ~5,000 steps, stop on WER plateau.

**Run 2 (if TORGO test WER is lagging):** Shift to 55/45. Everything else unchanged.

**Run 2 (if M08/M09/M10 causing loss instability):** Downweight very-low intel to 0.5×. Everything else unchanged.

Do not change two things at once between runs.