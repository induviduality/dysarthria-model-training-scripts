#!/usr/bin/env python3
"""
prepare_hf_dataset.py — Build a clean HF Hub dataset from the local manifest.

Reads manifest.jsonl from your local dysarthric-speech dataset, filters to
dysarthric speakers only, creates a proper train / validation / test split,
embeds audio bytes into the HF Audio feature, and pushes to HF Hub.

Run ONCE locally before training on RunPod:

    python prepare_hf_dataset.py \\
        --dataset_root  D:/Datasets/dysarthric-speech \\
        --output_repo   your-username/dysarthria-prepared \\
        --hf_token      hf_xxx

After running, update config.yaml:
    dataset:
      hf_dataset_id:    "your-username/dysarthria-prepared"
      train_split:      "train"
      validation_split: "validation"
      audio_column:     "audio"
      text_column:      "sentence"
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict

_ACRONYM_MAP = {
    "tv": "t v", "pc": "p c", "usa": "u s a", "uk": "u k",
    "id": "i d", "ok": "okay",
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root",  required=True,
                   help="Root of your local dysarthric-speech folder (contains manifest.jsonl).")
    # Output — provide exactly one of these two:
    p.add_argument("--output_dir",    default=None,
                   help="Save prepared dataset to a local folder (no HF account needed). "
                        "Use this path as dataset.local_path in config.yaml.")
    p.add_argument("--output_repo",   default=None,
                   help="HF Hub repo to push to, e.g. 'your-username/dysarthria-prepared'. "
                        "Requires --hf_token.")
    p.add_argument("--hf_token",      default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace write token (required only with --output_repo).")
    p.add_argument("--group",         default="dysarthric",
                   choices=["dysarthric", "control", "all"],
                   help="Which speaker group to include (default: dysarthric).")
    p.add_argument("--val_ratio",     type=float, default=0.10,
                   help="Fraction of train entries to use for validation (default: 10%%).")
    p.add_argument("--sampling_rate", type=int, default=16000)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--private",       action="store_true", default=True)
    return p.parse_args()


def load_manifest(root: Path, group: str) -> tuple[list[dict], list[dict]]:
    """Return (train_entries, test_entries) filtered by group."""
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"ERROR: manifest.jsonl not found at {manifest_path}")
        sys.exit(1)

    entries = [json.loads(line) for line in manifest_path.open()]

    # Remove 'xxx' transcripts — 34 TORGO utterances flagged as unintelligible
    # with no linguistic signal. Keeping them adds noise to the loss with zero benefit.
    before  = len(entries)
    entries = [e for e in entries if e.get("transcript", "").strip().lower() != "xxx"]
    removed = before - len(entries)
    if removed:
        print(f"  Removed {removed} entries with transcript='xxx' (no linguistic signal)")

    if group != "all":
        entries = [e for e in entries if e.get("group") == group]

    train_entries = [e for e in entries if e.get("split") == "train"]
    test_entries  = [e for e in entries if e.get("split") == "test"]

    return train_entries, test_entries


def entries_to_dataset(entries: list[dict], root: Path, sr: int) -> Dataset:
    """Convert manifest entries to a HuggingFace Dataset with Audio features."""
    # Validate the first few paths before doing any heavy lifting.
    missing = []
    for e in entries[:5]:
        p = (root / e["audio_path"]).resolve()
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("\nERROR: Audio files not found. Wrong --dataset_root?")
        print(f"  Tried: {missing[0]}")
        print(f"  Tip:   --dataset_root should be the folder that contains manifest.jsonl")
        print(f"         AND from which '{entries[0]['audio_path']}' resolves correctly.")
        sys.exit(1)

    # Use absolute paths so HF can load them correctly.
    audio_paths = [str((root / e["audio_path"]).resolve()) for e in entries]
    sentences   = [normalize_transcript(e["transcript"]) for e in entries]
    speaker_ids = [e.get("speaker_id", "")          for e in entries]
    datasets_   = [e.get("dataset",    "")          for e in entries]
    severities  = [e.get("severity",   "")          for e in entries]

    ds = Dataset.from_dict({
        "audio":      audio_paths,
        "sentence":   sentences,
        "speaker_id": speaker_ids,
        "dataset":    datasets_,
        "severity":   severities,
    })

    # Cast the audio column — HF reads each file and embeds the bytes.
    # This step takes a few minutes for large datasets.
    ds = ds.cast_column("audio", Audio(sampling_rate=sr))
    return ds


def print_stats(name: str, ds: Dataset):
    from collections import Counter
    print(f"  {name:12s}: {len(ds):6d} samples")
    if "speaker_id" in ds.column_names:
        by_spk  = Counter(ds["speaker_id"])
        by_data = Counter(ds["dataset"]) if "dataset" in ds.column_names else {}
        print(f"    datasets : {dict(by_data)}")
        print(f"    speakers : {dict(sorted(by_spk.items()))}")


def main():
    args = parse_args()

    if not args.output_dir and not args.output_repo:
        print("ERROR: provide --output_dir (local save) or --output_repo (HF Hub).")
        sys.exit(1)

    if args.output_repo and not args.hf_token:
        print("ERROR: --output_repo requires --hf_token (or HF_TOKEN env var).")
        print("       https://huggingface.co/settings/tokens")
        sys.exit(1)

    root = Path(args.dataset_root)
    if not root.exists():
        print(f"ERROR: dataset_root not found: {root}")
        sys.exit(1)

    print(f"Dataset root  : {root}")
    print(f"Output        : {args.output_dir or args.output_repo}")
    print(f"Group filter  : {args.group}")
    print(f"Val ratio     : {args.val_ratio:.0%}")

    # ── Load manifest ──────────────────────────────────────────────────────
    train_entries, test_entries = load_manifest(root, args.group)
    print(f"\nManifest entries — train: {len(train_entries)}, test: {len(test_entries)}")

    # ── Build datasets ─────────────────────────────────────────────────────
    # NOTE: cast_column reads every audio file → may take 10-20 min on 14k files.
    print("\nBuilding train dataset (reading + embedding audio)...")
    train_full = entries_to_dataset(train_entries, root, args.sampling_rate)

    print("Building test dataset...")
    test_ds = entries_to_dataset(test_entries, root, args.sampling_rate)

    # ── Carve validation from train ────────────────────────────────────────
    print(f"\nCarving validation split ({args.val_ratio:.0%}, seed={args.seed})...")
    tv = train_full.train_test_split(test_size=args.val_ratio, seed=args.seed)

    prepared = DatasetDict({
        "train":      tv["train"],
        "validation": tv["test"],
        "test":       test_ds,
    })

    print("\nFinal dataset:")
    for split, ds in prepared.items():
        print_stats(split, ds)

    # ── Save ───────────────────────────────────────────────────────────────
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving to local disk: {out}")
        print("(Embedding audio bytes — may take 10–20 min on 14k files.)")
        prepared.save_to_disk(str(out))
        print(f"\nDone! Dataset saved to {out}")
        print("\nUpdate config.yaml with:")
        print(f'  source:           "local"')
        print(f'  local_path:       "{out}"')
        print(f'  audio_column:     "audio"')
        print(f'  text_column:      "sentence"')
        print(f'  train_split:      "train"')
        print(f'  validation_split: "validation"')
    else:
        print(f"\nPushing to HF Hub: {args.output_repo} (private={args.private})")
        print("(Uploading audio bytes — may take several minutes on large datasets.)")
        prepared.push_to_hub(
            args.output_repo,
            token=args.hf_token,
            private=args.private,
        )
        print(f"\nDone! https://huggingface.co/datasets/{args.output_repo}")
        print("\nUpdate config.yaml with:")
        print(f'  source:           "huggingface"')
        print(f'  hf_dataset_id:    "{args.output_repo}"')
        print(f'  audio_column:     "audio"')
        print(f'  text_column:      "sentence"')
        print(f'  train_split:      "train"')
        print(f'  validation_split: "validation"')


if __name__ == "__main__":
    main()
