#!/usr/bin/env python3
"""
convert_to_ct2.py — Convert the fine-tuned Whisper turbo model to CTranslate2 format
for use with faster-whisper.

Reads merged_model_dir from config.yaml by default so you don't have to remember paths.

Usage:
    # Uses paths from config.yaml (recommended):
    python convert_to_ct2.py

    # Override paths:
    python convert_to_ct2.py --model_dir ./whisper-turbo-merged --output_dir ./whisper-turbo-ct2

    # Different quantization (float16 for GPU, int8 for CPU):
    python convert_to_ct2.py --quantization float16
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_merged_dir_from_config() -> str:
    if not CONFIG_PATH.exists():
        return "./whisper-turbo-merged"
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("output", {}).get("merged_model_dir", "./whisper-turbo-merged")


def parse_args():
    default_model_dir = load_merged_dir_from_config()

    p = argparse.ArgumentParser(
        description="Convert merged HF model to CTranslate2 format for faster-whisper."
    )
    p.add_argument(
        "--model_dir",
        default=default_model_dir,
        help=f"Path to merged HF model (default from config.yaml: {default_model_dir}).",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for CT2 model. Defaults to <model_dir>-ct2.",
    )
    p.add_argument(
        "--quantization",
        default="int8",
        choices=["float32", "float16", "int8", "int8_float16"],
        help="Quantization type (default: int8). Use float16 if running on GPU.",
    )
    return p.parse_args()


def fix_preprocessor_config(model_path: Path):
    """CT2 converter looks for preprocessor_config.json; newer HF saves processor_config.json."""
    processor_cfg    = model_path / "processor_config.json"
    preprocessor_cfg = model_path / "preprocessor_config.json"
    if processor_cfg.exists() and not preprocessor_cfg.exists():
        print("  [fix] Copying processor_config.json → preprocessor_config.json for CT2")
        shutil.copy(processor_cfg, preprocessor_cfg)


def run_conversion(model_path: Path, output_path: Path, quantization: str):
    tokenizer_files = [
        "tokenizer.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "normalizer.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "generation_config.json",
    ]
    copy_files = [f for f in tokenizer_files if (model_path / f).exists()]

    base_cmd = [
        "--model", str(model_path),
        "--output_dir", str(output_path),
        "--quantization", quantization,
        "--force",
    ]
    if copy_files:
        base_cmd += ["--copy_files"] + copy_files

    # Try the module form first, then the CLI entry-point
    attempts = [
        [sys.executable, "-m", "ctranslate2.converters.transformers"] + base_cmd,
        ["ct2-transformers-converter"] + base_cmd,
    ]

    for cmd in attempts:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            return True
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as e:
            print(f"  [error] Command failed: {' '.join(cmd[:2])}")
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)
            continue

    return False


def main():
    args = parse_args()

    model_path  = Path(args.model_dir).resolve()
    output_path = Path(args.output_dir).resolve() if args.output_dir else model_path.parent / (model_path.name + "-ct2")

    print("=" * 60)
    print("  Whisper Turbo → CTranslate2 Conversion")
    print("=" * 60)
    print(f"  Input  : {model_path}")
    print(f"  Output : {output_path}")
    print(f"  Quant  : {args.quantization}")
    print("=" * 60)

    if not model_path.exists():
        print(f"\n[error] Model directory not found: {model_path}")
        print("  Run finetune.py first — it saves the merged model when training completes.")
        sys.exit(1)

    if not (model_path / "config.json").exists():
        print(f"\n[error] config.json not found in {model_path}")
        print("  Make sure this is a merged HF model directory (not a LoRA checkpoint).")
        sys.exit(1)

    fix_preprocessor_config(model_path)

    print("\n  Running CT2 converter (may take a few minutes)...\n")
    ok = run_conversion(model_path, output_path, args.quantization)

    if not ok:
        print("\n[error] All conversion attempts failed.")
        print("  Install ctranslate2:  pip install ctranslate2")
        print("  Then run manually:")
        print(f'    ct2-transformers-converter --model "{model_path}" '
              f'--output_dir "{output_path}" --quantization {args.quantization} --force')
        sys.exit(1)

    ct2_bin = output_path / "model.bin"
    if ct2_bin.exists():
        size_mb = ct2_bin.stat().st_size / (1024 * 1024)
        print(f"  model.bin size : {size_mb:.1f} MB")

    print(f"\n  CT2 model saved to: {output_path}")
    print("\n" + "=" * 60)
    print("  Done! Use with faster-whisper:")
    print(f'    from faster_whisper import WhisperModel')
    print(f'    model = WhisperModel("{output_path}", device="cuda", compute_type="{args.quantization}")')
    print(f'    segments, info = model.transcribe("audio.wav", beam_size=5)')
    print("=" * 60 + "\n")
    print("  Or evaluate with eval_wer.py:")
    print(f'    python eval_wer.py --model_path "{output_path}" --model_type ct2')
    print()


if __name__ == "__main__":
    main()
