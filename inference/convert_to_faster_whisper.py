"""
convert_to_faster_whisper.py — Convert a fine-tuned Whisper model to CTranslate2 format.

faster-whisper uses CTranslate2 under the hood for fast inference.  This script
converts the merged HuggingFace model (from finetune_whisper.py) into the format
that faster-whisper can load directly.

Usage:
    python convert_to_faster_whisper.py [--model_dir ./whisper-dysarthria-finetuned/merged_model]
                                        [--output_dir ./whisper-dysarthria-ct2]
                                        [--quantization int8]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert fine-tuned Whisper to CTranslate2 format for faster-whisper."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./model-outputs/model-hf",
        help="Path to the merged HuggingFace model directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model-outputs/model-ct2",
        help="Where to save the CTranslate2-converted model.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="int8",
        choices=["float16", "float32", "int8", "int8_float16"],
        help="Quantization type for the converted model (default: int8).",
    )
    args = parser.parse_args()

    model_path  = Path(args.model_dir).resolve()
    output_path = Path(args.output_dir).resolve()

    print("=" * 60)
    print("  CTranslate2 Model Conversion for faster-whisper")
    print("=" * 60)
    print(f"  Input model   : {model_path}")
    print(f"  Output dir    : {output_path}")
    print(f"  Quantization  : {args.quantization}")
    print("=" * 60)

    # Validate input
    if not model_path.exists():
        print(f"\n[Error] Model directory not found: {model_path}")
        print("   Run finetune_whisper.py first to create the merged model.")
        sys.exit(1)

    # Check for required files
    required_files = ["config.json"]
    for f in required_files:
        if not (model_path / f).exists():
            print(f"\n[Error] Required file not found: {model_path / f}")
            sys.exit(1)

    # Workaround for HuggingFace renaming 'preprocessor_config.json' to 'processor_config.json'
    # ctranslate2 explicitly looks for 'preprocessor_config.json'
    processor_cfg = model_path / "processor_config.json"
    preprocessor_cfg = model_path / "preprocessor_config.json"
    if processor_cfg.exists() and not preprocessor_cfg.exists():
        print("   [Fix] Duplicating processor_config.json to preprocessor_config.json for ct2...")
        shutil.copy(processor_cfg, preprocessor_cfg)

    # Run the CTranslate2 converter
    print("\n   Running ct2-transformers-converter...")
    print("   (This may take a few minutes)\n")

    # Only pass files to copy that actually exist
    possible_files = [
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
    existing_files = [f for f in possible_files if (model_path / f).exists()]

    cmd = [
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", str(model_path),
        "--output_dir", str(output_path),
        "--quantization", args.quantization,
        "--force",
    ]
    
    if existing_files:
        cmd.extend(["--copy_files"] + existing_files)

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout)
            
        print("   Conversion successful!")
    except subprocess.CalledProcessError as e:
        print(f"   [Error] Conversion failed!")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")

        # Try alternative: use the ct2-transformers-converter CLI directly
        print("\n   Trying alternative conversion method...")
        alt_cmd = [
            "ct2-transformers-converter",
            "--model", str(model_path),
            "--output_dir", str(output_path),
            "--quantization", args.quantization,
            "--force",
        ]
        if existing_files:
            alt_cmd.extend(["--copy_files"] + existing_files)
        try:
            result = subprocess.run(alt_cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            print("   ✅ Conversion successful (alternative method)!")
        except (subprocess.CalledProcessError, FileNotFoundError) as e2:
            print(f"   [Error] Alternative method also failed: {e2}")
            print("\n   Try running manually:")
            print(f'   ct2-transformers-converter --model "{model_path}" '
                  f'--output_dir "{output_path}" --quantization {args.quantization} --force')
            sys.exit(1)

    # Verify output
    ct2_model_file = output_path / "model.bin"
    if ct2_model_file.exists():
        size_mb = ct2_model_file.stat().st_size / (1024 * 1024)
        print(f"\n   Converted model size: {size_mb:.1f} MB")
    else:
        print("\n   [Warning] model.bin not found in output directory.")

    print(f"\n   CTranslate2 model saved to: {output_path}")
    print("\n" + "=" * 60)
    print("  Conversion complete!")
    print(f"  Use in faster-whisper:")
    print(f'    model = WhisperModel("{output_path}")')
    print(f"  Or evaluate: python eval/eval_wer.py --model_path \"{output_path}\" --model_type ct2")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
