#!/usr/bin/env python3
"""
stream_transcribe.py — Keyboard-controlled microphone transcription
using a CTranslate2 faster-whisper model from a network volume.

Controls:
  SPACE  — start recording
  SPACE  — stop recording and send to model
  Q      — quit

Usage:
    python stream_transcribe.py --model_path /workspace/whisper-turbo-merged-ct2
    python stream_transcribe.py --model_path /workspace/whisper-turbo-merged-ct2 \
        --device cpu --compute_type int8 --beam_size 3
"""

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE   = 16_000
MIN_CLIP_S    = 0.2     # clips shorter than this are discarded


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Keyboard-controlled microphone transcription via faster-whisper CT2 model"
    )
    p.add_argument("--model_path", required=True,
                   help="Path to CTranslate2 model directory "
                        "(e.g. /workspace/whisper-turbo-merged-ct2)")
    p.add_argument("--device",       default=None,
                   help="'cuda' or 'cpu'. Auto-detected if omitted.")
    p.add_argument("--compute_type", default=None,
                   help="'float16' (GPU) or 'int8' (CPU). Auto-selected from device if omitted.")
    p.add_argument("--language",     default="en")
    p.add_argument("--beam_size",    type=int, default=5)
    p.add_argument("--initial_prompt", default="Complete sentence.",
                   help="Prompt fed to the Whisper decoder. "
                        "Use 'Single isolated word.' for single-word inputs.")
    p.add_argument("--device_index", type=int, default=None,
                   help="Microphone device index. Run with --list_devices to see options.")
    p.add_argument("--list_devices", action="store_true",
                   help="Print available audio input devices and exit.")
    return p.parse_args()


# ─── Keyboard (single keypress, no Enter needed) ──────────────────────────────

def _getch_unix() -> str:
    import termios, tty
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    except KeyboardInterrupt:
        return "q"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _getch_win() -> str:
    import msvcrt
    return msvcrt.getwch()


def getch() -> str:
    if sys.platform == "win32":
        return _getch_win()
    if not os.isatty(sys.stdin.fileno()):
        # Not a real terminal (piped/redirected) — fall back to line input.
        line = sys.stdin.readline()
        return line.strip()[0] if line.strip() else "q"
    return _getch_unix()


# ─── Audio recorder ───────────────────────────────────────────────────────────

class Recorder:
    """Accumulates microphone frames while recording is active."""

    def __init__(self):
        self._frames: list   = []
        self._lock           = threading.Lock()
        self.recording       = False

    def start(self):
        with self._lock:
            self._frames   = []
            self.recording = True

    def stop(self) -> np.ndarray:
        with self._lock:
            self.recording = False
            if not self._frames:
                return np.array([], dtype=np.float32)
            return np.concatenate(self._frames)

    def feed(self, chunk: np.ndarray):
        with self._lock:
            if self.recording:
                self._frames.append(chunk.copy())


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    # ── Device / compute type ──────────────────────────────────────────────
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else None
    except ImportError:
        has_cuda = False
        gpu_name = None

    device       = args.device       or ("cuda" if has_cuda else "cpu")
    compute_type = args.compute_type or ("float16" if device == "cuda" else "int8")

    # ── Load CT2 model ─────────────────────────────────────────────────────
    from faster_whisper import WhisperModel

    model_path = args.model_path
    if not Path(model_path).exists():
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)

    print(f"\nLoading model : {model_path}")
    print(f"  device       = {device}  ({gpu_name or 'CPU'})")
    print(f"  compute_type = {compute_type}")
    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    print("  Model ready.\n")

    # ── Inference worker (runs in background thread) ───────────────────────
    infer_q: queue.Queue = queue.Queue()

    def infer_worker():
        while True:
            audio = infer_q.get()
            if audio is None:
                infer_q.task_done()
                break

            dur = len(audio) / SAMPLE_RATE

            if dur < MIN_CLIP_S:
                print(f"\n  [skipped — {dur:.2f}s, too short]\n")
                _prompt()
                infer_q.task_done()
                continue

            print(f"\n  [transcribing {dur:.1f}s ...]")
            t0 = time.perf_counter()

            segments, _ = model.transcribe(
                audio,
                language        = args.language,
                beam_size       = args.beam_size,
                initial_prompt  = args.initial_prompt,
                vad_filter      = False,
                no_repeat_ngram_size = 3,
                repetition_penalty   = 1.2,
            )
            text    = " ".join(s.text for s in segments).strip()
            elapsed = time.perf_counter() - t0
            rtf     = elapsed / dur

            print(f"  RTF {rtf:.2f}x | {dur:.1f}s audio → {elapsed:.1f}s inference")
            print(f"  >>> {text or '[no speech detected]'}\n")
            _prompt()

            infer_q.task_done()

    worker = threading.Thread(target=infer_worker, daemon=True)
    worker.start()

    # ── Microphone stream ──────────────────────────────────────────────────
    recorder = Recorder()

    def audio_callback(indata, frames, time_info, status):
        recorder.feed(indata[:, 0])

    def _prompt():
        print("  SPACE = start/stop recording   Q = quit", flush=True)

    if args.device_index is not None:
        dev_info = sd.query_devices(args.device_index)
        print(f"  Mic : [{args.device_index}] {dev_info['name']}")

    with sd.InputStream(
        samplerate = SAMPLE_RATE,
        channels   = 1,
        dtype      = "float32",
        device     = args.device_index,
        callback   = audio_callback,
    ):
        print("Controls:  SPACE = start / stop & transcribe   Q = quit\n")
        _prompt()

        while True:
            ch = getch()

            if ch in ("q", "Q", "\x03"):   # Q or Ctrl+C
                if recorder.recording:
                    recorder.stop()
                print("\nQuitting...")
                break

            if ch == " ":
                if not recorder.recording:
                    recorder.start()
                    print("\r  [ RECORDING ... ]                    ", flush=True)
                else:
                    audio = recorder.stop()
                    dur   = len(audio) / SAMPLE_RATE
                    print(f"\r  [ STOPPED — {dur:.1f}s captured ]        ", flush=True)
                    infer_q.put(audio)

    # ── Clean shutdown ─────────────────────────────────────────────────────
    infer_q.join()
    infer_q.put(None)
    worker.join()


if __name__ == "__main__":
    main()
