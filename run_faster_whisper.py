import time
import sys
from pathlib import Path
from faster_whisper import WhisperModel

audio_file = r"D:/Datasets/dysarthric-speech/train/TORGO/dysarthric/F03/F03_Session2_0099.wav"

# Check if audio file exists
if not Path(audio_file).exists():
    print(f"ERROR: Audio file not found: {audio_file}", file=sys.stderr)
    sys.exit(1)

start_load = time.time()
model = WhisperModel("./model-outputs/model-ct2", device="cuda", compute_type="float16")
load_time = time.time() - start_load
print(f"Model loaded in {load_time:.2f}s")

start_transcribe = time.time()
segments, info = model.transcribe(audio_file)
transcribe_time = time.time() - start_transcribe
print(f"Transcription completed in {transcribe_time:.2f}s")

start_render = time.time()
for seg in segments:
    print(seg.text)
render_time = time.time() - start_render
print(f"\nRendering completed in {render_time:.2f}s")
print(f"Total time: {load_time + transcribe_time + render_time:.2f}s")