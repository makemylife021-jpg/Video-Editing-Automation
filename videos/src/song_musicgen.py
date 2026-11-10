"""Generate the background song with Meta's MusicGen (open source, via transformers).

MusicGen works in ~30 s windows, so the song is made in two passes:
  1. 30 s from the text prompt
  2. a continuation that uses the last 10 s as an audio prompt
The result is trimmed to 40 s with a short fade-out.

Note: the MusicGen model weights are licensed CC-BY-NC 4.0 (non-commercial).

Usage: python3 song_musicgen.py out.wav [model]   (default model: facebook/musicgen-small)
"""
import sys
import time
import wave

import numpy as np
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

PROMPT = (
    "Upbeat modern electronic tech track, 125 BPM, driving synth bass, bright arpeggiated "
    "synths, punchy drums, futuristic and optimistic mood, instrumental, for a technology "
    "explainer video"
)
DUR = 40.0
FIRST = 30.0
OVERLAP = 10.0
FRAME_RATE = 50  # MusicGen audio tokens per second


def main():
    out = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "facebook/musicgen-small"
    torch.manual_seed(42)
    torch.set_num_threads(torch.get_num_threads())
    processor = AutoProcessor.from_pretrained(name)
    model = MusicgenForConditionalGeneration.from_pretrained(name)
    model.eval()
    sr = model.config.audio_encoder.sampling_rate

    t0 = time.time()
    inputs = processor(text=[PROMPT], padding=True, return_tensors="pt")
    with torch.no_grad():
        part1 = model.generate(**inputs, do_sample=True, guidance_scale=3.0,
                               max_new_tokens=int(FIRST * FRAME_RATE))
    a1 = part1[0, 0].numpy()
    print(f"part 1: {len(a1) / sr:.1f}s in {time.time() - t0:.0f}s", flush=True)

    # continuation: condition on the last OVERLAP seconds
    tail = a1[-int(OVERLAP * sr):]
    need = DUR - len(a1) / sr + 1.0
    t0 = time.time()
    inputs = processor(audio=tail, sampling_rate=sr, text=[PROMPT], padding=True, return_tensors="pt")
    with torch.no_grad():
        part2 = model.generate(**inputs, do_sample=True, guidance_scale=3.0,
                               max_new_tokens=int((OVERLAP + need) * FRAME_RATE))
    a2 = part2[0, 0].numpy()[len(tail):]  # drop the echoed prompt
    print(f"part 2: {len(a2) / sr:.1f}s in {time.time() - t0:.0f}s", flush=True)

    song = np.concatenate([a1, a2])[: int(DUR * sr)]
    fade = int(1.5 * sr)
    song[-fade:] *= np.linspace(1, 0, fade)
    song = song / (np.abs(song).max() or 1) * 0.9
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((song * 32767).astype(np.int16).tobytes())
    print("wrote", out, f"{len(song) / sr:.1f}s")


if __name__ == "__main__":
    main()
