"""Generate the Kokoro TTS voiceover, one line per caption slot.

Each line is placed at its caption's start time and sped up (if needed) so it
finishes inside its slot, keeping the voice in sync with the on-screen captions.

Usage: python3 voice.py <kokoro-v1.0.onnx> <voices-v1.0.bin> out.wav [voice]
Model files: https://github.com/thewh1teagle/kokoro-onnx/releases (model-files-v1.0)
"""
import sys
import wave

import numpy as np
from kokoro_onnx import Kokoro

from render import CAPTIONS, DUR

# How each caption should be *spoken* (numbers spelled out for clean TTS).
SPOKEN = {
    "From one simple prompt each, Claude Fable 5.2 designed":
        "From one simple prompt each, Claude Fable five point two designed",
    "On a circuit-layout benchmark it jumped from 25% to 69%.":
        "On a circuit layout benchmark, it jumped from twenty-five percent to sixty-nine percent.",
    "An AI designed three gadgets… and they actually work.":
        "An AI designed three gadgets... and they actually work.",
}

MAX_SPEED = 1.4


def main():
    model, voices, out = sys.argv[1:4]
    voice = sys.argv[4] if len(sys.argv) > 4 else "am_michael"
    kokoro = Kokoro(model, voices)
    sr = 24000
    track = np.zeros(int(DUR * sr), dtype=np.float32)
    for st, en, text in CAPTIONS:
        spoken = SPOKEN.get(text, text)
        slot = en - st - 0.15
        speed = 1.1
        samples, sr = kokoro.create(spoken, voice=voice, speed=speed, lang="en-us")
        dur = len(samples) / sr
        if dur > slot:
            speed = min(MAX_SPEED, speed * dur / slot)
            samples, sr = kokoro.create(spoken, voice=voice, speed=speed, lang="en-us")
            dur = len(samples) / sr
        print(f"{st:5.1f}s  speed {speed:.2f}  {dur:4.2f}s / slot {slot:4.2f}s  {spoken}")
        i = int(st * sr)
        seg = samples[: len(track) - i]
        track[i : i + len(seg)] += seg
    peak = np.abs(track).max() or 1
    track = track / peak * 0.9
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((track * 32767).astype(np.int16).tobytes())
    print("wrote", out)


if __name__ == "__main__":
    main()
