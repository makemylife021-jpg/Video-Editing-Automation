"""Synthesize a 40 s, 125 BPM electronic "tech pulse" track plus SFX.

Pure numpy, no samples, so the result is original and royalty-free.
Usage: python3 music.py out.wav
"""
import sys
import wave

import numpy as np

SR = 44100
DUR = 40.0
BPM = 125
BEAT = 60 / BPM
N = int(SR * DUR)
t_all = np.arange(N) / SR
rng = np.random.default_rng(7)


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def lowpass(x, width):
    """Cheap lowpass: moving average of `width` samples."""
    if width <= 1:
        return x
    k = np.ones(width) / width
    return np.convolve(x, k, mode="same")


def add(buf, start, sig):
    i = int(start * SR)
    if i >= len(buf):
        return
    sig = sig[: len(buf) - i]
    buf[i : i + len(sig)] += sig


def env(n, a=0.002, d=0.2):
    t = np.arange(n) / SR
    e = np.exp(-t / d)
    na = max(1, int(a * SR))
    e[:na] *= np.linspace(0, 1, na)
    return e


def kick():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t / 0.03)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * env(n, 0.001, 0.12) * 1.0


def hat(open_=False):
    n = int((0.12 if open_ else 0.04) * SR)
    x = rng.standard_normal(n)
    x = x - lowpass(x, 6)  # crude highpass
    return x * env(n, 0.001, 0.05 if open_ else 0.012) * 0.25


def clap():
    n = int(0.2 * SR)
    x = rng.standard_normal(n)
    x = lowpass(x, 3) - lowpass(x, 30)
    return x * env(n, 0.001, 0.06) * 0.6


def saw(freq, n, detune=0.0):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for d in (-detune, 0.0, detune) if detune else (0.0,):
        p = (t * freq * (1 + d)) % 1.0
        out += 2 * p - 1
    return out / (3 if detune else 1)


def square(freq, n):
    t = np.arange(n) / SR
    return np.sign(np.sin(2 * np.pi * freq * t))


# Am  F  C  G  (one bar each = 4 beats)
CHORDS = [
    (57, [69, 72, 76]),  # A minor
    (53, [65, 69, 72]),  # F
    (48, [64, 67, 72]),  # C
    (55, [67, 71, 74]),  # G
]
BAR = 4 * BEAT

drums = np.zeros(N)
bass = np.zeros(N)
arp = np.zeros(N)
pad = np.zeros(N)
fx = np.zeros(N)

GROOVE_START = 3.0
BREAK_START, BREAK_END = 28.0, 30.84  # filter breakdown into "Now What"
END_HIT = 37.0

# ---- drums
t = GROOVE_START
k = kick()
while t < END_HIT:
    in_break = BREAK_START <= t < BREAK_END
    if not in_break:
        add(drums, t, k)
        add(drums, t + BEAT / 2, hat(open_=True))
        add(drums, t + BEAT / 4, hat())
        add(drums, t + 3 * BEAT / 4, hat())
        beat_idx = round((t - GROOVE_START) / BEAT)
        if beat_idx % 2 == 1:
            add(drums, t, clap())
    t += BEAT
add(drums, END_HIT, k * 1.3)
add(drums, END_HIT, clap() * 1.2)

# ---- bass, arp, pad
t = 0.0
bar_i = 0
while t < DUR:
    root, notes = CHORDS[bar_i % 4]
    # pad (whole song, quiet; louder at end)
    n = int(BAR * SR)
    p = np.zeros(n)
    for m in notes:
        p += saw(midi(m - 12), n, detune=0.006)
    p = lowpass(p, 40) * 0.05
    fade = np.minimum(1, np.arange(n) / (0.3 * SR))
    add(pad, t, p * fade)

    if t >= GROOVE_START - 1e-6 and t < END_HIT:
        # bass: eighth notes, octave bounce
        for i in range(8):
            st = t + i * BEAT / 2
            if BREAK_START <= st < BREAK_END:
                continue
            m = root - 12 + (12 if i % 2 else 0)
            nn = int(BEAT / 2 * SR * 0.9)
            s = lowpass(saw(midi(m), nn), 12) * env(nn, 0.003, 0.18) * 0.35
            add(bass, st, s)
        # arp: sixteenth notes over chord tones
        seq = [notes[0], notes[1], notes[2], notes[1] + 12 if False else notes[2] + 5]
        seq = [notes[0], notes[1], notes[2], notes[0] + 12]
        for i in range(16):
            st = t + i * BEAT / 4
            m = seq[i % 4] + (12 if (i // 4) % 2 else 0)
            nn = int(BEAT / 4 * SR)
            s = lowpass(square(midi(m), nn), 6) * env(nn, 0.002, 0.07) * 0.07
            add(arp, st, s)
    t += BAR
    bar_i += 1

# breakdown: muffle the arp + pad (filter-sweep feel)
i0, i1 = int(BREAK_START * SR), int(BREAK_END * SR)
for buf, w in ((arp, 60), (pad, 120)):
    seg = buf[i0:i1].copy()
    muffled = lowpass(seg, w)
    ramp = np.linspace(0, 1, len(seg)) ** 2  # opens back up
    buf[i0:i1] = muffled * (1 - ramp) + seg * ramp

# ---- FX
def riser(dur, start_f=200, end_f=4000):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    x = rng.standard_normal(n)
    w = np.linspace(40, 2, n).astype(int)
    out = np.empty(n)
    # sweep by blending a few filtered versions
    lo, mid, hi = lowpass(x, 40), lowpass(x, 10), lowpass(x, 2)
    a = tt / dur
    out = lo * (1 - a) ** 2 + mid * 2 * a * (1 - a) + hi * a ** 2
    tone = np.sin(2 * np.pi * np.cumsum(np.geomspace(start_f, end_f, n)) / SR) * 0.15
    return (out * 0.35 + tone) * a ** 1.5


def whoosh(dur=0.5):
    n = int(dur * SR)
    x = lowpass(rng.standard_normal(n), 8)
    e = np.sin(np.linspace(0, np.pi, n)) ** 2
    return x * e * 0.35


def zap(dur=0.35):
    n = int(dur * SR)
    f = np.geomspace(300, 2400, n)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.002, 0.15) * 0.25


def blip(f=1800):
    n = int(0.06 * SR)
    return np.sin(2 * np.pi * f * np.arange(n) / SR) * env(n, 0.001, 0.02) * 0.2


def impact():
    n = int(1.2 * SR)
    tt = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(40 + 60 * np.exp(-tt / 0.05)) / SR)
    noise = lowpass(rng.standard_normal(n), 20)
    return (boom * 0.9 + noise * 0.3) * env(n, 0.001, 0.35)


add(fx, 0.0, riser(3.0))
add(fx, 3.0, impact() * 0.8)
for ts in (12.0, 20.0, 28.0, 37.0):  # scene changes
    add(fx, ts - 0.25, whoosh())
add(fx, BREAK_END - 2.0, riser(2.0, 300, 3000) * 0.7)
for ts in (4.6, 6.6, 8.6, 10.2, 10.6, 11.0):  # device pops / chips
    add(fx, ts, blip(1600 + 200 * (ts % 3)))
add(fx, 14.2, zap())  # 25% bar
add(fx, 15.4, zap() * 1.2)  # 69% bar
add(fx, 25.5, impact() * 0.6)  # ZERO FIXES stamp
for ts in (31.0, 32.0, 33.0):
    add(fx, ts, blip(2200))

mix = drums * 0.8 + bass + arp + pad + fx
if len(sys.argv) > 2 and sys.argv[2] == "fx":  # SFX stem only (for layering over another song)
    mix = fx
# soft master: fade in/out, tanh limiter
mix *= np.minimum(1, t_all / 0.05)
mix *= np.minimum(1, (DUR - t_all) / 1.5)
mix = np.tanh(mix * 1.1) * 0.85

stereo = np.stack([mix, np.roll(mix, 220) * 0.98 + mix * 0.02], axis=1)
pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote", sys.argv[1])
