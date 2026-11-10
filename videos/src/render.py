"""Render the 40 s vertical explainer "Claude designs real hardware".

Motion graphics are drawn frame by frame with Pillow and piped to ffmpeg.
Usage: python3 render.py music.wav out.mp4
"""
import math
import random
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

from PIL import Image, ImageDraw, ImageFont

W, H, FPS, DUR = 1080, 1920, 30, 40.0
NFRAMES = int(FPS * DUR)

BG = (10, 14, 20)
TRACE = (34, 52, 46)
COPPER = (232, 131, 58)
GREEN = (43, 212, 125)
RED = (240, 82, 82)
WHITE = (245, 245, 240)
MUTED = (150, 160, 170)
BLUE = (90, 160, 255)

FONT_DIR = "/usr/share/fonts/opentype/inter/"


@lru_cache(maxsize=None)
def font(size, weight="Bold"):
    return ImageFont.truetype(f"{FONT_DIR}Inter-{weight}.otf", max(4, int(size)))


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, start, dur):
    return clamp((t - start) / dur)


def ease_out(x):
    return 1 - (1 - x) ** 3


def ease_back(x):
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def rgba(c, a):
    return (*c[:3], int(255 * clamp(a)))


class Frame:
    """Frame with alpha-aware drawing helpers."""

    def __init__(self, base):
        self.img = base.copy().convert("RGBA")

    def layer(self):
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        return lay, ImageDraw.Draw(lay)

    def put(self, lay):
        self.img = Image.alpha_composite(self.img, lay)

    def text(self, s, cx, cy, size, color=WHITE, a=1.0, weight="Bold", anchor="mm"):
        if a <= 0:
            return
        lay, d = self.layer()
        d.text((cx, cy), s, font=font(size, weight), fill=rgba(color, a), anchor=anchor)
        self.put(lay)

    def pill(self, s, cx, cy, size, fg, bg, a=1.0, pad=(34, 16)):
        if a <= 0:
            return
        f = font(size)
        l, t_, r, b = f.getbbox(s)
        w, h = r - l + 2 * pad[0], b - t_ + 2 * pad[1]
        lay, d = self.layer()
        d.rounded_rectangle(
            (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), radius=h / 2, fill=rgba(bg, a)
        )
        d.text((cx, cy), s, font=f, fill=rgba(fg, a), anchor="mm")
        self.put(lay)


# ---------------------------------------------------------------- background
def make_background():
    rnd = random.Random(3)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    grid = 60
    paths = []
    for _ in range(46):
        x = rnd.randrange(0, W // grid + 1) * grid
        y = rnd.randrange(0, H // grid + 1) * grid
        pts = [(x, y)]
        for _ in range(rnd.randint(3, 7)):
            dirn = rnd.choice([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1)])
            step = rnd.randint(2, 6) * grid
            x, y = x + dirn[0] * step, y + dirn[1] * step
            pts.append((x, y))
        paths.append(pts)
        d.line(pts, fill=TRACE, width=4, joint="curve")
        for px, py in (pts[0], pts[-1]):
            d.ellipse((px - 9, py - 9, px + 9, py + 9), outline=TRACE, width=4)
    # vignette-ish darkening top & bottom for text legibility
    shade = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(shade)
    for i in range(400):
        a = int(200 * (1 - i / 400) ** 2)
        sd.line((0, H - 1 - i, W, H - 1 - i), fill=a)
        sd.line((0, i, W, i), fill=a // 2)
    img = Image.composite(Image.new("RGB", (W, H), BG), img, shade)
    return img, paths


def point_on_path(pts, u):
    segs = list(zip(pts, pts[1:]))
    lens = [math.dist(a, b) for a, b in segs]
    total = sum(lens) or 1
    dist = u * total
    for (a, b), L in zip(segs, lens):
        if dist <= L:
            f = dist / L if L else 0
            return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
        dist -= L
    return pts[-1]


def draw_pulses(fr, paths, t):
    lay, d = fr.layer()
    for i, pts in enumerate(paths[::3]):
        u = ((t * 0.18 + i * 0.37) % 1.0)
        x, y = point_on_path(pts, u)
        for k, r in enumerate((16, 9, 5)):
            d.ellipse((x - r, y - r, x + r, y + r), fill=rgba(GREEN, 0.12 + 0.25 * k))
    fr.put(lay)


# ---------------------------------------------------------------- devices
def speakerphone(fr, cx, cy, r, a=1.0):
    lay, d = fr.layer()
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=rgba((236, 236, 232), a))
    d.ellipse((cx - r * 0.62, cy - r * 0.62, cx + r * 0.62, cy + r * 0.62), fill=rgba((200, 200, 196), a))
    for gx in range(-4, 5):
        for gy in range(-4, 5):
            px, py = cx + gx * r * 0.12, cy + gy * r * 0.12
            if math.dist((px, py), (cx, cy)) < r * 0.55:
                d.ellipse((px - r * 0.025, py - r * 0.025, px + r * 0.025, py + r * 0.025), fill=rgba((120, 120, 120), a))
    for i in range(16):
        ang = i / 16 * 2 * math.pi
        px, py = cx + math.cos(ang) * r * 0.8, cy + math.sin(ang) * r * 0.8
        d.ellipse((px - r * 0.045, py - r * 0.045, px + r * 0.045, py + r * 0.045), fill=rgba(GREEN, a))
    fr.put(lay)


def hub(fr, cx, cy, s, a=1.0):
    lay, d = fr.layer()
    w, h = 1.6 * s, 1.0 * s
    poly = [(cx - w / 2, cy + h / 2), (cx + w / 2, cy + h / 2), (cx + w / 2, cy + h * 0.1), (cx - w / 2, cy - h / 2)]
    d.polygon(poly, fill=rgba((40, 44, 52), a), outline=rgba(COPPER, a), width=max(2, int(s * 0.03)))
    for i in range(4):
        px = cx - w * 0.3 + i * w * 0.2
        py = cy + h * 0.28
        d.ellipse((px - s * 0.04, py - s * 0.04, px + s * 0.04, py + s * 0.04), fill=rgba(MUTED, a))
    d.line((cx - w * 0.38, cy - h * 0.18, cx + w * 0.38, cy + h * 0.05), fill=rgba(BLUE, a), width=max(2, int(s * 0.035)))
    fr.put(lay)


def pet(fr, cx, cy, s, a=1.0):
    lay, d = fr.layer()
    w, h = 0.95 * s, 1.25 * s
    d.rounded_rectangle((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), radius=w * 0.42, fill=rgba((242, 193, 78), a))
    sw, sh = w * 0.62, h * 0.38
    sx, sy = cx - sw / 2, cy - h * 0.3
    d.rounded_rectangle((sx, sy, sx + sw, sy + sh), radius=s * 0.05, fill=rgba((150, 190, 120), a))
    px = sw / 8
    face = ["........", "..#..#..", "........", ".#....#.", "..####..", "........"]
    for j, row in enumerate(face):
        for i, ch in enumerate(row):
            if ch == "#":
                x0, y0 = sx + i * px, sy + j * (sh / 6)
                d.rectangle((x0, y0, x0 + px, y0 + sh / 6), fill=rgba((40, 60, 40), a))
    for i in range(3):
        bx = cx - w * 0.25 + i * w * 0.25
        by = cy + h * 0.27
        d.ellipse((bx - s * 0.07, by - s * 0.07, bx + s * 0.07, by + s * 0.07), fill=rgba((60, 60, 70), a))
    fr.put(lay)


DEVICES = [speakerphone, hub, pet]


def draw_device(fr, i, cx, cy, s, a):
    if i == 0:
        speakerphone(fr, cx, cy, s * 0.6, a)
    elif i == 1:
        hub(fr, cx, cy, s * 0.75, a)
    else:
        pet(fr, cx, cy, s * 0.8, a)


# ---------------------------------------------------------------- captions
CAPTIONS = [
    (0.2, 3.0, "An AI just designed three gadgets… and they actually work."),
    (3.1, 7.4, "From one simple prompt each, Claude Fable 5.2 designed"),
    (7.4, 12.0, "a speakerphone, a voice-assistant hub, and a handheld virtual pet."),
    (12.1, 15.0, "The big leap is geometry."),
    (15.0, 20.0, "On a circuit-layout benchmark it jumped from 25% to 69%."),
    (20.1, 24.0, "When its wiring got tangled, it ripped it up and started over."),
    (24.0, 28.0, "It wrote its own router, and the factory built every board with zero fixes."),
    (28.1, 32.4, "They're still prototypes, with no certification yet."),
    (32.4, 37.0, "Next up: denser, faster boards, tested in simulation."),
    (37.1, 40.0, "From prompt to prototype."),
]


def wrap(words, f, maxw):
    lines, cur = [], []
    for w_ in words:
        trial = " ".join(cur + [w_])
        if cur and f.getlength(trial) > maxw:
            lines.append(cur)
            cur = [w_]
        else:
            cur.append(w_)
    if cur:
        lines.append(cur)
    return lines


def draw_caption(fr, t):
    for st, en, txt in CAPTIONS:
        if st <= t < en:
            break
    else:
        return
    a = min(prog(t, st, 0.15), 1 - prog(t, en - 0.12, 0.12))
    f = font(50, "SemiBold")
    words = txt.split()
    lines = wrap(words, f, W - 160)
    shown = int(len(words) * clamp((t - st) / ((en - st) * 0.8))) + 1
    lh = 66
    y0 = 1735 - (len(lines) - 1) * lh / 2
    lay, d = fr.layer()
    box_h = len(lines) * lh + 40
    d.rounded_rectangle((60, y0 - lh / 2 - 20, W - 60, y0 - lh / 2 - 20 + box_h), radius=26, fill=(0, 0, 0, int(150 * a)))
    k = 0
    for j, line in enumerate(lines):
        s = " ".join(line)
        x = W / 2 - f.getlength(s) / 2
        for w_ in line:
            col = WHITE if k < shown else (110, 115, 120)
            if k == shown - 1:
                col = COPPER
            d.text((x, y0 + j * lh), w_, font=f, fill=rgba(col, a), anchor="lm")
            x += f.getlength(w_ + " ")
            k += 1
    fr.put(lay)


# ---------------------------------------------------------------- scenes
def scene_alpha(t, st, en):
    return min(prog(t, st, 0.3), 1 - prog(t, en - 0.25, 0.25))


def label(fr, t, st, text, color):
    p = ease_out(prog(t, st, 0.45))
    fr.pill(text, W / 2 - 300 * (1 - p), 210, 46, BG, color, a=p * fr.sa)


def scene_hook(fr, t):
    fr.sa = scene_alpha(t, 0, 3.0)
    p = ease_back(prog(t, 0.25, 0.5))
    fr.text("AI BUILT", W / 2, 560, 150 * max(p, 0.01), WHITE, fr.sa * clamp(p * 2), "Black")
    fr.text("THIS?", W / 2, 730, 150 * max(p, 0.01), COPPER, fr.sa * clamp(p * 2), "Black")
    for i in range(3):
        q = ease_back(prog(t, 0.9 + i * 0.3, 0.45))
        if q > 0:
            draw_device(fr, i, 230 + i * 310, 1180, 260 * q, fr.sa)


def scene_what(fr, t):
    fr.sa = scene_alpha(t, 3.0, 12.0)
    label(fr, t, 3.0, "WHAT", COPPER)
    p = ease_out(prog(t, 3.3, 0.5))
    fr.text("3 DEVICES", W / 2, 340, 104, WHITE, fr.sa * p, "Black")
    fr.text("1 simple prompt each", W / 2, 440, 50, MUTED, fr.sa * p, "Medium")
    names = ["Speakerphone", "Voice-assistant hub", "Handheld virtual pet"]
    for i, nm in enumerate(names):
        st = 4.6 + 2.0 * i
        q = ease_back(prog(t, st, 0.5))
        y = 650 + i * 285
        if q > 0:
            draw_device(fr, i, 260, y, 250 * q, fr.sa)
            fr.text(nm, 440 + 40 * (1 - q), y, 52, WHITE, fr.sa * clamp(q), "Bold", anchor="lm")
    for i, (s, c) in enumerate([("BOARDS", GREEN), ("CASES", BLUE), ("FIRMWARE", COPPER)]):
        q = ease_back(prog(t, 10.2 + 0.4 * i, 0.4))
        if q > 0:
            fr.pill(s, 230 + i * 310, 1520, int(40 * max(q, 0.3)), BG, c, a=fr.sa * clamp(q))


def scene_sowhat1(fr, t):
    fr.sa = scene_alpha(t, 12.0, 20.0)
    label(fr, t, 12.0, "SO WHAT", GREEN)
    p = ease_out(prog(t, 12.3, 0.5))
    fr.text("Geometry got good.", W / 2, 340, 78, WHITE, fr.sa * p, "Black")
    fr.text("PCB Gym · circuit-layout benchmark", W / 2, 430, 42, MUTED, fr.sa * p, "Medium")
    base_y, max_h = 1380, 760
    bars = [("Fable 5", 25, 14.0, MUTED), ("Fable 5.2", 69, 15.2, GREEN)]
    lay, d = fr.layer()
    d.line((150, base_y, W - 150, base_y), fill=rgba(MUTED, fr.sa * p), width=4)
    fr.put(lay)
    for i, (nm, val, st, col) in enumerate(bars):
        q = ease_out(prog(t, st, 1.0))
        cx = 360 + i * 360
        h = max_h * val / 100 * q
        lay, d = fr.layer()
        if q > 0:
            d.rounded_rectangle((cx - 110, base_y - h, cx + 110, base_y), radius=18, fill=rgba(col, fr.sa))
        fr.put(lay)
        if q > 0:
            fr.text(f"{round(val * q)}%", cx, base_y - h - 60, 84, col, fr.sa, "Black")
        fr.text(nm, cx, base_y + 55, 46, WHITE, fr.sa * p, "SemiBold")
    q = ease_back(prog(t, 17.3, 0.5))
    if q > 0:
        fr.pill("2.8× BETTER", W / 2, 1530, int(50 * max(q, 0.3)), BG, COPPER, a=fr.sa * clamp(q))


def scene_sowhat2(fr, t):
    fr.sa = scene_alpha(t, 20.0, 28.0)
    label(fr, t, 20.0, "SO WHAT", GREEN)
    p = ease_out(prog(t, 20.3, 0.5))
    fr.text("“I should just rip these", W / 2, 340, 60, WHITE, fr.sa * p, "Bold")
    fr.text("traces and start fresh.”", W / 2, 420, 60, WHITE, fr.sa * p, "Bold")
    fr.text("— Claude, mid-design", W / 2, 495, 38, MUTED, fr.sa * p, "Medium")
    # board outline
    bx0, by0, bx1, by1 = 140, 600, W - 140, 1300
    lay, d = fr.layer()
    d.rounded_rectangle((bx0, by0, bx1, by1), radius=30, fill=rgba((16, 52, 38), fr.sa * p), outline=rgba(GREEN, fr.sa * p), width=5)
    for (x, y) in [(bx0 + 90, by0 + 90), (bx1 - 90, by1 - 90), (bx0 + 90, by1 - 90), (bx1 - 90, by0 + 90)]:
        d.rectangle((x - 50, y - 50, x + 50, y + 50), fill=rgba((30, 30, 34), fr.sa * p))
    fr.put(lay)
    # tangled traces (draw, then rip)
    rnd = random.Random(11)
    tangle = [(rnd.randint(bx0 + 60, bx1 - 60), rnd.randint(by0 + 60, by1 - 60)) for _ in range(26)]
    draw_p = prog(t, 20.6, 1.8)
    rip = prog(t, 22.6, 0.6)
    if draw_p > 0 and rip < 1:
        n = max(2, int(len(tangle) * draw_p))
        lay, d = fr.layer()
        jitter = 14 * rip
        pts = [(x + rnd.uniform(-jitter, jitter), y + rnd.uniform(-jitter, jitter) + 400 * rip ** 2) for x, y in tangle[:n]]
        d.line(pts, fill=rgba(RED, fr.sa * (1 - rip)), width=8, joint="curve")
        fr.put(lay)
        if 22.2 < t < 23.2:
            fr.text("TANGLED", W / 2, 950, 90, RED, fr.sa * (1 - rip), "Black")
    # clean routing: parallel matched pairs with tuning zigzags
    clean = prog(t, 23.2, 1.6)
    if clean > 0:
        lay, d = fr.layer()
        for k in range(5):
            y = by0 + 190 + k * 80
            xs = bx0 + 140
            xe = bx0 + 140 + (bx1 - bx0 - 280) * ease_out(clean)
            pts = [(xs, y)]
            x = xs
            while x < xe:
                if k % 2 == 0 and bx0 + 380 < x < bx0 + 560:
                    pts += [(x + 15, y - 22), (x + 30, y)]
                    x += 30
                else:
                    x += 30
                    pts.append((min(x, xe), y))
            d.line(pts, fill=rgba(COPPER, fr.sa), width=7, joint="curve")
        fr.put(lay)
    q = prog(t, 25.5, 0.35)
    if q > 0:
        s = 1 + 1.5 * (1 - ease_out(q))
        stamp = Image.new("RGBA", (900, 260), (0, 0, 0, 0))
        sd = ImageDraw.Draw(stamp)
        sd.rounded_rectangle((10, 10, 890, 250), radius=30, outline=rgba(GREEN, 1), width=12)
        sd.text((450, 130), "ZERO FIXES", font=font(130, "Black"), fill=rgba(GREEN, 1), anchor="mm")
        stamp = stamp.resize((int(900 * s * 0.8), int(260 * s * 0.8))).rotate(-8, expand=True, resample=Image.BICUBIC)
        alpha = stamp.split()[3].point(lambda v: int(v * fr.sa * clamp(q * 2)))
        stamp.putalpha(alpha)
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        lay.paste(stamp, (int(W / 2 - stamp.width / 2), int(1450 - stamp.height / 2)), stamp)
        fr.put(lay)


def scene_nowwhat(fr, t):
    fr.sa = scene_alpha(t, 28.0, 37.0)
    label(fr, t, 28.0, "NOW WHAT", BLUE)
    p = ease_out(prog(t, 28.3, 0.5))
    fr.text("Still prototypes", W / 2, 340, 90, WHITE, fr.sa * p, "Black")
    fr.text("Not yet validated or certified", W / 2, 430, 44, MUTED, fr.sa * p, "Medium")
    # stacking board layers (isometric)
    cols = [(16, 90, 60), (200, 120, 50), (60, 110, 200), (16, 140, 80)]
    for i in range(4):
        q = ease_out(prog(t, 28.8 + 0.35 * i, 0.5))
        if q <= 0:
            continue
        cy = 1020 - i * 70 - 300 * (1 - q)
        cx = W / 2
        w, h = 330, 150
        poly = [(cx, cy - h), (cx + w, cy), (cx, cy + h), (cx - w, cy)]
        lay, d = fr.layer()
        d.polygon(poly, fill=rgba(cols[i], fr.sa * q * 0.9), outline=rgba(WHITE, fr.sa * q * 0.6), width=3)
        fr.put(lay)
    for i, (s, c) in enumerate([("DENSER", GREEN), ("FASTER", COPPER), ("SIMULATED", BLUE)]):
        q = ease_back(prog(t, 31.0 + 1.0 * i, 0.45))
        if q > 0:
            fr.pill(s, W / 2, 1260 + i * 110, int(50 * max(q, 0.3)), BG, c, a=fr.sa * clamp(q))


def scene_end(fr, t):
    fr.sa = min(prog(t, 37.0, 0.3), 1.0)
    p = ease_back(prog(t, 37.05, 0.5))
    fr.text("PROMPT", W / 2, 560, 140 * max(p, 0.01), WHITE, fr.sa * clamp(p * 2), "Black")
    fr.text("↓", W / 2, 700, 100 * max(p, 0.01), COPPER, fr.sa * clamp(p * 2), "Black")
    fr.text("PROTOTYPE", W / 2, 840, 140 * max(p, 0.01), COPPER, fr.sa * clamp(p * 2), "Black")
    for i in range(3):
        q = ease_back(prog(t, 37.4 + i * 0.15, 0.4))
        if q > 0:
            draw_device(fr, i, 230 + i * 310, 1180, 240 * q, fr.sa)
    q = ease_out(prog(t, 38.0, 0.5))
    fr.text("Follow for more", W / 2, 1440, 52, MUTED, fr.sa * q, "SemiBold")


SCENES = [(0, 3.0, scene_hook), (3.0, 12.0, scene_what), (12.0, 20.0, scene_sowhat1),
          (20.0, 28.0, scene_sowhat2), (28.0, 37.0, scene_nowwhat), (37.0, 40.0, scene_end)]

BASE, PATHS = None, None


def init():
    global BASE, PATHS
    BASE, PATHS = make_background()


def render(i):
    t = i / FPS
    fr = Frame(BASE)
    draw_pulses(fr, PATHS, t)
    for st, en, fn in SCENES:
        if st <= t < en:
            fn(fr, t)
    draw_caption(fr, t)
    # fade from/to black
    edge = min(prog(t, 0, 0.25), 1 - prog(t, DUR - 0.4, 0.4))
    if edge < 1:
        fr.img = Image.blend(Image.new("RGBA", (W, H), (0, 0, 0, 255)), fr.img, edge)
    return fr.img.convert("RGB").tobytes()


def main():
    music, out = sys.argv[1], sys.argv[2]
    only = float(sys.argv[3]) if len(sys.argv) > 3 else None
    if only is not None:  # still preview
        init()
        Image.frombytes("RGB", (W, H), render(int(only * FPS))).save(out)
        return
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", music,
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    with Pool(initializer=init) as pool:
        for k, buf in enumerate(pool.imap(render, range(NFRAMES), chunksize=8)):
            ff.stdin.write(buf)
            if k % 150 == 0:
                print(f"frame {k}/{NFRAMES}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
