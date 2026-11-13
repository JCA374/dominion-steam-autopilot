"""Shared drawing and rendering for the films.

A film is a list of scene functions. Each takes the number of frames n and yields
n PIL images (1280x720). Scene k gets the narration voice/<slug>/audio/voiceover/chKK.mp3
as its sound and lasts as long as that narration plus a short gap.

Footage comes from a pilot recording (autopilot/pilot.py --record DIR): window frames
plus the events the pilot logged, so every overlay (what it read, what it decided,
where it clicked) is the pilot's own data at that moment, not a re-enactment.
"""
from __future__ import annotations

import bisect
import json
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 25
SW, SH = 1280, 720
GAP = 0.7                     # silence after each scene

BG = (16, 18, 24)
PANEL = (28, 31, 40)
INK = (234, 234, 238)
MUTED = (150, 153, 163)
ACC = (126, 162, 255)
GOOD = (105, 219, 124)
WARN = (255, 169, 77)
BAD = (255, 120, 120)
GOLD = (240, 200, 90)

_FONTS = "/usr/share/fonts/truetype/dejavu/"
_cache = {}


def font(size, bold=False, mono=False):
    name = ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if bold else "") + ".ttf"
    key = (name, size)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(_FONTS + name, size)
    return _cache[key]


def canvas(color=BG):
    return Image.new("RGB", (SW, SH), color)


def wrap(d, s, f, width):
    lines = []
    for para in s.split("\n"):
        cur = ""
        for w in para.split():
            t = (cur + " " + w).strip()
            if d.textlength(t, font=f) > width and cur:
                lines.append(cur)
                cur = w
            else:
                cur = t
        lines.append(cur)
    return lines


def text(d, xy, s, size=28, fill=INK, bold=False, width=None, mono=False, spacing=1.3):
    """Draw (wrapped) text; returns the y below it."""
    f = font(size, bold, mono)
    x, y = xy
    for line in (wrap(d, s, f, width) if width else s.split("\n")):
        d.text((x, y), line, font=f, fill=fill)
        y += int(size * spacing)
    return y


def header(d, title, sub=None):
    d.text((60, 40), title, font=font(40, True), fill=INK)
    if sub:
        d.text((62, 96), sub, font=font(22), fill=MUTED)


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def box(d, xy, fill=PANEL, outline=None, r=14, w=2):
    d.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=w)


def arrow(d, a, b, fill=MUTED, w=4, head=14):
    import math
    d.line([a, b], fill=fill, width=w)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    p1 = (b[0] - head * math.cos(ang - 0.45), b[1] - head * math.sin(ang - 0.45))
    p2 = (b[0] - head * math.cos(ang + 0.45), b[1] - head * math.sin(ang + 0.45))
    d.polygon([b, p1, p2], fill=fill)


def hold(img, n):
    for _ in range(n):
        yield img


def bullets(title, sub, items, n, reveal=True, size=30):
    """Title + bullet points that appear one by one over the scene."""
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, title, sub)
        y = 170
        shown = len(items) if not reveal else min(len(items), 1 + int(i / max(1, n * 0.85) * len(items)))
        for item in items[:shown]:
            d.ellipse((66, y + size * 0.42, 78, y + size * 0.42 + 12), fill=ACC)
            y = text(d, (100, y), item, size=size, width=1100) + 16
        yield img


def terminal_img(lines, title="terminal", size=22, w=1160, h=None, typed=None):
    """A terminal window. `typed` = number of characters of the last command shown (typing effect)."""
    f = font(size, mono=True)
    lh = int(size * 1.45)
    h = h or (70 + lh * max(4, len(lines)))
    img = Image.new("RGB", (w, h), (10, 11, 14))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, w, 34), fill=(42, 44, 52))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse((14 + i * 22, 11, 26 + i * 22, 23), fill=c)
    d.text((90, 8), title, font=font(16), fill=MUTED)
    y = 48
    for i, line in enumerate(lines):
        if typed is not None and i == len(lines) - 1:
            line = line[:typed]
        color = GOOD if line.startswith("$") else (WARN if line.startswith("#") else INK)
        d.text((18, y), line, font=f, fill=color)
        y += lh
    return img


def terminal_scene(title, sub, blocks, n):
    """Commands typed one block at a time; each block is a list of lines (first = command)."""
    per = n / max(1, len(blocks))
    for i in range(n):
        k = min(len(blocks) - 1, int(i / per))
        frac = (i - k * per) / per
        lines = [l for b in blocks[:k] for l in b]
        cur = blocks[k]
        cmd = cur[0]
        typed = int(len(cmd) * min(1.0, frac * 3))
        shown = lines + [cmd] if frac * 3 < 1 else lines + cur
        lines_out = shown[-14:]
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, title, sub)
        t = terminal_img(lines_out, typed=typed if frac * 3 < 1 else None)
        img.paste(t, (60, 150))
        yield img


# ---------------------------------------------------------------- recordings

class Recording:
    """Frames + events of one pilot recording."""

    def __init__(self, path):
        self.path = path
        names = sorted(os.listdir(os.path.join(path, "frames")))
        self.times = [int(n.split(".")[0]) / 1000 for n in names]
        self.files = [os.path.join(path, "frames", n) for n in names]
        self.events = [json.loads(l) for l in open(os.path.join(path, "events.jsonl"))]
        self.ev_t = [e["t"] for e in self.events]
        self._img_cache = {}

    def frame(self, t) -> Image.Image:
        i = max(0, bisect.bisect_right(self.times, t) - 1)
        f = self.files[i]
        if f not in self._img_cache:
            if len(self._img_cache) > 64:
                self._img_cache.clear()
            self._img_cache[f] = Image.open(f).convert("RGB")
        return self._img_cache[f]

    def between(self, t0, t1, kind=None):
        i = bisect.bisect_left(self.ev_t, t0)
        j = bisect.bisect_right(self.ev_t, t1)
        return [e for e in self.events[i:j] if kind is None or e["kind"] == kind]

    def last(self, t, kind, pred=None):
        i = bisect.bisect_right(self.ev_t, t)
        for e in reversed(self.events[:i]):
            if e["kind"] == kind and (pred is None or pred(e)):
                return e
        return None

    def find(self, kind, pred=None, after=0.0):
        for e in self.events:
            if e["t"] >= after and e["kind"] == kind and (pred is None or pred(e)):
                return e
        return None

    def games(self):
        """[(start_t, end_t, game_start_event, game_over_event or None)]"""
        starts = [e for e in self.events if e["kind"] == "game_start"]
        out = []
        for k, s in enumerate(starts):
            end = starts[k + 1]["t"] if k + 1 < len(starts) else self.times[-1]
            over = self.find("game_over", lambda e: e["t"] <= end, after=s["t"])
            out.append((s["t"], over["t"] + 4 if over else end, s, over))
        return out


def game_view(rec: Recording, t, overlay=True, scale_h=SH, ocr_window=1.6, click_window=1.0):
    """The game window at time t, with what the pilot read (boxes) and clicked (ring)."""
    src = rec.frame(t).copy()
    if overlay:
        d = ImageDraw.Draw(src)
        for e in rec.between(t - ocr_window, t, "ocr"):
            x0, y0, x1, y1 = e["region"]
            d.rectangle((x0, y0, x1, y1), outline=ACC, width=3)
            for txt, x, y, conf in e["texts"]:
                f = font(18, True)
                tw = d.textlength(txt, font=f)
                d.rectangle((x - tw / 2 - 4, y - 13, x + tw / 2 + 4, y + 11), fill=(20, 30, 70))
                d.text((x - tw / 2, y - 12), txt, font=f, fill=(200, 220, 255))
        for e in rec.between(t - click_window, t + 0.2, "click"):
            age = max(0.0, t - e["t"]) / click_window
            r = 16 + 30 * age
            d.ellipse((e["x"] - r, e["y"] - r, e["x"] + r, e["y"] + r), outline=GOLD, width=5)
            d.ellipse((e["x"] - 6, e["y"] - 6, e["x"] + 6, e["y"] + 6), fill=GOLD)
    w = int(src.width * scale_h / src.height)
    return src.resize((w, scale_h), Image.LANCZOS)


def log_lines(rec: Recording, t, n=14):
    evs = [e for e in rec.between(t - 600, t, "log")]
    return [e["text"] for e in evs[-n:]]


def footage_with_log(rec, t, title=None, note=None, speed=None, overlay=True):
    """Game on the left, the pilot's log on the right."""
    img = canvas()
    g = game_view(rec, t, overlay)
    img.paste(g, (0, 0))
    d = ImageDraw.Draw(img)
    x = g.width + 24
    width = SW - x - 24
    y = 24
    if title:
        y = text(d, (x, y), title, size=26, bold=True, width=width) + 6
    if note:
        y = text(d, (x, y), note, size=20, fill=MUTED, width=width) + 10
    d.line((x, y, SW - 24, y), fill=(60, 64, 76), width=2)
    y += 12
    f = font(15, mono=True)
    parts = [p for line in log_lines(rec, t, 40) for p in wrap(d, line, f, width)[:2]]
    room = max(1, (SH - 60 - y) // 20)
    for part in parts[-room:]:            # newest lines at the bottom, oldest dropped
        color = GOLD if "click" in part else (GOOD if "===" in part or "->" in part or "game over" in part else INK)
        d.text((x, y), part, font=f, fill=color)
        y += 20
    if speed:
        tag = f"{speed:g}x speed" if speed != 1 else "real time"
        d.rounded_rectangle((x, SH - 46, x + 150, SH - 16), radius=8, fill=(50, 54, 66))
        d.text((x + 12, SH - 42), tag, font=font(18, True), fill=WARN)
    return img


def play_span(rec, t0, t1, n, **kw):
    """n frames covering recording time t0..t1 (sped up or slowed to fit)."""
    speed = (t1 - t0) / max(1e-6, n / FPS)
    for i in range(n):
        t = t0 + (t1 - t0) * i / max(1, n - 1)
        yield footage_with_log(rec, t, speed=round(speed, 1), **kw)


# ---------------------------------------------------------------- render

def duration(path):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout
    return float(out)


def render_film(scenes, voice_dir, out_path, tmp_dir, only=(), preview=False, resume=False):
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    files = []
    for k, fn in enumerate(scenes, 1):
        out = os.path.join(tmp_dir, f"scene{k:02d}.mp4")
        files.append(out)
        if only and k not in only:
            continue
        if resume and os.path.exists(out):
            continue
        a = os.path.join(voice_dir, "audio", "voiceover", f"ch{k:02d}.mp3")
        a = a if os.path.exists(a) else None
        dur = (duration(a) if a else 12.0) + GAP
        n = int(dur * FPS)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{SW}x{SH}", "-r", str(FPS), "-i", "-"]
        if a:
            cmd += ["-i", a, "-af", f"apad=pad_dur={GAP + 1}", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "1"]
        else:
            cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-c:a", "aac", "-b:a", "64k"]
        cmd += ["-t", f"{n / FPS:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                "-pix_fmt", "yuv420p", "-r", str(FPS), out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        count, last = 0, None
        for img in fn(n):
            if count >= n:
                break
            if img.size != (SW, SH):
                img = img.resize((SW, SH))
            if preview and count % (n // 3 + 1) == 0:
                img.save(os.path.join(tmp_dir, f"preview{k:02d}_{count}.png"))
            p.stdin.write(img.tobytes())
            last = img
            count += 1
        while count < n:
            p.stdin.write(last.tobytes())
            count += 1
        p.stdin.close()
        p.wait()
        print(f"scene {k}: {fn.__name__} {dur:.1f} s", flush=True)
    if not only:
        lst = os.path.join(tmp_dir, "list.txt")
        open(lst, "w").write("".join(f"file '{os.path.abspath(f)}'\n" for f in files))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c", "copy", "-movflags", "+faststart", out_path], check=True)
        print("film:", out_path, f"{duration(out_path):.0f} s")
