"""Film: how to run the autopilot. Scenes match video/run/script.md (## sections) one to one.

    .venv/bin/python video/build_run.py [--only 3,4] [--preview] [--resume]
Needs: recordings (recordings/session2, else session1) and the voiceover voice/film-run.
Output: docs/video/how-to-run.mp4
"""
import argparse
import os
import sys

from PIL import ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (ACC, BAD, GOLD, GOOD, INK, MUTED, SH, SW, WARN, FPS, Recording, box, bullets,  # noqa
                    canvas, ease, font, footage_with_log, game_view, header, play_span, render_film,
                    terminal_img, terminal_scene, text)

ROOT = os.path.dirname(HERE)
REC_DIR = os.path.join(ROOT, "recordings", "session2")
if not os.path.isdir(REC_DIR):
    REC_DIR = os.path.join(ROOT, "recordings", "session1")
REC = Recording(REC_DIR)
GAME = REC.games()[0]
REPO = "https://github.com/JCA374/dominion-steam-autopilot"


def s_intro(n):
    t = GAME[0] + 40
    for i in range(n):
        img = canvas()
        g = game_view(REC, t + i / FPS * 0.5, overlay=True)
        img.paste(g, (SW - g.width, 0))
        d = ImageDraw.Draw(img)
        d.text((60, 220), "How to run the", font=font(46, True), fill=INK)
        d.text((60, 280), "Dominion autopilot", font=font(46, True), fill=INK)
        text(d, (62, 370), "Install once (about five minutes), then one command plays a whole game.", size=26,
             fill=MUTED, width=SW - g.width - 120)
        yield img


def s_needs(n):
    yield from bullets("What you need", None, [
        "A Linux desktop (built and tested on Wayland; plain X11 should work the same way)",
        "Steam with Dominion, running through Proton (Steam turns it on for this game)",
        "Python 3 and git",
        "No graphics card, no administrator rights: everything goes into one folder",
        "Only against the built-in computer opponents"], n)


def s_install(n):
    blocks = [
        [f"$ git clone --recurse-submodules {REPO}", "Cloning into 'dominion-steam-autopilot'...",
         "Submodule 'dominion-v1' registered ... checked out"],
        ["$ cd dominion-steam-autopilot"],
        ["$ python3 -m venv .venv"],
        ["$ .venv/bin/pip install -r requirements.txt",
         "# python-xlib, numpy, opencv-python-headless, pillow, rapidocr-onnxruntime",
         "Successfully installed ..."],
    ]
    yield from terminal_scene("1. Install", "Clone with the submodule, make a virtual environment, install five packages",
                              blocks, n)


def s_start(n):
    t = GAME[0] + 2
    for i in range(n):
        img = canvas()
        g = game_view(REC, t, overlay=False)
        img.paste(g, (0, 0))
        d = ImageDraw.Draw(img)
        x = g.width + 30
        d.text((x, 40), "2. Start a game", font=font(32, True), fill=INK)
        steps = ["Start Dominion from Steam", "Begin a game against the computer from the game's own menu",
                 "Any computer opponent will do", "Leave the window at its normal size",
                 "Keep it in plain view: clicks go to screen positions, so a window on top would get them"]
        y = 110
        for k, s in enumerate(steps):
            if i < n * k / (len(steps) + 1):
                break
            d.text((x, y), f"{k + 1}.", font=font(24, True), fill=ACC)
            y = text(d, (x + 40, y), s, size=24, fill=INK, width=SW - x - 70) + 18
        yield img


def s_run(n):
    logs = [e["text"] for e in REC.events if e["kind"] == "log"][:16]
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "3. Run it", "One command; the log shows every prompt, decision and click")
        k = 1 + int(min(1.0, i / max(1, n * 0.8)) * len(logs))
        lines = ["$ .venv/bin/python autopilot/pilot.py"] + logs[:k]
        term = terminal_img(lines[-16:], size=19, w=1160, h=540)
        img.paste(term, (60, 150))
        yield img


def s_watch(n):
    t0 = GAME[0] + 20
    yield from play_span(REC, t0, t0 + 70, n, title="While it plays",
                         note="Blue boxes: what it just read. Gold ring: its click. Hands off the mouse.")


def s_score(n):
    over = GAME[3]
    t0 = (over["t"] - 2) if over else GAME[1] - 6
    yield from play_span(REC, t0, t0 + 5, n, title="The score screen",
                         note=(f"It logs the result ({over['me']} to {over['opponent']}) and stops." if over else
                               "It logs the result and stops."))


def s_options(n):
    rows = [("--games N", "play N games, pressing Rematch in between"),
            ("--turn N", "joining a game already on turn N"),
            ("--dry", "read and decide, never click"),
            ("--record DIR", "save frames + events (how these films were made)"),
            ("--model FILE", "play a different strategy file")]
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "Useful options", "Add them after autopilot/pilot.py")
        y = 170
        for k, (a, b) in enumerate(rows):
            if i < n * k / (len(rows) + 1):
                break
            box(d, (60, y, 1220, y + 76))
            d.text((90, y + 20), a, font=font(28, True, mono=True), fill=GOLD)
            d.text((420, y + 24), b, font=font(26), fill=INK)
            y += 92
        yield img


def s_stuck(n):
    yield from bullets("When it gets stuck", None, [
        "An unknown question: it saves a screenshot to autopilot/unknown/ and prints \"don't know ...\"",
        "Answer it yourself in the game - it carries on by itself",
        "Stop it any time with Ctrl+C",
        "Its memory (turn, Provinces, cards owned, empty piles) is saved every turn in autopilot/run_<seed>.json",
        "Start it again: it picks up where it was"], n)


def s_trouble(n):
    rows = [("window not found", "the game isn't running, or has another title"),
            ("never clicks", "is the prompt line under the supply visible?"),
            ("clicks the wrong spots", "the window was resized - restore its size"),
            ("slow", "something else uses the processor; the text reader is CPU-only")]
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "Troubleshooting", None)
        y = 150
        for k, (a, b) in enumerate(rows):
            if i < n * k / (len(rows) + 1):
                break
            d.text((60, y), a, font=font(28, True), fill=WARN)
            d.text((60, y + 42), b, font=font(24), fill=INK)
            y += 110
        yield img


def s_outro(n):
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        d.text((60, 220), "That's it", font=font(50, True), fill=INK)
        y = 310
        for line in ["git clone --recurse-submodules ...", ".venv/bin/pip install -r requirements.txt",
                     "start a game vs the computer", ".venv/bin/python autopilot/pilot.py"]:
            d.text((64, y), line, font=font(26, mono=True), fill=GOOD)
            y += 44
        text(d, (64, y + 30), "Next: the film \"How it works\" explains every part, from screenshot to click.",
             size=24, fill=MUTED, width=1100)
        yield img


SCENES = [s_intro, s_needs, s_install, s_start, s_run, s_watch, s_score, s_options, s_stuck, s_trouble, s_outro]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    only = {int(x) for x in a.only.split(",") if x}
    render_film(SCENES, os.path.join(ROOT, "voice", "film-run"), os.path.join(ROOT, "docs", "video", "how-to-run.mp4"),
                os.path.join(ROOT, "video", "build", "run"), only, a.preview, a.resume)


if __name__ == "__main__":
    main()
