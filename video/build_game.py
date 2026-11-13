"""Film: one whole recorded game (recordings/session2), sped up, with commentary.
Scenes match video/game/script.md one to one; each plays its stretch of the game.

    .venv/bin/python video/build_game.py [--only 2] [--preview] [--resume]
Output: docs/video/full-game.mp4
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import Recording, play_span, render_film  # noqa: E402

ROOT = os.path.dirname(HERE)
REC = Recording(os.path.join(ROOT, "recordings", "session2"))


def turn_t(n):
    """Recording time at which turn n starts (its '=== turn' log line); turn 1 = start."""
    if n <= 1:
        return 0.0
    e = REC.find("log", lambda e: e["text"].startswith(f"=== turn {n} "))
    if e is None:   # turn 8 began in the restarted process, which logs no turn line for it
        e = [g for g in REC.events if g["kind"] == "game_start"][-1]
    return e["t"]


END = REC.find("log", lambda e: e["text"].startswith("game over"))["t"] + 3
KINGDOM = REC.find("game_start")["kingdom"]


def span(t0, t1, title, note):
    def scene(n):
        yield from play_span(REC, t0, t1, n, title=title, note=note)
    scene.__name__ = title
    return scene


def still(t, title, note):
    from common import footage_with_log
    def scene(n):
        img = footage_with_log(REC, t, title=title, note=note, overlay=False)
        for _ in range(n):
            yield img
    scene.__name__ = title
    return scene


SCENES = [
    still(1.0, "The game", "Very Easy AI. Kingdom: " + ", ".join(KINGDOM) +
         ". Never learned: Vassal, Workshop, Bureaucrat."),
    span(0, turn_t(6), "Turns 1-5: the opening", "Smithy, Silver, Gold, then the first Province on turn 5"),
    span(turn_t(6), turn_t(7) + 15, "Turn 6: the misread", "SMITHY read as AHLIWS; then stopped, fixed, restarted"),
    span(turn_t(8), turn_t(16), "Turns 8-15: the middle", "Silver while poor; Provinces on turns 11, 13 and 14"),
    span(turn_t(16), turn_t(30), "Turns 16-29: the late phase", "Estates; the empty Estate pile caught on turn 25"),
    span(turn_t(30), END, "Turns 30-39: the end", "Duchy, Silver, Copper; three empty piles end it, 55 to 35"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    only = {int(x) for x in a.only.split(",") if x}
    render_film(SCENES, os.path.join(ROOT, "voice", "film-game"), os.path.join(ROOT, "docs", "video", "full-game.mp4"),
                os.path.join(ROOT, "video", "build", "game"), only, a.preview, a.resume)


if __name__ == "__main__":
    main()
