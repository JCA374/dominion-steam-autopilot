"""Campaign: many games against harder and harder computer opponents, with statistics.

    .venv/bin/python autopilot/campaign.py --per 3              # 3 games each: Very Easy, Easy, Medium, Hard
    .venv/bin/python autopilot/campaign.py --levels Medium,Hard --per 2
    .venv/bin/python autopilot/campaign.py --no-publish          # don't commit/push the Stats page

For every game it sets up a new game through the menus (Main Menu > New Game, the opponent
set by a long hover on its tile), plays it with the autopilot in --auto mode (unknown
prompts are answered generically instead of waiting), and appends one record to
autopilot/stats/games.jsonl. After each game it rebuilds docs/stats.html and, unless
--no-publish, commits and pushes it. Kingdoms are the game's own random base-set kingdoms.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pilot as P  # noqa: E402
from click import click  # noqa: E402
from vision import ocr_engine  # noqa: E402

LEVELS = ["Very Easy", "Easy", "Medium", "Hard"]
LEVEL_BUTTON = {"Very Easy": (209, 472), "Easy": (209, 550), "Medium": (209, 628), "Hard": (209, 705)}
STATS = os.path.join(HERE, "stats", "games.jsonl")
GAME_LIMIT = 75 * 60        # give up on a game after 75 minutes
STALL_LIMIT = 8 * 60        # ... or after 8 minutes without a recognised prompt


def log(*a):
    P.log("[campaign]", *a)


class Menus:
    """Getting from wherever the game is to a new game against a chosen opponent."""

    def __init__(self):
        from Xlib import display
        from grab import find_window
        self.d = display.Display()
        self.win = find_window(self.d)

    def shot(self):
        from grab import grab
        return grab(self.d, self.win)

    def texts(self, img=None):
        img = self.shot() if img is None else img
        res, _ = ocr_engine()(img, use_cls=False)
        out = []
        for box, txt, _ in res or []:
            out.append((txt.strip(), int(sum(p[0] for p in box) / 4), int(sum(p[1] for p in box) / 4)))
        return out

    def click(self, x, y, long=False, wait=2.5, ref=False):
        """ref=True: (x, y) are reference-layout coordinates (927 x 1048), scaled to the window."""
        if ref:
            h, w = self.shot().shape[:2]
            x, y = int(x * w / 927), int(y * h / 1048)
        if long:
            click(self.d, self.win, x, y, hover=1.0, hold=0.3)
        else:
            click(self.d, self.win, x, y)
        time.sleep(wait)

    def find(self, texts, *words):
        norm = lambda s: s.lower().replace(" ", "")              # OCR often drops spaces ("NewGame")
        for t, x, y in texts:
            if any(norm(w) == norm(t) or (len(w) > 4 and norm(w) in norm(t)) for w in words):
                return x, y
        return None

    def to_new_game_dialog(self):
        """From the score screen, the finished board, a game in progress or the main menu."""
        for attempt in range(8):
            tx = self.texts()
            words = {t.lower().replace(" ", "") for t, _, _ in tx}
            if "newgame" in words and ("players" in words or "kingdom" in words):
                return True                                         # the dialog itself
            if (pos := self.find(tx, "Rematch")) and self.find(tx, "Done"):
                self.click(*self.find(tx, "Done"))                 # score screen -> board
                continue
            if (pos := self.find(tx, "New Game")) and self.find(tx, "Play Online"):
                self.click(*pos, long=True)                         # main menu
                continue
            if (pos := self.find(tx, "Main Menu")):
                self.click(*pos, long=True)                         # the in-game menu is open
                for t, x, y in self.texts():                        # leaving a game may ask to confirm
                    if t.lower() in ("yes", "resign", "quit", "leave", "ok", "confirm"):
                        self.click(x, y, long=True)
                continue
            self.click(36, 1010, long=True, ref=True)                        # open the in-game menu
        return False

    def start(self, level):
        if not self.to_new_game_dialog():
            raise RuntimeError("couldn't reach the New Game dialog")
        tx = self.texts()
        if (pos := self.find(tx, "Players")):
            self.click(*pos, long=True, wait=1.5)
        self.click(464, 440, long=True, ref=True)                            # the opponent tile -> Add Player
        tx = self.texts()
        pos = self.find(tx, level) if level != "Easy" else next(
            ((x, y) for t, x, y in tx if t.lower() == "easy"), None)
        self.click(*pos, long=True) if pos else self.click(*LEVEL_BUTTON[level], long=True, ref=True)
        h = self.shot().shape[0] / 1048
        tile = " ".join(t for t, x, y in self.texts() if 380 < x < 550 and 480 * h < y < 530 * h)
        log(f"opponent tile now reads '{tile}'")
        old = P.newest_save()
        old_m = os.path.getmtime(old) if old else 0
        self.click(463, 764, long=True, wait=1.0, ref=True)                  # Play
        for _ in range(90):
            time.sleep(1)
            new = P.newest_save()
            if new and (new != old or os.path.getmtime(new) > old_m + 1):
                time.sleep(4)
                return tile
        raise RuntimeError("no new game started after Play")


def play_one(level, model, game_no, setup=True):
    menus = Menus()
    tile = menus.start(level) if setup else level + " (game already running)"
    args = types.SimpleNamespace(turn=None, model=model, dry=False, once=False, poll=0.5, delay=0.8,
                                 games=1, record=None, auto=True)
    pl = P.Pilot(args)
    t0 = time.time()
    outcome = None
    while True:
        try:
            if pl.step() == "done":
                break
        except Exception as e:                                     # a vision hiccup must not end the campaign
            pl.problem("exception", error=repr(e)[:200])
            log("step failed:", repr(e)[:200])
            time.sleep(3)
        now = time.time()
        if now - t0 > GAME_LIMIT or now - pl.last_progress > STALL_LIMIT:
            why = "time limit" if now - t0 > GAME_LIMIT else "no progress"
            pl.problem("abandoned", why=why)
            log(f"abandoning the game ({why})")
            outcome = "abandoned"
            break
        time.sleep(args.poll)
    res = pl.result or {}
    rec = {
        "game": game_no, "date": time.strftime("%Y-%m-%d %H:%M"), "level": level, "opponent_tile": tile,
        "model": pl.brain.policy.name, "seed": pl.seed, "kingdom": pl.kingdom,
        "unknown_to_model": pl.brain.unknown + pl.brain.untrained,
        "outcome": outcome or res.get("outcome", "unknown"), "me": res.get("me"), "opponent": res.get("opponent"),
        "panel_turns": res.get("panel_turns"), "turns_counted": pl.brain.turn,
        "minutes": round((time.time() - t0) / 60, 1), "owned": dict(pl.brain.owned),
        "empty_piles": sorted(pl.brain.empty), "problems": pl.problems,
    }
    os.makedirs(os.path.dirname(STATS), exist_ok=True)
    with open(STATS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"game {game_no} vs {level}: {rec['outcome']} {rec['me']}-{rec['opponent']}, "
        f"{len(pl.problems)} problems, {rec['minutes']} min")
    return rec


def publish(msg):
    subprocess.run([sys.executable, os.path.join(HERE, "stats_report.py")], check=False)
    files = ["docs/stats.html", "docs/data/stats.json", "autopilot/stats/games.jsonl"]
    files += [f"docs/img/stats/{n}" for n in os.listdir(os.path.join(ROOT, "docs", "img", "stats"))] \
        if os.path.isdir(os.path.join(ROOT, "docs", "img", "stats")) else []
    subprocess.run(["git", "add", *files], cwd=ROOT, check=False)
    r = subprocess.run(["git", "commit", "-q", "-m", f"Stats: {msg}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"],
                       cwd=ROOT)
    if r.returncode == 0:
        if subprocess.run(["git", "push", "-q", "origin", "HEAD"], cwd=ROOT).returncode != 0:
            subprocess.run(["git", "pull", "-q", "--no-rebase", "--no-edit", "origin", "main"], cwd=ROOT)
            subprocess.run(["git", "push", "-q", "origin", "HEAD"], cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default=",".join(LEVELS), help="comma-separated, played in this order")
    ap.add_argument("--per", type=int, default=3, help="games per level")
    ap.add_argument("--model", help="released model (default: dominion-v1/models/DEFAULT)")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--continue-current", action="store_true",
                    help="the first game is already running (against the first level): play it, don't set one up")
    a = ap.parse_args()
    records = [json.loads(l) for l in open(STATS)] if os.path.exists(STATS) else []
    done = len(records)
    n = done
    for level in [l.strip() for l in a.levels.split(",")]:
        already = sum(1 for r in records if r.get("level") == level and r.get("outcome") in ("win", "loss", "tie"))
        for _ in range(max(0, a.per - already)):          # resume: only the games still missing
            n += 1
            try:
                rec = play_one(level, a.model, n, setup=not (a.continue_current and n == done + 1))
                msg = f"game {n} vs {level} AI: {rec['outcome']}"
            except Exception as e:
                log(f"game {n} vs {level} could not be played: {e!r}")
                with open(STATS, "a") as f:
                    f.write(json.dumps({"game": n, "date": time.strftime("%Y-%m-%d %H:%M"), "level": level,
                                        "outcome": "setup_failed", "error": repr(e)[:300], "problems": []}) + "\n")
                msg = f"game {n} vs {level} AI: setup failed"
            if not a.no_publish:
                publish(msg)


if __name__ == "__main__":
    main()
