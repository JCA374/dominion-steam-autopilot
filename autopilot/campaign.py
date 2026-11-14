"""Campaign: many games against harder and harder computer opponents, with statistics.

    .venv/bin/python autopilot/campaign.py --per 3              # 3 games each: Very Easy, Easy, Medium, Hard
    .venv/bin/python autopilot/campaign.py --levels Medium,Hard --per 2
    .venv/bin/python autopilot/campaign.py --no-publish          # don't commit/push the Stats page
    .venv/bin/python autopilot/campaign.py --games 2 --levels auto --kingdom auto --model auto --no-publish
    .venv/bin/python autopilot/campaign.py --games 1 --levels Easy --kingdom "Smithy,Witch,Market"

--kingdom: "random" (the game deals), "auto" (the best model's own training kingdom), or a
comma-separated card list typed into New Game > Kingdom > Cards (fewer than 10: the game
deals the rest). Without it the dialog's kingdom is left as it is. --levels auto: the easiest
level the model hasn't beaten yet (2 wins in its last 3 games there). --model auto (the
default): dominion-v1's model selector picks the best model for the dealt kingdom.

For every game it sets up a new game through the menus (Main Menu > New Game, the opponent
set by a long hover on its tile), plays it with the autopilot in --auto mode (unknown
prompts are answered generically instead of waiting), and appends one record to
autopilot/stats/games.jsonl. After each game it rebuilds docs/stats.html and, unless
--no-publish, commits and pushes it. The kingdom is the game's own random deal unless --kingdom says otherwise.
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
# The kingdom cards Steam deals from (base set, 2nd edition); any of them can be typed in.
STEAM_BASE_SET = [
    "Cellar", "Chapel", "Moat", "Harbinger", "Merchant", "Vassal", "Village", "Workshop",
    "Bureaucrat", "Gardens", "Militia", "Moneylender", "Poacher", "Remodel", "Smithy",
    "Throne Room", "Bandit", "Council Room", "Festival", "Laboratory", "Library", "Market",
    "Mine", "Sentry", "Witch", "Artisan",
]
# New Game > Kingdom tab, reference layout (927 x 1048)
KINGDOM_TAB, CARDS_BUTTON, CARD_FIELD, CLEAR_BUTTON = (463, 313), (617, 382), (425, 386), (248, 386)
# the 2 x 5 chosen-card slots: title rows and x range, in pixels of a 927 x 1014 window
SLOT_ROWS, SLOT_X, SLOT_WIN = (428, 520), (280, 770), (927, 1014)
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

    def set_kingdom(self, cards: list[str]) -> list[str]:
        """In the New Game dialog: Kingdom > Cards, clear the chosen cards, type in `cards`
        (none = the game deals a random kingdom). Returns the card names read back."""
        from click import key, type_text
        self.click(*KINGDOM_TAB, long=True, wait=1.5, ref=True)
        self.click(*CARDS_BUTTON, long=True, wait=1.5, ref=True)
        if (pos := self.find(self.texts(), "Clear")):
            self.click(*pos, long=True, wait=1.5)
        for card in cards:
            self.click(*CARD_FIELD, long=True, wait=0.8, ref=True)
            type_text(self.d, self.win, card)
            time.sleep(0.5)
            key(self.d, self.win, "Return")
            time.sleep(1.2)
        read = self.read_slots()
        got = [n for n in (slot_card(t) for t in read) if n]
        log(f"kingdom slots read {read} -> {got}")
        return got

    def read_slots(self, img=None) -> list[str]:
        """OCR the titles of the chosen-card slots, one upscaled strip per row (the small
        titles are missed on the full screenshot)."""
        import cv2
        img = self.shot() if img is None else img
        sy, sx = img.shape[0] / SLOT_WIN[1], img.shape[1] / SLOT_WIN[0]
        out = []
        for y in SLOT_ROWS:
            strip = img[int((y - 9) * sy):int((y + 9) * sy), int(SLOT_X[0] * sx):int(SLOT_X[1] * sx)]
            strip = cv2.resize(strip, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            res, _ = ocr_engine()(strip, use_cls=False)
            out += [r[1] for r in sorted(res or [], key=lambda r: r[0][0][0])]
        return out

    def start(self, level, kingdom=None):
        """kingdom: None leaves the dialog's kingdom as it is, [] = random, else these cards."""
        if not self.to_new_game_dialog():
            raise RuntimeError("couldn't reach the New Game dialog")
        if kingdom is not None:
            got = self.set_kingdom(kingdom)
            if len(got) != len(kingdom):
                log(f"warning: asked for {len(kingdom)} kingdom cards, read {len(got)} back")
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


def slot_card(text: str) -> str | None:
    """A chosen-card slot's OCR text ("THRONEROX") -> card name."""
    from vision import match_name
    names = {c.upper(): c for c in STEAM_BASE_SET} | {c.upper().replace(" ", ""): c for c in STEAM_BASE_SET}
    t = text.upper().replace(" ", "")
    return match_name(t, names) or next((c for u, c in names.items() if len(t) >= 6 and u.startswith(t[:6])), None)


def auto_level(model: str | None) -> str:
    """The easiest level `model` (None: any model) hasn't beaten yet: 2 wins in its last 3 games."""
    records = [json.loads(l) for l in open(STATS)] if os.path.exists(STATS) else []
    for level in LEVELS:
        games = [r for r in records if r.get("level") == level and r.get("outcome") in ("win", "loss", "tie")
                 and (model in (None, "auto") or r.get("model") == model)]
        if sum(r["outcome"] == "win" for r in games[-3:]) < 2:
            return level
    return LEVELS[-1]


def resolve_setup(kingdom_arg: str | None, model: str | None):
    """(kingdom list or None, model) from the --kingdom / --model choices."""
    if kingdom_arg is None:
        return None, model
    if kingdom_arg.lower() == "random":
        return [], model
    if kingdom_arg.lower() == "auto":
        from brain import best_kingdom
        m, k = best_kingdom()
        return k, (m["name"] if model in (None, "auto") else model)
    cards = [c.strip().title() for c in kingdom_arg.split(",") if c.strip()]
    bad = [c for c in cards if c not in STEAM_BASE_SET]
    if bad or len(cards) > 10:
        raise SystemExit(f"--kingdom: up to 10 of {', '.join(STEAM_BASE_SET)}; not known: {bad}")
    return cards, model


def play_one(level, model, game_no, setup=True, kingdom=None):
    menus = Menus()
    tile = menus.start(level, kingdom) if setup else level + " (game already running)"
    args = types.SimpleNamespace(turn=None, model=model, dry=False, once=False, poll=0.5, delay=0.8,
                                 games=1, record=None, auto=True)
    pl = P.Pilot(args)
    if kingdom and setup:
        missing = [c for c in kingdom if c not in pl.kingdom]
        if missing:
            pl.problem("kingdom_not_set", missing=missing)
            log(f"warning: the dealt kingdom lacks requested cards {missing}")
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
        "kingdom_requested": kingdom, "model_requested": model,
        "selector": pl.brain.ranking[0]["reason"] if pl.brain.ranking else None,
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
    ap.add_argument("--model", default="auto", help="model name, or 'auto' (the selector picks; default)")
    ap.add_argument("--kingdom", help="'random', 'auto' or a comma-separated card list (default: leave as is)")
    ap.add_argument("--games", type=int,
                    help="play exactly this many games per level (no resuming from games.jsonl)")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--continue-current", action="store_true",
                    help="the first game is already running (against the first level): play it, don't set one up")
    a = ap.parse_args()
    records = [json.loads(l) for l in open(STATS)] if os.path.exists(STATS) else []
    kingdom, model = resolve_setup(a.kingdom, a.model)
    if kingdom:
        log(f"kingdom: {', '.join(kingdom)}" + (f" (model {model})" if model != a.model else ""))
    done = len(records)
    n = done
    for level_arg in [l.strip() for l in a.levels.split(",")]:
        already = sum(1 for r in records if r.get("level") == level_arg and r.get("outcome") in ("win", "loss", "tie"))
        todo = a.games if a.games is not None else max(0, a.per - already)   # resume: only the games still missing
        for _ in range(todo):
            n += 1
            level = auto_level(model) if level_arg.lower() == "auto" else level_arg
            if level_arg.lower() == "auto":
                log(f"difficulty auto -> {level}")
            try:
                rec = play_one(level, model, n, setup=not (a.continue_current and n == done + 1),
                               kingdom=kingdom)
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
