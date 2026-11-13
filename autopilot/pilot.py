"""Autopilot: the evolved GA strategy plays Steam Dominion (vs the built-in AI).

    .venv/bin/python autopilot/pilot.py            # play until the game ends
    .venv/bin/python autopilot/pilot.py --turn 4   # joining a game already on turn 4
    .venv/bin/python autopilot/pilot.py --dry      # read + decide, never click
    .venv/bin/python autopilot/pilot.py --games 3  # play, press Rematch, play again (3 games)
    .venv/bin/python autopilot/pilot.py --record recordings/game1   # also save frames + events
    .venv/bin/python autopilot/pilot.py --list-models   # released dominion-v1 models
    .venv/bin/python autopilot/pilot.py --model k4-20260614         # play another model

Each loop: screenshot -> OCR the prompt line -> act on it. Anything it does not
understand is saved to autopilot/unknown/ and it waits for you to handle it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

import glob

import cv2
import numpy as np
from Xlib import display

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from brain import BASE, Brain, cards, print_models  # noqa: E402
from click import click  # noqa: E402
from grab import find_window, grab  # noqa: E402
from recorder import Recorder  # noqa: E402
from vision import Vision, ocr_engine  # noqa: E402

# Dominion's local saves inside the Proton prefix (app id 1131620). Snap Steam and the
# usual Steam installs keep steamapps in different places; use the first that exists.
_PREFIX = "steamapps/compatdata/1131620/pfx/drive_c/users/steamuser/AppData/Roaming/TempleGates/Dominion"
_STEAM_ROOTS = ["~/snap/steam/common/.local/share/Steam", "~/.local/share/Steam", "~/.steam/steam",
                "~/.var/app/com.valvesoftware.Steam/.local/share/Steam"]
SAVE_DIR = next((p for p in (os.path.expanduser(os.path.join(r, _PREFIX)) for r in _STEAM_ROOTS)
                 if os.path.isdir(p)), os.path.expanduser(os.path.join(_STEAM_ROOTS[0], _PREFIX)))
UNKNOWN_DIR = os.path.join(HERE, "unknown")
RESULTS_DIR = os.path.join(HERE, "results")


REC: Recorder | None = None


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(time.strftime("%H:%M:%S"), msg, flush=True)
    if REC:
        REC.event("log", text=msg)


def newest_save():
    """Each local game has its own gamelocal_*.xml; the newest is the current game."""
    saves = glob.glob(os.path.join(SAVE_DIR, "gamelocal_*.xml"))
    return max(saves, key=os.path.getmtime) if saves else None


def read_game_setup():
    """(seed, kingdom card names, opponent names) from the local game save."""
    root = ET.parse(newest_save()).getroot()
    cfg = root.find("config")
    kingdom = [k.strip().replace("_", " ").title() for k in cfg.findtext("kingdomData").split(",")]
    opps = [p.findtext("playerName") for p in root.findall("players")
            if p.findtext("playerType") == "AI"]
    return cfg.findtext("seed"), kingdom, opps


class Pilot:
    def __init__(self, args):
        self.args = args
        self.d = display.Display()
        self.win = find_window(self.d)
        if self.win is None:
            sys.exit("Dominion window not found - start the game first.")
        self.games_left = args.games
        self.new_game(args.turn)

    def new_game(self, turn=None):
        self.save_file = newest_save()
        self.seed, self.kingdom, opps = read_game_setup()
        log(f"game seed {self.seed} vs {', '.join(opps) or '?'}")
        log("kingdom:", ", ".join(self.kingdom))
        self.state_file = os.path.join(HERE, f"run_{self.seed}.json")
        self.brain = Brain(self.kingdom, self.args.model, turn=turn or 1)
        log(f"model {self.brain.policy.name} ({self.brain.policy.kind}), "
            f"trained on {', '.join(self.brain.policy.kingdom)}")
        self._load_state(turn)
        if self.brain.unknown:
            log("cards the GA never learned (never bought):", ", ".join(self.brain.unknown))
        if self.brain.untrained:
            log("cards not in this model's training kingdom (never bought):",
                ", ".join(self.brain.untrained))
        self.names = set(BASE) | set(self.kingdom)
        self.piles = {}
        self.plan = None          # remaining buys this turn
        self.last_kind = None
        self.unknown_count = 0
        self.zero_actions = 0
        self.blind_hand = 0
        self.problems = []        # what went wrong this game: for the stats
        self.result = None        # filled in on the score screen
        self.last_progress = time.time()
        self.hud_misses = 0
        self.pending = None       # (card, (coins, buys) before the click) of the last buy click
        if REC:
            REC.event("game_start", seed=self.seed, kingdom=self.kingdom, opponents=opps,
                      model=self.brain.policy.name,
                      unknown=self.brain.unknown, turn=self.brain.turn)

    # -- persistence (so a restart mid-game keeps the deck count) ------------

    def problem(self, kind, **data):
        """Note something that went wrong (kept per game, for the stats)."""
        self.problems.append({"kind": kind, "turn": self.brain.turn, **data})
        if REC:
            REC.event("problem", kind=kind, **data)

    def _load_state(self, turn_arg):
        if os.path.exists(self.state_file) and not turn_arg:
            s = json.load(open(self.state_file))
            self.brain.turn, self.brain.provinces = s["turn"], s["provinces"]
            self.brain.owned.clear()
            self.brain.owned.update(s["owned"])
            self.brain.empty = set(s.get("empty", []))
            log(f"resumed: turn {self.brain.turn}, deck {dict(self.brain.owned)}")

    def _save_state(self):
        json.dump({"turn": self.brain.turn, "provinces": self.brain.provinces,
                   "owned": dict(self.brain.owned),
                   "empty": sorted(self.brain.empty)}, open(self.state_file, "w"))

    # -- helpers --------------------------------------------------------------

    def shot(self):
        img = grab(self.d, self.win)
        if not hasattr(self, "vision") or img.shape[:2] != self.shape:
            self.shape = img.shape[:2]
            self.vision = Vision(img.shape[1], img.shape[0])
            if REC:
                self.vision.on_read = lambda region, texts, ms: REC.event(
                    "ocr", region=region, ms=ms, texts=[[t.text, t.x, t.y, round(t.conf, 3)] for t in texts])
            self.piles = {}
        return img

    def click(self, x, y, what):
        log(f"  click {what} @ ({x},{y})")
        if REC:
            REC.event("click", x=x, y=y, what=what)
        if not self.args.dry:
            click(self.d, self.win, x, y)
        time.sleep(self.args.delay)

    def press(self, img, *labels) -> bool:
        """Click the first button whose label contains one of `labels`."""
        buttons = self.vision.buttons(img)
        for want in labels:
            for b in buttons:
                if want.lower() in b.text.lower():
                    self.click(b.x, b.y, f"button '{b.text}'")
                    return True
        log(f"  no button among {[b.text for b in buttons]} matches {labels}")
        self.problem("no_button", want=list(labels), seen=[b.text for b in buttons])
        return False

    def pile_pos(self, img, card):
        if card in BASE:
            return self.vision.base_pile(card)
        if card not in self.piles:
            self.piles = self.vision.kingdom_piles(img, self.kingdom)
        return self.piles.get(card)

    def new_turn_check(self, kind):
        if kind in ("action", "treasure") and self.last_kind == "buy":
            self.brain.turn += 1
            self.plan = None
            self.pending = None
            self._save_state()
            log(f"=== turn {self.brain.turn} ({self.brain.phase()}) "
                f"provinces {self.brain.provinces} deck {sum(self.brain.owned.values())}")
        self.last_kind = kind

    # -- prompt handlers --------------------------------------------------------

    def on_action(self, img):
        s = self.vision.screen(img, self.names)
        hand = [t.text for t in s.hand]
        if s.actions == 0:
            self.zero_actions += 1
            if self.zero_actions < 4:          # turn still starting up; look again
                log(f"  hand {hand} shows 0 actions - waiting")
                return
        self.zero_actions = 0
        card = self.brain.choose_action(hand, s.actions or 1)
        has_action = any(cards.is_action(c) for c in hand)
        if card is None and not has_action and len(hand) < 5 and self.blind_hand < 2:
            # The game only asks for actions when you hold one, so a card was probably misread.
            self.blind_hand += 1
            log(f"  hand {hand}: no action card read, but the game asks for one - reading again")
            self.problem("misread_hand", hand=hand)
            return
        self.blind_hand = 0
        log(f"  hand {hand} actions {s.actions} -> {card or 'no action'}")
        if REC:
            REC.event("decide_action", hand=hand, actions=s.actions, coins=s.coins, buys=s.buys,
                      choice=card, turn=self.brain.turn, phase=self.brain.phase())
        if card:
            t = next(t for t in s.hand if t.text == card)
            self.click(t.x, t.y + int(55 * self.vision.sy), card)
            if card == "Moneylender" and "Copper" in hand:
                self.brain.trashed(["Copper"])
        else:
            self.press(img, "End Actions", "Treasures", "Done", "End")

    def on_treasure(self, img):
        self.press(img, "Play Treasures", "Treasures")

    def on_buy(self, img):
        a, coins, buys = self.vision.hud(img)
        if coins is None or buys is None:
            self.hud_misses += 1
            if self.hud_misses < 3:
                log("  couldn't read coins and buys - reading again")
                return
            coins, buys = coins or 0, buys or 1
        self.hud_misses = 0
        if self.pending and (coins, buys) == self.pending[1]:
            # Coins and buys didn't change, so the click bought nothing: that pile is empty.
            card = self.pending[0]
            log(f"  buying {card} did nothing - its pile must be empty; planning again")
            self.problem("empty_pile", card=card)
            self.brain.unbought(card)
            self.brain.empty.add(card)
            self.plan = None
        self.pending = None
        if self.plan is None:
            p = self.vision.provinces_left(img)
            if p is not None:
                self.brain.provinces = p
            self.plan = self.brain.choose_buys(coins, buys)
            log(f"  ${coins} {buys} buy(s), provinces {self.brain.provinces} -> buy {self.plan or 'nothing'}")
            if REC:
                REC.event("decide_buy", coins=coins, buys=buys, provinces=self.brain.provinces,
                          plan=list(self.plan), turn=self.brain.turn, phase=self.brain.phase(),
                          owned=dict(self.brain.owned), empty=sorted(self.brain.empty))
        if self.plan:
            card = self.plan.pop(0)
            pos = self.pile_pos(img, card)
            if pos is None:
                log(f"  can't find the {card} pile; ending buys")
                self.plan = []
            else:
                self.click(*pos, f"buy {card}")
                self.brain.bought(card)
                self.pending = (card, (coins, buys))
                return
        self.press(img, "End Buys", "End Turn", "End")
        self._save_state()

    def on_trash(self, img, prompt):
        s = self.vision.screen(img, self.names)
        hand = [t.text for t in s.hand]
        trash = self.brain.chapel_trash(hand)
        log(f"  trash prompt '{prompt}' hand {hand} -> trash {trash}")
        used = set()
        for card in trash:
            t = next(t for i, t in enumerate(s.hand) if t.text == card and i not in used
                     and not used.add(i))
            self.click(t.x, t.y + int(55 * self.vision.sy), f"trash {card}")
        self.brain.trashed(trash)
        self.press(self.shot(), "Trash", "Done", "Confirm", "OK")

    def on_discard(self, img, prompt):
        s = self.vision.screen(img, self.names)
        hand = [t.text for t in s.hand]
        m = re.search(r"(\d+)", prompt)
        keep = int(m.group(1)) if m else 3
        out = self.brain.discard_down_to(hand, keep)
        log(f"  discard prompt '{prompt}' hand {hand} -> discard {out}")
        used = set()
        for card in out:
            t = next(t for i, t in enumerate(s.hand) if t.text == card and i not in used
                     and not used.add(i))
            self.click(t.x, t.y + int(55 * self.vision.sy), f"discard {card}")
        self.press(self.shot(), "Discard", "Done", "Confirm", "OK")

    def on_unknown(self, img, prompt):
        self.unknown_count += 1
        if self.unknown_count == 3:     # same thing for a while: save it and say so
            os.makedirs(UNKNOWN_DIR, exist_ok=True)
            path = os.path.join(UNKNOWN_DIR, f"{int(time.time())}.png")
            cv2.imwrite(path, img)
            self.problem("unknown_prompt", prompt=prompt, screenshot=os.path.basename(path))
            if not getattr(self.args, "auto", False):
                log(f"  don't know how to handle prompt '{prompt}' - saved {path}; "
                    "waiting for you to resolve it in the game")
                return
            log(f"  don't know how to handle prompt '{prompt}' - saved {path}; trying generic answers")
        if getattr(self.args, "auto", False) and self.unknown_count >= 4 and self.unknown_count % 2 == 0:
            self.auto_resolve(img, prompt)

    def auto_resolve(self, img, prompt):
        """Campaign mode: answer an unknown prompt generically (a button, else the first hand card)."""
        tries = self.unknown_count // 2 - 1
        buttons = self.vision.buttons(img)
        generic = ("Done", "OK", "Confirm", "Skip", "No", "Yes", "End", "Play")
        for b in buttons:
            if any(g.lower() in b.text.lower() for g in generic):
                self.click(b.x, b.y, f"auto: button '{b.text}'")
                self.problem("auto_resolve", prompt=prompt, how=f"button {b.text}")
                return
        hand = self.vision.hand(img, self.names)
        if hand:
            t = hand[tries % len(hand)]
            self.click(t.x, t.y + int(55 * self.vision.sy), f"auto: card {t.text}")
            self.problem("auto_resolve", prompt=prompt, how=f"card {t.text}")

    # -- main loop ------------------------------------------------------------

    def stable_shot(self):
        """A screenshot taken once the screen has stopped animating."""
        prev = self.shot()
        for _ in range(10):
            time.sleep(0.35)
            img = self.shot()
            if img.shape == prev.shape and np.abs(img.astype(int) - prev).mean() < 0.5:
                return img
            prev = img
        return img

    def score_screen(self, img) -> bool:
        """The end-of-game Score dialog: log both scores."""
        texts = self.vision.read(img, (60, 250, 870, 800))
        if not any("Rematch" in t.text for t in texts):
            return False
        def num(x, y=618):
            """A number on the score panel: recognition on a tight crop first, else detection."""
            x0, y0 = self.vision.scale(x - 26, y - 22)
            x1, y1 = self.vision.scale(x + 26, y + 22)
            crop = cv2.resize(img[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            res, _ = ocr_engine()(crop, use_det=False, use_cls=False)
            d = re.sub(r"\D", "", res[0][0]) if res else ""
            if d:
                return int(d)
            x0, y0 = self.vision.scale(x - 30, y - 23)
            x1, y1 = self.vision.scale(x + 30, y + 27)
            res, _ = ocr_engine()(img[y0:y1, x0:x1])
            for r in res or []:
                d = re.sub(r"\D", "", r[1])
                if d:
                    return int(d)
            return None
        me, opp = num(200), num(586)
        my_rank, opp_rank = num(266), num(651)       # the laurel: 1 = first place
        my_n, opp_n = num(334), num(719)
        if my_rank and opp_rank and my_rank != opp_rank:
            outcome = "win" if my_rank < opp_rank else "loss"
        elif me is not None and opp is not None:
            outcome = "win" if me > opp else "loss" if me < opp else "tie"
        else:
            outcome = "tie" if my_rank == opp_rank and my_rank else "unknown"
        os.makedirs(RESULTS_DIR, exist_ok=True)   # keep the score screen as the record
        cv2.imwrite(os.path.join(RESULTS_DIR, f"score_{self.seed}.png"), img)
        self.result = {"outcome": outcome, "me": me, "opponent": opp, "my_rank": my_rank,
                       "opp_rank": opp_rank, "panel_turns": [my_n, opp_n], "turn": self.brain.turn}
        log(f"game over - {outcome}: me {me}, opponent {opp} (turn {self.brain.turn})")
        if REC:
            REC.event("game_over", me=me, opponent=opp, turn=self.brain.turn, outcome=outcome,
                      owned=dict(self.brain.owned), kingdom=self.kingdom)
        self._save_state()
        return True

    def step(self):
        img = self.stable_shot()
        prompt = self.vision.prompt(img)
        p = prompt.lower()
        if not prompt:
            if self.score_screen(img):
                return "done"
            return
        if "game over" in p or " wins" in p or "you won" in p or "you lost" in p:
            log("game finished:", prompt)
            self._save_state()
            return "done"
        kind = ("buy" if p.startswith("buy a card") else
                "treasure" if p.startswith("play treasure") else
                "action" if p.startswith("play action") else
                "trash" if "trash" in p and "up to" in p else
                "discard" if "discard" in p and "down to" in p else None)
        if kind is None:
            if "waiting" not in p and "turn" not in p:
                self.on_unknown(img, prompt)
            return
        self.unknown_count = 0
        self.last_progress = time.time()
        log(f"prompt: {prompt}")
        self.new_turn_check(kind)
        if kind in ("trash", "discard"):
            getattr(self, f"on_{kind}")(img, prompt)
        else:
            getattr(self, f"on_{kind}")(img)

    def rematch(self) -> bool:
        """Press Rematch on the Score screen and wait for the new game's save file."""
        img = self.shot()
        if not self.press(img, "Rematch"):
            return False
        old = self.save_file
        for _ in range(60):
            time.sleep(1)
            new = newest_save()
            if new != old or os.path.getmtime(new) > os.path.getmtime(old) + 1:
                time.sleep(2)
                self.new_game()
                return True
        log("no new game appeared after Rematch")
        return False

    def run(self):
        while True:
            if self.step() == "done":
                self.games_left -= 1
                if self.games_left <= 0 or not self.rematch():
                    break
            if self.args.once:
                break
            time.sleep(self.args.poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turn", type=int, help="current turn number when joining a game in progress")
    ap.add_argument("--model", help="released model name (see --list-models) or a path "
                                    "(default: dominion-v1/models/DEFAULT)")
    ap.add_argument("--list-models", action="store_true", help="list released models and exit")
    ap.add_argument("--dry", action="store_true", help="read and decide, but never click")
    ap.add_argument("--once", action="store_true", help="handle one prompt and exit")
    ap.add_argument("--poll", type=float, default=0.5, help="seconds between screen reads")
    ap.add_argument("--delay", type=float, default=0.8, help="seconds to wait after a click")
    ap.add_argument("--games", type=int, default=1, help="games to play, pressing Rematch in between")
    ap.add_argument("--record", metavar="DIR", help="save window frames and events for videos")
    ap.add_argument("--auto", action="store_true", help="answer unknown prompts generically instead of waiting")
    args = ap.parse_args()
    if args.list_models:
        print_models()
        return
    global REC
    if args.record:
        REC = Recorder(args.record)
    try:
        Pilot(args).run()
    finally:
        if REC:
            REC.close()


if __name__ == "__main__":
    main()
