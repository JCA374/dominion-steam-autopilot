"""Advisor: looks over your shoulder while you play Steam Dominion against its bots. It never clicks.

    .venv/bin/python autopilot/advisor.py               # opens the advisor window next to the game
    .venv/bin/python autopilot/advisor.py --no-window   # only the page: http://127.0.0.1:8643
    .venv/bin/python autopilot/advisor.py --install     # add "Dominion Advisor" to the app launcher
    .venv/bin/python autopilot/advisor.py --frames recordings/session2 --speed 4   # replay a recording (testing)

About once a second it takes a screenshot of the game window; when the screen has settled it reads it with
vision.py: the supply pile counts, the prompt line, coins/actions/buys, your hand, both scores, your draw and
discard piles and the opponent's "Added: ..." bar. `Tracker` keeps the game from those readings:
  - your cards = the starting 10 + every card that left the supply and was not the opponent's gain - your trashes
    (the cards that left your hand while a card asked you to trash);
  - the opponent's cards = the starting 10 + every "Added:" bar;
  - your draw pile = your cards minus your hand and the cards seen since the last shuffle, sized to the count shown.
The judge is the advisor network of the rl-v3 site's Play page (docs/playvs/spec.json "advisor"), run with the
same features (rlplay.py): your win probability, and its favourite answers to the question on screen.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import vision  # noqa: E402
from vision import BASE_PILES, Vision, match_name, ocr_engine, parse_added  # noqa: E402

PAGE = os.path.join(HERE, "advisor.html")
PORT = 8643
# The online Play page's advisor: its spec, its network files and its Python (rlplay.py + the engine).
ADVISOR_DIR = os.environ.get("DOMINION_ADVISOR_DIR",
                             os.path.join(ROOT, "..", "trainers", "rl-v3", "docs", "playvs"))
sys.path.insert(0, os.path.join(ADVISOR_DIR, "py"))
import rlplay  # noqa: E402
from dominion_engine import rules  # noqa: E402
from dominion_engine.cards import BY_NAME, CARDS, NAME, NUM_CARDS  # noqa: E402
from dominion_engine import game as eg  # noqa: E402

ID = {n: c.id for n, c in BY_NAME.items()}
BASE_NAMES = ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"]

# Steam's local saves (the same place pilot.py reads the kingdom from).
_PREFIX = "steamapps/compatdata/1131620/pfx/drive_c/users/steamuser/AppData/Roaming/TempleGates/Dominion"
_STEAM_ROOTS = ["~/snap/steam/common/.local/share/Steam", "~/.local/share/Steam", "~/.steam/steam",
                "~/.var/app/com.valvesoftware.Steam/.local/share/Steam"]
SAVE_DIR = next((p for p in (os.path.expanduser(os.path.join(r, _PREFIX)) for r in _STEAM_ROOTS)
                 if os.path.isdir(p)), os.path.expanduser(os.path.join(_STEAM_ROOTS[0], _PREFIX)))

# Where the numbers sit, in vision.py's reference coordinates (927 x 1048).
COUNT_AT = {"Province": (113, 95), "Gold": (181, 94), "Duchy": (114, 157), "Silver": (181, 156),
            "Estate": (113, 219), "Copper": (180, 219), "Curse": (113, 283)}
SLOT_X0, SLOT_W, SLOT_DX = 197, 128, 134.75        # kingdom piles: 5 per row, 2 rows
SLOT_Y0 = (45, 171)
HUD_AT = ((28, 758), (77, 758), (128, 757))          # actions, coins, buys
MY_VP_AT, OPP_VP_AT = (70, 805), (292, 18)
DRAW_AT, DISCARD_AT = (30, 927), (821, 922)
R_PROMPT_LINE = (150, 294, 780, 324)
READ_W = 1100                                       # screenshots are shrunk to this width before reading
OCR_THREADS = 2                                     # the reader's threads: leave the cores to Dominion itself

# The answer that means "no" for each kind of question (the Play page's words).
PASS_LABEL = {eg.PLAY_ACTION: "No action", eg.BUY: "Buy nothing", eg.THRONE: "None", eg.CHAPEL_TRASH: "Stop trashing",
              eg.MONEYLENDER_TRASH: "Don't trash", eg.MINE_TRASH: "Don't", eg.CELLAR_DISCARD: "Stop discarding"}
VERB = {eg.PLAY_ACTION: "Play", eg.BUY: "Buy", eg.THRONE: "Throne Room on", eg.CHAPEL_TRASH: "Trash",
        eg.MONEYLENDER_TRASH: "Trash", eg.MINE_TRASH: "Upgrade", eg.MINE_GAIN: "Gain", eg.REMODEL_TRASH: "Remodel",
        eg.REMODEL_GAIN: "Gain", eg.WORKSHOP_GAIN: "Gain", eg.ARTISAN_GAIN: "Gain", eg.ARTISAN_TOPDECK: "Put on top",
        eg.CELLAR_DISCARD: "Discard", eg.POACHER_DISCARD: "Discard", eg.MILITIA_DISCARD: "Discard",
        eg.BUREAUCRAT_TOPDECK: "Put on top"}
TREASURE_COINS = {"Copper": 1, "Silver": 2, "Gold": 3}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ── the network ─────────────────────────────────────────────────────────────────────────────────────

class Judge:
    """The Play page's advisor network: win probability and answer probabilities for a position."""

    def __init__(self):
        spec = json.load(open(os.path.join(ADVISOR_DIR, "spec.json")))
        self.file = spec["advisor"]
        self.net = rlplay.Net(np.load(os.path.join(ADVISOR_DIR, "models", self.file)).astype(np.float32))
        try:
            rows = json.load(open(os.path.join(ADVISOR_DIR, "models.json")))
            self.label = next((r["label"] for r in rows if r.get("file") == self.file), self.file)
        except (OSError, ValueError):
            self.label = self.file

    def ask(self, g, me, kind, options, may_pass, subject=-1, vp=None):
        """(win probability, [(card id or -1 for pass, probability), ...] best first)."""
        x = rlplay.features(g, me, kind, subject, self.net.nf > rlplay.NF0)
        if vp:                                  # the scores on screen beat the ones counted from the cards
            e = 5 * NUM_CARDS
            x[e + 4], x[e + 5], x[e + 6] = vp[0] / 20, vp[1] / 20, (vp[0] - vp[1]) / 10
        lg, v = self.net.forward(x)
        opts = list(options) + ([rlplay.PASS] if may_pass else [])
        if not opts:
            return (v + 1) / 2, []
        legal = np.array(opts)
        z = lg[legal] - lg[legal].max()
        pr = np.exp(z) / np.exp(z).sum()
        probs = sorted(((-1 if o == rlplay.PASS else int(o), float(q)) for o, q in zip(opts, pr)), key=lambda t: -t[1])
        return (v + 1) / 2, probs


class _Player:
    def __init__(self, deck=(), hand=(), discard=(), play=(), turns=0):
        self.deck, self.hand, self.discard, self.play, self.turns = list(deck), list(hand), list(discard), list(play), turns

    def all_cards(self):
        return self.deck + self.hand + self.discard + self.play


class _Game:
    """Just what rlplay.features reads from a game."""

    def __init__(self, players, supply, current, coins=0, buys=1, actions=1):
        self.players, self.supply, self.current = players, supply, current
        self.coins, self.buys, self.actions = coins, buys, actions


def ids(counter: Counter) -> list[int]:
    return [ID[n] for n, k in sorted(counter.items()) for _ in range(max(0, k))]


def spread(cands: Counter, total: int) -> Counter:
    """`total` cards in the proportions of `cands` (largest remainders)."""
    n = sum(cands.values())
    if total <= 0 or n <= 0:
        return Counter()
    exact = {c: k * total / n for c, k in cands.items() if k > 0}
    out = Counter({c: int(x) for c, x in exact.items()})
    for c in sorted(exact, key=lambda c: -(exact[c] - int(exact[c])))[:total - sum(out.values())]:
        out[c] += 1
    return +out


# ── the game, as read from the screen ───────────────────────────────────────────────────────────────

def newest_save():
    saves = glob.glob(os.path.join(SAVE_DIR, "gamelocal_*.xml"))
    return max(saves, key=os.path.getmtime) if saves else None


def read_kingdom(path) -> list[str]:
    cfg = ET.parse(path).getroot().find("config")
    return [k.strip().replace("_", " ").title() for k in cfg.findtext("kingdomData").split(",")]


class Tracker:
    """The game state, built from successive screen readings (see the module docstring)."""

    def __init__(self, kingdom: list[str]):
        self.kingdom = [k for k in kingdom if k in ID]
        self.names = set(self.kingdom) | set(BASE_NAMES)
        self.start = rules.supply([ID[k] for k in self.kingdom], 2)
        self.supply = list(self.start)
        self._maybe: dict[int, int] = {}          # a new pile count waits for a second reading
        self.opp_gained = Counter()
        self.banner = None                        # the opponent's gains shown on their bar this turn
        self.trashed = Counter()
        self.first = None                         # did I take the first turn?
        self.turns = 0                            # my turns completed
        self.last_kind = None
        self.started = False
        self.hand = Counter()
        self.hand_texts = []
        self.turn_seen = Counter()                # cards seen in my hand this turn
        self.seen = Counter()                     # cards gone to my discard pile since the last shuffle
        self.gained_at_shuffle = Counter()
        self.trash_hand = None                    # my hand when a card asked me to trash
        self.draw_n = self.discard_n = None
        self.vp = [None, None]
        self.coins = self.actions = self.buys = None
        self.prompt = ""
        self.slots: dict[str, tuple] = {}         # kingdom pile -> box (reference coordinates)
        self.wp = []                              # [turn, win probability]
        self.win = None
        self.notes = []

    # -- what I own ------------------------------------------------------------------------------
    def my_gained(self) -> Counter:
        out = Counter()
        for c, n in enumerate(self.start):
            if n > 0:
                left = n - self.supply[c] - self.opp_gained[NAME[c]]
                if left > 0:
                    out[NAME[c]] = left
        return out

    def mine(self) -> Counter:
        return +(Counter({"Copper": 7, "Estate": 3}) + self.my_gained() - self.trashed)

    def theirs(self) -> Counter:
        return Counter({"Copper": 7, "Estate": 3}) + self.opp_gained

    def draw_pile(self) -> Counter:
        mine = self.mine()
        gains = self.my_gained() - self.gained_at_shuffle
        cands = +(mine - self.hand - self.seen - gains)
        if not cands:
            cands = +(mine - self.hand)
        n = self.draw_n if self.draw_n is not None else sum(cands.values())
        return spread(cands, n)

    # -- readings --------------------------------------------------------------------------------
    def read_counts(self, counts: dict[str, int]):
        for name, n in counts.items():
            c = ID[name]
            if n is None or self.start[c] < 0 or n > self.supply[c]:
                continue                          # piles only shrink (base set): a larger number is a misread
            if n == self.supply[c]:
                self._maybe.pop(c, None)
            elif self._maybe.get(c) == n:
                self.supply[c] = n
                del self._maybe[c]
            else:
                self._maybe[c] = n

    def read_bar(self, text: str):
        """The opponent's bar: their name at the start of their turn, then "Added: <cards>" (growing during the
        turn: "Silver" -> "Silver and Copper"), still showing during my next turn. A misread changes nothing."""
        cards = parse_added(text, self.names)
        if cards is None:
            if len(re.sub(r"[^A-Za-z]", "", text)) >= 4 and "added" not in text.lower():
                self.banner = Counter()           # their name again: a new turn of theirs
            return
        if self.banner is None:
            self.banner = Counter()
        new = Counter(cards) - self.banner
        if new:
            self.banner |= Counter(cards)
            self.opp_gained += new
            log(f"opponent gained {sorted(new.elements())}")

    def phase(self, kind):
        """kind: 'action' | 'treasure' | 'buy' for my phase prompts."""
        if not self.started:
            self.started = True
            self.first = not self.opp_gained
            log(f"my first turn; I go {'first' if self.first else 'second'}")
        elif kind in ("action", "treasure") and self.last_kind == "buy":
            self.turns += 1
            self.seen += self.turn_seen
            self.turn_seen = Counter()
        self.last_kind = kind

    def read_hand(self, texts):
        self.hand_texts = texts
        self.hand = Counter()
        for t in texts:
            self.hand[t.text] += t.count
        for c, k in self.hand.items():
            self.turn_seen[c] = max(self.turn_seen[c], k)

    def read_piles(self, draw_n, discard_n):
        if draw_n is not None and self.draw_n is not None and draw_n > self.draw_n + 1:
            self.seen = Counter()                 # a shuffle: the discard pile became the draw pile
            self.turn_seen = Counter(self.hand)
            self.gained_at_shuffle = self.my_gained()
        if draw_n is not None:
            self.draw_n = draw_n
        if discard_n is not None:
            self.discard_n = discard_n

    def trash_prompt(self, active: bool):
        """Cards that leave my hand while a card asks me to trash are trashed."""
        if active and self.trash_hand is None:
            self.trash_hand = Counter(self.hand)
        elif not active and self.trash_hand is not None:
            gone = self.trash_hand - self.hand
            if gone:
                self.trashed += gone
                log(f"I trashed {sorted(gone.elements())}")
            self.trash_hand = None

    # -- a position for the network ----------------------------------------------------------
    def position(self, coins=0, buys=1, actions=1, hand=None):
        me = 0 if self.first in (True, None) else 1
        hand = Counter(self.hand) if hand is None else hand
        mine, theirs = self.mine(), self.theirs()
        deck = self.draw_pile()
        rest = +(mine - hand - deck)
        p = _Player(deck=ids(deck), hand=ids(hand), discard=ids(rest), turns=self.turns)
        o = _Player(discard=ids(theirs), turns=self.turns + (0 if me == 0 else 1))
        players = [p, o] if me == 0 else [o, p]
        return _Game(players, list(self.supply), me, coins, buys, actions), me


# ── the screen ──────────────────────────────────────────────────────────────────────────────────────

class Reader:
    """vision.py's readers, made fast enough to follow a game: a region is read again only when its pixels
    changed, and numbers and the prompt line at fixed places are read without the (slow) text detection."""

    def __init__(self, shape):
        self.shape = shape[:2]
        self.v = Vision(shape[1], shape[0])
        self._memo = {}
        read = self.v.read

        def memo_read(img, region, cls=True):
            x0, y0 = self.v.scale(*region[:2])
            x1, y1 = self.v.scale(*region[2:])
            thumb = cv2.resize(img[y0:y1, x0:x1], (48, 16), interpolation=cv2.INTER_AREA).astype(np.int16)
            hit = self._memo.get((region, cls))
            if hit is not None and np.abs(hit[0] - thumb).mean() < 3:    # the hand's glow pulses below this
                return hit[1]
            out = read(img, region, cls)
            self._memo[(region, cls)] = (thumb, out)
            return out
        self.v.read = memo_read

    def number(self, img, x, y, w=15, h=12):
        """A small number (recognition only on a magnified crop): fast and exact on the pile badges."""
        x0, y0 = self.v.scale(x - w, y - h)
        x1, y1 = self.v.scale(x + w - 2, y + h)
        crop = img[max(0, y0):y1, max(0, x0):x1]
        if crop.size == 0:
            return None
        hit = self._memo.get((x, y))
        if hit is not None and hit[0].shape == crop.shape and np.abs(hit[0] - crop.astype(np.int16)).mean() < 4:
            return hit[1]
        big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        res, _ = ocr_engine()(big, use_det=False, use_cls=False)
        d = re.sub(r"\D", "", res[0][0]) if res else ""
        n = int(d) if d and len(d) <= 2 else None
        self._memo[(x, y)] = (crop.astype(np.int16), n)
        return n

    def prompt(self, img):
        """The prompt line, recognition only; the full reader when that finds nothing sure."""
        x0, y0 = self.v.scale(*R_PROMPT_LINE[:2])
        x1, y1 = self.v.scale(*R_PROMPT_LINE[2:])
        crop = img[y0:y1, x0:x1]
        line = crop.astype(np.int16)
        hit = self._memo.get("prompt")
        if hit is not None and hit[0].shape == line.shape \
                and (np.abs(hit[0] - line).max(axis=2) > 40).sum() <= 4:   # not one letter changed: read before
            return hit[1]
        res, _ = ocr_engine()(crop, use_det=False, use_cls=False)
        if res and res[0][1] >= 0.75 and len(res[0][0]) >= 6:
            text = res[0][0].strip()
        else:
            text = self.v.prompt(img)
        self._memo["prompt"] = (line, text)
        return text

    def hud(self, img):
        return tuple(self.number(img, x, y) for x, y in HUD_AT)

    def counts(self, img, slots):
        out = {n: self.number(img, *xy) for n, xy in COUNT_AT.items()}
        for name, (x0, y0, x1, y1) in slots.items():
            out[name] = self.number(img, x1 - 8, y1 - 11)
        return out

    def slots(self, img, kingdom):
        """Kingdom pile -> its box (reference coordinates), found by the name banners."""
        out = {}
        for name, (x, y) in self.v.kingdom_piles(img, kingdom).items():
            rx, ry = self.v.unscale(x, y)
            ry -= 40
            col = int(round((rx - SLOT_X0 - SLOT_W / 2) / SLOT_DX))
            row = 0 if ry < 120 else 1
            x0 = SLOT_X0 + col * SLOT_DX
            out[name] = (x0, SLOT_Y0[row], x0 + SLOT_W, SLOT_Y0[row] + 120)
        return out


def prompt_kind(prompt: str, buttons: str) -> tuple[str | None, bool]:
    """My phase from the prompt line ('action' | 'treasure' | 'buy'), and whether the line said so (True) or only
    the buttons did (they stay on screen during the opponent's turn, and Play Treasures stays in the buy phase)."""
    q = prompt.lower().replace(" ", "")
    for key, kind in (("buyacard", "buy"), ("playtreasure", "treasure"), ("playaction", "action")):
        if key in q:
            return kind, True
    if not prompt or len(prompt) > 60:            # hidden under a big card: the buttons tell the phase
        for key, kind in (("end actions", "action"), ("play treasures", "treasure"), ("end buys", "buy")):
            if key in buttons:
                return kind, False
    return None, False


# ── the advisor: reading loop + advice ────────────────────────────────────────────────────────────────

class Advisor:
    def __init__(self, judge: Judge):
        self.judge = judge
        self.t: Tracker | None = None
        self.reader: Reader | None = None
        self.save = None
        self.state = {"status": "starting", "judge": judge.label}
        self.frame = None                         # the last reading's screenshot (JPEG bytes)
        self.frame_n = 0
        self.lock = threading.Lock()

    def new_game(self, kingdom, why=""):
        self.t = Tracker(kingdom)
        log(f"new game{why}: {', '.join(kingdom)}")

    # -- one settled screenshot ---------------------------------------------------------------
    def process(self, img):
        if img.shape[1] > READ_W:                 # the reader's own size: text detection costs ~ pixels
            img = cv2.resize(img, (READ_W, round(img.shape[0] * READ_W / img.shape[1])), interpolation=cv2.INTER_AREA)
        if self.reader is None or self.reader.shape != img.shape[:2]:
            self.reader = Reader(img.shape)
        r, t = self.reader, self.t
        if not t.slots or len(t.slots) < len(t.kingdom):
            t.slots.update(r.slots(img, t.kingdom))
        t.read_bar(r.v.opponent_bar(img))
        t.read_counts(r.counts(img, t.slots))
        prompt = r.prompt(img)
        kind, sure = prompt_kind(prompt, "")
        if kind is None and (not prompt or len(prompt) > 60):   # the line is hidden: ask the buttons
            kind, sure = prompt_kind(prompt, " | ".join(b.text.lower() for b in r.v.buttons(img)))
        t.prompt = prompt
        mine_now = kind is not None or self.card_prompt(prompt) is not None or "discard" in prompt.lower()
        if mine_now:
            t.read_hand(r.v.hand(img, t.names))
            a, c, b = r.hud(img)
            t.actions = a if a is not None and a <= 9 else None
            t.coins = c if c is not None and c <= 40 else None
            t.buys = b if b is not None and 1 <= b <= 6 else None
            t.read_piles(r.number(img, *DRAW_AT), r.number(img, *DISCARD_AT))
        if kind and sure:
            t.phase(kind)
        t.trash_prompt("trash" in prompt.lower() and kind is None)
        vp_me, vp_opp = r.number(img, *MY_VP_AT), r.number(img, *OPP_VP_AT)
        if vp_me is not None:
            t.vp[0] = vp_me
        if vp_opp is not None:
            t.vp[1] = vp_opp
        dealing = kind in ("action", "treasure") and not t.hand   # cards still flying in
        advice = self.advise(kind, prompt) if t.started and not dealing else None
        self.publish(img, kind, prompt, advice)

    def card_prompt(self, prompt):
        m = re.match(r"\s*([A-Za-z ]{3,20})\s*:", prompt)
        return match_name(m.group(1), {n.upper(): n for n in self.t.names}) if m else None

    # -- what the network says ------------------------------------------------------------------
    def advise(self, kind, prompt):
        """{'question', 'top': [[label, p]], 'plan': [card...], 'boxes': [...], 'win'} or None."""
        t = self.t
        coins, buys, actions = t.coins or 0, t.buys or 1, t.actions if t.actions is not None else 1
        hand = Counter(t.hand)
        vp = t.vp if None not in t.vp else None
        p = prompt.lower()
        card = self.card_prompt(prompt)
        if kind == "treasure":                    # the engine buys right after the treasures: say it now
            coins += sum(TREASURE_COINS.get(c, 0) * k for c, k in hand.items())
            for c in TREASURE_COINS:
                hand.pop(c, None)
            return self.buys(coins, buys, hand, vp, "After your treasures ($%d)" % coins)
        if kind == "buy":
            return self.buys(coins, buys, hand, vp, "Buy ($%d, %d buy%s)" % (coins, buys, "" if buys == 1 else "s"))
        if kind == "action":
            opts = sorted({ID[c] for c in hand if CARDS[ID[c]].is_action})
            return self.single(eg.PLAY_ACTION, opts, True, hand, vp, "Play an action", coins, buys, actions)
        gainable = lambda n: sorted(c for c in range(NUM_CARDS) if t.supply[c] > 0 and CARDS[c].cost <= n)  # noqa: E731
        hand_opts = sorted({ID[c] for c in hand})
        if card == "Chapel":
            return self.sequence(eg.CHAPEL_TRASH, hand, 4, True, vp, "Chapel: trash up to 4")
        if card == "Cellar":
            return self.sequence(eg.CELLAR_DISCARD, hand, 99, True, vp, "Cellar: discard, then draw as many")
        if card == "Workshop":
            return self.single(eg.WORKSHOP_GAIN, gainable(4), False, hand, vp, "Workshop: gain up to $4")
        if card == "Artisan":
            if "gain" in p:
                return self.single(eg.ARTISAN_GAIN, gainable(5), False, hand, vp, "Artisan: gain up to $5")
            return self.single(eg.ARTISAN_TOPDECK, hand_opts, False, hand, vp, "Artisan: put a card on your deck")
        if card == "Remodel" and "trash" in p:
            return self.single(eg.REMODEL_TRASH, hand_opts, False, hand, vp, "Remodel: trash a card")
        if card == "Mine" and "trash" in p:
            opts = sorted({ID[c] for c in hand if CARDS[ID[c]].is_treasure})
            return self.single(eg.MINE_TRASH, opts, True, hand, vp, "Mine: upgrade a treasure")
        if card == "Moneylender" and "Copper" in hand:
            return self.single(eg.MONEYLENDER_TRASH, [ID["Copper"]], True, hand, vp, "Moneylender: trash a Copper?")
        if card == "Throne Room":
            opts = sorted({ID[c] for c in hand if CARDS[ID[c]].is_action})
            return self.single(eg.THRONE, opts, True, hand, vp, "Throne Room: play which twice?")
        if card == "Bureaucrat" or ("victory" in p and "deck" in p):
            opts = sorted({ID[c] for c in hand if CARDS[ID[c]].is_victory})
            return self.single(eg.BUREAUCRAT_TOPDECK, opts, False, hand, vp, "Bureaucrat: put a Victory card on top")
        if "discard" in p:
            m = re.search(r"(\d+)", p)
            n = int(m.group(1)) if m else max(0, sum(hand.values()) - 3)
            k = eg.POACHER_DISCARD if card == "Poacher" else eg.MILITIA_DISCARD
            return self.sequence(k, hand, n, False, vp, f"Discard {n}", exact=True)
        return self.single(eg.BUY, [], False, hand, vp, None)     # no question I know: just the win chance

    def _ask(self, kind, opts, may_pass, hand, vp, coins=0, buys=1, actions=1, subject=-1):
        g, me = self.t.position(coins, buys, actions, hand)
        return self.judge.ask(g, me, kind, opts, may_pass, subject, vp)

    def _label(self, kind, o):
        return PASS_LABEL.get(kind, "No") if o == -1 else NAME[o]

    def single(self, kind, opts, may_pass, hand, vp, question, coins=0, buys=1, actions=1):
        win, probs = self._ask(kind, opts, may_pass, hand, vp, coins, buys, actions)
        if not question or not probs:
            return {"question": question, "win": win, "top": [], "verb": "", "plan": [], "where": kind}
        return {"question": question, "win": win, "verb": VERB.get(kind, ""), "where": kind,
                "top": [[self._label(kind, o), p] for o, p in probs[:3]],
                "plan": [NAME[probs[0][0]]] if probs[0][0] >= 0 else []}

    def buys(self, coins, buys, hand, vp, question):
        """The best buy now (top 3), then the plan for every buy this turn."""
        t = self.t
        supply = list(t.supply)
        opts = lambda c: [x for x in range(NUM_CARDS) if supply[x] > 0 and CARDS[x].cost <= c]  # noqa: E731
        win, probs = self._ask(eg.BUY, opts(coins), True, hand, vp, coins, buys)
        plan, c, b, saved = [], coins, buys, t.supply
        try:
            while b > 0 and opts(c):
                t.supply = supply
                _, pr = self._ask(eg.BUY, opts(c), True, hand, vp, c, b)
                if pr[0][0] < 0:
                    break
                plan.append(NAME[pr[0][0]])
                supply[pr[0][0]] -= 1
                c -= CARDS[pr[0][0]].cost
                b -= 1
        finally:
            t.supply = saved
        return {"question": question, "win": win, "verb": "Buy", "where": eg.BUY,
                "top": [[self._label(eg.BUY, o), p] for o, p in probs[:3]], "plan": plan}

    def sequence(self, kind, hand, n, may_pass, vp, question, exact=False):
        """Several picks from the hand (Chapel, Cellar, discard down): the network picks one at a time."""
        hand = Counter(hand)
        win, first = self._ask(kind, sorted({ID[c] for c in hand}), may_pass, hand, vp)
        plan = []
        while len(plan) < n and hand:
            _, pr = self._ask(kind, sorted({ID[c] for c in hand}), may_pass and not exact, hand, vp)
            if not pr or pr[0][0] < 0:
                break
            plan.append(NAME[pr[0][0]])
            hand[NAME[pr[0][0]]] -= 1
            hand = +hand
        return {"question": question, "win": win, "verb": VERB.get(kind, ""), "where": kind,
                "top": [[self._label(kind, o), p] for o, p in first[:3]], "plan": plan}

    # -- the page's state ------------------------------------------------------------------------
    def boxes(self, advice):
        """Where the suggested cards are on the game screen, as fractions of the window (for the page's picture)."""
        if not advice or not advice["plan"]:
            return []
        t, v = self.t, self.reader.v
        out = []
        supply_kinds = (eg.BUY, eg.WORKSHOP_GAIN, eg.ARTISAN_GAIN, eg.REMODEL_GAIN, eg.MINE_GAIN)
        boxes = []                                # (card, reference box)
        if advice["where"] in supply_kinds:
            for name in dict.fromkeys(advice["plan"]):
                if name in BASE_PILES:
                    x, y = BASE_PILES[name]
                    boxes.append((name, (x - 33, y - 32, x + 33, y + 32)))
                elif name in t.slots:
                    boxes.append((name, t.slots[name]))
        else:
            want = Counter(advice["plan"])
            for h in t.hand_texts:
                if want[h.text] > 0:
                    want[h.text] -= 1
                    rx, ry = v.unscale(h.x, h.y)
                    boxes.append((h.text, (rx - 58, ry - 18, rx + 58, min(1048, ry + 150))))
        W, H = self.reader.shape[1], self.reader.shape[0]
        for name, (x0, y0, x1, y1) in boxes:
            (a, b), (c, d) = v.scale(x0, y0), v.scale(x1, y1)
            out.append({"card": name, "box": [a / W, b / H, c / W, d / H]})
        return out

    def publish(self, img, kind, prompt, advice):
        t = self.t
        if advice:
            t.win = advice["win"]
            turn = t.turns + 1
            if t.wp and t.wp[-1][0] == turn:
                t.wp[-1][1] = t.win
            else:
                t.wp.append([turn, t.win])
        mine, theirs = t.mine(), t.theirs()
        empty = [NAME[c] for c, n in enumerate(t.supply) if n == 0]
        state = {
            "status": "your turn" if kind else ("your choice" if advice and advice["question"] else "watching"),
            "judge": self.judge.label, "turn": t.turns + 1, "first": t.first, "prompt": prompt,
            "win": t.win, "wp": t.wp, "advice": advice, "boxes": self.boxes(advice),
            "provinces": t.supply[ID["Province"]], "empty": empty,
            "vp": t.vp, "counted_vp": [rules.score(ids(mine)), rules.score(ids(theirs))],
            "mine": dict(sorted(mine.items())), "theirs": dict(sorted(theirs.items())),
            "draw": dict(sorted(t.draw_pile().items())), "hand": dict(sorted(t.hand.items())),
            "piles": {"draw": t.draw_n, "discard": t.discard_n, "total": sum(mine.values())},
            "hud": {"actions": t.actions, "coins": t.coins, "buys": t.buys}, "kingdom": t.kingdom,
            "trashed": dict(t.trashed), "time": time.time(), "aspect": img.shape[1] / img.shape[0],
        }
        h, w = img.shape[:2]
        small = cv2.resize(img, (480, int(480 * h / w)), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self.lock:
            self.state = state
            if ok:
                self.frame, self.frame_n = jpg.tobytes(), self.frame_n + 1
                self.state["frame"] = self.frame_n

    def idle(self, status):
        with self.lock:
            self.state = {**self.state, "status": status, "time": time.time()}

    # -- the loops ------------------------------------------------------------------------------
    def watch(self):
        """The live game: a screenshot every half second, read when it has settled."""
        from Xlib import display
        from grab import find_window, grab
        d = display.Display()
        win, prev = None, None
        while True:
            try:
                save = newest_save()
                if save and save != self.save:
                    self.save = save
                    self.new_game(read_kingdom(save), f" ({os.path.basename(save)})")
                if win is None:
                    win = find_window(d)
                    if win is None:
                        self.idle("waiting for Dominion")
                        time.sleep(2)
                        continue
                img = grab(d, win)
            except Exception as e:                # the window closed or moved: look again
                win, prev = None, None
                self.idle(f"waiting for Dominion ({type(e).__name__})")
                time.sleep(2)
                continue
            if self.t is None:
                self.idle("no game found (start a game in Dominion)")
            elif prev is not None and moving(img, prev) < 0.02:   # no cards flying: read it
                try:
                    self.process(img)
                except Exception as e:
                    log(f"reading failed: {e!r}")
            prev = img
            time.sleep(0.3)

    def replay(self, folder, speed):
        """A recording (recorder.py: frames/<ms>.jpg + events.jsonl), as if it were the live game."""
        starts = [json.loads(l) for l in open(os.path.join(folder, "events.jsonl")) if '"game_start"' in l]
        frames = sorted(glob.glob(os.path.join(folder, "frames", "*.jpg")))
        last_t, k, prev = None, 0, None
        for f in frames:
            ms = int(os.path.basename(f)[:-4])
            while k < len(starts) and starts[k]["t"] * 1000 <= ms:
                self.new_game(starts[k]["kingdom"], f" (recording, seed {starts[k]['seed']})")
                k += 1
            if self.t is None:
                continue
            if last_t is not None and speed:
                time.sleep(max(0, (ms - last_t) / 1000 / speed))
            last_t = ms
            img = cv2.imread(f)
            if prev is not None and np.abs(img.astype(int) - prev).mean() < 0.5:
                continue
            prev = img
            try:
                self.process(img)
            except Exception as e:
                log(f"reading failed on {f}: {e!r}")
        self.idle("recording finished")


def moving(a, b) -> float:
    """The share of the screen that changed between two screenshots. The hand's glow and the supply's shimmer move
    well under 1 %; a card flying across the board moves several %."""
    if a.shape != b.shape:
        return 1.0
    return float((np.abs(a[::4, ::4].astype(np.int16) - b[::4, ::4]).max(axis=2) > 40).mean())


# ── the window ───────────────────────────────────────────────────────────────────────────────────────

def serve(adv: Advisor, port: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self.send(open(PAGE, "rb").read(), "text/html; charset=utf-8")
            elif path == "/state":
                with adv.lock:
                    self.send(json.dumps(adv.state).encode(), "application/json")
            elif path == "/frame.jpg" and adv.frame:
                with adv.lock:
                    self.send(adv.frame, "image/jpeg")
            else:
                self.send_error(404)

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def x11_scale(d) -> float:
    """The desktop's scale factor as X clients see it (Xft.dpi / 96)."""
    from Xlib import Xatom
    prop = d.screen().root.get_full_property(d.intern_atom("RESOURCE_MANAGER"), Xatom.STRING)
    m = re.search(r"Xft\.dpi:\s*([\d.]+)", prop.value.decode() if prop else "")
    return float(m.group(1)) / 96 if m else 1.0


def open_window(url: str):
    """A browser app window (no tabs or address bar) beside the game window, or on the right half of the screen."""
    x = y = 0
    w, h = 560, 1000
    try:
        from Xlib import display
        from grab import find_window
        d = display.Display()
        root = d.screen()
        sw, sh = root.width_in_pixels, root.height_in_pixels
        win = find_window(d)
        if win is not None:
            g = win.get_geometry()
            pos = win.translate_coords(root.root, 0, 0)
            gx, gy = -pos.x, -pos.y
            if sw - (gx + g.width) >= 360:        # room on the right of the game
                x, y, w, h = gx + g.width, gy, sw - (gx + g.width), g.height
            else:
                x, y, w, h = sw // 2, 0, sw // 2, sh
        else:
            x, y, w, h = sw // 2, 0, sw // 2, sh
        scale = x11_scale(d)                      # the browser takes sizes in scaled pixels
        x, y, w, h = (int(round(v / scale)) for v in (x, y, w, h))
    except Exception:
        pass
    for exe in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        path = shutil.which(exe)
        if path:
            profile = os.path.expanduser("~/snap/chromium/common/dominion-advisor" if path.startswith("/snap/")
                                         else "~/.cache/dominion-advisor")
            os.makedirs(profile, exist_ok=True)
            return subprocess.Popen([path, f"--app={url}", f"--user-data-dir={profile}", "--ozone-platform=x11",
                                     f"--window-position={x},{y}", f"--window-size={w},{h}", "--no-first-run",
                                     "--no-default-browser-check"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if shutil.which("firefox"):
        return subprocess.Popen(["firefox", "--new-window", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"open {url} in a browser")
    return None


def install():
    """'Dominion Advisor' in the app launcher (~/.local/share/applications)."""
    path = os.path.expanduser("~/.local/share/applications/dominion-advisor.desktop")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    py = os.path.join(ROOT, ".venv", "bin", "python")
    with open(path, "w") as f:
        f.write("[Desktop Entry]\nType=Application\nName=Dominion Advisor\n"
                "Comment=Win chance and suggestions while you play Dominion on Steam\n"
                f"Exec={py} {os.path.abspath(__file__)}\nPath={ROOT}\nIcon=applications-games\n"
                "Terminal=false\nCategories=Game;\n")
    print(f"wrote {path}: search for 'Dominion Advisor' in the app launcher")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-window", action="store_true", help="don't open the window, only serve the page")
    ap.add_argument("--install", action="store_true", help="add 'Dominion Advisor' to the app launcher and exit")
    ap.add_argument("--frames", help="replay a recording folder (frames/ + events.jsonl) instead of the live game")
    ap.add_argument("--speed", type=float, default=4, help="replay speed (0 = as fast as it reads)")
    args = ap.parse_args()
    if args.install:
        return install()
    adv = Advisor(Judge())
    log(f"advisor network: {adv.judge.label} ({adv.judge.file})")
    try:
        serve(adv, args.port)
    except OSError:
        url = f"http://127.0.0.1:{args.port}/"
        print(f"the advisor is already running: {url}")
        if not args.no_window:
            open_window(url)
        return
    url = f"http://127.0.0.1:{args.port}/"
    log(f"page: {url}")
    vision._ocr = vision.RapidOCR(intra_op_num_threads=OCR_THREADS, inter_op_num_threads=1)   # loaded before the first screenshot
    if not args.no_window:
        open_window(url)
    try:
        if args.frames:
            adv.replay(args.frames, args.speed)
            while True:
                time.sleep(60)
        else:
            adv.watch()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
