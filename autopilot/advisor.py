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
Pile counts are only taken from piles that look like themselves (nothing covering them); a changed count must be
confirmed by a second, independent read, and a jump of more than two cards by 3 readings. At the start of each of my turns the
cards on screen (draw + discard + hand) are checked against my counted cards.
The game log (page + advisor_logs/<save>.log): my hand, plays, draws, discards, buys/gains (logged once the opponent's
bar has had time to claim the card), trashes, the opponent's gains and every suggestion. The tracker is kept in
advisor_logs/<save>.json, so a restarted advisor carries on with the game. A still screen that was already read is
not read again, and after 5 still seconds the advisor looks once a second.
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
from vision import BASE_PILES, REF_H, REF_W, Vision, match_name, ocr_engine, parse_added  # noqa: E402

PAGE = os.path.join(HERE, "advisor.html")
LOG_DIR = os.path.join(HERE, "advisor_logs")         # one game log per game (<date>_<save>.log, gitignored)
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
R_POPUP = (40, 40, 887, 118)                        # a card's window: Sentry's zone labels and the card titles
PILE_SAME = 0.8                                     # a pile still looks like its picture (ncc) above this
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
POPUP_CARDS = ("Sentry", "Vassal", "Library", "Harbinger", "Bandit")
SENTRY_LABEL = {eg.SENTRY_TRASH: "Trash", eg.SENTRY_DISCARD: "Discard", eg.SENTRY_KEEP: "Keep"}


def make_reader():
    """RapidOCR for the advisor. Its default text detection scales every crop UP until the short side is 736 px (a
    470x40 bar becomes ~2900x736: 9 s a read here); 'max' only scales down: the same text in 0.3-2 s."""
    return vision.RapidOCR(intra_op_num_threads=OCR_THREADS, inter_op_num_threads=1,
                           det_limit_type="max", det_limit_side_len=960)


_hand_ocr = None


def hand_reader():
    """The hand's reader: crops scaled up to a short side of 480 px. Without that the banners of a 7-card hand that
    overlap are missed; RapidOCR's own 736 finds the same cards 1.5x slower."""
    global _hand_ocr
    if _hand_ocr is None:
        _hand_ocr = vision.RapidOCR(intra_op_num_threads=OCR_THREADS, inter_op_num_threads=1,
                                    det_limit_type="min", det_limit_side_len=480)
    return _hand_ocr


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


def cards_text(cards: Counter) -> str:
    """Counter({'Copper': 3, 'Silver': 1}) -> '3 Copper, Silver' (the game log's style)."""
    return ", ".join(f"{k} {c}" if k > 1 else c for c, k in sorted(cards.items(), key=lambda x: (-CARDS[ID[x[0]]].cost, x[0]))
                     if k > 0) or "nothing"


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
        self.opp_moved = False                    # the opponent's bar changed since my last phase prompt
        self.trashed = Counter()
        self.in_play = Counter()                  # my cards in play this turn (played from the hand)
        self.last_trashed = None                  # the card I trashed last (Remodel / Mine: what may I gain?)
        self.opp_trashed = Counter()              # the opponent's trashes we know of (from their score)
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
        self.events = []                          # the game log: [turn, "who", text]
        self.log_file = None
        self.stable_hand = None                   # the hand once two readings in a row agree
        self._last_hand = None
        self.logged_mine = Counter()              # my gains already in the log
        self.pending = []                         # pile drops waiting to be logged: [card, turn, verb, readings]
        self.gains_hist = []                      # my logged gains in order: [turn, card]
        self.logged_vp = [None, None]
        self.deck_check = None                    # cards on screen (draw + discard + hand + play) vs counted

    # -- kept on disk, so a restarted advisor carries on with the game --------------------------------
    KEEP = ("supply", "opp_gained", "trashed", "first", "turns", "started", "seen", "gained_at_shuffle", "wp",
            "events", "logged_mine", "last_kind", "pending", "banner", "opp_moved", "gains_hist", "opp_trashed", "last_trashed", "in_play")

    def dump(self, path):
        data = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in vars(self).items() if k in self.KEEP}
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump({"kingdom": self.kingdom, **data}, f)
            os.replace(tmp, path)
        except OSError:
            pass

    def load(self, path) -> bool:
        try:
            data = json.load(open(path))
        except (OSError, ValueError):
            return False
        if data.get("kingdom") != self.kingdom:
            return False
        for k in self.KEEP:
            if k in data:
                v = data[k]
                setattr(self, k, Counter(v) if isinstance(getattr(self, k), Counter) or k == "banner" and v is not None
                        else v)
        return True

    # -- the game log -----------------------------------------------------------------------------
    def note(self, text, who="", turn=None):
        """One line of the game log (the page shows the newest; the file keeps them all)."""
        turn = turn or self.turns + 1
        self.events.append([turn, who, text])
        log(f"[turn {turn}] {who + ': ' if who else ''}{text}")
        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    f.write(f"{time.strftime('%H:%M:%S')}  turn {turn:2d}  {who + ': ' if who else ''}{text}\n")
            except OSError:
                pass

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
        return +(Counter({"Copper": 7, "Estate": 3}) + self.opp_gained - self.opp_trashed)

    def draw_pile(self) -> Counter:
        mine = self.mine()
        gains = self.my_gained() - self.gained_at_shuffle
        cands = +(mine - self.hand - self.seen - gains)
        if not cands:
            cands = +(mine - self.hand)
        n = self.draw_n if self.draw_n is not None else sum(cands.values())
        return spread(cands, n)

    # -- readings --------------------------------------------------------------------------------
    def read_counts(self, counts: dict[str, int], empty=()):
        """Pile counts from one reading (Reader.counts only passes a changed count that a second, independent read
        confirms, on a pile that still looks like itself). A drop of one or two cards counts after 2 equal readings
        in a row. Anything else - a bigger drop (the advisor started mid-game, or missed a turn) or a pile that
        GREW (an earlier misread: a "7" read as "0" while the bought card flew off it) - only after 3 readings and
        10 seconds of the same number."""
        moved = {ID[n]: k for n, k in counts.items()
                 if k is not None and self.start[ID[n]] >= 0 and k != self.supply[ID[n]]}
        for c in list(self._maybe):
            if c not in moved:
                del self._maybe[c]
        now = time.time()
        for c, n in moved.items():
            if n > self.start[c]:
                continue
            seen = self._maybe.get(c)
            if seen and seen[0] == n:
                k, turn, verb, since = seen[1] + 1, seen[2], seen[3], seen[4]
            else:                                 # first seen now: the turn and phase it happened in
                k, turn, verb, since = 1, self.turns + 1, "buy" if self.last_kind == "buy" else "gain", now
            small = 0 < self.supply[c] - n <= 2
            crossed = n == 0 and NAME[c] in empty     # the X is certain: no 10 s wait for a big drop
            if k >= 2 and (small or crossed) or k >= 3 and now - since >= 10:
                del self._maybe[c]
                if n > self.supply[c]:            # a correction: an earlier count was a misread
                    self.note(f"(correction: the {NAME[c]} pile has {n}, not {self.supply[c]})")
                    self.supply[c] = n
                    self.pending = [g for g in self.pending if g[0] != NAME[c]]
                    self.logged_mine[NAME[c]] = min(self.logged_mine[NAME[c]], self.my_gained()[NAME[c]])
                    self.gains_hist = [g for g in self.gains_hist if g[1] != NAME[c]] + \
                        [[turn, NAME[c]]] * self.logged_mine[NAME[c]]
                    continue
                if not small:
                    self.note(f"(the {NAME[c]} pile went from {self.supply[c]} to {n} between readings)")
                for _ in range(self.supply[c] - n):
                    self.pending.append([NAME[c], turn, verb, 4])
                self.supply[c] = n
            else:
                self._maybe[c] = (n, k, turn, verb, since)

    def settle_gains(self):
        """A pile that shrank is my gain unless the opponent's bar names that card: log it after 4 more readings
        without the bar claiming it, under the turn it happened in."""
        free = self.my_gained() - self.logged_mine
        keep = []
        for g in self.pending:
            if free[g[0]] <= 0:
                continue                          # the opponent's (their bar named it)
            g[3] -= 1
            if g[3] > 0:
                keep.append(g)
                continue
            free[g[0]] -= 1
            self.logged_mine[g[0]] += 1
            self.gains_hist.append([g[1], g[0]])
            self.note(f"{g[2]} {g[0]}", "You", turn=g[1])
        self.pending = keep

    def read_bar(self, text: str):
        """The opponent's bar: their name at the start of their turn, then "Added: <cards>" (growing during the
        turn: "Silver" -> "Silver and Copper"), still showing during my next turn. A misread changes nothing."""
        cards = parse_added(text, self.names)
        if cards is None:
            if len(re.sub(r"[^A-Za-z]", "", text)) >= 4 and "added" not in text.lower():
                if self.banner != Counter():      # their name again (after "Added: ..."): a new turn of theirs
                    self.opp_moved = True
                self.banner = Counter()
            return
        if self.banner is None:
            self.banner = Counter()
        new = Counter(cards) - self.banner
        if new:
            self.banner |= Counter(cards)
            self.opp_gained += new
            self.note(f"gains {cards_text(new)}", "Opponent")

    def phase(self, kind):
        """kind: 'action' | 'treasure' | 'buy' for my phase prompts."""
        if not self.started:
            self.started = True
            self.first = not self.opp_gained
            self.note(f"you go {'first' if self.first else 'second'}")
        elif kind in ("action", "treasure") and (self.last_kind == "buy" or self.opp_moved and self.last_kind):
            # my next turn: I was in my buy phase, or the opponent has moved since my last phase (a buy phase
            # that was never read between two screenshots)
            self.turns += 1
            self.seen += self.turn_seen
            self.turn_seen = Counter()
            self.stable_hand = self._last_hand = None
            self.in_play = Counter()
        if kind in ("action", "treasure", "buy"):
            self.opp_moved = False
        self.last_kind = kind

    def read_hand(self, texts, why=""):
        """My hand from one reading. The log follows the hand once two readings in a row agree: cards that leave it
        were played (or discarded, while a card asks for a discard), cards that join it were drawn."""
        self.hand_texts = texts
        self.hand = Counter()
        for t in texts:
            self.hand[t.text] += t.count
        for c, k in self.hand.items():
            self.turn_seen[c] = max(self.turn_seen[c], k)
        same = self._last_hand == self.hand
        self._last_hand = Counter(self.hand)
        if not same or self.hand == self.stable_hand:
            return
        before, self.stable_hand = self.stable_hand, Counter(self.hand)
        if before is None:
            if self.hand:
                self.note(f"hand: {cards_text(self.hand)}", "You")
                if self.last_kind in ("action", "treasure"):
                    self.check_due = True             # once the pile numbers have settled (read_piles)
            return
        gone, new = before - self.hand, self.hand - before
        self.check_due = False                    # cards moved: the start-of-turn moment is gone
        if gone and self.trash_hand is None:      # trashes are logged when the trash prompt ends
            verb = "discard" if why == "discard" else "play"
            self.note(f"{verb} {cards_text(gone)}", "You")
            if verb == "play":
                self.in_play += gone              # the network sees what I have in play this turn
        if new:
            self.note(f"draw {cards_text(new)}", "You")

    def sentry_window(self, pop):
        """Sentry trashes from the deck, not the hand: the cards in its Trash zone when the window closes."""
        if pop is not None:
            self.sentry_last, self.sentry_gone = pop, 0
            return
        self.sentry_gone = getattr(self, "sentry_gone", 0) + 1
        if self.sentry_gone < 2:                  # closed for two readings in a row (not one misread)
            return
        last, self.sentry_last = getattr(self, "sentry_last", None), None
        if last:
            gone = Counter(c for c, z in last["cards"] if z == "trash")
            if gone:
                self.trashed += gone
                self.note(f"trash {cards_text(gone)} (Sentry)", "You")
            put = Counter(c for c, z in last["cards"] if z == "discard")
            if put:
                self.note(f"discard {cards_text(put)} (Sentry)", "You")

    def reconcile(self):
        """At the start of my turn the screen's own numbers win over the counting: (1) my score below what my counted
        Victory cards give: Estates were trashed (a Remodel, a Chapel the reading missed); (2) fewer cards on screen
        (draw + discard + hand) than counted: my newest gains were the opponent's (pile drops the advisor saw only
        after their bar had moved on). Both only with stable readings, and Gardens off the table (it moves the score)."""
        mine = self.mine()
        if self.vp[0] is not None and getattr(self, "vp_same", False) and not mine["Gardens"]:
            extra = rules.score(ids(mine)) - self.vp[0]
            est = min(extra, mine["Estate"]) if extra > 0 else 0
            if est:
                self.trashed["Estate"] += est
                self.note(f"trash {cards_text(Counter({'Estate': est}))} (your score on screen says so)", "You")
        theirs = self.theirs()
        if self.vp[1] is not None and getattr(self, "vp_opp_same", False) and not theirs["Gardens"]:
            extra = rules.score(ids(theirs)) - self.vp[1]
            est = min(extra, theirs["Estate"]) if extra > 0 else 0
            if est:                               # their Remodel / Chapel: the trash is not on their bar
                self.opp_trashed["Estate"] += est
                self.note(f"trashes {cards_text(Counter({'Estate': est}))} (their score on screen says so)", "Opponent")
        shown = self.draw_n + self.discard_n + sum(self.hand.values())
        over = sum(self.mine().values()) - shown
        if over <= 0 or over != getattr(self, "prev_over", None):   # the same surplus two turns running only
            self.prev_over = over if over > 0 else None
            return
        self.prev_over = None
        moved = Counter()
        while over > 0 and self.gains_hist:
            turn, card = self.gains_hist.pop()
            if self.hand[card] >= self.mine()[card]:
                continue                          # it is in my hand: mine for sure
            self.opp_gained[card] += 1
            self.logged_mine[card] -= 1
            moved[card] += 1
            over -= 1
        if moved:
            self.note(f"{cards_text(moved)}: the opponent's, not yours (the screen shows fewer cards)", "Opponent")

    def check_deck(self):
        """At the start of my turn: the cards the screen shows (draw pile + discard pile + hand) against my cards
        as counted. A difference means a gain or a trash was missed."""
        if self.draw_n is None or self.discard_n is None:
            return
        self.reconcile()
        shown, counted = self.draw_n + self.discard_n + sum(self.hand.values()), sum(self.mine().values())
        off = shown - counted
        persistent = bool(off) and off == getattr(self, "prev_off", 0)   # the same gap two turns running
        self.deck_check = [shown, counted] if persistent or not off else self.deck_check
        if persistent:
            self.note(f"(the screen shows {shown} cards, counted {counted}: a gain or trash was missed)")
        self.prev_off = off

    def read_piles(self, draw_n, discard_n):
        settled = (draw_n, discard_n) == getattr(self, "_piles_prev", None) and None not in (draw_n, discard_n)
        self._piles_prev = (draw_n, discard_n)
        if getattr(self, "check_due", False) and settled and self.hand == self.stable_hand:
            self.check_due = False
            self._read_piles(draw_n, discard_n)
            self.check_deck()
            return
        self._read_piles(draw_n, discard_n)

    def _read_piles(self, draw_n, discard_n):
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
                self.last_trashed = max(gone, key=lambda c: CARDS[ID[c]].cost)   # Remodel / Mine gain from its cost
                self.trashed += gone
                self.note(f"trash {cards_text(gone)}", "You")
            self.trash_hand = None

    # -- a position for the network ----------------------------------------------------------
    def position(self, coins=0, buys=1, actions=1, hand=None):
        me = 0 if self.first in (True, None) else 1
        hand = Counter(self.hand) if hand is None else hand
        mine, theirs = self.mine(), self.theirs()
        deck = self.draw_pile()
        play = self.in_play & +(mine - hand - deck)
        rest = +(mine - hand - deck - play)
        p = _Player(deck=ids(deck), hand=ids(hand), discard=ids(rest), play=ids(play), turns=self.turns)
        o = _Player(discard=ids(theirs), turns=self.turns + (0 if me == 0 else 1))
        players = [p, o] if me == 0 else [o, p]
        return _Game(players, list(self.supply), me, coins, buys, actions), me


# ── the screen ──────────────────────────────────────────────────────────────────────────────────────

def _read_number(crop, fx):
    big = cv2.resize(crop, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    res, _ = ocr_engine()(big, use_det=False, use_cls=False)
    d = re.sub(r"\D", "", res[0][0]) if res else ""
    return int(d) if d and len(d) <= 2 else None


class Reader:
    """vision.py's readers, made fast enough to follow a game: a region is read again only when its pixels
    changed, and numbers and the prompt line at fixed places are read without the (slow) text detection."""

    def __init__(self, shape):
        self.shape = shape[:2]
        self.v = Vision(shape[1], shape[0])
        self._memo = {}
        self._pile_ref, self._pile_runs = {}, {}
        self.deferred = False                     # a changed region was left for the next reading
        self.empty = set()                        # piles showing the empty X in the last reading
        self.slots_at = 0.0                       # when the kingdom piles were last looked for
        read = self.v.read

        def memo_read(img, region, cls=True):
            x0, y0 = self.v.scale(*region[:2])
            x1, y1 = self.v.scale(*region[2:])
            hit = self._memo.get((region, cls))
            if region == vision.R_HAND:           # the hand's glow pulses: compare a small blurred thumbnail
                look = cv2.resize(img[y0:y1, x0:x1], (48, 16), interpolation=cv2.INTER_AREA).astype(np.int16)
                same = hit is not None and np.abs(hit[0] - look).mean() < 3
            else:                                 # text: one changed word ("Smithy" -> "Witch") must count
                look = img[y0:y1, x0:x1].astype(np.int16)
                same = hit is not None and hit[0].shape == look.shape \
                    and int((np.abs(hit[0] - look).max(axis=2) > 60).sum()) <= max(6, look.shape[0] * look.shape[1] // 400)
            if same:
                return hit[1]
            if hit is not None and time.time() - hit[2] < 1.0:   # read under a second ago: next time
                self.deferred = True
                return hit[1]
            if region == vision.R_HAND:           # overlapping banners in a big hand need the crop scaled up
                fast, vision._ocr = vision._ocr, hand_reader()
                try:
                    out = read(img, region, cls)
                finally:
                    vision._ocr = fast
            else:
                out = read(img, region, cls)
            self._memo[(region, cls)] = (look, out, time.time())
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
        n = _read_number(crop, 3)
        self._memo[(x, y)] = (crop.astype(np.int16), n)
        return n

    def number_again(self, img, x, y, w=15, h=12):
        """The same number read a second, independent way (a wider crop, magnified 4x): a changed pile count must
        agree with it. A memoised read repeated over identical frames is not a second opinion."""
        x0, y0 = self.v.scale(x - w - 3, y - h - 2)
        x1, y1 = self.v.scale(x + w + 1, y + h + 2)
        crop = img[max(0, y0):y1, max(0, x0):x1]
        return _read_number(crop, 4) if crop.size else None

    def confirms(self, img, x, y, n):
        """Does an independent read agree that the number at (x, y) is n? (two tries: wider crop 4x, tight crop 2x)"""
        if self.number_again(img, x, y) == n:
            return True
        (x0, y0), (x1, y1) = self.v.scale(x - 15, y - 12), self.v.scale(x + 13, y + 12)
        crop = img[max(0, y0):y1, max(0, x0):x1]
        return bool(crop.size) and _read_number(crop, 2) == n

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

    def pile_size(self, img, x, y):
        """My draw or discard pile's count: the number on it, or 0 when the spot is the empty (flat brown) mat."""
        n = self.number(img, x, y)
        if n is not None:
            return n
        (x0, y0), (x1, y1) = self.v.scale(x - 16, y - 16), self.v.scale(x + 16, y + 16)
        spot = cv2.cvtColor(img[max(0, y0):y1, max(0, x0):x1], cv2.COLOR_BGR2GRAY)
        return 0 if spot.size and float(spot.std()) < 14 else None

    def hud(self, img):
        return tuple(self.number(img, x, y) for x, y in HUD_AT)

    def counts(self, img, slots, supply):
        """Pile counts, only for piles that are on screen: a pile's picture is taken the first time its number
        reads as expected (or the same number 5 times running), and later a pile counts only while it still looks
        like that picture (a card shown big, a popup or the hover zoom hides it)."""
        boxes = {n: (x - 33, y - 30, x + 33, y + 8) for n, (x, y) in BASE_PILES.items()}
        boxes.update({n: (x0, y0, x1, y0 + 75) for n, (x0, y0, x1, y1) in slots.items()})
        at = dict(COUNT_AT)
        at.update({n: (x1 - 8, y1 - 11) for n, (x0, y0, x1, y1) in slots.items()})
        out = {}
        self.empty = set()
        for name, box in boxes.items():
            if self.crossed(img, box):            # an empty pile: its picture becomes a dark card with an orange X
                self.empty.add(name)
                out[name] = 0
                continue
            look = self.look(img, box)
            ref = self._pile_ref.get(name)
            if ref is not None and ncc(ref, look) < PILE_SAME:
                continue
            n = self.number(img, *at[name])
            if ref is None:
                runs = self._pile_runs.get(name, (None, 0))
                runs = (n, runs[1] + 1 if runs[0] == n else 1)
                self._pile_runs[name] = runs
                if n is None or (n != supply[ID[name]] and (runs[1] < 2 or not self.confirms(img, *at[name], n))):
                    continue
                self._pile_ref[name] = look
            if n is not None and n != supply[ID[name]] and not self.confirms(img, *at[name], n):
                continue                          # a changed count needs a second opinion
            out[name] = n
        return out

    def popup(self, img, names):
        """A card's window above the board: Sentry's (zones Discard / Deck / Trash, the cards it looks at), or the
        one card or row of cards that Vassal, Library, Harbinger and Bandit show. {'zones': {...}, 'cards':
        [(name, zone)] left to right} or None."""
        upper = {n.upper().replace(" ", ""): n for n in names}
        texts = sorted(self.v.read(img, R_POPUP, cls=False), key=lambda t: t.x)
        zones = {}
        for t in texts:
            for z in ("discard", "deck", "trash"):
                if t.text.lower().strip() == z:
                    zones[z] = t.x
        cards = []
        for t in texts:
            name = match_name(t.text.upper().replace(" ", ""), upper)
            rx = self.v.unscale(t.x, t.y)[0]
            if not 185 <= rx <= 870:              # the supply's own labels left (and right) of the window
                continue
            if name and t.text.lower().strip() not in zones:
                zone = min(zones, key=lambda z: abs(zones[z] - t.x)) if len(zones) == 3 else None
                cards.append((name, zone, abs(rx - REF_W / 2)))
        if len(zones) != 3:                       # one card in the middle (Vassal, Library) comes first
            cards.sort(key=lambda c: c[2])
        return {"zones": zones, "cards": [c[:2] for c in cards]} if cards else None

    def crossed(self, img, box):
        """The empty pile's orange X on a dark card (measured: 12-14% orange and >50% dark when empty, at most 1%
        orange on every pile with cards)."""
        (x0, y0), (x1, y1) = self.v.scale(*box[:2]), self.v.scale(*box[2:])
        c = img[max(0, y0):y1, max(0, x0):x1].astype(np.int16)
        if c.size == 0:
            return False
        b, g, r = c[..., 0], c[..., 1], c[..., 2]
        orange = ((r > 170) & (g > 40) & (g < 130) & (b < 80)).mean()
        return orange > 0.06 and (c.mean(axis=2) < 45).mean() > 0.35

    def look(self, img, box):
        (x0, y0), (x1, y1) = self.v.scale(*box[:2]), self.v.scale(*box[2:])
        g = cv2.cvtColor(img[max(0, y0):y1, max(0, x0):x1], cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (24, 16), interpolation=cv2.INTER_AREA).astype(np.float32)

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
    q = re.sub(r"[^a-z]", "", prompt.lower())
    for key, kind in (("buyacard", "buy"), ("playtreasure", "treasure"), ("playaction", "action")):
        if 0 <= q.find(key) <= 1:                # at the start: "Vassal: Play Action or decline." is no phase
            return kind, True
    from difflib import SequenceMatcher            # OCR slips: "lay Actoncards", "Play Treasure card"
    for phrase, kind in (("buyacard", "buy"), ("playtreasurecards", "treasure"), ("playactioncards", "action")):
        if len(q) <= len(phrase) + 4 and SequenceMatcher(None, q, phrase).ratio() >= 0.8:
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
        self.said = None                          # the last suggestion written to the game log
        self.popup = None                         # a card's window on screen (Reader.popup)
        self.keep = None                          # where this game's tracker is kept (a restart carries on)
        self.lock = threading.Lock()

    def new_game(self, kingdom, why="", name=None):
        self.t = Tracker(kingdom)
        self.reader = None                        # the pile pictures belong to the old game's board
        os.makedirs(LOG_DIR, exist_ok=True)
        self.keep = os.path.join(LOG_DIR, f"{name}.json") if name else None
        if self.keep and self.t.load(self.keep):  # the advisor was restarted during this game
            self.t.log_file = os.path.join(LOG_DIR, f"{name}.log")
            log(f"carrying on with {name}: turn {self.t.turns + 1}, {sum(self.t.mine().values())} cards")
            return
        self.t.log_file = os.path.join(LOG_DIR, f"{name or time.strftime('%Y-%m-%d_%H%M')}.log")
        if os.path.exists(self.t.log_file):
            os.replace(self.t.log_file, self.t.log_file[:-4] + time.strftime("_%H%M%S.old.log"))
        log(f"new game{why}: {', '.join(kingdom)}")
        self.t.note(f"new game{why}. Kingdom: {', '.join(kingdom)}")

    # -- one settled screenshot ---------------------------------------------------------------
    def process(self, img):
        if img.shape[1] > READ_W:                 # the reader's own size: text detection costs ~ pixels
            img = cv2.resize(img, (READ_W, round(img.shape[0] * READ_W / img.shape[1])), interpolation=cv2.INTER_AREA)
        if self.reader is None or self.reader.shape != img.shape[:2]:
            self.reader = Reader(img.shape)
        r, t = self.reader, self.t
        r.deferred = False
        if len(t.slots) < len(t.kingdom) and time.time() - r.slots_at > 5:   # hidden piles: look again in 5 s
            r.slots_at = time.time()
            t.slots.update(r.slots(img, t.kingdom))
        t.read_bar(r.v.opponent_bar(img))         # first: the opponent's name on it starts my next turn
        prompt = r.prompt(img)
        kind, sure = prompt_kind(prompt, "")
        if kind is None and (not prompt or len(prompt) > 60):   # the line is hidden: ask the buttons
            kind, sure = prompt_kind(prompt, " | ".join(b.text.lower() for b in r.v.buttons(img)))
        t.prompt = prompt
        if kind and sure:
            t.phase(kind)
        card = self.card_prompt(prompt)
        self.popup = None
        if kind is None and (card in POPUP_CARDS or card is None):   # a card's window may hide the prompt line
            self.popup = r.popup(img, t.names)
        t.sentry_window(self.popup if self.popup and len(self.popup["zones"]) == 3 else None)
        mine_now = kind is not None or card is not None or "discard" in prompt.lower() or self.popup is not None
        if mine_now:
            t.read_piles(r.pile_size(img, *DRAW_AT), r.pile_size(img, *DISCARD_AT))
            t.read_hand(r.v.hand(img, t.names), "discard" if "discard" in prompt.lower() else "")
            a, c, b = r.hud(img)
            t.actions = a if a is not None and a <= 9 else None
            t.coins = c if c is not None and c <= 40 else None
            t.buys = b if b is not None and 1 <= b <= 6 else None
        t.trash_prompt("trash" in prompt.lower() and kind is None)
        vp_me, vp_opp = r.number(img, *MY_VP_AT), r.number(img, *OPP_VP_AT)
        t.vp_same = vp_me is not None and vp_me == t.vp[0]   # two readings agree: the score can correct the count
        if vp_me is not None:
            t.vp[0] = vp_me
        t.vp_opp_same = vp_opp is not None and vp_opp == t.vp[1]
        if vp_opp is not None:
            t.vp[1] = vp_opp
        dealing = kind in ("action", "treasure") and not t.hand   # cards still flying in
        asked = t.started or self.popup is not None or card is not None   # a card's question needs no phase seen
        advice = self.advise(kind, prompt) if asked and not dealing else None
        self.publish(img, kind, prompt, advice)   # the suggestion first: the pile counts take up to 2 s more
        before = (tuple(t.supply), sum(t.opp_gained.values()))
        t.read_counts(r.counts(img, t.slots, t.supply), r.empty)
        t.settle_gains()
        if t._maybe or t.pending or any(k == 1 for _, k in r._pile_runs.values() if _ is not None):
            r.deferred = True                     # a count waits for a second reading: read this screen again
        if self.keep:
            t.dump(self.keep)
        if (tuple(t.supply), sum(t.opp_gained.values())) != before:   # a gain changed the deck: ask again
            advice = self.advise(kind, prompt) if asked and not dealing else None
            self.publish(img, kind, prompt, advice)

    def card_prompt(self, prompt):
        """The card a prompt line belongs to ("Remodel: Gain a card ..."), also when OCR drops the colon or the
        spaces ("RemodelGainacard...")."""
        names = {n.upper(): n for n in self.t.names}
        m = re.match(r"\s*([A-Za-z ]{3,20})\s*[:;.]", prompt)
        if m and match_name(m.group(1), names):
            return match_name(m.group(1), names)
        head = re.sub(r"[^A-Z]", "", prompt.upper())[:14]
        for n in sorted(names, key=len, reverse=True):       # the line starts with a card name
            k = n.replace(" ", "")
            if len(head) >= len(k) and _similar(head[:len(k)], k) >= 0.85:
                return names[n]
        return None

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
            a = self.single(eg.PLAY_ACTION, opts, True, hand, vp, "Play an action", coins, buys, actions)
            best = a["plan"][0] if a["plan"] else None
            if best:
                rest = Counter(hand)
                rest[best] -= 1
                a["then"] = self.followup(best, +rest, vp)
            return a
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
        if card in ("Remodel", "Mine") and "gain" in p:
            m = re.search(r"up\s*to\s*\$?\s*(\d+)", p)
            up = int(m.group(1)) if m else (CARDS[ID[t.last_trashed]].cost + (2 if card == "Remodel" else 3)
                                             if t.last_trashed else None)
            if up is not None:
                if card == "Mine":
                    opts = [c for c in gainable(up) if CARDS[c].is_treasure]
                    return self.single(eg.MINE_GAIN, opts, False, hand, vp, f"Mine: gain a treasure up to ${up}",
                                       subject=ID[t.last_trashed] if t.last_trashed else -1)
                return self.single(eg.REMODEL_GAIN, gainable(up), False, hand, vp, f"Remodel: gain a card up to ${up}",
                                   subject=ID[t.last_trashed] if t.last_trashed else -1)
        if card == "Remodel" and "trash" in p:
            a = self.single(eg.REMODEL_TRASH, hand_opts, False, hand, vp, "Remodel: trash a card")
            if a["plan"]:
                a["then"] = self.upgrade("Remodel", hand, vp, trash=a["plan"][0])
            return a
        if card == "Mine" and "trash" in p:
            opts = sorted({ID[c] for c in hand if CARDS[ID[c]].is_treasure})
            a = self.single(eg.MINE_TRASH, opts, True, hand, vp, "Mine: upgrade a treasure")
            if a["plan"]:
                a["then"] = self.upgrade("Mine", hand, vp, trash=a["plan"][0])
            return a
        if card == "Moneylender" and "Copper" in hand:
            return self.single(eg.MONEYLENDER_TRASH, [ID["Copper"]], True, hand, vp, "Moneylender: trash a Copper?")
        if card == "Throne Room":
            opts = sorted({ID[c] for c in hand if CARDS[ID[c]].is_action})
            a = self.single(eg.THRONE, opts, True, hand, vp, "Throne Room: play which twice?")
            if a["plan"]:
                rest = Counter(hand)
                rest[a["plan"][0]] -= 1
                a["then"] = self.followup(a["plan"][0], +rest, vp, twice=True)
            return a
        if card == "Bureaucrat" or ("victory" in p and "deck" in p):
            opts = sorted({ID[c] for c in hand if CARDS[ID[c]].is_victory})
            return self.single(eg.BUREAUCRAT_TOPDECK, opts, False, hand, vp, "Bureaucrat: put a Victory card on top")
        pop = self.popup
        if pop and len(pop["zones"]) == 3 or card == "Sentry" and pop:
            return self.sentry([c for c, _ in pop["cards"]][::-1], hand, vp)    # shown bottom..top: top first
        if pop and card in (None, "Vassal", "Library", "Harbinger", "Bandit"):
            shown = [c for c, _ in pop["cards"]]
            if card and len(shown) > 1:
                shown = [c for c in shown if c != card] or shown   # the window's title is the card itself
            if card == "Harbinger" or (card is None and len(shown) > 1 and "Harbinger" in t.turn_seen):
                return self.single(eg.HARBINGER_TOPDECK, sorted({ID[c] for c in shown}), True, hand, vp,
                                   "Harbinger: put a card from your discard pile on your deck")
            if card == "Bandit" or (card is None and all(CARDS[ID[c]].is_treasure for c in shown) and len(shown) == 2
                                    and "Bandit" not in t.turn_seen):
                return self.single(eg.BANDIT_TRASH, sorted({ID[c] for c in shown}), False, hand, vp,
                                   "Bandit: which of your treasures to lose")
            if card == "Vassal" or (card is None and "Vassal" in t.turn_seen and CARDS[ID[shown[0]]].is_action):
                a = self.yes_no(eg.VASSAL_PLAY, ID[shown[0]], hand, vp, f"Vassal: play the {shown[0]}?",
                                f"Play {shown[0]}", "Don't")
                if a["top"] and a["top"][0][0] != "Don't":
                    a["then"] = self.followup(shown[0], hand, vp)
                return a
            if card == "Library" or (card is None and "Library" in t.turn_seen and CARDS[ID[shown[0]]].is_action):
                return self.yes_no(eg.LIBRARY_SKIP, ID[shown[0]], hand, vp, f"Library drew a {shown[0]}",
                                   "Set it aside", "Keep it")
        if "discard" in p:
            m = re.search(r"(\d+)", p)
            n = int(m.group(1)) if m else max(0, sum(hand.values()) - 3)
            k = eg.POACHER_DISCARD if card == "Poacher" else eg.MILITIA_DISCARD
            return self.sequence(k, hand, n, False, vp, f"Discard {n}", exact=True)
        return self.single(eg.BUY, [], False, hand, vp, None)     # no question I know: just the win chance

    def _ask(self, kind, opts, may_pass, hand, vp, coins=0, buys=1, actions=1, subject=-1):
        g, me = self.t.position(coins, buys, actions, hand)
        return self.judge.ask(g, me, kind, opts, may_pass, subject, vp)

    def followup(self, card, hand, vp, twice=False, depth=0):
        """What to do with `card` once it is played, from the hand left after it (`hand`): 'Throne Room' -> 'play
        Smithy twice', 'Chapel' -> 'trash Estate, 2 Copper', 'Workshop' -> 'gain Silver'. None for cards whose choice
        waits on cards not seen yet (draws, Vassal, Sentry, Library...): the next suggestion comes once they show.
        `twice` (a Throne Room's pick): the choice made twice, from the same hand (a good enough guess)."""
        t, hand = self.t, Counter(hand)
        gainable = lambda n: [c for c in range(NUM_CARDS) if t.supply[c] > 0 and CARDS[c].cost <= n]  # noqa: E731
        x2 = " (twice)" if twice else ""
        if card == "Throne Room" and depth == 0:
            opts = sorted({ID[c] for c in hand if CARDS[ID[c]].is_action})
            if not opts:
                return "nothing to play twice"
            _, probs = self._ask(eg.THRONE, opts, True, hand, vp)
            if probs[0][0] < 0:
                return None
            pick = NAME[probs[0][0]]
            hand[pick] -= 1
            more = self.followup(pick, +hand, vp, depth=1)
            return f"play {pick} twice" + (f" ({more})" if more else "")
        if card in ("Remodel", "Mine"):
            up = self.upgrade(card, hand, vp)
            return up + x2 if up else None
        if card in ("Chapel", "Cellar"):
            kind = eg.CHAPEL_TRASH if card == "Chapel" else eg.CELLAR_DISCARD
            plan = self.sequence(kind, hand, 4 if card == "Chapel" else 99, True, vp, "")["plan"]
            verb = "trash" if card == "Chapel" else "discard"
            return f"{verb} {cards_text(Counter(plan))}" if plan else f"{verb} nothing"
        if card == "Workshop":
            _, probs = self._ask(eg.WORKSHOP_GAIN, gainable(4), False, hand, vp)
            return f"gain {NAME[probs[0][0]]}{x2}" if probs else None
        if card == "Artisan":
            _, probs = self._ask(eg.ARTISAN_GAIN, gainable(5), False, hand, vp)
            if not probs:
                return None
            gain = NAME[probs[0][0]]
            hand[gain] += 1
            _, top = self._ask(eg.ARTISAN_TOPDECK, sorted({ID[c] for c in hand}), False, hand, vp)
            return f"gain {gain}, put {NAME[top[0][0]]} on your deck{x2}" if top else f"gain {gain}{x2}"
        if card == "Moneylender" and "Copper" in hand:
            _, probs = self._ask(eg.MONEYLENDER_TRASH, [ID["Copper"]], True, hand, vp)
            return ("trash a Copper" if probs[0][0] >= 0 else "keep your Copper") + x2
        if card == "Bureaucrat":
            return "gain a Silver onto your deck" + x2
        return None

    def upgrade(self, card, hand, vp, trash=None):
        """Remodel / Mine in one go: which card to trash (the network's pick, or `trash`) and what to gain for it.
        'trash Estate, gain Silver', or None (Mine with no treasure, or the network would not trash)."""
        t = self.t
        hand = Counter(hand)
        if trash is None:
            if card == "Remodel":
                opts, kind, may_pass = sorted({ID[c] for c in hand}), eg.REMODEL_TRASH, False
            else:
                opts, kind, may_pass = sorted({ID[c] for c in hand if CARDS[ID[c]].is_treasure}), eg.MINE_TRASH, True
            if not opts:
                return None
            _, probs = self._ask(kind, opts, may_pass, hand, vp)
            if probs[0][0] < 0:
                return None
            trash = NAME[probs[0][0]]
        hand[trash] -= 1
        up = CARDS[ID[trash]].cost + (2 if card == "Remodel" else 3)
        opts = [c for c in range(NUM_CARDS) if t.supply[c] > 0 and CARDS[c].cost <= up
                and (card == "Remodel" or CARDS[c].is_treasure)]
        if not opts:
            return f"trash {trash} (nothing to gain)"
        _, probs = self._ask(eg.REMODEL_GAIN if card == "Remodel" else eg.MINE_GAIN, opts, False, +hand, vp,
                             subject=ID[trash])
        return f"trash {trash}, gain {NAME[probs[0][0]]}"

    def steady_win(self):
        """The headline win chance: one number per turn, taken at the start of my turn (the dealt hand, no coins
        yet, always the same question) and held until my next turn. Within a turn the network's number moves with
        every card played (median 14, up to 26 points on a recorded game); that live number stays with each
        suggestion as "this move"."""
        t = self.t
        turn = t.turns + 1
        held = getattr(self, "_steady", (None, None))
        if held[0] == turn and held[1] is not None:
            return held[1]
        if not t.stable_hand:                     # the hand is still being dealt: keep last turn's number
            return held[1] if held[1] is not None else t.win
        vp = t.vp if None not in t.vp else None
        w, _ = self._ask(eg.PLAY_ACTION, [], True, Counter(t.stable_hand), vp, 0, 1, 1)
        self._steady = (turn, w)
        return w

    def yes_no(self, kind, c, hand, vp, question, yes, no):
        """A question about one card (Vassal: play it? Library: set it aside?)."""
        win, probs = self._ask(kind, [c], True, hand, vp, subject=c)
        top = [[yes if o >= 0 else no, p] for o, p in probs]
        return {"question": question, "win": win, "verb": "", "where": kind, "top": top, "plan": []}

    def sentry(self, cards, hand, vp):
        """Sentry: trash, discard or keep each card it shows (top card first)."""
        if not cards:
            return None
        plan, top, win = [], [], None
        for c in cards:
            w, probs = self._ask(eg.SENTRY_CARD, [eg.SENTRY_TRASH, eg.SENTRY_DISCARD, eg.SENTRY_KEEP], False, hand, vp,
                                 subject=ID[c])
            win = w if win is None else win
            plan.append(f"{SENTRY_LABEL[probs[0][0]]} {c}")
            top.append([f"{SENTRY_LABEL[probs[0][0]]} {c}", probs[0][1]])
        return {"question": f"Sentry: {', '.join(cards)} (top first)", "win": win, "verb": "", "where": eg.SENTRY_CARD,
                "top": top[:1], "plan": plan if len(plan) > 1 else []}

    def _label(self, kind, o):
        return PASS_LABEL.get(kind, "No") if o == -1 else NAME[o]

    def single(self, kind, opts, may_pass, hand, vp, question, coins=0, buys=1, actions=1, subject=-1):
        win, probs = self._ask(kind, opts, may_pass, hand, vp, coins, buys, actions, subject)
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
        steady = self.steady_win() if advice else None
        if steady is None and advice:             # no start-of-turn number yet: the live one
            steady = advice["win"]
        if advice and advice["question"] and advice["top"]:
            label, p = advice["top"][0]
            said = (t.turns, advice["question"], label)
            if said != self.said:
                self.said = said
                best = label if advice["plan"] == [] else f"{advice['verb']} {label}"
                plan = f" (plan: {', '.join(advice['plan'])})" if len(advice["plan"]) > 1 else ""
                plan += f" → then {advice['then']}" if advice.get("then") else ""
                t.note(f"{advice['question']}: {best} {round(100 * p)}%{plan} · win {round(100 * steady)}%",
                       "Advisor")
        if advice:
            t.win_raw = advice["win"]
            t.win = steady
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
            "win": t.win, "win_raw": getattr(t, "win_raw", None), "wp": t.wp, "advice": advice, "boxes": self.boxes(advice),
            "provinces": t.supply[ID["Province"]], "empty": empty,
            "vp": t.vp, "counted_vp": [rules.score(ids(mine)), rules.score(ids(theirs))],
            "mine": dict(sorted(mine.items())), "theirs": dict(sorted(theirs.items())),
            "draw": dict(sorted(t.draw_pile().items())), "hand": dict(sorted(t.hand.items())),
            "piles": {"draw": t.draw_n, "discard": t.discard_n, "total": sum(mine.values())},
            "hud": {"actions": t.actions, "coins": t.coins, "buys": t.buys}, "kingdom": t.kingdom,
            "trashed": dict(t.trashed), "time": time.time(), "aspect": img.shape[1] / img.shape[0],
            "events": t.events[-60:], "deck_check": t.deck_check, "log_file": t.log_file,
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

    def spots(self, shape):
        """The places the advisor reads, in pixels of a full-size screenshot (for nothing_new)."""
        key = (shape, tuple(sorted(self.t.slots.items())) if self.t else ())
        if getattr(self, "_spots", (None,))[0] != key:
            v = Vision(shape[1], shape[0])
            boxes = [R_PROMPT_LINE, vision.R_OPPONENT, (0, 735, 200, 782)]
            at = list(COUNT_AT.values()) + [MY_VP_AT, OPP_VP_AT, DRAW_AT, DISCARD_AT]
            if self.t:
                at += [(x1 - 8, y1 - 11) for x0, y0, x1, y1 in self.t.slots.values()]
            boxes += [(x - 15, y - 12, x + 13, y + 12) for x, y in at]
            px = []
            for b in boxes:
                (x0, y0), (x1, y1) = v.scale(*b[:2]), v.scale(*b[2:])
                px.append((max(0, x0), max(0, y0), x1, y1))
            self._spots = (key, px)
        return self._spots[1]

    # -- the loops ------------------------------------------------------------------------------
    def watch(self):
        """The live game: a screenshot every half second, read when it has settled."""
        from Xlib import display
        from grab import find_window, grab
        d = display.Display()
        win, prev = None, None
        done, quiet, pause = None, 0.0, 0.3       # the last screenshot read; how long the screen has been still
        extra = 0                                 # readings of an unchanged screen asked for by the reader
        last_read = 0.0
        while True:
            try:
                save = newest_save()
                if save and save != self.save:
                    self.save = save
                    self.new_game(read_kingdom(save), f" ({os.path.basename(save)})",
                                  os.path.splitext(os.path.basename(save))[0])
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
                waiting = self.t._maybe and time.time() - last_read > 3   # a count to confirm: look every 3 s
                if done is not None and not waiting and nothing_new(img, done, self.spots(img.shape)):
                    quiet += pause
                    with self.lock:
                        self.state["time"] = time.time()
                else:
                    quiet = 0.0
                    last_read = time.time()
                    try:
                        self.process(img)
                    except Exception as e:
                        log(f"reading failed: {e!r}")
                    again = self.reader is not None and self.reader.deferred and extra < 4
                    extra = extra + 1 if again else 0
                    done = None if again else img   # read the same screen again (at most 4 times)
            else:
                quiet = 0.0
            prev = img
            pause = 0.3 if quiet < 5 else 0.5     # a still screen (the player thinking): look less often
            time.sleep(pause)

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


def _similar(a, b) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def ncc(a, b) -> float:
    """Normalised correlation of two small grey pictures: 1 = the same picture (also dimmed or brightened)."""
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum()) / d if d > 1e-6 else 0.0


def nothing_new(a, b, spots) -> bool:
    """The screen read last time, give or take animation: no spot the advisor reads (`spots`: prompt line, counters,
    bar, scores, pile numbers, in pixels) has more than a few changed pixels, under 1% of the rest of the screen
    changed (a glowing card, the supply's shimmer), and under 3% of the hand (its pulsing glow)."""
    if a.shape != b.shape:
        return False
    for x0, y0, x1, y1 in spots:
        if int((np.abs(a[y0:y1, x0:x1].astype(np.int16) - b[y0:y1, x0:x1]).max(axis=2) > 40).sum()) > 3:
            return False
    ch = np.abs(a[::4, ::4].astype(np.int16) - b[::4, ::4]).max(axis=2) > 40
    cut = int(ch.shape[0] - (REF_H - 820) * (a.shape[1] / REF_W) / 4)
    return float(ch[:cut].mean()) < 0.01 and float(ch[cut:].mean()) < 0.03


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
                    state = {**adv.state, "page": os.path.getmtime(PAGE)}   # a newer page: the window reloads
                    self.send(json.dumps(state).encode(), "application/json")
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
            right, left = sw - (gx + g.width), gx
            if right >= 360 and right >= left:    # beside the game, on its wider free side: never on top of it
                x, y, w, h = gx + g.width, gy, right, g.height
            elif left >= 360:
                x, y, w, h = 0, gy, left - 220, g.height   # room for the dock and the window frame
            else:                                 # no room: the half the game's middle is not on
                x = 0 if gx + g.width // 2 > sw // 2 else sw // 2
                y, w, h = 0, sw // 2, sh
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
    vision._ocr = make_reader()                   # loaded before the first screenshot
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
