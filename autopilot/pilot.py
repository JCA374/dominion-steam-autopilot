"""Autopilot: the evolved GA strategy plays Steam Dominion (vs the built-in AI).

    .venv/bin/python autopilot/pilot.py            # play until the game ends
    .venv/bin/python autopilot/pilot.py --turn 4   # joining a game already on turn 4
    .venv/bin/python autopilot/pilot.py --dry      # read + decide, never click
    .venv/bin/python autopilot/pilot.py --games 3  # play, press Rematch, play again (3 games)
    .venv/bin/python autopilot/pilot.py --record recordings/game1   # also save frames + events
    .venv/bin/python autopilot/pilot.py --list-models   # released dominion-v1 models
    .venv/bin/python autopilot/pilot.py --model k4-20260614         # play this model
    .venv/bin/python autopilot/pilot.py --model auto     # the model selector picks (the default)

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
from vision import Vision, match_name, ocr_engine, parse_added  # noqa: E402

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
# What a played action adds to this turn's counters (the treasure-phase coin count needs it)
ACTION_COINS = {"Festival": 2, "Market": 1, "Vassal": 2, "Poacher": 1}
ACTION_BUYS = {"Festival": 1, "Market": 1, "Council Room": 1}
GAINS_ON_PLAY = {"Bandit": "Gold", "Bureaucrat": "Silver"}     # gained with no prompt


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
        if self.brain.policy.kind == "ga-portfolio":       # it picks a specialist for this kingdom
            self.brain.policy.member(self.brain.obs())
            ch = self.brain.policy.chosen
            log(f"portfolio plays {ch['member']}; trial vs {', '.join(ch['bots'])}: "
                + ", ".join(f"{n} {100 * v:.0f}%" for n, v in
                            sorted(ch["scores"].items(), key=lambda x: -x[1])[:4]))
        if self.brain.ranking:
            pick = self.brain.ranking[0]
            log(f"model selector: {pick['name']} ({pick['reason']}"
                + (f"; benchmark {100 * pick['score']:.0f}%" if pick["score"] is not None else "") + ")")
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
        self.blank = 0            # consecutive passes with no prompt line
        self.expected = None      # (coins, buys) counted from the hand in the treasure phase
        self.action_coins = self.action_buys = self.merchants = 0   # from my own action plays this turn
        self.pending = None       # (card, (coins, buys) before the click) of the last buy click
        self.resolving = None     # the action card I just played, while its prompts may come
        self.opp_banner = None    # the opponent "Added:" banner on screen (counted once)
        self.plan_seen = False    # I have reached my first buy (to tell who moved first)
        self.card_step = 0        # how many of its prompts were answered
        self.throned = False      # the resolving card was played by Throne Room (prompts twice)
        if REC:
            REC.event("game_start", seed=self.seed, kingdom=self.kingdom, opponents=opps,
                      model=self.brain.policy.name,
                      member=(self.brain.policy.chosen or {}).get("member")
                      if self.brain.policy.kind == "ga-portfolio" else None,
                      unknown=self.brain.unknown, turn=self.brain.turn)

    # -- persistence (so a restart mid-game keeps the deck count) ------------

    def problem(self, kind, **data):
        """Note something that went wrong (kept per game, for the stats)."""
        self.problems.append({"kind": kind, "turn": self.brain.turn, **data})
        if REC:
            REC.event("problem", kind=kind, **data)

    def _load_state(self, turn_arg):
        if os.path.exists(self.state_file) and not turn_arg:
            try:
                s = json.load(open(self.state_file))
            except (OSError, ValueError):
                log("  the game memory file is unreadable - starting from turn 1")
                return
            self.brain.turn, self.brain.provinces = s["turn"], s["provinces"]
            self.brain.owned.clear()
            self.brain.owned.update(s["owned"])
            self.brain.empty = set(s.get("empty", []))
            if "opp_owned" in s:
                self.brain.opp_owned.clear()
                self.brain.opp_owned.update(s["opp_owned"])
                self.brain.first = s.get("first", True)
            log(f"resumed: turn {self.brain.turn}, deck {dict(self.brain.owned)}")

    def _save_state(self):
        # Write to a temporary file and rename: a full disk or a kill can't leave a broken file.
        tmp = self.state_file + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump({"turn": self.brain.turn, "provinces": self.brain.provinces,
                           "owned": dict(self.brain.owned), "empty": sorted(self.brain.empty),
                           "opp_owned": dict(self.brain.opp_owned), "first": self.brain.first}, f)
            os.replace(tmp, self.state_file)
        except OSError as e:
            log(f"  couldn't save the game memory: {e}")

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

    def key(self, name):
        """Send one key press to the game window (XTEST)."""
        from Xlib import X, XK
        from Xlib.ext import xtest
        if self.args.dry:
            return
        code = self.d.keysym_to_keycode(XK.string_to_keysym(name))
        self.win.set_input_focus(X.RevertToParent, X.CurrentTime)
        xtest.fake_input(self.d, X.KeyPress, code)
        self.d.sync()
        time.sleep(0.1)
        xtest.fake_input(self.d, X.KeyRelease, code)
        self.d.sync()
        time.sleep(1.0)

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

    def pile_shows_x(self, img, card) -> bool:
        """An empty supply pile is covered by a big orange-red X."""
        pos = self.pile_pos(img, card)
        if not pos:
            return False
        r = int((22 if card in BASE else 30) * self.vision.sy)
        x, y = pos
        c = img[max(0, y - r):y + r, max(0, x - r):x + r].astype(int)
        red = (c[..., 2] > 190) & (c[..., 1] < 130) & (c[..., 0] < 90)
        return red.mean() > 0.08

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
            self.expected = None
            self.action_coins = self.action_buys = self.merchants = 0
            self._save_state()
            log(f"=== turn {self.brain.turn} ({self.brain.phase()}) "
                f"provinces {self.brain.provinces} deck {sum(self.brain.owned.values())}")
        self.last_kind = kind

    # -- prompt handlers --------------------------------------------------------

    def on_action(self, img):
        s = self.vision.screen(img, self.names)
        hand = [t.text for t in s.hand for _ in range(t.count)]
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
            self.count_play(card, hand)
            t = next(t for t in s.hand if t.text == card)
            self.click(t.x, t.y + int(55 * self.vision.sy), card)
            self.resolving = card               # its own prompts come next (on_card)
            self.card_step = 0
            self.throned = False
            if card == "Moneylender" and "Copper" in hand:
                self.brain.trashed(["Copper"])          # Steam trashes it without asking
        else:
            self.press(img, "End Actions", "Treasures", "Done", "End")

    def count_play(self, card, hand):
        """Bookkeeping for one play of an action card: coins, buys, gains without a prompt."""
        self.action_coins += ACTION_COINS.get(card, 0) + (3 if card == "Moneylender" and "Copper" in hand else 0)
        self.action_buys += ACTION_BUYS.get(card, 0)
        self.merchants += card == "Merchant"
        if card in GAINS_ON_PLAY and GAINS_ON_PLAY[card] not in self.brain.empty:
            self.brain.gained(GAINS_ON_PLAY[card])

    def on_treasure(self, img):
        # Count the coins now: on the buy prompt the counters are sometimes hidden under the
        # treasures just played, and then this is the fallback.
        s = self.vision.screen(img, self.names)
        value = {"Copper": 1, "Silver": 2, "Gold": 3}
        hand = [t.text for t in s.hand for _ in range(t.count)]
        coins = sum(value.get(c, 0) for c in hand) + self.action_coins
        if "Silver" in hand:
            coins += self.merchants                       # Merchant: +$1 on the first Silver
        self.expected = (coins, 1 + self.action_buys)
        self.press(img, "Play Treasures", "Treasures")

    def on_buy(self, img):
        a, coins, buys = self.vision.hud(img)
        read = coins is not None and buys is not None
        if coins is None or buys is None:
            self.hud_misses += 1
            if self.hud_misses < 3:
                log("  couldn't read coins and buys - reading again")
                return
            if coins is None and self.expected:
                coins, buys = self.expected[0], buys or self.expected[1]
                log(f"  counters unreadable - using the coins counted from the hand: ${coins}")
                self.problem("hud_unreadable", used=coins)
            coins, buys = coins or 0, buys or 1
        self.hud_misses = 0
        if self.plan is None and self.expected and coins is not None and coins > self.expected[0] + 2:
            # The counter can be misread (a 3 read as 13); the treasures counted from the hand
            # (plus a little for bonuses like Merchant) are the safer number.
            log(f"  counter says ${coins} but the hand counted ${self.expected[0]} - using ${self.expected[0]}")
            self.problem("coins_disagree", read=coins, counted=self.expected[0])
            coins = self.expected[0]
        if self.pending and not read:
            # Counters unreadable: the fallback numbers never change, so they can't tell whether
            # the last click bought anything. Trust it (a missed click just costs that buy).
            card, before, _ = self.pending
            self.pending = None
            coins = max(0, before[0] - (cards.cost(card) or 0))
            buys = max(0, before[1] - 1)
            if buys == 0:
                self.press(img, "End Buys", "End Turn", "End")
                self._save_state()
                return
        if self.pending and (coins, buys) == self.pending[1]:
            # Coins and buys didn't change, so the click bought nothing: either the click didn't
            # register (it happens) or the pile is empty. An empty pile shows a big red X.
            card, before, tries = self.pending
            self.brain.unbought(card)
            if self.pile_shows_x(img, card):
                log(f"  the {card} pile shows the empty X; planning again")
                self.problem("empty_pile", card=card)
                self.brain.empty.add(card)
                self.plan = None
            elif tries < 3:
                log(f"  buying {card} did nothing - clicking it once more")
                self.problem("buy_retry", card=card)
                self.click(*self.pile_pos(img, card), f"buy {card} (again)")
                self.brain.bought(card)
                self.pending = (card, before, tries + 1)
                return
            else:
                log(f"  buying {card} failed three times (pile not empty) - ending buys this turn")
                os.makedirs(UNKNOWN_DIR, exist_ok=True)
                shot = f"buyfail_{int(time.time())}.png"
                cv2.imwrite(os.path.join(UNKNOWN_DIR, shot), img)
                self.problem("buy_failed", card=card, coins_read=coins, screenshot=shot)
                self.plan = []
        self.pending = None
        if self.plan is None:
            p = self.vision.provinces_left(img)
            if p is not None and 0 < p <= 8:          # 8 in a two-player game; more is a misread
                self.brain.provinces = p
            self.plan_seen = True
            self.plan = self.brain.choose_buys(coins, buys)
            log(f"  ${coins} {buys} buy(s), provinces {self.brain.provinces} -> buy {self.plan or 'nothing'}")
            if REC:
                REC.event("decide_buy", coins=coins, buys=buys, provinces=self.brain.provinces,
                          plan=list(self.plan), turn=self.brain.turn, phase=self.brain.phase(),
                          owned=dict(self.brain.owned), empty=sorted(self.brain.empty))
        if self.plan and self.pile_shows_x(img, self.plan[0]):
            card = self.plan[0]
            log(f"  the {card} pile shows the empty X; planning again")
            self.problem("empty_pile", card=card)
            self.brain.empty.add(card)
            self.plan = self.brain.choose_buys(coins, buys)
            log(f"  -> buy {self.plan or 'nothing'}")
        if self.plan:
            card = self.plan.pop(0)
            pos = self.pile_pos(img, card)
            if pos is None:
                log(f"  can't find the {card} pile; ending buys")
                self.plan = []
            else:
                self.click(*pos, f"buy {card}")
                self.brain.bought(card)
                self.pending = (card, (coins, buys), 1)
                return
        self.press(img, "End Buys", "End Turn", "End")
        self._save_state()

    def on_trash(self, img, prompt):
        s = self.vision.screen(img, self.names)
        hand = [t.text for t in s.hand for _ in range(t.count)]
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
        hand = [t.text for t in s.hand for _ in range(t.count)]
        low = prompt.lower()
        m = (re.search(r"down\s*to\s*(\d+)", low) or re.search(r"discard\s*(\d+)", low)
             or re.search(r"(\d+)\s*card", low))      # not the "(0 of 2)" counter of repeated attacks
        n = int(m.group(1)) if m else None
        keep = n if (n is not None and "down to" in low) else (
            len(hand) - n if n is not None else 3)
        out = self.brain.discard_down_to(hand, keep)
        log(f"  discard prompt '{prompt}' hand {hand} -> discard {out}")
        for card in out:
            t = self.card_slot(s.hand, card, out)
            if t:
                self.click(t.x, t.y + int(55 * self.vision.sy), f"discard {card}")
        if not self.press_ok_if_any("discard", prefer=out):
            self.press(self.shot(), "Discard", "Done", "Confirm", "OK")

    def center_card(self, img):
        """The name on the big card shown in the middle (an attack being resolved), if any."""
        names = []
        texts = sorted(self.vision.read(img, (0, 315, 927, 370), cls=False), key=lambda t: t.x)
        for t in texts:                               # a row of played cards, the newest on the right
            name = match_name(t.text, {n.upper(): n for n in self.names})
            if name:
                names.append(name)
        for attack in ("Militia", "Bureaucrat", "Bandit", "Witch"):     # the attack being resolved
            if attack in names:
                return attack
        return names[-1] if names else None

    def my_phase_buttons(self, img, labels=None) -> bool:
        """True when my own turn's buttons are showing (then a big card is just the last one played)."""
        labels = labels if labels is not None else " | ".join(b.text.lower() for b in self.vision.buttons(img))
        return any(w in labels for w in ("end actions", "play treasures", "end buys"))

    def on_attack(self, img, prompt) -> bool:
        """An opponent's attack that needs an answer, when its prompt is hidden under the played cards.
        One card per pass: the screen is read again before the next, so repeated attacks (a Throne
        Room on a Militia) and one-card prompts work the same way."""
        card = self.center_card(img)
        if card not in ("Militia", "Bureaucrat"):
            return False
        s = self.vision.screen(img, self.names)
        hand = [t.text for t in s.hand for _ in range(t.count)]
        if card == "Militia":
            if len(hand) <= 3:
                return self.press_ok_if_any(card)
            out = self.brain.discard_down_to(hand, 3)       # all at once: a click only selects
        else:
            victory = [c for c in ("Estate", "Duchy", "Gardens", "Province") if c in hand]
            if not victory:
                return self.press_ok_if_any(card)
            out = victory[:1]
        log(f"  {card} attack (prompt hidden: '{prompt}') hand {hand} -> {out}")
        self.problem("attack_answered", card=card, prompt=prompt, chosen=out)
        for c in out:
            t = self.card_slot(s.hand, c, out)
            if t:
                self.click(t.x, t.y + int(55 * self.vision.sy), f"{card}: {c}")
        self.press_ok_if_any(card, prefer=out)
        self.unknown_count = 0
        self.last_progress = time.time()
        return True

    # -- the cards' own prompts (protocol 2) -----------------------------------------------

    def prompt_card(self, prompt):
        """The card a prompt line is about: Steam writes '<Card>: <what to do>'."""
        head = prompt.split(":")[0] if ":" in prompt else ""
        return match_name(head.upper().replace(" ", ""),
                          {n.upper().replace(" ", ""): n for n in self.names}) if head else None

    def hand_now(self, img):
        s = self.vision.screen(img, self.names)
        return s, [t.text for t in s.hand for _ in range(t.count)]

    @staticmethod
    def handler_name(card):
        return "card_" + card.lower().replace(" ", "_")

    def on_card(self, img, prompt, card) -> bool:
        """A prompt of a card I played (or that attacks me). False: not handled."""
        handler = getattr(self, self.handler_name(card), None)
        if handler is None:
            return False
        log(f"prompt: {prompt or '(hidden)'} [{card}]")
        done = handler(img, prompt)
        if done is not False:
            self.card_step += 1
            if REC:
                REC.event("card_prompt", card=card, prompt=prompt, step=self.card_step)
        return done is not False

    def refresh_empty(self, img):
        """Mark supply piles that show the empty X (gains and Poacher depend on them)."""
        for c in list(self.kingdom) + [c for c in BASE if c not in ("Copper", "Curse")]:
            if c not in self.brain.empty and self.pile_shows_x(img, c):
                log(f"  the {c} pile is empty")
                self.brain.empty.add(c)

    def gain_pile(self, card, what):
        """Click a supply pile to gain `card` (Workshop, Remodel, Artisan, Mine)."""
        img = self.shot()
        last = getattr(self, "last_gain", None)
        if last == (card, self.brain.turn, what, self.card_step - 1) and not self.throned:
            # The same gain prompt came back after clicking this pile: the click did nothing,
            # so the pile is empty (or unreachable). Undo the count and choose again.
            log(f"  gaining {card} did nothing - treating the pile as empty")
            self.problem("gain_failed", card=card)
            self.brain.empty.add(card)
            self.brain.owned[card] -= 1
            self.last_gain = None
            return False
        pos = self.pile_pos(img, card)
        if pos is None:
            log(f"  can't find the {card} pile to gain")
            self.problem("gain_pile_missing", card=card)
            return False
        self.click(*pos, f"{what}: gain {card}")
        self.brain.gained(card)
        self.last_gain = (card, self.brain.turn, what, self.card_step)
        return True

    def card_workshop(self, img, prompt):
        self.refresh_empty(img)
        s, hand = self.hand_now(img)
        g = self.brain.workshop(hand)
        log(f"  Workshop -> gain {g}")
        return self.gain_pile(g, "Workshop") if g else self.press(img, "Done", "OK")

    @staticmethod
    def says_gain(prompt) -> bool:
        """The prompt asks to gain (OCR of a half-hidden prompt: 'Goin', 'Gaim'...)."""
        return bool(re.search(r"g[oa][il1][nmr]", prompt.lower().replace(" ", "")))

    @staticmethod
    def says_trash(prompt) -> bool:
        return bool(re.search(r"tr[ao][s5c]", prompt.lower().replace(" ", "")))

    def click_hand(self, s, card, what, chosen=None):
        t = self.card_slot(s.hand, card, chosen if chosen is not None else [card])
        if t is None:
            log(f"  {card} is not in the hand I read")
            self.problem("card_not_in_hand", card=card, what=what)
            return False
        self.click(t.x, t.y + int(55 * self.vision.sy), f"{what}: {card}")
        return True

    def card_remodel(self, img, prompt):
        s, hand = self.hand_now(img)
        gain_step = self.says_gain(prompt) or (self.card_step % 2 == 1 and not self.says_trash(prompt))
        if gain_step:
            self.refresh_empty(img)
            g = getattr(self, "planned_gain", None)
            if g is None or g in self.brain.empty:
                g = self.brain.remodel(hand + [getattr(self, "remodel_trashed", "Copper")])["gain"]
            self.planned_gain = None
            log(f"  Remodel -> gain {g}")
            return self.gain_pile(g, "Remodel")
        self.refresh_empty(img)
        r = self.brain.remodel(hand)
        log(f"  Remodel hand {hand} -> trash {r['trash']}, then gain {r['gain']}")
        self.remodel_trashed = r["trash"]
        if not self.click_hand(s, r["trash"], "Remodel trash"):
            return False
        self.brain.trashed([r["trash"]])
        self.planned_gain = r["gain"]
        return True

    def popup_cards(self, img, region=(0, 55, 927, 100)):
        """Card titles in a pop-up row (Harbinger's discard pile, Sentry's look): [(name, x, y)]."""
        names = {n.upper().replace(" ", ""): n for n in self.names}
        out = []
        for t in sorted(self.vision.read(img, region, cls=False), key=lambda t: t.x):
            name = match_name(t.text.upper().replace(" ", ""), names)
            if name:
                out.append((name, t.x, t.y))
        return out

    def select_and_ok(self, s, cards, what):
        """Click each card (a multi-select from the hand), then the OK/Done button. Choosing
        nothing is the Decline button."""
        if not cards:
            return self.press(self.shot(), "Decline", "Done", "OK", "Confirm")
        for c in cards:
            self.click_hand(s, c, what, chosen=cards)
        img = self.shot()
        if self.ok_button(img) is None and not any(
                w in b.text.lower() for b in self.vision.buttons(img) for w in ("discard", "trash")):
            return True                      # a one-card choice resolves on the click (Poacher)
        if not self.press_ok_if_any(what, prefer=cards):
            self.press(self.shot(), "OK", "Done", "Confirm", "Discard", "Trash")
        return True

    def card_cellar(self, img, prompt):
        s, hand = self.hand_now(img)
        out = self.brain.cellar(hand)
        log(f"  Cellar hand {hand} -> discard {out or 'nothing'}")
        return self.select_and_ok(s, out, "Cellar discard")

    def card_poacher(self, img, prompt):
        s, hand = self.hand_now(img)
        self.refresh_empty(img)
        m = re.search(r"(\d+)", prompt)
        n = int(m.group(1)) if m else max(1, len([c for c in self.brain.empty if c in self.kingdom]))
        out = self.brain.poacher(hand, n)
        log(f"  Poacher: discard {n} -> {out}")
        return self.select_and_ok(s, out, "Poacher discard")

    def card_vassal(self, img, prompt):
        shown = self.popup_cards(img)
        if not shown:
            log("  Vassal: no card window read")
            return self.press(img, "Decline", "Done", "OK")
        card, x, y = shown[0]
        a, _, _ = self.vision.hud(img)
        play = self.brain.vassal(card, actions=a or 0)
        log(f"  Vassal discarded {card} -> {'play it' if play else 'decline'}")
        if not play:
            return self.press(img, "Decline", "Done", "OK")
        self.click(x, y + int(90 * self.vision.sy), f"Vassal: play {card}")
        self.resolving, self.card_step = card, -1        # its own prompts come next
        return True

    def drag(self, x0, y0, x1, y1, what):
        from click import drag
        log(f"  drag {what} ({x0},{y0}) -> ({x1},{y1})")
        if REC:
            REC.event("drag", x0=x0, y0=y0, x1=x1, y1=y1, what=what)
        if not self.args.dry:
            drag(self.d, self.win, x0, y0, x1, y1)
        time.sleep(self.args.delay)

    def sentry_layout(self, img):
        """Sentry's window: {zone: x} from its Discard / Deck / Trash labels, and the cards
        [(name, x, y, zone)] left to right (in the Deck zone, left = bottom, right = top)."""
        zones = {}
        for t in self.vision.read(img, (60, 45, 870, 80), cls=False):
            w = t.text.lower()
            for z in ("discard", "deck", "trash"):
                if z in w:
                    zones[z] = (t.x, t.y)
        cards = []
        if len(zones) == 3:
            for name, x, y in self.popup_cards(img, (60, 80, 870, 115)):
                zone = min(zones, key=lambda z: abs(zones[z][0] - x))
                cards.append((name, x, y, zone))
        return zones, cards

    def card_sentry(self, img, prompt):
        zones, cards = self.sentry_layout(img)
        if len(zones) < 3 or not cards:
            log(f"  Sentry window not read (zones {list(zones)}, cards {cards})")
            return False
        looked = [c[0] for c in cards]
        plan = self.brain.sentry(list(reversed(looked)))            # top card first
        log(f"  Sentry looks at {looked} (bottom..top) -> {plan}")
        down = int(90 * self.vision.sy)
        for zone in ("trash", "discard"):
            for name in plan[zone]:
                zones_now, now = self.sentry_layout(self.shot())
                card = next((c for c in now if c[0] == name and c[3] == "deck"), None)
                if card:
                    zx, zy = zones_now[zone]
                    self.drag(card[1], card[2] + down, zx, zy + 2 * down, f"Sentry: {name} to {zone}")
        keep = plan["keep"]
        if len(keep) == 2 and keep[0] != keep[1]:
            zones_now, now = self.sentry_layout(self.shot())
            deck = [c for c in now if c[3] == "deck"]
            if len(deck) == 2 and deck[-1][0] != keep[0]:             # the wanted top card is on the left
                left, right = deck
                self.drag(left[1], left[2] + down, right[1] + int(80 * self.vision.sx), right[2] + down,
                          f"Sentry: {left[0]} on top")
        _, now = self.sentry_layout(self.shot())
        placed = {z: [c[0] for c in now if c[3] == z] for z in ("trash", "discard", "deck")}
        log(f"  Sentry window now: {placed}")
        want = {"trash": sorted(plan["trash"]), "discard": sorted(plan["discard"])}
        if any(sorted(placed[z]) != want[z] for z in want):
            self.problem("sentry_mismatch", want=plan, seen=placed)
        self.brain.trashed(plan["trash"])
        return self.press(self.shot(), "OK", "Done")

    def card_mine(self, img, prompt):
        s, hand = self.hand_now(img)
        gain_step = self.says_gain(prompt) or (self.card_step % 2 == 1 and not self.says_trash(prompt))
        if gain_step:
            self.refresh_empty(img)
            g = getattr(self, "planned_gain", None)
            if g is None or g in self.brain.empty:
                g = self.brain.mine(hand + [getattr(self, "mine_trashed", "Copper")])["gain"]
            self.planned_gain = None
            log(f"  Mine -> gain {g} (to hand)")
            return self.gain_pile(g, "Mine") if g else self.press(img, "Decline", "Done", "OK")
        self.refresh_empty(img)
        r = self.brain.mine(hand)
        log(f"  Mine hand {hand} -> trash {r['trash']}, then gain {r['gain']}")
        if r["trash"] is None:
            return self.press(img, "Decline", "Done", "OK")
        if not self.click_hand(s, r["trash"], "Mine trash"):
            return False
        self.brain.trashed([r["trash"]])
        self.mine_trashed, self.planned_gain = r["trash"], r["gain"]
        return True

    def card_artisan(self, img, prompt):
        s, hand = self.hand_now(img)
        q = prompt.lower().replace(" ", "")
        topdeck_step = ("onto" in q or "deck" in q or "put" in q or
                        (self.card_step % 2 == 1 and not self.says_gain(prompt)))
        if topdeck_step:
            top = getattr(self, "planned_top", None)
            if top is None or top not in hand:
                top = self.brain.artisan(hand)["topdeck"]
                top = top if top in hand else (hand[0] if hand else None)
            self.planned_top = None
            log(f"  Artisan -> put {top} onto the deck")
            return self.click_hand(s, top, "Artisan topdeck") if top else self.press(img, "Done", "OK")
        self.refresh_empty(img)
        r = self.brain.artisan(hand)
        log(f"  Artisan hand {hand} -> gain {r['gain']}, then topdeck {r['topdeck']}")
        self.planned_top = r["topdeck"]
        return self.gain_pile(r["gain"], "Artisan") if r["gain"] else self.press(img, "Done", "OK")

    def card_moneylender(self, img, prompt):
        s, hand = self.hand_now(img)
        yes = self.brain.moneylender(hand)
        log(f"  Moneylender hand {hand} -> {'trash a Copper' if yes else 'decline'}")
        if yes and self.click_hand(s, "Copper", "Moneylender trash"):
            return True                       # on_action already counted the Copper as trashed
        self.brain.gained("Copper")           # ... so declining puts it back in the count
        return self.press(img, "Decline", "Done", "OK")

    def card_library(self, img, prompt):
        shown = self.popup_cards(img)
        s, hand = self.hand_now(img)
        a, _, _ = self.vision.hud(img)
        card = shown[0][0] if shown else next(
            (c for c in self.names if c.lower() in prompt.lower().replace(" ", "")), None)
        if card is None:
            log(f"  Library: which card? (prompt '{prompt}', window {shown})")
            return False
        skip = self.brain.library(card, hand, actions=a or 0)
        log(f"  Library drew {card} -> {'set it aside' if skip else 'keep it'}")
        labels = [b.text for b in self.vision.buttons(img)]
        if skip:
            return self.press(img, "Set Aside", "Aside", "Skip", "Yes")
        return self.press(img, "Keep", "Take", "Decline", "No")

    def card_throne_room(self, img, prompt):
        s, hand = self.hand_now(img)
        t = self.brain.throne_target(hand)
        log(f"  Throne Room hand {hand} -> {t or 'decline'}")
        if t is None or not self.click_hand(s, t, "Throne Room: play twice"):
            return self.press(img, "Decline", "Done", "OK")
        self.count_play(t, hand)
        self.count_play(t, hand)
        self.resolving, self.card_step = t, -1           # its prompts come next, twice
        self.throned = True
        return True

    def card_moat(self, img, prompt):
        """Attacked while holding a Moat: always reveal it (the engine does too). Not yet seen on
        Steam; if its prompt differs, the pilot saves it as an unknown prompt."""
        if self.my_phase_buttons(img):
            return False                       # my own Moat play: no prompt of its own
        log("  Moat: reveal it")
        return self.press(img, "Reveal", "Yes", "OK", "Done") or False

    def card_bandit(self, img, prompt):
        """Attacked by Bandit with two treasures to choose from (only then is there a choice).
        Not yet seen on Steam."""
        if self.my_phase_buttons(img):
            return False
        shown = [n for n, _, _ in self.popup_cards(img)]
        pick = self.brain.bandit(shown)
        log(f"  Bandit revealed {shown} -> trash {pick}")
        if pick is None:
            return self.press(img, "OK", "Done") or False
        _, x, y = next(p for p in self.popup_cards(img) if p[0] == pick)
        self.click(x, y + int(90 * self.vision.sy), f"Bandit: trash {pick}")
        return True

    def card_harbinger(self, img, prompt):
        s, hand = self.hand_now(img)
        pile = self.popup_cards(img)
        if not pile:
            log("  Harbinger: no discard pile window read")
            return self.press(img, "Decline", "Done", "OK")
        top = self.brain.harbinger(hand, [n for n, _, _ in pile])
        log(f"  Harbinger discard pile {[n for n, _, _ in pile]} -> {top or 'decline'}")
        if top is None:
            return self.press(img, "Decline", "Done", "OK")
        _, x, y = next(p for p in pile if p[0] == top)
        self.click(x, y + int(90 * self.vision.sy), f"Harbinger: topdeck {top}")
        return True

    def card_slot(self, hand_texts, card, chosen):
        """Where to click to select one more `card`: a stacked card is clicked once per copy;
        separate copies are clicked left to right."""
        self._slot_used = getattr(self, "_slot_used", {})
        if self._slot_used.get("_for") is not chosen:
            self._slot_used = {"_for": chosen}
        k = self._slot_used.get(card, 0)
        self._slot_used[card] = k + 1
        slots = [t for t in hand_texts if t.text == card]
        if not slots:
            return None
        if len(slots) == 1:
            return slots[0]
        return slots[min(k, len(slots) - 1)]

    def ok_button(self, img):
        for b in self.vision.buttons(img):
            if b.text.lower() in ("ok", "done", "confirm"):
                return b
        return None

    def ok_active(self, img, b) -> bool:
        """An enabled OK button is green; a disabled one is grey."""
        x0, y0 = b.x - int(45 * self.vision.sx), b.y - int(10 * self.vision.sy)
        x1, y1 = b.x + int(45 * self.vision.sx), b.y + int(10 * self.vision.sy)
        bgr = img[y0:y1, x0:x1].reshape(-1, 3).mean(0)
        return bgr[1] - bgr[0] > 25

    def press_ok_if_any(self, card, prefer=()) -> bool:
        """Press OK once the selection is valid (green). If it isn't - a stacked hand whose counts
        were misread, or a selection left over - click the stacks one at a time, in discard
        order, until OK turns green."""
        img = self.shot()
        b = self.ok_button(img)
        if b is None:
            return False
        if not self.ok_active(img, b):
            s = self.vision.screen(img, self.names)
            order = [c for c in dict.fromkeys(prefer)] + [t.text for t in s.hand if t.text not in prefer]
            fixed = False
            for c in order:
                t = next((t for t in s.hand if t.text == c), None)
                if t is None:
                    continue
                for _ in range(t.count + 1):
                    self.click(t.x, t.y + int(55 * self.vision.sy), f"{card}: adjust {c}")
                    img = self.shot()
                    if self.ok_active(img, b):
                        fixed = True
                        break
                if fixed:
                    break
            self.problem("selection_adjusted", card=card, fixed=fixed)
            if not fixed:
                return False
        self.click(b.x, b.y, f"{card}: button '{b.text}'")
        return True

    def on_unknown(self, img, prompt):
        self.unknown_count += 1
        if self.unknown_count == 3:     # same thing for a while: save it and say so
            os.makedirs(UNKNOWN_DIR, exist_ok=True)
            path = os.path.join(UNKNOWN_DIR, f"{int(time.time())}.png")
            cv2.imwrite(path, img)
            self.problem("unknown_prompt", prompt=prompt, card=self.center_card(img),
                         screenshot=os.path.basename(path))
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
            """A number on the score panel. Recognition on a tight crop and detection on a wider one
            each drop digits now and then: keep the longer reading (recognition wins a tie)."""
            reads = []
            x0, y0 = self.vision.scale(x - 26, y - 22)
            x1, y1 = self.vision.scale(x + 26, y + 22)
            crop = cv2.resize(img[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            res, _ = ocr_engine()(crop, use_det=False, use_cls=False)
            reads.append(re.sub(r"\D", "", res[0][0]) if res else "")
            x0, y0 = self.vision.scale(x - 30, y - 23)
            x1, y1 = self.vision.scale(x + 30, y + 27)
            res, _ = ocr_engine()(img[y0:y1, x0:x1])
            reads.append(next((re.sub(r"\D", "", r[1]) for r in res or [] if re.sub(r"\D", "", r[1])), ""))
            reads = [r for r in reads if r and int(r) <= 150]       # no game reaches 150 points
            best = max(reads, key=len) if reads else ""
            return int(best) if best else None
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

    # -- the opponent's cards (the top bar shows "Added: <cards>" after each of their gains) -------

    def parse_added(self, text):
        return parse_added(text, self.names)

    def track_opponent(self, img):
        text = self.vision.opponent_bar(img)
        cards = self.parse_added(text)
        if cards is None:
            self.opp_banner = None
            return
        if cards == getattr(self, "opp_banner", None):
            return                                   # the same banner still showing
        self.opp_banner = cards
        if self.brain.turn == 1 and not self.plan_seen:
            self.brain.first = False                 # they gained before my first buy: they went first
        self.brain.opp_owned.update(cards)
        log(f"  opponent gained {cards} -> {sum(self.brain.opp_owned.values())} cards")
        if REC:
            REC.event("opponent_gain", cards=cards, text=text)

    def step(self):
        img = self.stable_shot()
        try:
            self.track_opponent(img)
        except Exception as e:                       # tracking must never stop the game
            log(f"  opponent tracking failed: {e!r}")
        prompt = self.vision.prompt(img)
        p = prompt.lower()
        if not prompt:
            if self.score_screen(img):
                return "done"
            if not self.my_phase_buttons(img) and self.on_attack(img, "(blank)"):
                self.blank = 0
                return
            card = self.resolving or self.center_card(img)
            if (card and hasattr(self, self.handler_name(card)) and not self.my_phase_buttons(img)
                    and self.on_card(img, "", card)):
                self.blank = 0
                self.last_progress = time.time()
                return
            self.blank += 1
            if self.blank % 3 == 0:
                # Usually a card viewer (a big card with arrows, opened by a long press) covers the
                # board; Escape closes it.
                os.makedirs(UNKNOWN_DIR, exist_ok=True)
                cv2.imwrite(os.path.join(UNKNOWN_DIR, f"blank_{int(time.time())}.png"), img)
                log("  no prompt line for a while - pressing Escape to close any overlay")
                self.problem("overlay_closed")
                self.key("Escape")
            return
        self.blank = 0
        if "game over" in p or " wins" in p or "you won" in p or "you lost" in p:
            log("game finished:", prompt)
            self._save_state()
            return "done"
        q = p.replace(" ", "")      # OCR drops spaces, and stray pile digits can come before the text
        named = self.prompt_card(prompt)          # '<Card>: ...' goes to that card's handler first
        if named and hasattr(self, self.handler_name(named)) and self.on_card(img, prompt, named):
            self.unknown_count = 0
            self.last_progress = time.time()
            return
        kind = ("buy" if "buyacard" in q else
                "treasure" if "playtreasure" in q else
                "action" if "playaction" in q else
                "trash" if "trash" in p and "up to" in p else
                "discard" if "discard" in p and re.search(r"\d", p) and len(p) < 45 else None)
        # (a long text is a card's own text from a hover tooltip, not a prompt)
        if kind is None and "wait" not in p and "turn" not in p:
            # The last card played stays big in the middle and can hide the prompt line.
            # Then the buttons tell the phase: an OK button means a choice is wanted.
            labels = " | ".join(b.text.lower() for b in self.vision.buttons(img))
            if not self.my_phase_buttons(img, labels):
                if self.on_attack(img, prompt):
                    return
            elif "end actions" in labels:
                kind = "action"
            elif "play treasures" in labels:
                kind = "treasure"
            elif "end buys" in labels:
                kind = "buy"
            if kind:
                prompt = f"({kind} phase, prompt hidden)"
        if kind is None and "wait" not in p and "turn" not in p:
            card = self.prompt_card(prompt) or self.resolving or self.center_card(img)
            if card and self.on_card(img, prompt, card):
                self.unknown_count = 0
                self.last_progress = time.time()
                return
        if kind is None:
            if "wait" not in p and "turn" not in p:
                self.on_unknown(img, prompt)
            return
        self.resolving = None
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
    ap.add_argument("--model", default="auto",
                    help="model name (see --list-models), a path, or 'auto': the model selector picks "
                         "the best model for the dealt kingdom (default)")
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
