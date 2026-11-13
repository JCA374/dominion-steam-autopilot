"""Decisions for the Steam autopilot, taken from the evolved GA strategy.

Buying reuses dominion-v1's own play_buy_phase on a stand-in game state, so the
purchases are exactly what the trained strategy would make. Cards from the Steam
kingdom that the GA never learned are left out of the supply: it never buys them.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

REPO = os.path.join(os.path.dirname(__file__), "..", "dominion-v1")
sys.path.insert(0, os.path.abspath(REPO))

from core.cards import ALL_CARDS, CardType  # noqa: E402
from core.engine import GameState, play_buy_phase  # noqa: E402
from core.strategy import (get_current_phase, get_chapel_trash_priority,  # noqa: E402
                           load_strategy)

BASE = ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"]


class Brain:
    def __init__(self, kingdom: list[str], model: str | None = None, turn: int = 1):
        self.strategy = load_strategy(model or os.path.join(REPO, "best_model", "strategy.json"))
        self.kingdom = kingdom
        self.known = [c for c in kingdom if c in ALL_CARDS]
        self.unknown = [c for c in kingdom if c not in ALL_CARDS]
        self.owned = Counter({"Copper": 7, "Estate": 3})
        self.turn = turn
        self.provinces = 8
        self.empty: set[str] = set()   # piles seen to be empty (a buy click did nothing)

    def phase(self) -> str:
        return get_current_phase(self.turn, self.provinces, self.strategy.transitions)

    # -- action phase --------------------------------------------------------

    def choose_action(self, hand: list[str], actions: int) -> str | None:
        """Next action card to play: non-terminals first, then terminals."""
        if actions < 1:
            return None
        for prio in (self.strategy.nonterminal_priority, self.strategy.terminal_priority):
            for card in prio:
                if card in hand and not self._useless(card, hand):
                    return card
        return None

    def _useless(self, card: str, hand: list[str]) -> bool:
        if card == "Moneylender":
            return "Copper" not in hand
        if card == "Throne Room":
            return not any(self.throne_target(hand, exclude_self=True))
        return False

    def throne_target(self, hand: list[str], exclude_self=False):
        rest = list(hand)
        if exclude_self and "Throne Room" in rest:
            rest.remove("Throne Room")
        for card in self.strategy.throne_room_priority:
            if card in rest and ALL_CARDS[card].card_type == CardType.ACTION:
                yield card

    def chapel_trash(self, hand: list[str]) -> list[str]:
        trash, rest = [], list(hand)
        for card in get_chapel_trash_priority(self.strategy, self.phase()):
            if card == "STOP":
                break
            while card in rest and len(trash) < min(self.strategy.chapel_max_trash, 4):
                rest.remove(card)
                trash.append(card)
        return trash

    def discard_down_to(self, hand: list[str], keep: int) -> list[str]:
        """Cards to discard to an attack like Militia (same order as the GA engine)."""
        from core.engine import militia_discard
        n = len(hand) - keep
        extra = [c for c in hand if c not in ALL_CARDS]  # cards the GA doesn't know go first
        if len(extra) >= n:
            return extra[:n]
        # militia_discard empties a hand down to 3; pad so it stops at `keep`
        st = GameState(hand=[c for c in hand if c in ALL_CARDS] + ["Province"] * (3 - keep))
        militia_discard(st, self.strategy)
        return (extra + st.discard)[:n]

    # -- buy phase -----------------------------------------------------------

    def choose_buys(self, coins: int, buys: int) -> list[str]:
        supply = {c: 10 for c in BASE + self.known}
        supply["Province"] = self.provinces
        for c in self.empty:
            supply[c] = 0
        st = GameState(supply=supply, coins=coins, buys=buys, turn=self.turn,
                       deck=list(self.owned.elements()))
        play_buy_phase(st, self.strategy)
        return list(st.discard)

    def bought(self, card: str):
        self.owned[card] += 1
        if card == "Province":
            self.provinces -= 1

    def unbought(self, card: str):
        if self.owned[card] > 0:
            self.owned[card] -= 1
        if card == "Province":
            self.provinces += 1

    def trashed(self, cards: list[str]):
        for c in cards:
            if self.owned[c] > 0:
                self.owned[c] -= 1
