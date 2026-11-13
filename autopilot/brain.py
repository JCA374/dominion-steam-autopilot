"""Decisions for the Steam autopilot, asked of a released dominion-v1 model.

The autopilot keeps the game state it can see (turn, deck, Provinces, empty
piles) and asks the model through dominion-v1's `policy` protocol
(dominion-v1/docs/PROTOCOL.md). Any released model works: `--model NAME`, a
path, or models/DEFAULT. A GA model plays exactly as in training; cards the
model never learned are never bought.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dominion-v1"))
sys.path.insert(0, REPO)

from policy import PROTOCOL_VERSION, cards, list_models, load_policy, observation  # noqa: E402

BASE = cards.BASE
NEEDS_PROTOCOL = 1
if PROTOCOL_VERSION != NEEDS_PROTOCOL:
    raise ImportError(f"dominion-v1 speaks protocol {PROTOCOL_VERSION}, autopilot needs "
                      f"{NEEDS_PROTOCOL}; update the submodule or the autopilot")


class Brain:
    def __init__(self, kingdom: list[str], model: str | None = None, turn: int = 1):
        self.policy = load_policy(model)
        self.kingdom = kingdom
        self.unknown = [c for c in kingdom if not cards.known(c)]
        self.untrained = [c for c in kingdom if cards.known(c) and c not in self.policy.kingdom]
        self.owned = Counter({"Copper": 7, "Estate": 3})
        self.turn = turn
        self.provinces = 8
        self.empty: set[str] = set()   # piles seen to be empty (a buy click did nothing)

    def obs(self, hand=(), actions=1, coins=0, buys=1) -> dict:
        return observation(turn=self.turn, provinces_left=self.provinces, kingdom=self.kingdom,
                           empty_piles=sorted(self.empty), owned=dict(self.owned),
                           hand=list(hand), actions=actions, coins=coins, buys=buys)

    def phase(self) -> str:
        return self.policy.phase(self.obs()) or "-"

    # -- decisions -----------------------------------------------------------

    def choose_action(self, hand: list[str], actions: int) -> str | None:
        return self.policy.play_action(self.obs(hand, actions=actions))

    def throne_target(self, hand: list[str]) -> str | None:
        return self.policy.throne_target(self.obs(hand))

    def chapel_trash(self, hand: list[str]) -> list[str]:
        return self.policy.trash(self.obs(hand))

    def discard_down_to(self, hand: list[str], keep: int) -> list[str]:
        """Cards to discard to an attack like Militia."""
        return self.policy.discard(self.obs(hand), keep)

    def choose_buys(self, coins: int, buys: int) -> list[str]:
        return self.policy.buy(self.obs(coins=coins, buys=buys))

    # -- bookkeeping ---------------------------------------------------------

    def bought(self, card: str):
        self.owned[card] += 1
        if card == "Province":
            self.provinces -= 1

    def unbought(self, card: str):
        if self.owned[card] > 0:
            self.owned[card] -= 1
        if card == "Province":
            self.provinces += 1

    def trashed(self, trashed: list[str]):
        for c in trashed:
            if self.owned[c] > 0:
                self.owned[c] -= 1


def print_models():
    for m in list_models():
        print(f"{m['name']:18s} {m['kind']:12s} {', '.join(m.get('kingdom', []))}")
        if m.get("description"):
            print(f"{'':18s} {m['description']}")
