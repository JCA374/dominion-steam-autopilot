"""Decisions for the Steam autopilot, asked of a released dominion-v1 model.

The autopilot keeps the game state it can see (turn, deck, Provinces, empty
piles) and asks the model through dominion-v1's `policy` protocol
(dominion-v1/docs/PROTOCOL.md). Any model works: `--model NAME`, a path,
`auto` (dominion-v1's model selector picks the best model for the dealt
kingdom) or None (models/DEFAULT). A GA model plays exactly as in training;
cards the model never learned are never bought.

Models are plug-in folders: dominion-v1/models/ plus this repo's models/
(added to DOMINION_MODELS here), so a new model is a new folder, no code change.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.join(ROOT, "dominion-v1")
PLUGIN_MODELS = os.path.join(ROOT, "models")
sys.path.insert(0, REPO)
if PLUGIN_MODELS not in os.environ.get("DOMINION_MODELS", "").split(os.pathsep):
    os.environ["DOMINION_MODELS"] = os.pathsep.join(
        p for p in (os.environ.get("DOMINION_MODELS"), PLUGIN_MODELS) if p)

from policy import (PROTOCOL_VERSION, best_kingdom, cards, key_cards, list_models,  # noqa: E402
                    load_policy, observation, rank_models, read_bench)

AUTO = "auto"

BASE = cards.BASE
NEEDS_PROTOCOL = 1
if PROTOCOL_VERSION != NEEDS_PROTOCOL:
    raise ImportError(f"dominion-v1 speaks protocol {PROTOCOL_VERSION}, autopilot needs "
                      f"{NEEDS_PROTOCOL}; update the submodule or the autopilot")


class Brain:
    def __init__(self, kingdom: list[str], model: str | None = None, turn: int = 1):
        self.ranking = None
        if model == AUTO:                          # the selector picks for this kingdom
            self.ranking = rank_models(kingdom)
            model = self.ranking[0]["name"]
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


def model_rows() -> list[dict]:
    """Every model with what the UI and --list-models show: key cards, stored score."""
    rows = []
    for m in list_models():
        bench = read_bench(m) or {}
        rows.append({"name": m["name"], "kind": m.get("kind"), "kingdom": m.get("kingdom", []),
                     "key_cards": key_cards(m), "score": bench.get("score"),
                     "random_score": bench.get("random_score"), "bench_date": bench.get("date"),
                     "description": m.get("description", ""), "retired": bool(m.get("retired")),
                     "folder": m.get("dir")})
    return rows


def fmt_keys(keys) -> str:
    return "?" if keys is None else (" + ".join(f"{n}x {c}" for c, n in keys.items()) or "-")


def print_models():
    for r in model_rows():
        score = "  -  " if r["score"] is None else f"{100 * r['score']:4.0f}%"
        print(f"{r['name']:18s} {score}  key cards {fmt_keys(r['key_cards'])}"
              + ("  (retired)" if r["retired"] else ""))
        print(f"{'':18s} trained on {', '.join(r['kingdom'])}")
        if r["description"]:
            print(f"{'':18s} {r['description']}")
