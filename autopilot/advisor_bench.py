"""How fast and how right is the advisor? It replays a recording at real speed (the advisor's live loop, missing
frames while it reads as it would live) and measures, for every new question it answers (a buy, an action, a card's
question):

  - latency: from the first frame that showed the question (prompt line and coin counter as they were when it was
    read) to the moment the suggestion was published, in the recording's clock;
  - holds: suggestions held back because a key reading was unsure (coins, a pile, the board after a resize);
  - readings against the recording's own log: the coins of my last buy question each turn against the pilot's
    decide_buy, and the advisor's own corrections and deck-check gaps.

    .venv/bin/python autopilot/advisor_bench.py recordings/session2 [--speed 1] [--no-teacher] [--out x.json]
    .venv/bin/python autopilot/advisor_bench.py recordings/session1b --resize 60:1100x1100   # a window resize at 60 s

--resize T:WxH shows every frame from T seconds on as the game would draw it in a WxH window (the board scaled with
the width, the top half pinned to the top and the bottom half to the bottom, as vision.py assumes).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import advisor as A  # noqa: E402
import vision  # noqa: E402


def resized(img, w, h):
    """A frame as the game draws it in a w x h window."""
    s = w / A.REF_W
    big = cv2.resize(img, (w, round(A.REF_H * s)), interpolation=cv2.INTER_AREA)
    half = round(A.REF_H / 2 * s)
    if h >= big.shape[0]:
        fill = np.repeat(big[half:half + 1], h - big.shape[0], axis=0)
        return np.concatenate([big[:half], fill, big[half:]])
    cut = big.shape[0] - h
    return np.concatenate([big[:half - cut // 2], big[half + cut - cut // 2:]])


def compare_coins(qs, decides):
    """My coins at each of the pilot's buys (its decide_buy events: time, coins) against the advisor's: the most any
    buy question said that appeared since the pilot's previous buy (the first one, before a buy spends coins)."""
    wrong, n, prev = [], 0, -1e9
    for t, coins in decides:
        said = [int(m.group(1)) for q in qs if prev < q["appeared"] <= t + 1000 and q["kind"] in ("buy", "treasure")
                for m in [re.search(r"\$(\d+)", q["question"])] if m]
        prev = t
        if said:
            n += 1
            if max(said) != coins:
                wrong.append({"at_s": round(t / 1000), "advisor": max(said), "pilot": coins})
    return {"compared": n, "of": len(decides), "wrong": wrong}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--resize", help="T:WxH, a window resize T seconds into the recording")
    ap.add_argument("--out", help="write the measurements here (JSON)")
    a = ap.parse_args()

    frames = sorted(os.listdir(os.path.join(a.folder, "frames")))
    times = [int(f[:-4]) for f in frames]
    index = {ms: i for i, ms in enumerate(times)}
    hook = None
    if a.resize:
        t, size = a.resize.split(":")
        rw, rh = (int(v) for v in size.split("x"))
        hook = lambda ms, img: resized(img, rw, rh) if ms >= float(t) * 1000 else img   # noqa: E731

    def load(i):
        img = cv2.imread(os.path.join(a.folder, "frames", frames[i]))
        return hook(times[i], img) if hook else img

    def spots(img):
        v = vision.Vision(img.shape[1], img.shape[0])
        out = []
        for b in (A.R_PROMPT_LINE, A.vision.R_HUD):
            (x0, y0), (x1, y1) = v.scale(*b[:2]), v.scale(*b[2:])
            out.append(img[y0:y1, x0:x1].astype(np.int16))
        return out

    def appeared(ms):
        """The first frame (going back from ms) with the same prompt line and counters: the question's start."""
        i = index[ms]
        ref = spots(load(i))
        while i > 0:
            prev = load(i - 1)
            sp = spots(prev)
            if any(p.shape != r.shape or int((np.abs(p - r).max(axis=2) > 40).sum()) > 3 for p, r in zip(sp, ref)):
                break
            i -= 1
        return times[i]

    A.vision._ocr = A.make_reader()
    adv = A.Advisor(A.Judge())
    if not a.no_teacher:
        adv.attach_teacher()
    reads, pubs = [], []
    process, publish = adv.process, adv.publish

    def timed_process(img):
        t0 = time.time()
        process(img)
        reads.append({"ms": adv.frame_ms, "secs": round(time.time() - t0, 3)})

    def logged_publish(img, kind, prompt, advice, hold=None):
        publish(img, kind, prompt, advice, hold)
        t = adv.t
        pubs.append({"at": adv.clock(), "ms": adv.frame_ms, "kind": kind, "turn": t.turns + 1, "game": adv.game_n,
                     "question": advice and advice.get("question"), "hold": hold and list(hold),
                     "top": advice and advice["top"][:1], "coins": t.coins, "provinces": t.supply[A.ID["Province"]]})

    adv.process, adv.publish = timed_process, logged_publish
    adv.game_n = 0
    new_game = adv.new_game

    def counted_new_game(*args, **kw):
        adv.game_n += 1
        new_game(*args, **kw)
    adv.new_game = counted_new_game
    t0 = time.time()
    adv.replay(a.folder, a.speed, frame_hook=hook)
    wall = time.time() - t0

    # every new question once, at its first suggestion; a hold before it (same game, turn and phase) is its start
    qs, seen, holds = [], set(), {}
    for p in pubs:
        if p["hold"]:
            holds.setdefault((p["game"], p["turn"], p["kind"]), p)
            continue
        if not p["question"]:
            continue
        key = (p["game"], p["turn"], p["question"])
        if key in seen:
            continue
        seen.add(key)
        h = holds.pop((p["game"], p["turn"], p["kind"]), None)
        qs.append({"game": p["game"], "turn": p["turn"], "question": p["question"], "kind": p["kind"],
                   "ms": (h or p)["ms"], "shown": p["at"], "held": h is not None, "why": h and h["hold"][1],
                   "top": p["top"]})
    for q in qs:
        q["appeared"] = appeared(q["ms"])
        q["latency"] = round((q["shown"] - q["appeared"]) / 1000, 2)
    unshown = [{"game": h["game"], "turn": h["turn"], "ms": h["ms"], "why": h["hold"][1]} for h in holds.values()]

    # coins: at each of the pilot's buys (its own reading, logged when it decided)
    events = [json.loads(l) for l in open(os.path.join(a.folder, "events.jsonl"))]
    out_coins = compare_coins(qs, [(e["t"] * 1000, e["coins"]) for e in events if e["kind"] == "decide_buy"])

    lat = np.array([q["latency"] for q in qs])
    secs = np.array([r["secs"] for r in reads])
    notes = [e[2] for e in (adv.t.events if adv.t else [])]
    pct = lambda x, p: round(float(np.percentile(x, p)), 2) if len(x) else None   # noqa: E731
    out = {
        "folder": a.folder, "speed": a.speed, "teacher": not a.no_teacher, "resize": a.resize, "wall_s": round(wall),
        "questions": len(qs), "held_then_shown": sum(q["held"] for q in qs), "held_never_shown": unshown,
        "held_why": {w: sum(q["why"] == w for q in qs) for w in sorted({q["why"] for q in qs if q["why"]})},
        "latency_s": {"median": pct(lat, 50), "p90": pct(lat, 90), "max": pct(lat, 100)} if len(lat) else None,
        "read_s": {"n": int(len(secs)), "median": pct(secs, 50), "p90": pct(secs, 90), "max": pct(secs, 100)},
        "coins": out_coins,
        "last_game_notes": {"corrections": sum("correction" in n for n in notes),
                            "deck_gaps": sum("a gain or trash was missed" in n for n in notes),
                            "big_drops": sum("between readings" in n for n in notes)},
        "slowest": sorted(({k: q[k] for k in ("game", "turn", "question", "latency", "why")} for q in qs),
                          key=lambda q: -q["latency"])[:8],
    }
    print(json.dumps(out, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({**out, "all": qs, "reads": reads}, f, indent=1)
    os._exit(0)                                   # the teacher's thread


if __name__ == "__main__":
    main()
