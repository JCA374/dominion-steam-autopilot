"""Film: how the autopilot works. Scenes match video/works/script.md (## sections) one to one.

    .venv/bin/python video/build_works.py [--only 3,4] [--preview] [--resume]
Needs: a recording (recordings/session1) and the exported voiceover voice/film-works.
Output: docs/video/how-it-works.mp4
"""
import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "autopilot"))
from common import (ACC, BAD, BG, GOLD, GOOD, INK, MUTED, PANEL, SH, SW, WARN, FPS, Recording, arrow,  # noqa
                    box, bullets, canvas, ease, font, footage_with_log, game_view, header, play_span,
                    render_film, terminal_img, text, wrap)

ROOT = os.path.dirname(HERE)
REC = Recording(os.path.join(ROOT, "recordings", "session1"))
STRAT = json.load(open(os.path.join(ROOT, "dominion-v1", "best_model", "strategy.json")))
GAME = REC.games()[0]
REC2 = Recording(os.path.join(ROOT, "recordings", "session2"))   # the clean game, with a score screen
GAME2 = REC2.games()[0]


def first(kind, pred=None, after=None):
    return REC.find(kind, pred, after=GAME[0] if after is None else after)


def blurred_frame(t, dim=0.45):
    g = game_view(REC, t, overlay=False, scale_h=SH)
    bg = canvas()
    big = g.resize((SW, int(g.height * SW / g.width))).filter(ImageFilter.GaussianBlur(14))
    bg.paste(big, (0, -(big.height - SH) // 2))
    return Image.blend(bg, canvas(), 1 - dim)


# ---------------------------------------------------------------- scenes

def s_intro(n):
    t_mid = (GAME[0] + GAME[1]) / 2
    base = blurred_frame(t_mid)
    for i in range(n):
        img = base.copy()
        d = ImageDraw.Draw(img)
        a = ease(i / (FPS * 1.2))
        d.text((80, 230), "How the Dominion autopilot works", font=font(54, True), fill=INK)
        d.text((84, 310), "An evolved strategy plays the Steam game: screen in, clicks out", font=font(28), fill=MUTED)
        if a > 0.5:
            text(d, (84, 400), "Every game frame, box, click and log line in this film is from a real recording "
                 "of the autopilot playing.", size=24, fill=GOLD, width=1000)
        yield img


def s_problem(n):
    t = GAME[0] + 30
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "The problem", "Two different programs, no bridge between them")
        box(d, (60, 170, 560, 520), outline=ACC)
        d.text((90, 190), "The genetic algorithm", font=font(30, True), fill=ACC)
        y = 250
        for line in ["Its own small Dominion engine", "(Python, plus C for speed)", "Thousands of simulated games",
                     "Output: best_model/strategy.json", "15 base-set kingdom cards"]:
            d.text((90, y), line, font=font(24), fill=INK)
            y += 44
        box(d, (720, 170, 1220, 520), outline=WARN)
        d.text((750, 190), "Steam Dominion", font=font(30, True), fill=WARN)
        y = 250
        for line in ["Closed Windows program", "Runs on Linux via Proton", "No bot API, no mods",
                     "No file with your hand", "Hundreds of kingdom cards"]:
            d.text((750, y), line, font=font(24), fill=INK)
            y += 44
        if i > n * 0.35:
            arrow(d, (570, 345), (710, 345), fill=MUTED, w=5)
            d.text((588, 300), "?", font=font(40, True), fill=BAD)
        if i > n * 0.6:
            text(d, (60, 560), "The only way in: what a person does. Look at the screen, decide, click.",
                 size=30, fill=GOLD, bold=True, width=1160)
        yield img


def s_pipeline(n):
    labels = [("EYES", "screenshot + text reading", ACC), ("BRAIN", "the evolved strategy", GOOD),
              ("HANDS", "mouse moves + clicks", GOLD)]
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "Three parts in a loop", "One pass handles one thing the game asks for")
        xs = [80, 480, 880]
        on = int(i / (n / 4))
        for k, (name, sub, c) in enumerate(labels):
            x = xs[k]
            box(d, (x, 220, x + 320, 400), outline=c if on >= k else (60, 64, 76), w=4)
            d.text((x + 30, 250), name, font=font(44, True), fill=c if on >= k else MUTED)
            text(d, (x + 30, 320), sub, size=24, fill=INK, width=270)
            if k < 2:
                arrow(d, (x + 330, 310), (xs[k + 1] - 10, 310), fill=MUTED)
        d.line((1040, 410, 1040, 480), fill=MUTED, width=4)
        d.line((1040, 480, 240, 480), fill=MUTED, width=4)
        arrow(d, (240, 480), (240, 410), fill=MUTED)
        d.text((520, 490), "next screenshot", font=font(22), fill=MUTED)
        if on >= 3:
            prompts = ["Play Action cards.", "Play Treasure cards.", "Buy a card."]
            y = 560
            d.text((80, y), "Prompts it acts on:", font=font(24, True), fill=INK)
            for k, p in enumerate(prompts):
                d.text((360 + k * 290, y), f'"{p}"', font=font(24), fill=GOLD)
        yield img


def s_screenshots(n):
    t = first("decide_action")["t"] - 1.0
    g = game_view(REC, t, overlay=False)
    for i in range(n):
        img = canvas()
        img.paste(g, (0, 0))
        d = ImageDraw.Draw(img)
        x = g.width + 30
        d.text((x, 40), "The eyes: a screenshot", font=font(30, True), fill=INK)
        y = 100
        for k, (a, b) in enumerate([("Window", '"Dominion", 927 x 1048'),
                                    ("Layer", "XWayland (X11 inside Wayland)"),
                                    ("Call", "XGetImage on that window"),
                                    ("Time", "about 50 ms per screenshot"),
                                    ("Permission", "none needed")]):
            if i > n * k / 6:
                d.text((x, y), a, font=font(20), fill=MUTED)
                d.text((x, y + 26), b, font=font(26, True), fill=ACC)
                y += 84
        yield img


def _ocr_moment():
    e = first("decide_action")
    return e["t"] + 0.05


def s_ocr(n):
    t = _ocr_moment()
    reads = REC.between(t - 25, t, "ocr")[-4:]
    for i in range(n):
        img = canvas()
        g = game_view(REC, t, overlay=True, ocr_window=25)
        img.paste(g, (0, 0))
        d = ImageDraw.Draw(img)
        x = g.width + 30
        d.text((x, 30), "What it reads (blue boxes)", font=font(28, True), fill=INK)
        y = 84
        names = ["prompt line", "hand", "counters: actions, coins, buys", "buttons"]
        for k, e in enumerate(reads):
            if i < n * k / 5:
                break
            found = ", ".join(tx[0] for tx in e["texts"]) or "(nothing)"
            label = names[k] if k < len(names) else "region"
            d.text((x, y), label, font=font(20), fill=MUTED)
            y = text(d, (x, y + 26), found, size=22, fill=ACC, width=SW - x - 30) + 18
        text(d, (x, SH - 90), "Rapid OCR, running locally on the processor", size=20, fill=MUTED, width=SW - x - 30)
        yield img


def s_matching(n):
    from vision import match_name
    names = sorted(set(GAME[2]["kingdom"]) | {"Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"})
    upper = {x.upper(): x for x in names}
    seen = {}
    for e in REC.events:
        if e["kind"] != "ocr":
            continue
        for tx in e["texts"]:
            raw = tx[0]
            m = match_name(raw, upper)
            if m and raw.upper().replace(" ", "") != m.upper().replace(" ", "") or (m and raw.upper() != m.upper()):
                seen.setdefault(raw, m)
    seen.setdefault("STATE", "Estate")          # read in the very first test (whole-window OCR)
    examples = list(seen.items())[:5] + [("AHLIWS", "Smithy, after the fix")]
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "Fixing small reading mistakes", "Each read name is matched to this game's card list (at least 80% similar)")
        d.text((120, 170), "OCR read", font=font(22, True), fill=MUTED)
        d.text((620, 170), "matched card", font=font(22, True), fill=MUTED)
        y = 215
        for k, (raw, m) in enumerate(examples):
            if i < n * k / (len(examples) + 2):
                break
            d.text((120, y), f'"{raw}"', font=font(30, mono=True), fill=WARN)
            arrow(d, (480, y + 18), (580, y + 18))
            d.text((620, y), m, font=font(30, True), fill=GOOD)
            y += 58
        text(d, (120, SH - 120), "Numbers are read as plain digits. If the dimmed Province pile can't be read, "
             "the last count seen is kept.", size=22, fill=MUTED, width=1040)
        yield img


def s_kingdom(n):
    gs = GAME[2]
    lines = ["<GameSetup_Save>", "  <players>", "    <playerType>AI</playerType>",
             f"    <playerName>{gs['opponents'][0]}</playerName>", "  </players>", "  <config>",
             f"    <seed>{gs['seed']}</seed>", "    <setupData>-e base</setupData>",
             "    <kingdomData>" + ", ".join(k.lower().replace(" ", "_") for k in gs["kingdom"]) + "</kingdomData>",
             "  </config>"]
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "The kingdom comes from the save file", "TempleGates/Dominion/gamelocal_*.xml inside the Proton folder")
        term = terminal_img(lines, title="gamelocal_0_-2.xml (excerpt)", size=19, w=1160)
        img.paste(term, (60, 150))
        y = 150 + term.height + 30
        if i > n * 0.4:
            d.text((60, y), "Read once per game:", font=font(24, True), fill=INK)
            y += 40
            text(d, (60, y), " · ".join(gs["kingdom"]), size=24, fill=GOOD, width=1160)
        yield img


def s_strategy(n):
    tr = STRAT["transitions"]
    cols = [("EARLY", f"turns 1-{tr['early_to_mid_turn']}", STRAT["early_buy_priority"]),
            ("MIDDLE", f"until {tr['mid_to_late_provinces']} Provinces left or turn {tr['mid_to_late_turn']}",
             STRAT["mid_buy_priority"]),
            ("LATE", "to the end", STRAT["late_buy_priority"])]
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "The brain: the evolved strategy", "dominion-v1/best_model/strategy.json: one buy list per phase")
        for k, (name, sub, lst) in enumerate(cols):
            x = 60 + k * 400
            if i < n * k / 3.5:
                break
            box(d, (x, 160, x + 370, 680), outline=[ACC, GOOD, GOLD][k])
            d.text((x + 20, 175), name, font=font(28, True), fill=[ACC, GOOD, GOLD][k])
            text(d, (x + 20, 212), sub, size=18, fill=MUTED, width=330)
            y = 270
            for j, c in enumerate(lst):
                if c == "PASS":
                    d.text((x + 20, y), f"{j + 1:2d}. PASS (stop here)", font=font(20, True), fill=BAD)
                    break
                d.text((x + 20, y), f"{j + 1:2d}. {c}", font=font(20), fill=INK)
                y += 30
        yield img


def s_buying(n):
    e = first("decide_buy", lambda e: e["coins"] >= 6) or first("decide_buy")
    owned = ", ".join(f"{v} {k}" for k, v in e["owned"].items() if v)
    steps = [("read", f"coins {e['coins']}, buys {e['buys']}, Provinces left {e['provinces']}", ACC),
             ("stand-in state", f"GameState(coins={e['coins']}, buys={e['buys']}, deck = {owned})", INK),
             ("original code", "core.engine.play_buy_phase(state, strategy)", GOOD),
             ("plan", f"buy {', '.join(e['plan']) or 'nothing'}  (turn {e['turn']}, {e['phase']} phase)", GOLD),
             ("click", "the pile of each card in the plan", GOLD)]
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "Buying: the strategy's own code decides", "A real decision from the recording")
        y = 170
        for k, (a, b, c) in enumerate(steps):
            if i < n * k / (len(steps) + 1.5):
                break
            box(d, (60, y, 1220, y + 74), outline=c)
            d.text((84, y + 10), a, font=font(18), fill=MUTED)
            text(d, (84, y + 34), b, size=22, fill=c, width=1110, mono=True)
            if k < len(steps) - 1:
                arrow(d, (640, y + 76), (640, y + 96))
            y += 98
        if i > n * 0.75:
            text(d, (60, SH - 60), "Hidden rules come along: buy targets (max copies) and the terminal-density check.",
                 size=21, fill=MUTED, width=1160)
        yield img


def s_unknown(n):
    gs = GAME[2]
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "Cards it never learned are left out", f"This game's kingdom vs the 15 cards the strategy knows")
        for k, c in enumerate(gs["kingdom"]):
            x = 60 + (k % 5) * 236
            y = 190 + (k // 5) * 170
            unk = c in gs["unknown"]
            col = (70, 72, 80) if unk else GOOD
            if i < n * k / 12:
                break
            box(d, (x, y, x + 216, y + 140), outline=col, w=4)
            d.text((x + 16, y + 20), c, font=font(24, True), fill=MUTED if unk else INK)
            d.text((x + 16, y + 80), "never bought" if unk else "known", font=font(20), fill=BAD if unk else GOOD)
        text(d, (60, 560), "An unknown card is removed from the stand-in supply, so the strategy can't buy it, "
             "and the autopilot never needs to know how to play it.", size=24, fill=INK, width=1160)
        yield img


def s_actions(n):
    e = first("decide_action", lambda e: e["choice"])
    t = e["t"]
    for i in range(n):
        img = canvas()
        g = game_view(REC, t + 0.05 + min(1.2, i / FPS * 0.2))
        img.paste(g, (0, 0))
        d = ImageDraw.Draw(img)
        x = g.width + 30
        d.text((x, 30), "Playing actions", font=font(30, True), fill=INK)
        d.text((x, 86), "1. Non-terminals first (give an action back)", font=font(20, True), fill=ACC)
        text(d, (x + 20, 116), " > ".join(STRAT["nonterminal_priority"]), size=20, fill=INK, width=SW - x - 50)
        d.text((x, 190), "2. Then terminals", font=font(20, True), fill=ACC)
        text(d, (x + 20, 220), " > ".join(STRAT["terminal_priority"]), size=20, fill=INK, width=SW - x - 50)
        d.text((x, 300), "This moment:", font=font(22, True), fill=INK)
        text(d, (x, 334), f"hand: {', '.join(e['hand'])}", size=20, fill=MUTED, width=SW - x - 30)
        d.text((x, 400), f"actions: {e['actions']}  ->  play {e['choice']}", font=font(24, True), fill=GOLD)
        text(d, (x, 460), "Moneylender is skipped with no Copper in hand. After each play: a fresh screenshot, "
             "because the hand has changed.", size=20, fill=MUTED, width=SW - x - 30)
        yield img


def s_clicks(n):
    c = first("click", lambda e: "button" not in e["what"] and "buy" not in e["what"])
    c = c or first("click")
    t0, t1 = c["t"] - 0.3, c["t"] + 1.6
    for i in range(n):
        t = t0 + (t1 - t0) * min(1.0, i / max(1, n * 0.7))
        img = canvas()
        g = game_view(REC, t, click_window=1.6)
        img.paste(g, (0, 0))
        d = ImageDraw.Draw(img)
        x = g.width + 30
        d.text((x, 30), "The hands: a patient click", font=font(28, True), fill=INK)
        steps = [("glide", "the last few pixels to the target"), ("rest", "about 0.35 s, so the game sees a hover"),
                 ("press", "hold 0.15 s, then release"), ("via", "XTEST, passed on by XWayland")]
        y = 100
        for k, (a, b) in enumerate(steps):
            if i < n * k / 5:
                break
            d.text((x, y), a, font=font(24, True), fill=GOLD)
            text(d, (x + 110, y + 2), b, size=22, fill=INK, width=SW - x - 140)
            y += 70
        text(d, (x, SH - 130), f"Here: {c['what']} at ({c['x']}, {c['y']}). The first version clicked "
             "instantly and the game ignored it.", size=20, fill=MUTED, width=SW - x - 30)
        yield img


def s_timing(n):
    # the start of a turn: the 'turn' log line, footage just after it
    e = REC.find("log", lambda e: e["text"].startswith("=== turn"), after=GAME[0] + 60)
    t0 = e["t"] - 6
    for i in range(n):
        t = t0 + 8 * min(1.0, i / max(1, n * 0.6))
        img = canvas()
        g = game_view(REC, t, overlay=False)
        img.paste(g, (0, 0))
        d = ImageDraw.Draw(img)
        x = g.width + 30
        d.text((x, 30), "Wait for a still screen", font=font(28, True), fill=INK)
        text(d, (x, 90), "At the start of a turn the game animates: the hand slides in, the opponent's last card "
             "is still shown. Clicks then are lost.", size=21, fill=INK, width=SW - x - 30)
        rules = ["1. Two screenshots 0.35 s apart, until they barely differ", "2. In the action phase: wait until "
                 "the actions counter shows at least 1"]
        y = 280
        for k, r in enumerate(rules):
            if i > n * (0.3 + 0.25 * k):
                y = text(d, (x, y), r, size=22, fill=GOLD, bold=True, width=SW - x - 30) + 16
        if i > n * 0.8:
            text(d, (x, SH - 90), "With both rules, Smithy plays on the first click.", size=22, fill=GOOD,
                 width=SW - x - 30)
        yield img


def s_empty(n):
    t = REC.times[-1] - 1       # the end of session1: stalled at turn 52, both piles empty
    stuck = [e for e in REC.events if e["kind"] == "decide_buy" and e["turn"] >= 25]
    for i in range(n):
        img = canvas()
        g = game_view(REC, t, overlay=False)
        img.paste(g, (0, 0))
        d = ImageDraw.Draw(img)
        x = g.width + 30
        d.text((x, 30), "Bug: empty piles", font=font(30, True), fill=BAD)
        text(d, (x, 84), "Duchy and Estate show a red X: empty. The strategy kept asking for them.", size=21,
             fill=INK, width=SW - x - 30)
        y = 170
        d.text((x, y), "turn  coins  plan", font=font(18, True, mono=True), fill=MUTED)
        y += 28
        for e in stuck[:min(len(stuck), 1 + int(i / max(1, n * 0.5) * 12))][:12]:
            d.text((x, y), f"{e['turn']:>4}  ${e['coins'] if e['coins'] is not None else '?':<5} {', '.join(e['plan'])}",
                   font=font(18, mono=True), fill=WARN if e["plan"] and e["plan"][0] in ("Duchy", "Estate") else INK)
            y += 26
        if i > n * 0.6:
            text(d, (x, SH - 130), "Fix: after a buy click, if coins and buys didn't change, the pile is empty - "
                 "note it, ask the strategy again.", size=21, fill=GOOD, width=SW - x - 30)
        yield img


def s_turn(n):
    a = first("decide_action", lambda e: e["choice"] == "Smithy", after=GAME[0] + 40) or first("decide_action")
    b = REC.find("click", lambda e: e["what"].startswith("buy"), after=a["t"])
    t0, t1 = a["t"] - 2.5, b["t"] + 2.0
    yield from play_span(REC, t0, t1, n, title=f"One whole turn (turn {a['turn']})",
                         note="Action, treasures, buy - each step: screenshot, read, decide, click")


def s_memory(n):
    seed = GAME2[2]["seed"]
    path = os.path.join(ROOT, "autopilot", f"run_{seed}.json")
    data = json.load(open(path)) if os.path.exists(path) else {}
    body = json.dumps(data, indent=2).split("\n")
    lines = [f"$ cat autopilot/run_{seed}.json"] + body
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        header(d, "A small memory, saved every turn", "Turn (for the phase), Provinces left, cards owned, empty piles")
        term = terminal_img(lines[:min(len(lines), 2 + int(i / max(1, n * 0.5) * len(lines)))], size=18, w=700,
                            h=540)
        img.paste(term, (60, 150))
        text(d, (800, 180), "Why: the buy rules count owned cards (buy targets, terminal density), and the phase "
             "depends on the turn.", size=22, fill=INK, width=420)
        text(d, (800, 380), "Stop it and start it again mid-game: it reloads this file and carries on.", size=22,
             fill=GOOD, width=420)
        yield img


def s_ending(n):
    over = REC2.find("log", lambda e: e["text"].startswith("game over"))
    t0 = over["t"] - 14
    yield from play_span(REC2, t0, over["t"] + 3, n, title="The score screen",
                         note="It finds Rematch, reads the points, saves a picture of the screen. "
                              "This game: won 55 to 35.")


def s_limits(n):
    yield from bullets("Limits", "What it can't do (yet)", [
        "Knows only the 15 base-set cards the strategy was evolved with",
        "A prompt it doesn't understand: it saves a screenshot to autopilot/unknown/ and waits for you",
        "Assumes the window keeps roughly its size (927 x 1048 was used here)",
        "Only for games against the built-in computer opponents, never against people online",
        "Inside those limits: a whole game on its own, first treasure to score screen"], n)


SCENES = [s_intro, s_problem, s_pipeline, s_screenshots, s_ocr, s_matching, s_kingdom, s_strategy, s_buying,
          s_unknown, s_actions, s_clicks, s_timing, s_empty, s_turn, s_memory, s_ending, s_limits]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    only = {int(x) for x in a.only.split(",") if x}
    render_film(SCENES, os.path.join(ROOT, "voice", "film-works"), os.path.join(ROOT, "docs", "video", "how-it-works.mp4"),
                os.path.join(ROOT, "video", "build", "works"), only, a.preview, a.resume)


if __name__ == "__main__":
    main()
