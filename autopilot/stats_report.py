"""Build the Stats tab (docs/stats.html + docs/data/stats.json) from autopilot/stats/games.jsonl.

    .venv/bin/python autopilot/stats_report.py
Every number on the page comes from that file. Small samples are marked as such.
"""
from __future__ import annotations

import html
import json
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GAMES = os.path.join(HERE, "stats", "games.jsonl")
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(ROOT, "docs", "stats.html")
IMG = os.path.join(ROOT, "docs", "img", "stats")
LEVELS = ["Very Easy", "Easy", "Medium", "Hard"]

# Dominion base set (2nd edition) kingdom cards; first-edition cards included in case they turn up.
BASE_KINGDOM = ["Artisan", "Bandit", "Bureaucrat", "Cellar", "Chapel", "Council Room", "Festival", "Gardens",
                "Harbinger", "Laboratory", "Library", "Market", "Merchant", "Militia", "Mine", "Moat",
                "Moneylender", "Poacher", "Remodel", "Sentry", "Smithy", "Throne Room", "Vassal", "Village",
                "Witch", "Workshop"]
ATTACKS = {"Bandit", "Bureaucrat", "Militia", "Witch"}
# Logged events that are normal handling, not problems.
HANDLED = {"misread_hand", "attack_answered"}


def esc(s):
    return html.escape(str(s))


def load():
    if not os.path.exists(GAMES):
        return []
    games = [json.loads(l) for l in open(GAMES) if l.strip()]
    for g in games:                     # an OCR misread of a point shield (e.g. 742): not plausible
        for k in ("me", "opponent"):
            if g.get(k) is not None and g[k] > 150:
                g[k] = None
    return games


def played(g):
    return g.get("outcome") in ("win", "loss", "tie")


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "–"


def cards_in_text(text):
    found = []
    for c in BASE_KINGDOM:
        if re.search(r"\b" + re.escape(c) + r"\b", text, re.I):
            found.append(c)
    return found


def problem_label(p):
    k = p["kind"]
    if k == "unknown_prompt":
        return f"unknown prompt: “{p.get('prompt', '')}”"
    if k == "auto_resolve":
        return f"answered generically ({p.get('how', '')}): “{p.get('prompt', '')}”"
    if k == "empty_pile":
        return f"empty pile noticed: {p.get('card')}"
    if k == "misread_hand":
        return "hand misread, read again"
    if k == "no_button":
        return f"expected button missing ({'/'.join(p.get('want', []))})"
    if k == "abandoned":
        return f"game abandoned ({p.get('why')})"
    if k == "exception":
        return f"error: {p.get('error', '')[:80]}"
    return k


def score_image(g):
    """A small copy of the saved score screen, for the game list."""
    src = os.path.join(RESULTS, f"score_{g.get('seed')}.png")
    if not g.get("seed") or not os.path.exists(src):
        return None
    os.makedirs(IMG, exist_ok=True)
    name = f"score_{g['game']}.jpg"
    dst = os.path.join(IMG, name)
    if not os.path.exists(dst):
        from PIL import Image
        im = Image.open(src).convert("RGB").crop((60, 250, 870, 810))
        im.thumbnail((480, 480))
        im.save(dst, quality=78)
    return f"img/stats/{name}"


def build(games):
    done = [g for g in games if played(g)]
    by_level = defaultdict(list)
    for g in games:
        by_level[g["level"]].append(g)

    # ---- per level
    level_rows = []
    for lv in LEVELS + sorted(set(by_level) - set(LEVELS)):
        gs = by_level.get(lv, [])
        if not gs:
            continue
        p = [g for g in gs if played(g)]
        w = sum(g["outcome"] == "win" for g in p)
        l = sum(g["outcome"] == "loss" for g in p)
        ab = sum(g["outcome"] in ("abandoned", "setup_failed", "unknown") for g in gs)
        margins = [g["me"] - g["opponent"] for g in p if g.get("me") is not None and g.get("opponent") is not None]
        turns = [g["panel_turns"][0] for g in p if g.get("panel_turns") and g["panel_turns"][0]]
        probs = sum(len([x for x in g.get("problems", []) if x["kind"] not in HANDLED]) for g in gs)
        attacks = sum(len([x for x in g.get("problems", []) if x["kind"] == "attack_answered"]) for g in gs)
        clean = [g for g in p if not g.get("bug")]
        cw = sum(g["outcome"] == "win" for g in clean)
        level_rows.append(dict(level=lv, games=len(gs), played=len(p), wins=w, losses=l, other=ab,
                               bugged=len(p) - len(clean), clean_winrate=f"{pct(cw, len(clean))} ({cw}/{len(clean)})",
                               winrate=pct(w, len(p)),
                               margin=f"{sum(margins) / len(margins):+.1f} (n={len(margins)})" if margins else "–",
                               turns=f"{sum(turns) / len(turns):.0f}" if turns else "–",
                               minutes=(lambda m: f"{sum(m) / len(m):.0f}" if m else "–")(
                                   [g["minutes"] for g in gs if g.get("minutes")]),
                               problems=f"{probs / len(gs):.1f}", attacks=f"{attacks / len(gs):.1f}"))

    # ---- per card
    card_rows = []
    prob_by_card = Counter()
    for g in games:
        for p in g.get("problems", []):
            if p["kind"] in ("unknown_prompt", "auto_resolve", "abandoned"):
                for c in cards_in_text(p.get("prompt", "")):
                    prob_by_card[c] += 1
    # Level-adjusted: each game's result minus the usual win rate at its level (bug-hit games left out),
    # so a card that only appeared against Hard isn't blamed for Hard being hard.
    clean_done = [g for g in done if not g.get("bug")]
    base = {}
    for lv in set(g["level"] for g in clean_done):
        gl = [g for g in clean_done if g["level"] == lv]
        base[lv] = sum(g["outcome"] == "win" for g in gl) / len(gl)
    for c in BASE_KINGDOM:
        with_c = [g for g in done if c in g.get("kingdom", [])]
        adj_g = [g for g in clean_done if c in g.get("kingdom", [])]
        adjusted = (sum((g["outcome"] == "win") - base[g["level"]] for g in adj_g) / len(adj_g)) if adj_g else None
        without = [g for g in done if c not in g.get("kingdom", [])]
        if not with_c and not prob_by_card[c]:
            continue
        wc = sum(g["outcome"] == "win" for g in with_c)
        wo = sum(g["outcome"] == "win" for g in without)
        known = not any(c in g.get("unknown_to_model", []) for g in games if c in g.get("kingdom", []))
        lv = Counter(g["level"] for g in with_c)
        card_rows.append(dict(card=c, n=len(with_c), win_with=pct(wc, len(with_c)), win_without=pct(wo, len(without)),
                              adjusted=adjusted,
                              delta=(wc / len(with_c) - wo / len(without)) if with_c and without else None,
                              problems=prob_by_card[c], known=known, attack=c in ATTACKS,
                              levels=", ".join(f"{k} {v}" for k, v in lv.items())))
    card_rows.sort(key=lambda r: (r["adjusted"] if r["adjusted"] is not None else 9, -r["problems"]))

    # ---- problems
    prob_counter = Counter()
    prompt_counter = Counter()
    for g in games:
        for p in g.get("problems", []):
            prob_counter[p["kind"]] += 1
            if p["kind"] == "unknown_prompt":
                prompt_counter[(p.get("prompt", ""), g["level"])] += 1

    # ---- findings (generated from the numbers above)
    findings = []
    if done:
        w = sum(g["outcome"] == "win" for g in done)
        findings.append(f"{len(games)} games so far ({len(done)} finished): {w} won, "
                        f"{sum(g['outcome'] == 'loss' for g in done)} lost.")
    for r in level_rows:
        if r["played"]:
            findings.append(f"{r['level']}: {r['wins']} of {r['played']} won ({r['winrate']}), "
                            f"about {r['turns']} turns per game, {r['attacks']} attacks answered and "
                            f"{r['problems']} problems per game.")
    worst = [r for r in card_rows if r["adjusted"] is not None and r["n"] >= 2 and r["adjusted"] < 0]
    if worst:
        findings.append("Below the level average when present (at least 2 games): " + "; ".join(
            f"{r['card']} {100 * r['adjusted']:+.0f} pts (n={r['n']})" for r in worst[:4]) + ".")
    if prompt_counter:
        top = prompt_counter.most_common(3)
        findings.append("Prompts it didn't understand: " + "; ".join(
            f"“{p}” ({lv}, {n}×)" for (p, lv), n in top) + ".")
    ab = [g for g in games if g.get("outcome") in ("abandoned", "setup_failed")]
    if ab:
        findings.append(f"{len(ab)} game(s) couldn't be finished by the autopilot (see the game list).")
    bugged = [g for g in games if g.get("bug")]
    if bugged:
        findings.append(f"{len(bugged)} game(s) were hurt by an autopilot bug, not by the model (marked in the game "
                        f"list; the bug was fixed afterwards). The 'without bug-hit games' column leaves them out.")
    findings.append("Samples are small: a few games per level. Treat card win rates as hints, not measurements.")
    return dict(level_rows=level_rows, card_rows=card_rows, prob_counter=prob_counter,
                prompt_counter=prompt_counter, findings=findings)


def analysis_html():
    """A written analysis (autopilot/stats/analysis.html), shown above the computed tables when present."""
    path = os.path.join(HERE, "stats", "analysis.html")
    return open(path).read() if os.path.exists(path) else ""


def render(games, s):
    nav = ('<nav class="top"><a class="brand" href="index.html">🃏 Dominion Autopilot</a><a href="index.html">Home</a>'
           '<a href="setup.html">Setup</a><a href="training.html">Training</a><a href="how.html">How it works</a><a href="run.html">Run it</a><a href="videos.html">Videos</a>'
           '<a class="on" href="stats.html">Stats</a><a href="podcast/index.html">Podcast</a><a href="lessons.html">Lessons</a></nav>')
    done = [g for g in games if played(g)]
    wins = sum(g["outcome"] == "win" for g in done)
    hardest = next((r["level"] for r in reversed(s["level_rows"])), "–")
    out = [f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dominion Autopilot: Stats</title><link rel="stylesheet" href="style.css"><link rel="stylesheet" href="site.css">
</head><body>{nav}<main>
<h1>Stats</h1>
<p class="lede">The autopilot (model <code>{esc(games[-1].get('model', 'k1-steam')) if games else 'k1-steam'}</code>) plays the
built-in computer opponents, from Very Easy up to Hard, on the game's random <b>base-set</b> kingdoms. After every game this
page is rebuilt from <code>autopilot/stats/games.jsonl</code>: who won, by how much, and what the model had trouble with.
Updated {esc(games[-1]['date']) if games else '–'}.</p>
<div class="cards">
<div class="card"><div class="k">Games</div><div class="v">{len(games)}</div><div class="s">{len(done)} finished</div></div>
<div class="card"><div class="k">Won</div><div class="v">{wins}</div><div class="s">{pct(wins, len(done))} of finished</div></div>
<div class="card"><div class="k">Hardest level so far</div><div class="v" style="font-size:1.2rem">{esc(hardest)}</div></div>
<div class="card"><div class="k">Problems logged</div><div class="v">{sum(v for k, v in s['prob_counter'].items() if k not in HANDLED)}</div><div class="s">{esc(', '.join(f'{k} {v}' for k, v in s['prob_counter'].most_common() if k not in HANDLED))}</div></div>
</div>
{analysis_html()}
<h2>Findings so far (computed)</h2><div class="box"><ul>{''.join(f'<li>{esc(f)}</li>' for f in s['findings'])}</ul></div>
<h2>By opponent level</h2>
<div class="tablewrap"><table class="t"><tr><th>Level</th><th>Games</th><th>Won</th><th>Lost</th><th>Not finished</th>
<th>Win rate</th><th>Win rate without bug-hit games</th><th>Avg turns</th><th>Avg minutes</th><th>Attacks answered / game</th><th>Problems / game</th></tr>"""]
    for r in s["level_rows"]:
        out.append(f"<tr><td>{esc(r['level'])}</td><td class='num'>{r['games']}</td><td class='num'>{r['wins']}</td>"
                   f"<td class='num'>{r['losses']}</td><td class='num'>{r['other']}</td><td class='num'>{r['winrate']}</td>"
                   f"<td class='num'>{r['turns']}</td><td class='num'>{r['minutes']}</td>"
                   f"<td class='num'>{r['attacks']}</td><td class='num'>{r['problems']}</td></tr>")
    out.append("""</table></div>
<p class="small muted">Win or loss comes from the score screen's placement laurels (always read correctly so far).
Turns = the score panel's turn count. Problems exclude hand misreads that a second read fixed and attacks answered normally
(those are counted in their own column). What counts: unknown prompts, generic answers, selections that needed
adjusting, unreadable counters, empty piles, missing buttons, games given up.</p>
<h2>By kingdom card</h2>
<p>Each base-set card that has appeared: win rate in finished games with it in the kingdom vs without it, and how many
problems named it. <b>vs level average</b>: how much more or less often it won than the usual win rate at
the same opponent levels (games hit by autopilot bugs left out), so a card that only turned up against Hard isn't
blamed for Hard being hard. Sorted by that (worst first). <b>unknown</b> = the model never learned the card and never buys it.</p>
<div class="tablewrap"><table class="t"><tr><th>Card</th><th>Games with it</th><th>Win with</th><th>Win without</th><th>vs level average</th>
<th>Problems naming it</th><th>Model</th><th>Levels</th></tr>""")
    for r in s["card_rows"]:
        tag = "attack" if r["attack"] else ""
        out.append(f"<tr><td>{esc(r['card'])}{' <span class=tag>attack</span>' if tag else ''}</td><td class='num'>{r['n']}</td>"
                   f"<td class='num'>{r['win_with']}</td><td class='num'>{r['win_without']}</td>"
                   f"<td class='num'>{'–' if r['adjusted'] is None else f'{100 * r[chr(97) + "djusted"]:+.0f} pts'}</td>"
                   f"<td class='num'>{r['problems']}</td><td>{'knows' if r['known'] else 'unknown'}</td>"
                   f"<td class='small'>{esc(r['levels'])}</td></tr>")
    out.append("</table></div><h2>Prompts it didn't understand</h2>")
    if s["prompt_counter"]:
        out.append("<div class='tablewrap'><table class='t'><tr><th>Prompt (as read)</th><th>Level</th><th>Times</th></tr>")
        for (p, lv), n in s["prompt_counter"].most_common():
            out.append(f"<tr><td>{esc(p)}</td><td>{esc(lv)}</td><td class='num'>{n}</td></tr>")
        out.append("</table></div>")
    else:
        out.append("<p class='muted'>None yet.</p>")
    out.append("<h2>Every game</h2>")
    for g in reversed(games):
        img = score_image(g)
        probs = g.get("problems", [])
        res = g.get("outcome", "?")
        pts = f"{g.get('me') if g.get('me') is not None else '?'}–{g.get('opponent') if g.get('opponent') is not None else '?'}"
        out.append(f"""<div class="ev"><div class="body"><div class="t">Game {g['game']} · {esc(g['date'])} · {g.get('minutes', '–')} min</div>
<h4>vs {esc(g['level'])} AI: <span class="result {'win' if res == 'win' else ''}">{esc(res)}</span> <span class="small muted">points ≈ {pts} (OCR; the screen below is the record)</span></h4>
<p><b>Kingdom:</b> {esc(', '.join(g.get('kingdom', [])))}<br>
<b>Model doesn't know:</b> {esc(', '.join(g.get('unknown_to_model', [])) or 'none')}<br>
<b>Bought:</b> {esc(', '.join(f'{v} {k}' for k, v in sorted((g.get('owned') or {}).items(), key=lambda kv: -kv[1])))}</p>
{'<ul>' + ''.join(f'<li>turn {p.get("turn")}: {esc(problem_label(p))}</li>' for p in probs[:12]) + '</ul>' if probs else '<p class="muted">No problems.</p>'}
{f'<p class="small muted">{esc(g.get("error", ""))}</p>' if g.get('error') else ''}
{f'<p class="small muted">Note: {esc(g["note"])}</p>' if g.get('note') else ''}
{f'<p><span class="tag crash">autopilot bug</span> {esc(g["bug"])}</p>' if g.get('bug') else ''}</div>
{f'<img src="{img}" alt="score screen of game {g["game"]}">' if img else ''}</div>""")
    out.append("""<p class="small muted">The point shields are hard for the OCR (digits get dropped or changed), so points are shown as
approximate; win or loss comes from the placement laurels, and every game's score screen is shown.</p>
</main><footer>Dominion Autopilot · statistics from games against the built-in computer opponents only.</footer></body></html>""")
    return "\n".join(out)


def main():
    games = load()
    s = build(games)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(render(games, s))
    os.makedirs(os.path.join(ROOT, "docs", "data"), exist_ok=True)
    json.dump({"games": games, "levels": s["level_rows"], "cards": s["card_rows"], "findings": s["findings"]},
              open(os.path.join(ROOT, "docs", "data", "stats.json"), "w"), indent=1)
    print(f"stats: {len(games)} games -> {OUT}")


if __name__ == "__main__":
    main()
