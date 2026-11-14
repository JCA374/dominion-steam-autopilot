# Dominion Steam Autopilot

**🌐 Live site: https://jca374.github.io/dominion-steam-autopilot/**

An evolved Dominion strategy (from [dominion-v1](https://github.com/JCA374/dominion-v1), a genetic algorithm)
plays the **Steam version of Dominion** on its own. Steam Dominion has no bot interface, so the autopilot works like a
person: it takes screenshots of the game window, reads them with OCR, asks the evolved strategy what to do, and clicks.

Use it **only against the built-in computer opponents**, never against people online.

Recorded results so far, all against *Very Easy AI*: won 58–8, won 76–7 (after an empty-pile stall that led to a
fix), and more on the site.

## Quick start

```bash
git clone --recurse-submodules https://github.com/JCA374/dominion-steam-autopilot
cd dominion-steam-autopilot
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# start Dominion from Steam (main menu is fine), then open the control panel:
.venv/bin/python autopilot/ui.py          # http://127.0.0.1:8642
# or, with a game against the computer already running:
.venv/bin/python autopilot/pilot.py
```

The **control panel** (`autopilot/ui.py`) is a local web page: pick the kingdom (Auto, whatever the game deals, or
up to 10 cards of your choice), the difficulty (Auto or Very Easy..Hard) and the model (Auto or any model), see
which model would play and why, press Play. It sets up New Game (kingdom cards are typed into Kingdom > Cards),
plays, and records each game in `autopilot/stats/games.jsonl`. The same from the command line:
`autopilot/campaign.py --games 1 --levels auto --kingdom auto --model auto --no-publish`.

Keep the game window at its normal size and in plain view, and keep your hands off the mouse while it plays.

| Option | |
|---|---|
| `--games N` | play N games, pressing Rematch in between |
| `--turn N` | joining a game already on turn N |
| `--dry` | read and decide, never click |
| `--record DIR` | save frames and events (the films are built from these) |
| `--model NAME` | play this model (or a path); default `auto`: the model selector picks for the dealt kingdom |
| `--list-models` | list the models: key cards, benchmark score, training kingdom |

Needs Linux (built on GNOME/Wayland, where the game runs in XWayland), Steam with Dominion via Proton, Python 3.10+.

## How it works

- **Eyes** (`autopilot/grab.py`, `vision.py`): an X11 screenshot of the XWayland window (~50 ms), then RapidOCR on a
  few fixed regions: the prompt line, the hand, the actions/coins/buys counters, the buttons.
- **Brain** (`autopilot/brain.py`): keeps track of the deck, turn and Provinces and asks a released dominion-v1 model
  through its `policy` protocol (see below). GA models decide with dominion-v1's own engine functions, so they play
  exactly as in training. The kingdom is read from the game's save file (`gamelocal_*.xml`). Cards a model never
  learned are never bought.
- **Hands** (`autopilot/click.py`): XTEST clicks with a short hover first, because the game ignores instant clicks.
- **Loop** (`autopilot/pilot.py`): one prompt per pass, only on a still screen; a memory file per game; empty-pile
  detection; score screen and rematch.

## Which model plays

Every model knows its **key cards**, the kingdom cards its game is built around (e.g. 2× Witch + 2× Market +
1× Merchant), measured from its own games. With `--model auto` (the default) the selector in dominion-v1
(`policy.select_model`) picks, after the deal:

1. a model trained on exactly the dealt kingdom;
2. else the model with the best stored benchmark score among those whose key cards are all dealt, whatever the
   other cards are;
3. else the model with the largest share of its key cards dealt. Never a Big Money fallback.

The scores come from the **benchmark runner**, `autopilot/bench_models.py` (button *Benchmark new models* in the
control panel): every model against Big Money, Smithy-BM and Witch-BM on Steam-like kingdoms holding its key cards,
stored as `bench.json` in the model's folder.

## New models

Models are **plug-ins**: a model is a folder. Drop one into `models/` here (see `models/README.md`) or release it
into `dominion-v1/models/`, run `autopilot/bench_models.py`, and it shows up in the control panel and the selector;
no code changes.

The autopilot never reaches into dominion-v1's internals. It loads models with `policy.load_policy(name)` and asks
them JSON questions (`action`, `buy`, `trash`, `discard`, `throne_room`), as described in
[dominion-v1/docs/PROTOCOL.md](https://github.com/JCA374/dominion-v1/blob/main/docs/PROTOCOL.md). To play a newly
trained model:

```bash
# in dominion-v1: release the training output as an immutable, named model
python -m policy.release best_model k4-gen52000 --description "..." && git add models && git commit && git push
# here: pull it in and play it
git -C dominion-v1 pull origin main && git add dominion-v1 && git commit -m "dominion-v1: k4-gen52000"
.venv/bin/python autopilot/pilot.py --model k4-gen52000
```

A model that isn't a GA strategy (for example a neural net with its own dependencies) can be a `subprocess` model
that speaks the same protocol over stdin/stdout. The autopilot doesn't change.

Details, films, podcast and lessons: **https://jca374.github.io/dominion-steam-autopilot/**

## Repository

| Path | |
|---|---|
| `autopilot/` | the autopilot (`ui.py` control panel, `pilot.py` player, `campaign.py` game setup, `bench_models.py`) |
| `models/` | plug-in models (folders), next to `dominion-v1/models/` |
| `dominion-v1/` | submodule: the genetic algorithm, its engine, the released models (`models/`) and the `policy` protocol |
| `video/` | film scripts and builders (`common.py`, `build_*.py`); every frame comes from a recording |
| `podcast/` | podcast episode scripts (audio in `docs/podcast/`) |
| `docs/` | the GitHub Pages site |

Not affiliated with the makers of Dominion. Dominion is played through the official Steam client; nothing in the
game is modified or redistributed.
