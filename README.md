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

# start Dominion from Steam, begin a game against the computer, then:
.venv/bin/python autopilot/pilot.py
```

Keep the game window at its normal size and in plain view, and keep your hands off the mouse while it plays.

| Option | |
|---|---|
| `--games N` | play N games, pressing Rematch in between |
| `--turn N` | joining a game already on turn N |
| `--dry` | read and decide, never click |
| `--record DIR` | save frames and events (the films are built from these) |
| `--model NAME` | another released dominion-v1 model (or a path); default `dominion-v1/models/DEFAULT` |
| `--list-models` | list the released models and the kingdom each was trained on |

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

## New models

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
| `autopilot/` | the autopilot |
| `dominion-v1/` | submodule: the genetic algorithm, its engine, the released models (`models/`) and the `policy` protocol |
| `video/` | film scripts and builders (`common.py`, `build_*.py`); every frame comes from a recording |
| `podcast/` | podcast episode scripts (audio in `docs/podcast/`) |
| `docs/` | the GitHub Pages site |

Not affiliated with the makers of Dominion. Dominion is played through the official Steam client; nothing in the
game is modified or redistributed.
