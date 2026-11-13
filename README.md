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
| `--model FILE` | another strategy file |

Needs Linux (built on GNOME/Wayland, where the game runs in XWayland), Steam with Dominion via Proton, Python 3.10+.

## How it works

- **Eyes** (`autopilot/grab.py`, `vision.py`): an X11 screenshot of the XWayland window (~50 ms), then RapidOCR on a
  few fixed regions: the prompt line, the hand, the actions/coins/buys counters, the buttons.
- **Brain** (`autopilot/brain.py`): the evolved `strategy.json`. Buys are made by calling dominion-v1's own
  `play_buy_phase` on a stand-in game state, so they match training exactly. The kingdom is read from the game's
  save file (`gamelocal_*.xml`). Cards the strategy never learned are never bought.
- **Hands** (`autopilot/click.py`): XTEST clicks with a short hover first, because the game ignores instant clicks.
- **Loop** (`autopilot/pilot.py`): one prompt per pass, only on a still screen; a memory file per game; empty-pile
  detection; score screen and rematch.

Details, films, podcast and lessons: **https://jca374.github.io/dominion-steam-autopilot/**

## Repository

| Path | |
|---|---|
| `autopilot/` | the autopilot |
| `dominion-v1/` | submodule: the genetic algorithm, its engine and the evolved strategy |
| `video/` | film scripts and builders (`common.py`, `build_*.py`); every frame comes from a recording |
| `podcast/` | podcast episode scripts (audio in `docs/podcast/`) |
| `docs/` | the GitHub Pages site |

Not affiliated with the makers of Dominion. Dominion is played through the official Steam client; nothing in the
game is modified or redistributed.
