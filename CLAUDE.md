# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An evolved Dominion strategy plays the **Steam** version of Dominion (built-in computer opponents only, never online
players). Steam Dominion has no bot API, so the autopilot works like a person: X11 screenshot of the XWayland window →
RapidOCR on fixed screen regions → ask a released model what to do → XTEST click. Repo:
`JCA374/dominion-steam-autopilot`; live site (GitHub Pages from `docs/`): https://jca374.github.io/dominion-steam-autopilot/

`dominion-v1/` is a **git submodule** (`JCA374/dominion-v1`): the genetic algorithm, its Python/C engine, the released
models (`models/`) and the `policy` protocol. It has its own README/HOWTO. `../dominion-GA1` is a separate checkout of
the same dominion-v1 repo (where training runs); it is not part of this repo.

## Commands

All Python runs through the repo venv (`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`).
There is no test suite here; dominion-v1 has its own (`cd dominion-v1 && python -m pytest -q tests`).

```bash
.venv/bin/python autopilot/ui.py                     # control panel http://127.0.0.1:8642: kingdom/difficulty/model or Auto, Play
.venv/bin/python autopilot/bench_models.py [--all]   # benchmark runner: bench.json per model (the selector's scores)
.venv/bin/python autopilot/pilot.py                  # play the current game (Dominion must be open in Steam)
.venv/bin/python autopilot/pilot.py --dry            # read + decide, never click (safe for testing vision/brain)
.venv/bin/python autopilot/pilot.py --once           # handle one prompt and exit
.venv/bin/python autopilot/pilot.py --games 3 --record recordings/x --model k4-20260614 --list-models --auto
.venv/bin/python autopilot/debug_screen.py [shot.png]  # print what vision reads from a screenshot or the live window
.venv/bin/python autopilot/campaign.py --per 3       # games vs Very Easy..Hard AI; commits+pushes Stats page unless --no-publish
.venv/bin/python autopilot/campaign.py --games 1 --levels auto --kingdom auto|random|"Smithy,Witch" --model auto --no-publish
autopilot/stop_campaign.sh                           # stop a running campaign
.venv/bin/python autopilot/stats_report.py           # rebuild docs/stats.html + docs/data/stats.json
.venv/bin/python video/build_game.py [--only N] [--preview] [--resume]   # also build_run.py, build_works.py
.venv/bin/python video/stills.py                     # site stills (docs/img/) from recordings
```

Anything pilot.py doesn't recognise is saved to `autopilot/unknown/` and it waits (unless `--auto`).
Running the pilot, a campaign or Play in the UI clicks in the real game window: only do it when the user asks.

## Architecture

- **`autopilot/pilot.py`** – the main loop: one prompt per pass, only acting on a still screen; prompt handlers,
  per-game memory file, empty-pile detection, score screen, rematch. Reads the kingdom from the game's own save file
  (`gamelocal_*.xml` inside the Proton prefix, app id 1131620; several Steam install roots are tried).
- **`vision.py`** – all screen coordinates are for a reference window size (`REF_W x REF_H`) and scaled to the live
  window; OCR text is fuzzy-matched to card names. **`grab.py` / `click.py`** – screenshot and click (clicks hover
  first; the game ignores instant clicks). **`recorder.py`** – `--record DIR` writes `frames/<ms>.jpg` +
  `events.jsonl` from a second X connection on a thread.
- **`brain.py`** – tracks turn/deck/Provinces/empty piles and asks the model. **The autopilot never reaches into
  dominion-v1 internals**: it imports only `policy` (`load_policy`, `observation`, `list_models`, `cards`) and checks
  `PROTOCOL_VERSION` against `NEEDS_PROTOCOL`. Questions are JSON (`action`, `buy`, `trash`, `discard`,
  `throne_room`), specified in `dominion-v1/docs/PROTOCOL.md`. Cards a model never learned are never bought.
  `--model auto` (default) asks `policy.rank_models(kingdom)` after the deal: exact training kingdom, else key
  cards all dealt + best `bench.json` score, else most key cards dealt; never Big Money; `"retired"` models skipped.
- **Models are plug-ins**: folders in `dominion-v1/models/` and `models/` (brain.py adds the latter to
  `DOMINION_MODELS`). Nothing in the autopilot or UI lists models by name; `brain.model_rows()` reads them all.
- **Control panel** – `ui.py` (stdlib HTTP server + `ui.html`) runs `campaign.py` / `bench_models.py` as child
  processes and shows their output. Auto kingdom = `policy.best_kingdom()` (best model's training kingdom); auto
  difficulty = easiest level the model hasn't won 2 of its last 3 games at (`campaign.auto_level`).
- **Campaign/stats** – `campaign.py` sets up each game via the menus (kingdom: Kingdom > Cards > Clear, type each
  name + Enter into "Specify Card", read the slots back), runs pilot in `--auto` mode, appends to
  `autopilot/stats/games.jsonl`; `stats_report.py` builds the Stats tab only from that file
  (`autopilot/stats/analysis.html` is the hand-written summary embedded in it).
- **Films** (`video/`) – a film is a list of scene functions yielding 1280×720 PIL frames; scene k is paired with
  narration `voice/<slug>/audio/voiceover/chKK.mp3` and follows `video/<film>/script.md` one to one. Every overlay
  comes from a recording's events, never a re-enactment. Output goes to `docs/video/`.
- **Site** – `docs/*.html` are hand-written except `stats.html`. Podcast scripts live in `podcast/`, audio in
  `docs/podcast/`, config in `podcast.conf` (used by the `project-podcast` skill; pronunciations in
  `voice/glossary.md`).

## Using a new model

Models are released in dominion-v1, then pulled in here by bumping the submodule:

```bash
# in dominion-v1 (or ../dominion-GA1): release training output as an immutable, named model
python -m policy.release best_model <name> --description "..." && git add models && git commit && git push
# here
git -C dominion-v1 pull origin main && git add dominion-v1 && git commit -m "dominion-v1: <name>"
.venv/bin/python autopilot/bench_models.py           # so the selector has its score
.venv/bin/python autopilot/pilot.py --model <name>
```

Or skip the release: drop the model folder into `models/` (see `models/README.md`) and run `bench_models.py`.

`dominion-v1/models/DEFAULT` is `k1-steam`, the model behind the recorded wins. Non-GA models (e.g. a neural net in
its own venv) can be `subprocess` models speaking the same protocol over stdin/stdout, with no autopilot changes.

## Conventions

- The user commits to this repo in parallel (podcast, video, site files): commit only files you changed.
- Run records (`autopilot/run_*.json`, `results/`, `unknown/`, `*.log`), `recordings/`, `video/build/` and `voice/`
  are gitignored.
