# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An evolved Dominion strategy plays the **Steam** version of Dominion (built-in computer opponents only, never online
players). Steam Dominion has no bot API, so the autopilot works like a person: X11 screenshot of the XWayland window →
RapidOCR on fixed screen regions → ask a released model what to do → XTEST click. Repo:
`JCA374/dominion-steam-autopilot`; live site (GitHub Pages from `docs/`): https://jca374.github.io/dominion-steam-autopilot/

`dominion-v1/` is a **git submodule** (`JCA374/dominion-v1`): the genetic algorithm, the released models (`models/`)
and the `policy` protocol. It has its own submodule `dominion-v1/engine/` (`JCA374/dominion-engine`, the game with
Steam's rules), so clone/update with `--recursive` / `git submodule update --init --recursive`. `../trainers/ga-v1`
is a separate checkout of dominion-v1 (where training runs) and `../engine` of the engine; neither is part of this repo.

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
.venv/bin/python autopilot/advisor.py                # advisor window beside the game while YOU play (never clicks)
.venv/bin/python autopilot/advisor.py --frames recordings/session2   # the advisor on a recording (testing)
.venv/bin/python autopilot/advisor.py --install      # "Dominion Advisor" in the app launcher
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
  `--model auto` (default) asks `policy.rank_models(kingdom)` after the deal: a `ga-portfolio` model first, else exact training kingdom, else key
  cards all dealt + best `bench.json` score, else most key cards dealt; never Big Money; `"retired"` models skipped.
- **Models are plug-ins**: folders in `dominion-v1/models/` and `models/` (brain.py adds the latter to
  `DOMINION_MODELS`). Nothing in the autopilot or UI lists models by name; `brain.model_rows()` reads them all.
- **Advisor** (`advisor.py` + `advisor.html`, port 8643) – for a person playing: reads the screen (vision.py; pile
  counts, HUD and scores by recognition-only OCR on fixed spots, other regions re-read only when their pixels change),
  keeps the game (`Tracker`: my cards = start + supply drops not on the opponent's "Added:" bar - trashes; turns from
  buy → action/treasure prompts; pile counts only from uncovered piles, ≤2 cards per confirmed change; deck check
  draw+discard+hand vs counted each turn; game log + tracker per save in `autopilot/advisor_logs/`, so a restart
  carries on; opponent plays are not logged — the prompt line shows my next turn during theirs) and asks the rl-v3 Play page's advisor net (`../trainers/rl-v3/docs/playvs/`: spec.json,
  models/, py/rlplay.py; override with `DOMINION_ADVISOR_DIR`) for the win chance and top answers. Opens a Chromium
  app window right of the game (sizes divided by Xft.dpi/96).
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
# in dominion-v1 (or ../trainers/ga-v1): release training output as an immutable, named model
python -m policy.release best_model <name> --description "..." && git add models && git commit && git push
# here
git -C dominion-v1 pull origin main && git -C dominion-v1 submodule update --init
git add dominion-v1 && git commit -m "dominion-v1: <name>"
.venv/bin/python autopilot/bench_models.py           # so the selector has its score
.venv/bin/python autopilot/pilot.py --model <name>
```

Or skip the release: drop the model folder into `models/` (see `models/README.md`) and run `bench_models.py`.

`dominion-v1/models/DEFAULT` is `portfolio-v2` (kind `ga-portfolio`, since 2026-11-15): 26 specialists trained on
random kingdoms with all 26 cards against the bots and a hall of strong specialists; at game start it plays each on the
dealt kingdom in the engine (~2 s) and the best one plays (pilot logs `portfolio plays <member>`). Protocol 2: the
pilot handles every card's own prompt (`card_<name>` handlers in pilot.py, dispatched from a '<Card>:' prompt line,
the card just played, or the rightmost big card; Sentry is drag and drop via `click.drag`). Moat reveal and the Bandit
victim's choice were never met on Steam yet. `k1-steam` is the model behind the
recorded wins. Training progress: `docs/training.html` from `autopilot/training_log.py` (reads
`../trainers/ga-v1/runs/*/progress.json` + hand-written `docs/data/training_notes.json`; `--push` commits only those). Non-GA models (e.g. a neural net in
its own venv) can be `subprocess` models speaking the same protocol over stdin/stdout, with no autopilot changes.

## Conventions

- The user commits to this repo in parallel (podcast, video, site files): commit only files you changed.
- Run records (`autopilot/run_*.json`, `results/`, `unknown/`, `*.log`), `recordings/`, `video/build/` and `voice/`
  are gitignored.
