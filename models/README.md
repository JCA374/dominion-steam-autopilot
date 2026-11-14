# Plug-in models

Drop a model folder here and the autopilot, the model selector and the UI pick it up. No code
changes. This folder sits next to `dominion-v1/models/` (the released GA models) on dominion-v1's
model search path (`DOMINION_MODELS`, set in `autopilot/brain.py`). A name that already exists
in `dominion-v1/models/` wins over the same name here.

```
models/
  my-model/
    model.json       required: the manifest (dominion-v1/docs/PROTOCOL.md)
    strategy.json    for "kind": "ga-strategy"
    bench.json       written by autopilot/bench_models.py (or shipped, for kinds the engine can't play)
```

Minimal `model.json` for a GA strategy:

```json
{"protocol": 1, "name": "my-model", "kind": "ga-strategy", "file": "strategy.json",
 "kingdom": ["Chapel", "Festival", "..."],
 "key_cards": {"Smithy": 2, "Witch": 1},
 "description": "what it is"}
```

- `kingdom`: the 10 cards it was trained on. A model trained on exactly the dealt kingdom is
  always the selector's choice.
- `key_cards`: the kingdom cards its game is built around. The selector lets it play any
  kingdom holding all of them. Leave it out for a GA model and the benchmark runner measures
  it (`python -m policy.keycards` in dominion-v1 shows what it measures).
- A `subprocess` model (any program speaking the JSON-lines protocol) works too; it has to
  declare `key_cards` and ship a `bench.json` with a `"score"` (0-1) to be ranked fairly.

Then run `.venv/bin/python autopilot/bench_models.py` (or press *Benchmark new models* in the
UI) so the selector has its score.
