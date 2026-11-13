# autopilot/

See the top-level README and the site's "How it works" page. Files:

- `pilot.py`: the loop, prompt handlers, memory, rematch, options (`--help`)
- `vision.py`: regions, OCR, name matching
- `brain.py`: decisions from the evolved strategy (dominion-v1)
- `grab.py`, `click.py`: screenshot and click on the XWayland window
- `recorder.py`: `--record DIR` (frames + events)
- `debug_screen.py`: print what the vision layer reads from a screenshot
