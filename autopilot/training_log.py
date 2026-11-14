"""Build docs/data/training.json for the site's Training log page (docs/training.html).

    python autopilot/training_log.py                        # runs from ../trainers/ga-v1/runs
    python autopilot/training_log.py --runs PATH --push     # also commit + push the data file

Reads every specialist run's progress.json (ga/specialist.py in the GA trainer) and the
hand-written log entries in docs/data/training_notes.json, and writes one JSON the page
renders. --push commits only docs/data/training*.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
DEFAULT_RUNS = os.path.join(os.path.dirname(ROOT), "trainers", "ga-v1", "runs")
NOTES = os.path.join(DOCS, "data", "training_notes.json")
OUT = os.path.join(DOCS, "data", "training.json")


def runs(path: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(path, "*", "progress.json"))):
        p = json.load(open(f))
        hist = p.get("history", [])
        out.append({"name": p["name"], "must": p.get("must", []), "banned": bool(p.get("banned")),
                    "gen": p.get("gen", 0), "gens": p.get("gens"), "best": p.get("best_val"),
                    "bots": hist[-1]["bots"] if hist else {},
                    "curve": [[h["gen"], h["best"]] for h in hist],
                    "done": bool(p.get("done")), "updated": p.get("updated") or p.get("started"),
                    "minutes": round(p.get("elapsed_s", 0) / 60, 1)})
    return out


def build(runs_path: str) -> dict:
    notes = json.load(open(NOTES)) if os.path.exists(NOTES) else {"entries": []}
    data = {"generated": time.strftime("%Y-%m-%d %H:%M"), "runs": runs(runs_path), **notes}
    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    return data


def push(message: str) -> None:
    files = [os.path.relpath(OUT, ROOT), os.path.relpath(NOTES, ROOT)]
    subprocess.run(["git", "add", *files], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--", *files], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "-m", message, "--", *files], cwd=ROOT, check=True)
    subprocess.run(["git", "pull", "--rebase", "--autostash", "-q"], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "-q"], cwd=ROOT, check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args(argv)
    d = build(a.runs)
    done = sum(r["done"] for r in d["runs"])
    print(f"{OUT}: {len(d['runs'])} runs ({done} done), {len(d.get('entries', []))} log entries")
    if a.push:
        push(f"Training log: {done}/{len(d['runs'])} specialists done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
