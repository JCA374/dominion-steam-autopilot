"""Benchmark runner for the autopilot: score every model (dominion-v1/models and this repo's
models/) and store the result in each model's folder (bench.json), where the model selector
reads it.

    .venv/bin/python autopilot/bench_models.py              # only models with no up-to-date bench.json
    .venv/bin/python autopilot/bench_models.py --all        # every model again
    .venv/bin/python autopilot/bench_models.py my-model     # just these

It runs dominion-v1's `python -m policy.bench --store` (vs Big Money, Smithy-BM and Witch-BM on
Steam-like kingdoms that hold the model's key cards) with this repo's models/ folder added.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from brain import REPO  # noqa: E402  (also puts models/ on DOMINION_MODELS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--all", action="store_true", help="re-run models that already have results")
    a = ap.parse_args(argv)
    cmd = [sys.executable, "-u", "-m", "policy.bench", "--store", *a.models]
    if not a.all:
        cmd.append("--missing")
    return subprocess.run(cmd, cwd=REPO, env=os.environ.copy()).returncode


if __name__ == "__main__":
    sys.exit(main())
