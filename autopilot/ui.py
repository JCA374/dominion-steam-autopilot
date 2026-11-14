"""Control panel: pick the kingdom, the difficulty and the model (or let it choose), then play.

    .venv/bin/python autopilot/ui.py            # http://127.0.0.1:8642
    .venv/bin/python autopilot/ui.py --port 9000

A local web page (stdlib server, no extra packages). Everything it lists comes from the model
folders when the page asks (dominion-v1/models/ and models/), so a new model folder shows up
on reload, with no change here. "Play" starts autopilot/campaign.py with the chosen setup,
"Benchmark new models" starts autopilot/bench_models.py; their output is shown on the page.
It only clicks in the game window while a run started here is going.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from brain import AUTO, best_kingdom, fmt_keys, model_rows, rank_models  # noqa: E402
from campaign import LEVELS, STATS, STEAM_BASE_SET, auto_level  # noqa: E402

PAGE = os.path.join(HERE, "ui.html")


class Job:
    """One child process at a time (a game run or a benchmark), output kept in memory."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.what = ""
        self.lines: deque[str] = deque(maxlen=400)
        self.lock = threading.Lock()

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, what: str, cmd: list[str]):
        with self.lock:
            if self.running():
                raise RuntimeError(f"already running: {self.what}")
            self.what = what
            self.lines.clear()
            self.lines.append("$ " + " ".join(cmd[1:]))
            self.proc = subprocess.Popen(cmd, cwd=os.path.dirname(HERE), stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc):
        for line in proc.stdout:
            if "Warning" not in line:          # onnxruntime chatter
                self.lines.append(line.rstrip())
        self.lines.append(f"[{self.what} finished, exit code {proc.wait()}]")

    def stop(self):
        if self.running():
            self.proc.terminate()
            self.lines.append(f"[{self.what}: stop requested]")


JOB = Job()


def model_stats() -> dict:
    """Real Steam results per model from autopilot/stats/games.jsonl."""
    out = {}
    if os.path.exists(STATS):
        for line in open(STATS):
            r = json.loads(line)
            if r.get("outcome") in ("win", "loss", "tie") and r.get("model"):
                s = out.setdefault(r["model"], {"win": 0, "loss": 0, "tie": 0})
                s[r["outcome"]] += 1
    return out


def state() -> dict:
    models = model_rows()
    try:
        bm, bk = best_kingdom()
        best = {"model": bm["name"], "kingdom": bk}
    except LookupError:
        best = None
    return {"models": models, "cards": STEAM_BASE_SET, "levels": LEVELS, "best": best,
            "auto_level": {m["name"]: auto_level(m["name"]) for m in models} | {AUTO: auto_level(None)},
            "steam": model_stats(), "running": JOB.running(), "job": JOB.what}


def preview(kingdom_mode: str, cards: list[str], model: str) -> dict:
    """Who would play, and why, for the current choices."""
    if model != AUTO:
        return {"note": f"{model}: chosen by hand", "ranking": []}
    if kingdom_mode == "auto":
        bm, bk = best_kingdom()
        cards = bk
        note = f"Auto kingdom = the training kingdom of {bm['name']}, the best model on its own kingdom"
    elif kingdom_mode == "random" or not cards:
        return {"note": "The game deals the kingdom; the selector picks the model right after the deal "
                        "(exact training kingdom, else key cards dealt + best benchmark).", "ranking": []}
    else:
        note = (f"{len(cards)} of 10 cards chosen; the game deals the other {10 - len(cards)}, "
                "so this ranking can still change" if len(cards) < 10 else "All 10 cards chosen")
    ranking = [{k: r[k] for k in ("name", "tier", "score", "reason", "missing")}
               | {"key_cards": fmt_keys(r["key_cards"])} for r in rank_models(cards)]
    return {"note": note, "ranking": ranking, "kingdom": cards}


def start_games(body: dict):
    mode = body.get("kingdom_mode", "random")
    kingdom = {"random": "random", "auto": "auto"}.get(mode) or ",".join(body.get("cards", []))
    if not kingdom:
        kingdom = "random"
    level = body.get("level", "auto")
    if level != "auto" and level not in LEVELS:
        raise ValueError(f"unknown level {level!r}")
    games = max(1, min(50, int(body.get("games", 1))))
    cmd = [sys.executable, "-u", os.path.join(HERE, "campaign.py"), "--games", str(games),
           "--levels", level, "--kingdom", kingdom, "--model", body.get("model", AUTO)]
    if not body.get("publish"):
        cmd.append("--no-publish")
    JOB.start(f"{games} game(s)", cmd)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code: int, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send(200, open(PAGE, "rb").read(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self.send(200, state())
        elif self.path == "/api/log":
            self.send(200, {"running": JOB.running(), "job": JOB.what, "lines": list(JOB.lines)})
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/preview":
                self.send(200, preview(body.get("kingdom_mode", "random"), body.get("cards", []),
                                       body.get("model", AUTO)))
            elif self.path == "/api/start":
                start_games(body)
                self.send(200, {"ok": True})
            elif self.path == "/api/bench":
                cmd = [sys.executable, "-u", os.path.join(HERE, "bench_models.py")]
                JOB.start("benchmark", cmd + (["--all"] if body.get("all") else []))
                self.send(200, {"ok": True})
            elif self.path == "/api/stop":
                JOB.stop()
                self.send(200, {"ok": True})
            else:
                self.send(404, {"error": "not found"})
        except Exception as e:
            self.send(400, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8642)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Dominion autopilot control panel: http://127.0.0.1:{a.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        JOB.stop()


if __name__ == "__main__":
    main()
