"""The teacher: bv-v1's paired-rollout search (trainers/bv-v1, C bv_advise in bv.so) checks the advisor's pick at buys
and action plays. Every candidate (the advisor's pick, PASS and the five next best of RL-v3 c2) is played to the end of
the game on the same hidden-state draws, c2 continuing both players; the opponent's unseen cards are pooled and dealt
again (only what a player can know). It overrides the pick when the best candidate wins more often by more than
MARGIN and by more than Z standard errors (the paired difference): the settings of sg-v1's ablation, where c2 with
this teacher scored 58.8% against plain c2 (600 games). Each question gets at most SECS seconds.

Runs in a thread of its own (the C code is single-threaded: one question at a time); the newest question replaces a
waiting one.
"""
from __future__ import annotations

import ctypes
import json
import os
import threading
import time

import numpy as np

TEACHER_DIR = os.environ.get("DOMINION_TEACHER_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                                   "..", "trainers", "bv-v1"))
NET = "rl3_c2_2654.npy"
SECS = 3.0                       # the time cap per question
K0 = K1 = 128                    # hidden-state draws per candidate, as many as fit in SECS (sg-v1's ablation used
                                 # 32: +-9 points per option; 128 halves that, ~3 s for a buy with 11 options)
TOPN, MARGIN, Z = 5, 0.02, 2.0   # candidates beyond the pick and PASS; the override gate
MIN_DRAWS = 32                   # fewer draws in the time (a busy machine): no override
DUMP = os.environ.get("DOMINION_TEACHER_DUMP")    # a file: every question and answer as a JSON line (testing)
BUY, ACTION = 1, 0               # engine decision kinds (DE_BUY, DE_PLAY_ACTION)

_I, _F, _U8 = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_ubyte)


def _ints(xs):
    return (ctypes.c_int * len(xs))(*xs)


class Teacher:
    def __init__(self, on_result):
        """on_result(job, result): called from the teacher's thread when a question is graded."""
        so = os.path.join(TEACHER_DIR, "bv.so")
        self.lib = lib = ctypes.CDLL(so)
        lib.bv_advise.restype = ctypes.c_int
        lib.bv_advise.argtypes = [_F, ctypes.c_int, ctypes.c_int, _I, ctypes.c_int, _I, _I, _I, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                  ctypes.c_uint64, _F, _F, _U8, _F]
        lib.bv_set_cfg.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float]
        lib.rl_num_params.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.bv_set_cfg(K0, K1, 1.5, TOPN, MARGIN)
        self.na = lib.rl_num_actions()
        self.prm = np.ascontiguousarray(np.load(os.path.join(TEACHER_DIR, "models", NET)).astype(np.float32))
        self.nh, self.nf = self._shape(self.prm)
        self.on_result = on_result
        self.cache: dict = {}                    # job key -> result (the same question asked again: no new rollouts)
        self.job = None
        self.cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True, name="teacher").start()

    def _shape(self, p):
        for nf in (self.lib.rl_num_features(), self.lib.rl_num_features_base()):
            b = nf + self.na + 4
            nh = int(round((-b + (b * b + 4 * (p.size - self.na - 1)) ** 0.5) / 2))
            if nh > 0 and self.lib.rl_num_params(nh, nf) == p.size:
                return nh, nf
        raise ValueError(f"{NET}: {p.size} parameters fit no net shape")

    def ask(self, job):
        """job: {"key", "supply", "me", "zones" (8 lists), "turns", "kind", "coins", "buys", "actions", "merchants",
        "inc"}. Returns the cached result at once if this question was graded before, else None (graded later)."""
        if job["key"] in self.cache:
            return self.cache[job["key"]]
        with self.cv:
            self.job = job
            self.cv.notify()
        return None

    def _run(self):
        while True:
            with self.cv:
                while self.job is None:
                    self.cv.wait()
                job, self.job = self.job, None
            try:
                r = self.grade(job)
            except Exception as e:               # never take the advisor down
                r = {"draws": -1, "error": repr(e)}
            if len(self.cache) > 500:
                self.cache.clear()
            self.cache[job["key"]] = r
            if DUMP:
                with open(DUMP, "a") as f:
                    f.write(json.dumps({"job": {k: v for k, v in job.items() if k != "key"}, "result": r}) + "\n")
            self.on_result(job, r)

    def grade(self, job):
        na = self.na
        q, se = np.full(na, np.nan, np.float32), np.full(na, np.nan, np.float32)
        cand, out = np.zeros(na, np.uint8), np.zeros(5, np.float32)
        flat = [c for z in job["zones"] for c in z]
        t = time.time()
        k = self.lib.bv_advise(self.prm.ctypes.data_as(_F), self.nh, self.nf, _ints(job["supply"]), job["me"],
                               _ints(flat or [0]), _ints([len(z) for z in job["zones"]]), _ints(job["turns"]),
                               job["kind"], job["coins"], job["buys"], job["actions"], job["merchants"], job["inc"],
                               SECS, int(t * 1e6) & (2**64 - 1), q.ctypes.data_as(_F), se.ctypes.data_as(_F),
                               cand.ctypes.data_as(_U8), out.ctypes.data_as(_F))
        inc, pas = int(out[0]), na - 1
        r = {"draws": int(k), "secs": round(time.time() - t, 2), "inc": inc}
        if k < 1:
            return r
        opts = sorted((int(a) for a in np.where(cand)[0]), key=lambda a: -q[a])
        b = opts[0]
        gain = float(q[b] - q[inc])
        override = (b != inc and k >= MIN_DRAWS and gain > MARGIN and gain > Z * float(se[b]))
        r.update({"best": b, "gain": round(gain, 4), "se": round(float(se[b]), 4), "override": bool(override),
                  "pick": b if override else inc, "pass": pas,
                  "q": [[a, round(float(q[a]), 4)] for a in opts]})
        return r
