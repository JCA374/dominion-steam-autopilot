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
K0 = K1 = 128                    # hidden-state draws per candidate PER WORKER, as many as fit in SECS (sg-v1's ablation used
                                 # 32: +-9 points per option; 128 halves that, ~3 s for a buy with 11 options)
TOPN, MARGIN, Z = 5, 0.02, 2.0   # candidates beyond the pick and PASS; the override gate
MIN_DRAWS = 32                   # fewer draws in the time (a busy machine): no override
DUMP = os.environ.get("DOMINION_TEACHER_DUMP")    # a file: every question and answer as a JSON line (testing)
BUY, ACTION = 1, 0               # engine decision kinds (DE_BUY, DE_PLAY_ACTION)

_I, _F, _U8 = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_ubyte)


def _ints(xs):
    return (ctypes.c_int * len(xs))(*xs)


WORKERS = int(os.environ.get("DOMINION_TEACHER_WORKERS", "3"))   # processes grading each question together (pooled draws)


def _load(so, net):
    lib = ctypes.CDLL(so)
    lib.bv_advise.restype = ctypes.c_int
    lib.bv_advise.argtypes = [_F, ctypes.c_int, ctypes.c_int, _I, ctypes.c_int, _I, _I, _I, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                              ctypes.c_uint64, _F, _F, _U8, _F]
    lib.bv_set_cfg.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float]
    lib.rl_num_params.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.bv_set_cfg(K0, K1, 1.5, TOPN, MARGIN)
    prm = np.ascontiguousarray(np.load(net).astype(np.float32))
    na = lib.rl_num_actions()
    for nf in (lib.rl_num_features(), lib.rl_num_features_base()):
        b = nf + na + 4
        nh = int(round((-b + (b * b + 4 * (prm.size - na - 1)) ** 0.5) / 2))
        if nh > 0 and lib.rl_num_params(nh, nf) == prm.size:
            return lib, prm, nh, nf, na
    raise ValueError(f"{net}: {prm.size} parameters fit no net shape")


def _advise(lib, prm, nh, nf, na, job, secs, seed):
    """One process's share: bv_advise on the job -> (draws, q, se, cand, out) as numpy arrays."""
    q, se = np.full(na, np.nan, np.float32), np.full(na, np.nan, np.float32)
    cand, out = np.zeros(na, np.uint8), np.zeros(5, np.float32)
    flat = [c for z in job["zones"] for c in z]
    k = lib.bv_advise(prm.ctypes.data_as(_F), nh, nf, _ints(job["supply"]), job["me"], _ints(flat or [0]),
                      _ints([len(z) for z in job["zones"]]), _ints(job["turns"]), job["kind"], job["coins"],
                      job["buys"], job["actions"], job["merchants"], job["inc"], secs, seed & (2**64 - 1),
                      q.ctypes.data_as(_F), se.ctypes.data_as(_F), cand.ctypes.data_as(_U8), out.ctypes.data_as(_F))
    return int(k), q, se, cand, out


_W: dict = {}


def _winit(so, net):
    os.nice(5)                                   # below the advisor's own reading of the screen
    _W["m"] = _load(so, net)


def _wjob(args):
    job, secs, seed = args
    k, q, se, cand, out = _advise(*_W["m"], job, secs, seed)
    return k, q.tolist(), se.tolist(), cand.tolist(), out.tolist()


def pool_draws(parts):
    """Merge the processes' answers to one question into one, exactly: parts = [(K_i draws, q_i per-option means,
    se_i SEs of the paired difference with the incumbent, inc)], each over its own K_i draws. A part with K_i < 1 (no
    time, an error) is left out. -> (K, q, se) over all draws (K = 0, None, None if no part has a draw)."""
    parts = [p for p in parts if p[0] > 0]
    if not parts:
        return 0, None, None
    K = sum(p[0] for p in parts)
    q = sum(k * np.asarray(qi, np.float64) for k, qi, _, _ in parts) / K
    d = [np.asarray(qi, np.float64) - np.asarray(qi, np.float64)[inc] for _, qi, _, inc in parts]   # mean differences
    e2 = sum(k * (np.asarray(si, np.float64) ** 2 * k + di ** 2) for (k, _, si, _), di in zip(parts, d)) / K
    D = sum(k * di for (k, _, _, _), di in zip(parts, d)) / K       # E[x^2]_i = population variance + mean^2
    return K, q.astype(np.float32), np.sqrt(np.maximum(e2 - D ** 2, 0) / K).astype(np.float32)


class Teacher:
    def __init__(self, on_result):
        """on_result(job, result): called from the teacher's thread when a question is graded."""
        so, net = os.path.join(TEACHER_DIR, "bv.so"), os.path.join(TEACHER_DIR, "models", NET)
        self.lib, self.prm, self.nh, self.nf, self.na = _load(so, net)
        self.pool = None
        if WORKERS > 1:                          # each process deals its own draws; the answers are pooled exactly
            import multiprocessing as mp
            self.pool = mp.get_context("spawn").Pool(WORKERS, _winit, (so, net))
        self.on_result = on_result
        self.cache: dict = {}                    # job key -> result (the same question asked again: no new rollouts)
        self.job = None
        self.cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True, name="teacher").start()

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
        t = time.time()
        seed = int(t * 1e6)
        if self.pool is None:
            k, q, se, cand, out = _advise(self.lib, self.prm, self.nh, self.nf, na, job, SECS, seed)
        else:                                    # WORKERS processes, different deals, the same deadline
            parts = self.pool.map(_wjob, [(job, SECS, seed + 7919 * i) for i in range(WORKERS)])
            out = np.asarray(parts[0][4], np.float32)
            inc = int(out[0])
            cand = np.asarray(parts[0][3], np.uint8)
            if min(p[0] for p in parts) < 0:     # an engine error or a one-option position in any part
                k = min(p[0] for p in parts)
            else:
                k, q, se = pool_draws([(p[0], p[1], p[2], inc) for p in parts])
        inc, pas = int(out[0]), na - 1
        r = {"draws": int(k), "secs": round(time.time() - t, 2), "inc": inc, "workers": WORKERS}
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
