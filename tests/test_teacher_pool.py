"""The advisor's teacher with several processes: pooled draws are exactly the draws of one big sample (the merge
math), and a pooled answer on a real position has about WORKERS times the draws of one process in the same time."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "autopilot"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dominion-v1", "engine"))
import teacher  # noqa: E402


def test_pool_math_is_exact():
    rng = np.random.default_rng(1)
    na, inc = 6, 2
    V = rng.random((300, na)) < rng.random(na)          # per draw, per option: win 1 / loss 0
    V = V.astype(np.float64)

    def part(rows):
        q = rows.mean(0)
        d = rows - rows[:, [inc]]
        se = np.sqrt(np.maximum((d ** 2).mean(0) - d.mean(0) ** 2, 0) / len(rows))
        return len(rows), q, se, inc

    K, q, se = teacher.pool_draws([part(V[:50]), part(V[50:170]), part(V[170:])])
    _, q1, se1, _ = part(V)
    assert K == 300
    assert np.allclose(q, q1, atol=1e-6) and np.allclose(se, se1, atol=1e-6)
    assert teacher.pool_draws([(0, None, None, inc)]) == (0, None, None)


def _job():
    from dominion_engine.cards import ID, NUM_CARDS
    s = [-1] * NUM_CARDS
    for c, n in {"Copper": 46, "Silver": 40, "Gold": 30, "Estate": 8, "Duchy": 8, "Province": 8, "Curse": 10}.items():
        s[ID[c]] = n
    for c in ["Village", "Smithy", "Market", "Militia", "Moat", "Cellar", "Chapel", "Festival", "Laboratory", "Witch"]:
        s[ID[c]] = 10
    i = lambda *n: [ID[x] for x in n]  # noqa: E731
    start = i(*["Copper"] * 7, *["Estate"] * 3)
    return {"key": "t", "supply": s, "me": 0, "turns": [0, 0], "kind": teacher.BUY, "coins": 5, "buys": 1, "actions": 1,
            "merchants": 0, "inc": -1, "zones": [i("Copper", "Copper", "Estate", "Estate", "Estate"), [], [],
                                                 i(*["Copper"] * 5), start[:5], start[5:], [], []]}


def test_pooled_answer_has_more_draws():
    job = _job()
    teacher.SECS = 1.0
    one = teacher.Teacher.__new__(teacher.Teacher)
    one.lib, one.prm, one.nh, one.nf, one.na = teacher._load(os.path.join(teacher.TEACHER_DIR, "bv.so"),
                                                             os.path.join(teacher.TEACHER_DIR, "models", teacher.NET))
    one.pool = None
    r1 = one.grade(job)
    import multiprocessing as mp
    many = teacher.Teacher.__new__(teacher.Teacher)
    many.lib, many.prm, many.nh, many.nf, many.na = one.lib, one.prm, one.nh, one.nf, one.na
    many.pool = mp.get_context("spawn").Pool(3, teacher._winit, (os.path.join(teacher.TEACHER_DIR, "bv.so"),
                                                                 os.path.join(teacher.TEACHER_DIR, "models", teacher.NET)))
    many.pool.map(teacher._wjob, [(job, 0.05, i) for i in range(3)])     # warm: the processes have loaded
    t = time.time()
    r3 = many.grade(job)
    many.pool.terminate()
    print(r1["draws"], r3["draws"], r1["q"][:3], r3["q"][:3])
    assert r3["draws"] > r1["draws"]             # ~3x on an idle machine; less when every core is busy
    assert time.time() - t < teacher.SECS + 1.0
    assert r1["inc"] == r3["inc"]


def test_card_choice_pooled():
    """A Workshop gain graded through Teacher with ts-v1's ts.so and 3 processes: every gainable card is an option and
    the answer is pooled from all three."""
    ts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "trainers", "ts-v1")
    if not os.path.exists(os.path.join(ts_dir, "ts.so")):
        return
    from dominion_engine.cards import ID
    teacher.TEACHER_DIR, teacher.SO, teacher.SECS = ts_dir, "ts.so", 1.0
    job = _job()
    hand = [ID["Workshop"], ID["Copper"], ID["Copper"], ID["Estate"], ID["Copper"]]
    opts = sorted(c for c, n in enumerate(job["supply"]) if n > 0 and c in
                  [ID[x] for x in ("Copper", "Silver", "Estate", "Curse", "Village", "Smithy", "Militia", "Moat", "Cellar", "Chapel")])
    job.update(kind=0, coins=0, actions=1, inc=ID["Silver"], zones=[job["zones"][0], hand, [], [], *job["zones"][4:]],
               sub={"card": ID["Workshop"], "hk": [], "ho": [], "kind": 9, "opts": opts, "may_pass": False}, key="w")
    import multiprocessing as mp
    t = teacher.Teacher.__new__(teacher.Teacher)
    net = os.path.join(ts_dir, "bv-v1", "models", teacher.NET)
    t.lib, t.prm, t.nh, t.nf, t.na = teacher._load(os.path.join(ts_dir, "ts.so"), net)
    t.pool = mp.get_context("spawn").Pool(3, teacher._winit, (os.path.join(ts_dir, "ts.so"), net))
    r = t.grade(job)
    t.pool.terminate()
    print(r)
    assert r["draws"] > 0 and r["inc"] == ID["Silver"]
    assert {a for a, _ in r["q"]} == set(opts)
