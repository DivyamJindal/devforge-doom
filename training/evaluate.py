"""Held-out scorecard: run the SUBMISSION agent (submission/agent.py, exactly as the scorer would) on maps it never
trained on, and measure the hidden scorer's categories.

  python training/evaluate.py --weights submission/weights/egomem.pt --per-tier 12 --procs 8

Maps: freshly generated levels per tier with seeds outside the training range, plus 6 Freedoom 2 maps excluded
from training. Metrics per map: exit reached, keycards, doors / locked doors opened, area coverage (128-unit cells),
kills, death. `proxy` = our guess at the scorer's weighting (exit largest, then progression, exploration, combat).
"""
from __future__ import annotations

import argparse, json, os, shutil, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
EVAL_SEED0 = 900_000_000                       # training seeds are drawn below 2**30 at random; these are fixed


def run_one(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch; torch.set_num_threads(1)
    from doom_env import make_env
    from ppo import Shaper, TIER_TICKS
    import vizdoom as vd
    src, idx, weights, tmp = job
    if src == "freedoom":
        wad, m, ticks = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad"), idx, 4200
    else:
        from mapgen import generate
        tier = int(src[-1]); wad = f"{tmp}/{src}_{idx}.wad"; generate(EVAL_SEED0 + idx, tier, wad); m, ticks = "MAP01", TIER_TICKS[tier]
    sys.path.insert(0, str(ROOT / "submission"))
    import agent as agent_mod
    if Path(weights).name == "egomem.pt":
        agent_mod.WEIGHTS_DIR = Path(weights).parent
    env = make_env(wad, m, episode_ticks=ticks, render_buffers=False)
    ag = agent_mod.DoomAgent(env.observation_space, env.action_space, "cpu")
    if Path(weights).name != "egomem.pt":
        import torch as T
        ag.net.load_state_dict(T.load(weights, map_location="cpu", weights_only=True)["state"])
    obs, _ = env.reset(seed=idx if isinstance(idx, int) else 0); ag.reset(); sh = Shaper(env)
    t0 = time.time(); n = 0; worst = 0.0
    while True:
        t1 = time.time(); a = ag.act(obs); worst = max(worst, time.time() - t1); n += 1
        obs, _, term, trunc, _ = env.step(a); ps = env.privileged_state(); sh(ps)
        if term or trunc:
            break
    agent_ms = (time.time() - t0) / n * 1000
    env.close()
    r = {"src": src, "map": str(idx), "exit": float(ps["exit_reached"]), "keys": int(np.sum(ps["keys"])), "doors": sh.n_doors,
         "locks": sh.n_locks, "cells": len(sh.cells), "kills": ps["kills"], "dead": float(ps["dead"]), "tics": ps["tic"],
         "ms_per_step": agent_ms, "worst_ms": worst * 1000}
    r["proxy"] = 50 * r["exit"] + 8 * r["keys"] + 3 * r["doors"] + 6 * r["locks"] + 0.4 * min(r["cells"], 100) + 2 * r["kills"] - 10 * r["dead"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "submission" / "weights" / "egomem.pt"))
    ap.add_argument("--per-tier", type=int, default=12); ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--no-freedoom", action="store_true"); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    from ppo import FREEDOOM_EVAL
    tmp = "/tmp/egomem_eval"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    jobs = [(f"gen{t}", i, a.weights, tmp) for t in (1, 2, 3, 4) for i in range(a.per_tier)]
    if not a.no_freedoom:
        jobs += [("freedoom", m, a.weights, tmp) for m in FREEDOOM_EVAL]
    with Pool(a.procs) as p:
        res = p.map(run_one, jobs, chunksize=1)
    print(f"weights: {a.weights}")
    print(f"{'set':9s} {'n':>3s} {'exit':>5s} {'keys':>5s} {'doors':>5s} {'locks':>5s} {'cells':>6s} {'kills':>5s} {'dead':>5s} {'proxy':>6s} {'ms/step':>8s}")
    rows = {}
    for src in sorted({r["src"] for r in res}):
        R = [r for r in res if r["src"] == src]
        rows[src] = {k: float(np.mean([r[k] for r in R])) for k in ("exit", "keys", "doors", "locks", "cells", "kills", "dead", "proxy", "ms_per_step")}
        x = rows[src]
        print(f"{src:9s} {len(R):3d} {x['exit']:5.2f} {x['keys']:5.2f} {x['doors']:5.1f} {x['locks']:5.2f} {x['cells']:6.1f} {x['kills']:5.1f} {x['dead']:5.2f} {x['proxy']:6.1f} {x['ms_per_step']:8.2f}")
    w = {"gen1": 1, "gen2": 2, "gen3": 3, "gen4": 4, "freedoom": 4}
    total = sum(w[s] * rows[s]["proxy"] for s in rows) / sum(w[s] for s in rows)
    print(f"tier-weighted proxy score: {total:.1f}   worst act(): {max(r['worst_ms'] for r in res):.1f} ms")
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "weighted_proxy": total, "maps": res}, indent=1))


if __name__ == "__main__":
    main()
