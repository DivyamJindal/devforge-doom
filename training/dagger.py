"""Imitation learning with DAgger: the student (EgoMemNet, official observations only) learns from the privileged
teacher (training/expert.py). Afterwards PPO fine-tunes it (ppo.py --init).

  python training/dagger.py --workers 8 --envs-per-worker 4 --iters 8 --steps-per-iter 200000 --out runs/dagger

Iteration i: each worker plays with a mixture policy (teacher with probability beta_i, otherwise the student's own
sampled action, chosen per episode segment), the teacher labels EVERY visited state, and the student is trained
with cross-entropy on the aggregated dataset (BPTT over 64-step segments). beta decays 1 -> 0, so later data comes
from the states the student itself reaches: that is what fixes compounding errors of plain behaviour cloning.
"""
from __future__ import annotations

import argparse, json, os, random, sys, threading, time
from collections import defaultdict, deque
from pathlib import Path
import multiprocessing as mp

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submission")); sys.path.insert(0, str(ROOT / "training"))
from policy import EgoMemNet, Features, stack  # noqa: E402
from ppo import Shaper, TIER_TICKS  # noqa: E402

KEYS = ("ray", "ent", "mask", "scal")


def worker(conn, wid, n, seed, mix, shared, hidden):
    os.environ["OMP_NUM_THREADS"] = "1"; torch.set_num_threads(1)
    from doom_env import make_env
    from mapgen import generate
    from expert import Expert
    rng = random.Random(seed); wdir = Path(f"/tmp/egomem_dagger/{seed}"); wdir.mkdir(parents=True, exist_ok=True)
    boot = threading.Lock(); cnt = [0]
    envs = [None] * n; experts = [None] * n; feats = [Features() for _ in range(n)]; shapers = [None] * n
    spare = [None] * n; threads = [None] * n; src = [None] * n; plays = [0] * n

    def build(i, s):
        r = random.Random(s)
        for _ in range(5):
            try:
                tier = int(r.choices(list(mix), weights=list(mix.values()))[0][-1]); cnt[0] += 1
                path = wdir / f"w{wid}_{i}_{cnt[0]}.wad"; rows = generate(r.randrange(1 << 30), tier, path)
                with boot:
                    e = make_env(path, "MAP01", episode_ticks=TIER_TICKS[tier], render_buffers=False)
                    obs, _ = e.reset(seed=r.randrange(1 << 30))
                spare[i] = (e, obs, rows, f"gen{tier}"); return
            except Exception as ex:
                print(f"[worker {wid}] map failed: {ex}", flush=True)

    def prefetch(i):
        threads[i] = threading.Thread(target=build, args=(i, rng.randrange(1 << 30)), daemon=True); threads[i].start()

    def new_episode(i):
        if envs[i] is not None and plays[i] < 2:
            obs, _ = envs[i].reset(seed=rng.randrange(1 << 30)); plays[i] += 1; rows = experts[i].g
        else:
            threads[i].join()
            if envs[i] is not None:
                old = envs[i].wad; envs[i].close(); Path(old).unlink(missing_ok=True)
            envs[i], obs, rows, src[i] = spare[i]; plays[i] = 1; prefetch(i)
        experts[i] = Expert(rows, rng.randrange(1 << 30)); feats[i].reset(); shapers[i] = Shaper(envs[i])
        return obs

    for i in range(n):
        prefetch(i)
    obs = [new_episode(i) for i in range(n)]
    net = EgoMemNet(hidden); net.eval(); h = torch.zeros(n, hidden)
    student_drives = [False] * n; seg = [0] * n
    conn.send("ready")
    while True:
        cmd, T, beta = conn.recv()
        if cmd == "close":
            break
        net.load_state_dict(shared)
        buf = defaultdict(list); infos = []; agree = []
        start = np.zeros(n, np.float32)
        for t in range(T):
            f = [feats[i](obs[i]) for i in range(n)]; x = stack(f)
            with torch.no_grad():
                logits, _, h = net.step({k: torch.from_numpy(v) for k, v in x.items()}, h)
                sa = torch.distributions.Categorical(logits=logits).sample().numpy()
            labels = np.array([experts[i].act(envs[i].privileged_state(), obs[i]) for i in range(n)], np.int64)
            for k in KEYS:
                buf[k].append(x[k].astype(np.float16))
            buf["label"].append(labels); buf["start"].append(start.copy())
            start = np.zeros(n, np.float32)
            for i in range(n):
                if seg[i] <= 0:                                # who drives is re-drawn every ~16 steps
                    student_drives[i] = rng.random() > beta; seg[i] = 16
                seg[i] -= 1
                a = int(sa[i]) if student_drives[i] else int(labels[i])
                agree.append(int(sa[i]) == int(labels[i]))
                ob, _, te, tr, _ = envs[i].step(feats[i].button_vector(a))
                ps = envs[i].privileged_state(); shapers[i](ps)
                if te or tr:
                    s = shapers[i]
                    infos.append({"src": src[i], "exit": float(ps["exit_reached"]), "dead": float(ps["dead"]), "kills": ps["kills"],
                                  "keys": int(np.sum(ps["keys"])), "doors": s.n_doors, "cells": len(s.cells), "beta": beta})
                    obs[i] = new_episode(i); start[i] = 1.0; h[i] = 0
                else:
                    obs[i] = ob
        conn.send(({k: np.stack(v) for k, v in buf.items()}, infos, float(np.mean(agree))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--envs-per-worker", type=int, default=4)
    ap.add_argument("--iters", type=int, default=8); ap.add_argument("--steps-per-iter", type=float, default=2e5)
    ap.add_argument("--T", type=int, default=64); ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--train-steps", type=int, default=1500); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--cap", type=float, default=4e6)
    ap.add_argument("--mix", default="gen0:0.15,gen1:0.2,gen2:0.25,gen3:0.25,gen4:0.15")
    ap.add_argument("--init", default=None); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="runs/dagger")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = EgoMemNet(a.hidden).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev, weights_only=True)["state"])
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    mix = {k: float(v) for k, v in (x.split(":") for x in a.mix.split(","))}
    shared = {k: v.detach().cpu().clone().share_memory_() for k, v in net.state_dict().items()}
    import torch.multiprocessing as tmp
    ctx = tmp.get_context("spawn"); pipes = []
    for w in range(a.workers):
        p1, p2 = ctx.Pipe(); ctx.Process(target=worker, args=(p2, w, a.envs_per_worker, a.seed * 7919 + w, mix, shared, a.hidden), daemon=True).start()
        pipes.append(p1)
    for p in pipes:
        assert p.recv() == "ready"
    N = a.workers * a.envs_per_worker; T = a.T
    data = deque(maxlen=int(a.cap // T))                       # segments: dict of [T, ...] arrays
    hist = deque(maxlen=400); t0 = time.time(); total = 0
    betas = [max(0.0, 1.0 - i / max(a.iters - 2, 1)) for i in range(a.iters)]
    print(f"device {dev}  envs {N}  betas {[round(b, 2) for b in betas]}", flush=True)
    for it, beta in enumerate(betas):
        n_chunks = max(1, int(a.steps_per_iter // (N * T))); agrees = []
        for c in range(n_chunks):
            for p in pipes:
                p.send(("collect", T, beta))
            for p in pipes:
                d, infos, ag = p.recv(); hist.extend(infos); agrees.append(ag)
                for e in range(d["label"].shape[1]):
                    data.append({k: v[:, e] for k, v in d.items()})
            total += N * T
        # ---- train the student on the aggregate ----
        net.train(); losses, accs = [], []
        for s in range(a.train_steps):
            idx = np.random.randint(0, len(data), a.batch)
            seq = {k: torch.from_numpy(np.stack([data[i][k] for i in idx], 1)).to(dev) for k in (*KEYS, "label", "start")}
            x = {k: seq[k].float() for k in KEYS}
            st = seq["start"].clone(); st[0] = 1.0
            logits, _ = net.sequence(x, torch.zeros(a.batch, a.hidden, device=dev), st)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), seq["label"].reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            losses.append(loss.item()); accs.append((logits.argmax(-1) == seq["label"]).float().mean().item())
        net.eval()
        sd = net.state_dict()
        for k in shared:
            shared[k].copy_(sd[k].detach().cpu())
        H = [x for x in hist if x["beta"] == beta] or list(hist)
        by = defaultdict(list)
        for x in H:
            by[x["src"]].append(x)
        line = (f"iter {it} beta {beta:.2f} data {len(data) * T / 1e6:.2f}M steps total {total / 1e6:.2f}M ({total / (time.time() - t0):.0f}/s) "
                f"loss {np.mean(losses[-200:]):.3f} acc {np.mean(accs[-200:]):.3f} student-agrees {np.mean(agrees):.3f}")
        for s_ in sorted(by):
            R = by[s_]
            line += f" | {s_}: exit {np.mean([r['exit'] for r in R]):.2f} keys {np.mean([r['keys'] for r in R]):.2f} doors {np.mean([r['doors'] for r in R]):.1f} kills {np.mean([r['kills'] for r in R]):.1f} dead {np.mean([r['dead'] for r in R]):.2f} cells {np.mean([r['cells'] for r in R]):.0f}"
        print(line, flush=True)
        ck = {"state": {k: v.cpu() for k, v in net.state_dict().items()}, "cfg": {"hidden": a.hidden}, "steps": total, "iter": it}
        torch.save(ck, out / "egomem.pt"); torch.save(ck, out / f"egomem_it{it}.pt")
    for p in pipes:
        p.send(("close", 0, 0))


if __name__ == "__main__":
    main()
