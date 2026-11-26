"""Recurrent PPO for the EgoMem agent, written from scratch for this track.

  python training/ppo.py --workers 8 --envs-per-worker 2 --steps 2000000 --out runs/local

Design
- Worker processes each run several official `doom_env` instances (same observations as evaluation, pixels off).
  Every episode starts on a NEW level: procedurally generated (training/mapgen.py, tiers 1-4) or a Freedoom 2 map,
  so the agent can only succeed by generalising, never by memorising a layout.
- Intrinsic episodic curiosity (in the spirit of NGU): a small bonus whenever the agent's own place memory says the
  current surroundings look new. It uses only the agent's observations, so it generalises beyond our level generator.
- Rewards are shaped from `privileged_state()` (training only; the agent never sees it): new areas reached,
  kills, pickups, keycards, doors and locked doors opened, the exit (plus a time/health bonus), minus damage
  and death. This mirrors the hidden scorer's categories (exit > progression > exploration > combat).
- PPO with GAE, a GRU policy trained by backprop-through-time over whole rollout segments, return-based reward
  scaling, entropy annealing. Batched inference on the GPU; environments on all CPU cores.
"""
from __future__ import annotations

import argparse, json, math, os, random, sys, time
from collections import defaultdict, deque
from pathlib import Path
import multiprocessing as mp

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submission")); sys.path.insert(0, str(ROOT / "training"))
from policy import EgoMemNet, Features, N_ACT, stack  # noqa: E402

TIER_TICKS = {0: 2100, 1: 2100, 2: 3150, 3: 4200, 4: 4200}
FREEDOOM_TRAIN = [f"MAP{i:02d}" for i in range(1, 33) if i not in (2, 7, 11, 16, 23, 29)]  # those 6 held out for eval
FREEDOOM_EVAL = ["MAP02", "MAP07", "MAP11", "MAP16", "MAP23", "MAP29"]


class Shaper:
    """Shaped reward from ground truth. Mirrors the evaluation categories."""

    W = {"cell": 0.06, "kill": 1.0, "item": 0.15, "key": 4.0, "dmg": 0.006, "door": 1.0, "lock": 3.0, "exit": 15.0, "death": 4.0, "time": 0.002, "curio": 0.02}

    def __init__(self, env, cell=128.0, w: dict | None = None):
        self.env, self.cell = env, cell
        self.w = {**Shaper.W, **(w or {})}
        ps = env.privileged_state()
        self.cells = {self._c(ps)}; self.kills = ps["kills"]; self.items = ps["items"]; self.keys = int(np.sum(ps["keys"]))
        self.health = ps["health"]; self.door_sec = dict(getattr(env.mapinfo, "door_sectors", {}) or {})
        self.opened = set(); self.n_doors = 0; self.n_locks = 0

    def _c(self, ps):
        return (int(ps["x"] // self.cell), int(ps["y"] // self.cell))

    def __call__(self, ps) -> float:
        w = self.w; r = -w["time"]
        c = self._c(ps)
        if c not in self.cells:
            self.cells.add(c); r += w["cell"]
        r += w["kill"] * (ps["kills"] - self.kills); self.kills = ps["kills"]
        r += w["item"] * max(0, ps["items"] - self.items); self.items = ps["items"]
        k = int(np.sum(ps["keys"])); r += w["key"] * max(0, k - self.keys); self.keys = k
        r -= w["dmg"] * max(0.0, self.health - ps["health"]); self.health = ps["health"]
        ceil, flo = ps["sector_ceiling"], ps["sector_floor"]
        if ceil is not None:
            for s, lock in self.door_sec.items():
                if s not in self.opened and s < len(ceil) and ceil[s] - flo[s] > 56:
                    self.opened.add(s); r += w["lock"] if lock else w["door"]; self.n_locks += bool(lock); self.n_doors += 1
        if ps["exit_reached"]:
            r += w["exit"] + 5.0 * (1 - ps["tic"] / ps["episode_ticks"]) + 2.0 * max(ps["health"], 0) / 100
        if ps["dead"]:
            r -= w["death"]
        return r


def make_map(rng: random.Random, weights: dict, wdir: Path, tag: str):
    from mapgen import generate
    import vizdoom as vd
    src = rng.choices(list(weights), weights=list(weights.values()))[0]
    if src == "freedoom":
        return os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad"), rng.choice(FREEDOOM_TRAIN), 4200, src
    tier = int(src[-1]); path = wdir / f"{tag}.wad"
    generate(rng.randrange(1 << 30), tier, path)
    return str(path), "MAP01", TIER_TICKS[tier], src


def worker(conn, wid: int, n: int, seed: int, weights: dict, shared: dict, hidden: int, shape: dict, repeat: int = 2):
    """Steps n environments. Level loading is double-buffered: while a level is being played, the next one is generated
    and booted in a background thread (ViZDoom releases the GIL), so a finished episode never stalls the batch.
    Each level is played `repeat` times (a reset costs ~3 ms, a fresh level ~180 ms).
    The worker runs whole T-step rollouts with its own CPU copy of the policy (refreshed from shared memory at the start of
    each rollout), so there is no per-step lock-step with the learner: a slow level load only delays this worker."""
    import threading
    os.environ["OMP_NUM_THREADS"] = "1"; torch.set_num_threads(1)
    from doom_env import make_env
    rng = random.Random(seed); wdir = Path(f"/tmp/egomem_maps/{seed}"); wdir.mkdir(parents=True, exist_ok=True)
    envs = [None] * n; feats = [Features() for _ in range(n)]; shapers = [None] * n; meta = [None] * n; ret = [0.0] * n
    plays = [0] * n; spare = [None] * n; threads = [None] * n; counter = [0]
    boot = threading.Lock()                       # ViZDoom instances must not boot concurrently in one process

    def build(i, s):
        r = random.Random(s)
        for _ in range(5):
            try:
                counter[0] += 1
                wad, m, ticks, src = make_map(r, weights, wdir, f"w{wid}_{i}_{counter[0]}")
                with boot:
                    e = make_env(wad, m, episode_ticks=ticks, render_buffers=False)
                    obs, _ = e.reset(seed=r.randrange(1 << 30))
                spare[i] = (e, obs, src); return
            except Exception as ex:  # a rare bad level: just make another
                print(f"[worker {wid}] map failed: {ex}", flush=True)

    def prefetch(i):
        threads[i] = threading.Thread(target=build, args=(i, rng.randrange(1 << 30)), daemon=True); threads[i].start()

    def new_episode(i):
        if envs[i] is not None and plays[i] < repeat:
            obs, _ = envs[i].reset(seed=rng.randrange(1 << 30)); plays[i] += 1
        else:
            threads[i].join()
            if envs[i] is not None:
                old = envs[i].wad; envs[i].close()
                if old.startswith(str(wdir)):
                    Path(old).unlink(missing_ok=True)
            envs[i], obs, meta[i] = spare[i]; plays[i] = 1
            prefetch(i)
        feats[i].reset(); shapers[i] = Shaper(envs[i], w=shape); ret[i] = 0.0
        return feats[i](obs)

    for i in range(n):
        prefetch(i)
    cur = [new_episode(i) for i in range(n)]
    net = EgoMemNet(hidden); net.eval()
    h = torch.zeros(n, hidden); start = np.ones(n, np.float32)
    conn.send("ready")
    while True:
        cmd, T = conn.recv()
        if cmd == "close":
            break
        net.load_state_dict(shared)
        buf = defaultdict(list); h0 = h.clone(); infos = []
        for t in range(T):
            x = stack(cur)
            with torch.no_grad():
                logits, val, h = net.step({k: torch.from_numpy(v) for k, v in x.items()}, h)
                dist = torch.distributions.Categorical(logits=logits); act = dist.sample(); logp = dist.log_prob(act)
            for k, v in x.items():
                buf[k].append(v)
            buf["act"].append(act.numpy()); buf["logp"].append(logp.numpy()); buf["val"].append(val.numpy()); buf["start"].append(start.copy())
            rews, dones = np.zeros(n, np.float32), np.zeros(n, np.float32)
            for i in range(n):
                obs, _, term, trunc, _ = envs[i].step(feats[i].button_vector(int(act[i])))
                ps = envs[i].privileged_state(); r = shapers[i](ps); ret[i] += r; rews[i] = r
                if term or trunc:
                    sh = shapers[i]
                    infos.append({"src": meta[i], "exit": float(ps["exit_reached"]), "dead": float(ps["dead"]), "kills": ps["kills"],
                                  "keys": int(np.sum(ps["keys"])), "doors": sh.n_doors, "locks": sh.n_locks, "cells": len(sh.cells),
                                  "ret": ret[i], "len": ps["tic"]})
                    dones[i] = 1.0; cur[i] = new_episode(i)
                else:
                    cur[i] = feats[i](obs)
                    rews[i] += shapers[i].w["curio"] * feats[i].novel   # episodic curiosity: reward places that look new
            buf["rew"].append(rews); buf["done"].append(dones)
            start = dones.copy(); h = h * (1 - torch.from_numpy(dones)).unsqueeze(-1)
        with torch.no_grad():
            _, last_v, _ = net.step({k: torch.from_numpy(v) for k, v in stack(cur).items()}, h)
        outd = {k: np.stack(v) for k, v in buf.items()}
        outd["h0"] = h0.numpy(); outd["last_v"] = last_v.numpy()
        conn.send((outd, infos))


class RunningStd:
    def __init__(self):
        self.n, self.mean, self.m2 = 1e-4, 0.0, 1.0

    def update(self, x):
        x = np.asarray(x, np.float64).ravel(); bn, bm, bv = len(x), x.mean(), x.var()
        d = bm - self.mean; tot = self.n + bn
        self.mean += d * bn / tot; self.m2 = self.m2 * self.n + bv * bn + d * d * self.n * bn / tot; self.m2 /= tot; self.n = tot

    @property
    def std(self):
        return math.sqrt(max(self.m2, 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--envs-per-worker", type=int, default=2)
    ap.add_argument("--steps", type=float, default=2e6); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--minibatches", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95); ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.02); ap.add_argument("--ent-end", type=float, default=0.003)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--mix", default="gen1:0.2,gen2:0.25,gen3:0.3,gen4:0.15,freedoom:0.1")
    ap.add_argument("--init", default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shape", default="", help="reward weight overrides, e.g. death=8,dmg=0.02")
    ap.add_argument("--out", default="runs/local"); ap.add_argument("--save-every", type=int, default=50)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    weights = {k: float(v) for k, v in (x.split(":") for x in a.mix.split(","))}
    shape = {k: float(v) for k, v in (x.split("=") for x in a.shape.split(",") if x)}
    net = EgoMemNet(a.hidden).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev, weights_only=True)["state"])
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    import torch.multiprocessing as tmp
    shared = {k: v.detach().cpu().clone().share_memory_() for k, v in net.state_dict().items()}
    ctx = tmp.get_context("spawn"); pipes, procs = [], []
    for w in range(a.workers):
        p1, p2 = ctx.Pipe(); pr = ctx.Process(target=worker, args=(p2, w, a.envs_per_worker, a.seed * 100003 + w, weights, shared, a.hidden, shape), daemon=True)
        pr.start(); pipes.append(p1); procs.append(pr)
    for p in pipes:
        assert p.recv() == "ready"
    N = a.workers * a.envs_per_worker; T = a.T
    rstd = RunningStd(); disc = np.zeros(N); hist = defaultdict(lambda: deque(maxlen=300))
    n_updates = int(a.steps // (N * T)); t0 = time.time(); total = 0
    print(f"device {dev}  envs {N}  updates {n_updates}  params {sum(p.numel() for p in net.parameters()):,}  (async rollouts, policy lag 1)", flush=True)
    for p in pipes:
        p.send(("rollout", T))
    for upd in range(1, n_updates + 1):
        frac = (upd - 1) / max(n_updates - 1, 1)
        for g in opt.param_groups:
            g["lr"] = a.lr * (1 - 0.8 * frac)
        ent_coef = a.ent + (a.ent_end - a.ent) * frac
        tr0 = time.time(); res = [p.recv() for p in pipes]; t_env = time.time() - tr0
        sd = net.state_dict()                                  # workers are idle now: publish the latest weights,
        for k in shared:                                       # then immediately start the next rollout while we learn
            shared[k].copy_(sd[k].detach().cpu())
        for p in pipes:
            p.send(("rollout", T))
        for r in res:
            for info in r[1]:
                hist[info["src"]].append(info)
        buf = {k: np.concatenate([r[0][k] for r in res], axis=1) for k in res[0][0] if k not in ("h0", "last_v")}
        h0 = torch.from_numpy(np.concatenate([r[0]["h0"] for r in res])).to(dev)
        last_v = np.concatenate([r[0]["last_v"] for r in res])
        for t in range(T):
            disc = disc * a.gamma + buf["rew"][t]; rstd.update(disc); disc[buf["done"][t] > 0] = 0
        t_roll = t_env; tu0 = time.time()
        rew = buf["rew"] / rstd.std; done = buf["done"]; val = buf["val"]
        adv = np.zeros_like(rew); gae = np.zeros(N)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val[t + 1]
            delta = rew[t] + a.gamma * nv * (1 - done[t]) - val[t]
            gae = delta + a.gamma * a.lam * (1 - done[t]) * gae; adv[t] = gae
        ret = adv + val
        B = {k: torch.from_numpy(buf[k]).to(dev) for k in ("ray", "ent", "mask", "scal")}
        act_t = torch.from_numpy(buf["act"]).to(dev); oldlp = torch.from_numpy(buf["logp"]).to(dev)
        adv_t = torch.from_numpy(adv).float().to(dev); ret_t = torch.from_numpy(ret).float().to(dev)
        st_t = torch.from_numpy(buf["start"]).to(dev)
        idx = np.arange(N); mb = N // a.minibatches; stats = defaultdict(list)
        for _ in range(a.epochs):
            np.random.shuffle(idx)
            for j in range(a.minibatches):
                e = torch.from_numpy(idx[j * mb:(j + 1) * mb]).to(dev)
                logits, v = net.sequence({k: x[:, e] for k, x in B.items()}, h0[e], st_t[:, e])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(act_t[:, e]); ratio = torch.exp(lp - oldlp[:, e])
                A = adv_t[:, e]; A = (A - A.mean()) / (A.std() + 1e-8)
                pl = -torch.min(ratio * A, ratio.clamp(1 - a.clip, 1 + a.clip) * A).mean()
                vl = 0.5 * ((v - ret_t[:, e]) ** 2).mean(); en = dist.entropy().mean()
                loss = pl + 0.5 * vl - ent_coef * en
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
                stats["pl"].append(pl.item()); stats["vl"].append(vl.item()); stats["ent"].append(en.item())
                stats["kl"].append(((ratio - 1) - torch.log(ratio)).mean().item())
        total += N * T; t_upd = time.time() - tu0
        if upd % 5 == 0 or upd == 1:
            fps = total / (time.time() - t0)
            line = f"upd {upd}/{n_updates} steps {total / 1e6:.2f}M fps {fps:.0f} ent {np.mean(stats['ent']):.2f} kl {np.mean(stats['kl']):.4f} vl {np.mean(stats['vl']):.3f} [waited for rollout {t_env:.1f}s, update {t_upd:.1f}s]"
            for src in sorted(hist):
                H = hist[src]
                if H:
                    line += (f" | {src}: n{len(H)} exit {np.mean([x['exit'] for x in H]):.2f} keys {np.mean([x['keys'] for x in H]):.2f} "
                             f"doors {np.mean([x['doors'] for x in H]):.1f} kills {np.mean([x['kills'] for x in H]):.1f} "
                             f"cells {np.mean([x['cells'] for x in H]):.0f} dead {np.mean([x['dead'] for x in H]):.2f}")
            print(line, flush=True)
            (out / "stats.json").write_text(json.dumps({"steps": total, "fps": fps, "by_src": {s: {k: float(np.mean([x[k] for x in H])) for k in ("exit", "keys", "doors", "locks", "kills", "cells", "dead", "ret")} for s, H in hist.items() if H}}))
        if upd % a.save_every == 0 or upd == n_updates:
            ck = {"state": {k: v.cpu() for k, v in net.state_dict().items()}, "cfg": {"hidden": a.hidden}, "steps": total}
            torch.save(ck, out / "egomem.pt"); torch.save(ck, out / f"egomem_{total // 1000000}M.pt")
    for p in pipes:
        p.recv(); p.send(("close", None))


if __name__ == "__main__":
    main()
