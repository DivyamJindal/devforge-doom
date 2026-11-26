"""Privileged teacher for imitation learning (TRAINING ONLY: never shipped, never used at evaluation).

It knows the generated level's grid (from mapgen) and the true position/angle (privileged_state), plans with BFS
to the current objective (missing keycard -> exit), opens doors in front of it, and engages visible enemies.
The student (submission/policy.py) only ever sees the official observations and learns to imitate these choices
(behaviour cloning + DAgger). The rules allow any training-time scaffolding; the submitted agent is the neural network.
"""
from __future__ import annotations

import math
import random
from collections import deque

import numpy as np

CELL = 64
# macro-action indices (see submission/policy.py ACTIONS)
NOOP, FWD, FWD_L, FWD_R, LEFT, RIGHT, BACK, STRAFE_L, STRAFE_R, ATK, ATK_L, ATK_R, FWD_ATK, FWD_USE, USE = range(15)
KEY_IDX = {"b": (0, 3), "y": (1, 4), "r": (2, 5)}     # vitals/privileged key flags: blue, yellow, red (card, skull)
ENEMY_TYPES = (1, 2, 3)


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class Expert:
    def __init__(self, rows: list[str], seed: int = 0):
        self.g = rows; self.H = len(rows); self.W = len(rows[0])
        self.rng = random.Random(seed)
        self.exit = next(((x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c == "E"), None)
        self.keys = {c: (x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c in "rby"}
        self._cache = {}
        self.hist = deque(maxlen=16); self.unstick = 0; self.unstick_a = BACK; self.use_cd = 0

    def cell(self, ps):
        return int(ps["x"] // CELL), self.H - 1 - int(ps["y"] // CELL)

    def _field(self, goal, held: frozenset):
        k = (goal, held)
        if k in self._cache:
            return self._cache[k]
        dist = {goal: 0}; q = deque([goal])
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.W and 0 <= ny < self.H) or (nx, ny) in dist:
                    continue
                c = self.g[ny][nx]
                if c == "#" or (c in "RBY" and c.lower() not in held):
                    continue
                dist[(nx, ny)] = dist[(x, y)] + 1; q.append((nx, ny))
        self._cache[k] = dist
        return dist

    def _held(self, ps):
        k = ps["keys"]
        return frozenset(c for c, (i, j) in KEY_IDX.items() if k[i] or k[j])

    def goal_field(self, ps):
        held = self._held(ps); here = self.cell(ps)
        if self.exit is not None:
            f = self._field(self.exit, held)
            if here in f:
                return f
        best = None
        for c, pos in self.keys.items():
            if c in held:
                continue
            f = self._field(pos, held)
            if here in f and (best is None or f[here] < best[0]):
                best = (f[here], f)
        if best:
            return best[1]
        return self._field(self.exit, frozenset("rby")) if self.exit else None

    def act(self, ps, obs) -> int:
        here = self.cell(ps); px, py = ps["x"], ps["y"]; ang = ps["angle"]
        self.hist.append((px, py)); self.use_cd = max(0, self.use_cd - 1)
        if self.unstick > 0:
            self.unstick -= 1; return self.unstick_a
        # --- combat: engage the nearest visible enemy in range ---
        ve = obs["visible_entities"]; m = np.asarray(ve["mask"]); t = np.asarray(ve["entity_type"]); fe = np.asarray(ve["features"])
        for i in range(len(m)):
            if m[i] and t[i] in ENEMY_TYPES and fe[i, 0] * 1024 < 800:
                a = math.degrees(math.atan2(fe[i, 1], fe[i, 2]))
                if abs(a) < 5:
                    return ATK
                return ATK_L if a > 0 else ATK_R
        # --- doors right in front: press USE once, then give it time to open ---
        rays = obs["rays"]; ht = np.asarray(rays["hit_type"]); d = np.asarray(rays["distance"]) * 1024
        if ht[0] in (2, 3) and d[0] < 96:
            if self.use_cd == 0:
                self.use_cd = 30; return FWD_USE
            return FWD
        # --- stuck detection ---
        if len(self.hist) == self.hist.maxlen:
            x0, y0 = self.hist[0]
            if math.hypot(px - x0, py - y0) < 24:
                self.unstick = self.rng.randint(3, 7); self.unstick_a = self.rng.choice([BACK, STRAFE_L, STRAFE_R, LEFT, RIGHT])
                self.hist.clear(); return self.unstick_a
        # --- navigation along the BFS gradient to the objective ---
        f = self.goal_field(ps)
        if f is None or here not in f:
            return FWD if d[0] > 0.15 * 1024 / 1024 * 150 else LEFT
        wp = here
        for _ in range(3):                          # look 3 cells ahead along the descending distance field
            nb = [(wp[0] + dx, wp[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            nb = [c for c in nb if c in f and f[c] < f[wp]]
            if not nb:
                break
            wp = min(nb, key=lambda c: f[c])
        if wp == here:
            return FWD
        wx, wy = (wp[0] + 0.5) * CELL, (self.H - 1 - wp[1] + 0.5) * CELL
        diff = _wrap(math.degrees(math.atan2(wy - py, wx - px)) - ang)
        if abs(diff) > 50:
            return LEFT if diff > 0 else RIGHT
        if abs(diff) > 12:
            return FWD_L if diff > 0 else FWD_R
        return FWD


if __name__ == "__main__":                           # sanity check: how good is the teacher on fresh levels?
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent / "submission"))
    from mapgen import generate
    from doom_env import make_env
    from policy import Features
    from ppo import Shaper, TIER_TICKS
    for tier in (0, 1, 2, 3, 4):
        res = []
        for s in range(6):
            rows = generate(10_000 + s, tier, "/tmp/egomem_expert.wad")
            env = make_env("/tmp/egomem_expert.wad", "MAP01", episode_ticks=TIER_TICKS[tier], render_buffers=False)
            obs, _ = env.reset(seed=s); ex = Expert(rows, s); fe = Features(); sh = Shaper(env)
            while True:
                a = ex.act(env.privileged_state(), obs)
                obs, _, te, tr, _ = env.step(fe.button_vector(a)); ps = env.privileged_state(); sh(ps)
                if te or tr:
                    break
            res.append((ps["exit_reached"], int(np.sum(ps["keys"])), sh.n_doors, ps["kills"], ps["dead"], len(sh.cells)))
            env.close()
        r = np.array(res, float)
        print(f"tier {tier}: exit {r[:, 0].mean():.2f} keys {r[:, 1].mean():.2f} doors {r[:, 2].mean():.1f} kills {r[:, 3].mean():.1f} dead {r[:, 4].mean():.2f} cells {r[:, 5].mean():.0f}")
