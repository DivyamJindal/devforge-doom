"""EgoMem agent: observation encoder, episodic place memory, and recurrent actor-critic.

Shared by training (training/ppo.py) and evaluation (submission/agent.py) so features are bit-identical in both.
Everything here works only from the official observation dictionary (no privileged information).

Architecture (all trained end-to-end with our own recurrent PPO):
  rays      64 ego-centric rays -> circular 1-D convolutions (the ray ring has no start or end, so the convolution
            wraps around: an obstacle behind-left is encoded the same way wherever it sits on the ring)
  entities  up to 32 visible things -> per-entity MLP -> masked attention pooling + max pooling
            (permutation-invariant, any number of enemies/items)
  vitals    health, armour, weapon, ammo, keys, own velocity
  memory    (1) episodic place memory: a rotation-invariant signature of the surroundings (FFT magnitude of the ray
                ring + hit-type histogram) is compared with every place visited this episode -> novelty/familiarity
                signals that tell the agent "I have been here" without any map coordinates;
            (2) event memory: locked door seen / how long ago, door seen, keys held, recent damage;
            (3) a GRU that integrates all of it over time;
            (4) optional ego-centric occupancy map (omap): heading is dead-reckoned from our own turn presses with
                DOOM's exact turn model (5 slow tics at 1.76 deg, then 3.52 deg/tic; verified exact against ground
                truth), position from the observed velocity. Rays paint free space, walls and doors into a world-
                frame grid; a 21x21 crop rotated to the agent's heading (64 units/cell, +-672 units) goes to a CNN.
Actions are 15 macro-actions over the 9 buttons (USE is pulsed so holding it re-triggers doors correctly).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

NUM_RAYS, MAX_ENT, N_ETYPE, N_HIT = 64, 32, 12, 6
F, B, SL, SR, TL, TR, ATK, USE, NXT = range(9)
ACTIONS = [(), (F,), (F, TL), (F, TR), (TL,), (TR,), (B,), (SL,), (SR,), (ATK,), (ATK, TL), (ATK, TR), (F, ATK), (F, USE), (USE,)]
N_ACT = len(ACTIONS)
MEM_CAP, MEM_EVERY, NOVEL_THR = 512, 2, 0.09
SIG_DIM = 12 + N_HIT + 1
N_SCALAR = 24 + N_ACT + 10
OM_CELL, OM_N, OM_R, OM_C = 64.0, 256, 10, 4          # grid cell size, world grid side, crop radius, channels
RAY_RANGE = 1024.0
_RAY_ANG = np.arange(NUM_RAYS) * (2 * np.pi / NUM_RAYS)
_FRAC = np.linspace(0.1, 0.9, 9)
_OFF = np.stack(np.meshgrid(np.arange(OM_R, -OM_R - 1, -1), np.arange(-OM_R, OM_R + 1), indexing="ij"), -1).reshape(-1, 2) * OM_CELL  # (fwd, right)


def turn_degrees(held_tics: int) -> float:
    """Heading change of one 4-tic step with TURN held, given how many tics it was already held (DOOM SLOWTURNTICS)."""
    return sum(1.7578125 if held_tics + i + 1 <= 5 else 3.515625 for i in range(4))


class OccupancyMap:
    """Dead-reckoned world-frame grid; channels: visited, free, wall, door/locked door. The grid wraps around
    (torus) so no level size can index out of bounds. ponytail: no loop closure, drift grows ~0.6 units/step."""

    def __init__(self):
        self.g = np.zeros((OM_C, OM_N, OM_N), np.float32); self.x = self.y = self.h = 0.0
        self.held = 0; self.pw = np.zeros(2)

    def _idx(self, wx, wy):
        return (np.floor(wx / OM_CELL).astype(np.int64) % OM_N, np.floor(wy / OM_CELL).astype(np.int64) % OM_N)

    def update(self, prev_action: int, vel_fr, dist, hit, first: bool) -> np.ndarray:
        a = ACTIONS[prev_action]
        if not first:
            d = 1 if TL in a else -1 if TR in a else 0
            if d:
                self.h += np.radians(d * turn_degrees(self.held)); self.held += 4
            else:
                self.held = 0
        c, s = np.cos(self.h), np.sin(self.h)
        w = np.array([vel_fr[0] * c + vel_fr[1] * s, vel_fr[0] * s - vel_fr[1] * c])
        if not first:                                   # per-step displacement fit on ground truth (resid 0.6 units)
            dx, dy = 2.76 * w + 1.56 * self.pw; self.x += dx; self.y += dy
        self.pw = w
        ang = self.h + _RAY_ANG; r = dist * RAY_RANGE
        fx = self.x + np.outer(r, _FRAC) * np.cos(ang)[:, None]; fy = self.y + np.outer(r, _FRAC) * np.sin(ang)[:, None]
        self.g[(1, *self._idx(fx.ravel(), fy.ravel()))] = 1.0
        solid = (dist < 0.999) & (hit >= 1) & (hit <= 3)
        if solid.any():
            hx = self.x + r[solid] * np.cos(ang[solid]); hy = self.y + r[solid] * np.sin(ang[solid])
            ch = np.where(hit[solid] == 1, 2, 3)
            self.g[(ch, *self._idx(hx, hy))] = 1.0
        ix, iy = self._idx(np.array([self.x]), np.array([self.y]))
        self.g[0, ix, iy] = min(self.g[0, ix[0], iy[0]] + 0.25, 1.0)
        ox = self.x + _OFF[:, 0] * c + _OFF[:, 1] * s; oy = self.y + _OFF[:, 0] * s - _OFF[:, 1] * c
        return self.g[(slice(None), *self._idx(ox, oy))].reshape(OM_C, 2 * OM_R + 1, 2 * OM_R + 1)


def place_signature(dist: np.ndarray, hit: np.ndarray) -> np.ndarray:
    """Rotation-invariant description of the surroundings from one ray scan."""
    spec = np.abs(np.fft.rfft(dist))[:12] / NUM_RAYS
    hist = np.bincount(hit, minlength=N_HIT)[:N_HIT] / NUM_RAYS
    return np.concatenate([spec, hist, [dist.mean()]]).astype(np.float32)


class Features:
    """Per-episode stateful feature extractor (memory lives here). Call reset() before every map."""

    def __init__(self, vision: bool = False, omap: bool = False):
        self.vision, self.omap = vision, omap
        self.reset()

    def reset(self):
        self.mem = np.zeros((MEM_CAP, SIG_DIM), np.float32); self.n_mem = 0; self.t = 0
        self.since_novel = 0; self.prev_action = 0; self.use_down = False
        self.locked_seen = 0.0; self.since_locked = 999; self.door_seen = 0.0; self.since_door = 999
        self.prev_health = None; self.dmg_recent = 0.0; self.novel = False
        self.om = OccupancyMap() if self.omap else None

    def button_vector(self, a: int) -> np.ndarray:
        """Macro-action -> 9 buttons. USE fires on the button-down edge, so a held USE is released every other step."""
        v = np.zeros(9, np.int8)
        for b in ACTIONS[a]:
            v[b] = 1
        if v[USE]:
            if self.use_down:
                v[USE] = 0
            self.use_down = bool(v[USE])
        else:
            self.use_down = False
        self.prev_action = a
        return v

    def __call__(self, obs: dict) -> dict:
        rays = obs["rays"]; dist = np.asarray(rays["distance"], np.float32); hit = np.asarray(rays["hit_type"], np.int64)
        ray = np.zeros((1 + N_HIT, NUM_RAYS), np.float32); ray[0] = dist; ray[1 + hit, np.arange(NUM_RAYS)] = 1.0
        ve = obs["visible_entities"]; m = np.asarray(ve["mask"], np.float32)
        ent = np.zeros((MAX_ENT, 5 + N_ETYPE), np.float32)
        ent[:, :5] = np.clip(np.asarray(ve["features"], np.float32), -10, 10)
        ent[np.arange(MAX_ENT), 5 + np.asarray(ve["entity_type"], np.int64)] = 1.0
        ent *= m[:, None]
        v = np.asarray(obs["vitals"], np.float32)
        wep = np.zeros(8, np.float32); wep[int(np.clip(v[2], 0, 7))] = 1
        vit = np.concatenate([[v[0] / 100, v[1] / 100], wep, [min(v[3], 200) / 50], np.minimum(v[4:8], 400) / [200, 50, 50, 300],
                              v[8:14], np.clip(v[14:17] / 10, -3, 3)]).astype(np.float32)
        # ---- episodic place memory ----
        sig = place_signature(dist, hit)
        if self.n_mem:
            d = np.sqrt(((self.mem[: self.n_mem] - sig) ** 2).sum(1)); nov = float(d.min()); fam = float((d < NOVEL_THR).sum())
        else:
            nov, fam = 1.0, 0.0
        self.novel = nov > NOVEL_THR                 # read by training for the episodic-curiosity bonus
        if self.novel:
            self.since_novel = 0
        else:
            self.since_novel += 1
        if self.t % MEM_EVERY == 0:
            self.mem[self.n_mem % MEM_CAP] = sig; self.n_mem += 1
        # ---- event memory ----
        if (hit == 3).any():
            self.locked_seen, self.since_locked = 1.0, 0
        else:
            self.since_locked += 1
        if (hit == 2).any():
            self.door_seen, self.since_door = 1.0, 0
        else:
            self.since_door += 1
        if self.prev_health is not None and v[0] < self.prev_health:
            self.dmg_recent = min(1.0, self.dmg_recent + (self.prev_health - v[0]) / 50)
        self.dmg_recent *= 0.85; self.prev_health = float(v[0])
        pa = np.zeros(N_ACT, np.float32); pa[self.prev_action] = 1
        mem = np.array([min(nov, 1.0) * 4, min(fam, 50) / 10, min(self.since_novel, 300) / 100, min(self.t, 1050) / 525,
                        self.locked_seen, min(self.since_locked, 500) / 250, self.door_seen, min(self.since_door, 500) / 250,
                        self.dmg_recent, min(self.n_mem, MEM_CAP) / MEM_CAP], np.float32)
        self.t += 1
        et = np.asarray(ve["entity_type"], np.int64)
        enemy = bool(((et >= 1) & (et <= 3) & (m > 0)).any())
        front = float(min(dist[0], dist[1], dist[-1]))
        am = np.ones(N_ACT, np.float32)
        if not enemy or ATK in ACTIONS[self.prev_action]:     # no target, or fired last step (held pistol fire spreads)
            am[[i for i, a in enumerate(ACTIONS) if ATK in a]] = 0.0
        if front > 0.12:                                      # nothing within reach ahead: USE is pointless
            am[[i for i, a in enumerate(ACTIONS) if USE in a]] = 0.0
        out = {"ray": ray, "ent": ent, "mask": m, "scal": np.concatenate([vit, pa, mem]), "amask": am}
        if self.om is not None:
            out["omap"] = self.om.update(self.prev_action, v[14:16], dist, hit, self.t == 1)
        if self.vision:                                # 4 x 30 x 40 uint8: RGB + depth, every 4th pixel
            rgb = np.asarray(obs.get("rgb", np.zeros((120, 160, 3), np.uint8)))[::4, ::4]
            dep = np.asarray(obs.get("depth", np.zeros((120, 160), np.uint8)))[::4, ::4]
            out["img"] = np.concatenate([rgb.transpose(2, 0, 1), dep[None]]).astype(np.uint8)
        return out


def stack(feats: list[dict]) -> dict:
    return {k: np.stack([f[k] for f in feats]) for k in feats[0]}


def warm_start(net: "EgoMemNet", state: dict) -> None:
    """Load a checkpoint into a (possibly wider) net: extra fuse inputs (the vision branch) start at zero, so the
    warm-started policy behaves exactly like the old one until training teaches it to use the new input."""
    own = net.state_dict()
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
        elif k == "fuse.0.weight":
            w = torch.zeros_like(own[k]); w[:, : v.shape[1]] = v; own[k] = w
    net.load_state_dict(own)


# Encoders subclass nn.Sequential so parameter names (ray.0.weight, ...) match every checkpoint we trained.
class RayRingEncoder(nn.Sequential):
    """64-ray ring (distance + one-hot hit type) -> 256. Circular padding: the ring has no seam."""

    def __init__(self):
        super().__init__(
            nn.Conv1d(1 + N_HIT, 32, 5, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(32, 48, 5, stride=2, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(48, 48, 3, stride=2, padding=1, padding_mode="circular"), nn.ReLU(),
            nn.Flatten(), nn.Linear(48 * 16, 256), nn.ReLU())


class EntityEncoder(nn.Sequential):
    """One visible thing (distance, bearing, velocity, type) -> 64; shared over all slots, pooled by EgoMemNet."""

    def __init__(self):
        super().__init__(nn.Linear(5 + N_ETYPE, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())


class VisionEncoder(nn.Sequential):
    """4 x 30 x 40 RGB + depth -> 128: sees what rays cannot (exit pads and switches, key and door colours)."""

    def __init__(self):
        super().__init__(nn.Conv2d(4, 16, 5, stride=2, padding=2), nn.ReLU(), nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                         nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 4 * 5, 128), nn.ReLU())


class OccupancyEncoder(nn.Sequential):
    """4 x 21 x 21 ego occupancy crop -> 128: walls, doors and unexplored gaps beyond the current view."""

    def __init__(self):
        super().__init__(nn.Conv2d(OM_C, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                         nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 6 * 6, 128), nn.ReLU())


class EgoMemNet(nn.Module):
    def __init__(self, hidden: int = 256, vision: bool = False, omap: bool = False):
        super().__init__()
        self.hidden, self.vision, self.omap = hidden, vision, omap
        self.ray = RayRingEncoder()
        self.ent = EntityEncoder()
        self.ent_score = nn.Linear(64, 1)
        self.scal = nn.Sequential(nn.Linear(N_SCALAR, 128), nn.ReLU())
        if vision:   # sees what rays cannot: exit pads/switches, door and key colours, lighting
            self.vis = VisionEncoder()
        if omap:     # ego occupancy map: where walls, doors and unexplored gaps are, beyond the current view
            self.om = OccupancyEncoder()
        self.fuse = nn.Sequential(nn.Linear(256 + 128 + 128 + (128 if vision else 0) + (128 if omap else 0), hidden), nn.ReLU())
        self.gru = nn.GRUCell(hidden, hidden)
        self.pi = nn.Linear(hidden, N_ACT); self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.zeros_(self.pi.bias)

    def embed(self, ray, ent, mask, scal, img=None, omap=None):
        e = self.ent(ent)                                                   # [N,32,64]
        s = self.ent_score(e).squeeze(-1).masked_fill(mask < 0.5, -1e9)
        att = torch.softmax(s, -1) * (mask.sum(-1, keepdim=True) > 0)
        pooled = torch.cat([(att.unsqueeze(-1) * e).sum(1), (e - (1 - mask.unsqueeze(-1)) * 1e4).max(1).values.clamp(min=0)], -1)
        parts = [self.ray(ray), pooled, self.scal(scal)]
        if self.vision:
            parts.append(self.vis(img.float() / 255.0))
        if self.omap:
            parts.append(self.om(omap))
        return self.fuse(torch.cat(parts, -1))

    def masked(self, logits, amask):
        return logits if amask is None else logits.masked_fill(amask < 0.5, -1e9)

    def step(self, x: dict, h):
        z = self.embed(x["ray"], x["ent"], x["mask"], x["scal"], x.get("img"), x.get("omap"))
        h = self.gru(z, h)
        return self.masked(self.pi(h), x.get("amask")), self.v(h).squeeze(-1), h

    def sequence(self, x: dict, h0, starts):
        """x: tensors [T,B,...]; starts [T,B] = 1 where a new episode begins (hidden reset). -> logits, values [T,B]."""
        T, Bn = starts.shape
        r = lambda k: x[k].reshape(T * Bn, *x[k].shape[2:]) if k in x else None
        z = self.embed(r("ray"), r("ent"), r("mask"), r("scal"), r("img"), r("omap")).reshape(T, Bn, -1)
        h, logits, values = h0, [], []
        for t in range(T):
            h = h * (1 - starts[t]).unsqueeze(-1)
            h = self.gru(z[t], h)
            logits.append(self.masked(self.pi(h), x["amask"][t] if "amask" in x else None)); values.append(self.v(h).squeeze(-1))
        return torch.stack(logits), torch.stack(values)
