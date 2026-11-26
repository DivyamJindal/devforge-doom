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
            (3) a GRU that integrates all of it over time.
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


def place_signature(dist: np.ndarray, hit: np.ndarray) -> np.ndarray:
    """Rotation-invariant description of the surroundings from one ray scan."""
    spec = np.abs(np.fft.rfft(dist))[:12] / NUM_RAYS
    hist = np.bincount(hit, minlength=N_HIT)[:N_HIT] / NUM_RAYS
    return np.concatenate([spec, hist, [dist.mean()]]).astype(np.float32)


class Features:
    """Per-episode stateful feature extractor (memory lives here). Call reset() before every map."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mem = np.zeros((MEM_CAP, SIG_DIM), np.float32); self.n_mem = 0; self.t = 0
        self.since_novel = 0; self.prev_action = 0; self.use_down = False
        self.locked_seen = 0.0; self.since_locked = 999; self.door_seen = 0.0; self.since_door = 999
        self.prev_health = None; self.dmg_recent = 0.0

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
        if nov > NOVEL_THR:
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
        return {"ray": ray, "ent": ent, "mask": m, "scal": np.concatenate([vit, pa, mem])}


def stack(feats: list[dict]) -> dict:
    return {k: np.stack([f[k] for f in feats]) for k in feats[0]}


class EgoMemNet(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.hidden = hidden
        self.ray = nn.Sequential(
            nn.Conv1d(1 + N_HIT, 32, 5, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(32, 48, 5, stride=2, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(48, 48, 3, stride=2, padding=1, padding_mode="circular"), nn.ReLU(),
            nn.Flatten(), nn.Linear(48 * 16, 256), nn.ReLU())
        self.ent = nn.Sequential(nn.Linear(5 + N_ETYPE, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.ent_score = nn.Linear(64, 1)
        self.scal = nn.Sequential(nn.Linear(N_SCALAR, 128), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(256 + 128 + 128, hidden), nn.ReLU())
        self.gru = nn.GRUCell(hidden, hidden)
        self.pi = nn.Linear(hidden, N_ACT); self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.zeros_(self.pi.bias)

    def embed(self, ray, ent, mask, scal):
        e = self.ent(ent)                                                   # [N,32,64]
        s = self.ent_score(e).squeeze(-1).masked_fill(mask < 0.5, -1e9)
        att = torch.softmax(s, -1) * (mask.sum(-1, keepdim=True) > 0)
        pooled = torch.cat([(att.unsqueeze(-1) * e).sum(1), (e - (1 - mask.unsqueeze(-1)) * 1e4).max(1).values.clamp(min=0)], -1)
        return self.fuse(torch.cat([self.ray(ray), pooled, self.scal(scal)], -1))

    def step(self, x: dict, h):
        z = self.embed(x["ray"], x["ent"], x["mask"], x["scal"])
        h = self.gru(z, h)
        return self.pi(h), self.v(h).squeeze(-1), h

    def sequence(self, x: dict, h0, starts):
        """x: tensors [T,B,...]; starts [T,B] = 1 where a new episode begins (hidden reset). -> logits, values [T,B]."""
        T, Bn = starts.shape
        z = self.embed(*(x[k].reshape(T * Bn, *x[k].shape[2:]) for k in ("ray", "ent", "mask", "scal"))).reshape(T, Bn, -1)
        h, logits, values = h0, [], []
        for t in range(T):
            h = h * (1 - starts[t]).unsqueeze(-1)
            h = self.gru(z[t], h)
            logits.append(self.pi(h)); values.append(self.v(h).squeeze(-1))
        return torch.stack(logits), torch.stack(values)
