"""Evaluation entry point: the EgoMem agent (see policy.py for the architecture).

Every action comes from the trained network: we sample from its action distribution (a little stochasticity helps
it escape dead ends, and is what it was trained with), using a fixed seed so runs are reproducible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
sys.path.insert(0, str(SUBMISSION_DIR))
from policy import EgoMemNet, Features  # noqa: E402

TEMPERATURE = 0.75                                     # held-out: 0.5 -> 29.2, 0.75 -> 30.2, 1.0 -> 29.6


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        self.observation_space = observation_space
        self.action_space = action_space
        torch.set_num_threads(1)
        self.device = "cpu"                                    # tiny network: CPU is faster than a GPU round trip
        path = WEIGHTS_DIR / "egomem.pt"
        if path.stat().st_size < 4096:                         # an un-fetched Git LFS pointer: use the plain-git copy
            path = WEIGHTS_DIR / "egomem_nolfs.raw"
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.net = EgoMemNet(**{k: v for k, v in ck.get("cfg", {}).items() if k != "masks"})
        self.net.load_state_dict(ck["state"]); self.net.eval()
        self.feat = Features(vision=self.net.vision)
        self.use_masks = bool(ck.get("cfg", {}).get("masks", False))   # only for checkpoints trained with action masks
        self.h = torch.zeros(1, self.net.hidden)
        self.gen = torch.Generator().manual_seed(0)
        with torch.no_grad():                                   # warm up kernels outside the timed act() calls
            self.reset(); self._decide(self._dummy())
        self.reset()

    def _dummy(self):
        return {"rays": {"distance": np.ones(64, np.float32), "hit_type": np.zeros(64, np.int64)},
                "visible_entities": {"features": np.zeros((32, 5), np.float32), "entity_type": np.zeros(32, np.int64),
                                     "mask": np.zeros(32, np.int8)},
                "vitals": np.array([100] + [0] * 16, np.float32)}

    def reset(self) -> None:
        self.feat.reset()
        self.h = torch.zeros(1, self.net.hidden)
        self.gen.manual_seed(0)

    @torch.no_grad()
    def _decide(self, observation) -> int:
        f = self.feat(observation)
        x = {k: torch.from_numpy(np.ascontiguousarray(v))[None] for k, v in f.items() if k != "amask" or self.use_masks}
        logits, _, self.h = self.net.step(x, self.h)
        p = torch.softmax(logits[0] / TEMPERATURE, -1)
        return int(torch.multinomial(p, 1, generator=self.gen))

    def act(self, observation: dict) -> np.ndarray:
        return self.feat.button_vector(self._decide(observation))
