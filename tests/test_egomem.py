"""Tests for our agent, features, memory, level generator and training utilities. Run: uv run pytest -q tests/test_egomem.py"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submission")); sys.path.insert(0, str(ROOT / "training"))

from policy import ACTIONS, MEM_CAP, N_ACT, N_SCALAR, NUM_RAYS, EgoMemNet, Features, place_signature, stack  # noqa: E402
from mapgen import generate, to_udmf, layout, populate, _bfs  # noqa: E402

USE = 7


def fake_obs(dist=None, hit=None, n_ent=0, health=100.0, keys=(0, 0, 0, 0, 0, 0)):
    dist = np.ones(NUM_RAYS, np.float32) if dist is None else dist
    hit = np.zeros(NUM_RAYS, np.int64) if hit is None else hit
    feats = np.zeros((32, 5), np.float32); types = np.zeros(32, np.int64); mask = np.zeros(32, np.int8)
    for i in range(n_ent):
        feats[i] = [0.2 + 0.1 * i, 0.0, 1.0, 0.0, 0.0]; types[i] = 2; mask[i] = 1
    vit = np.zeros(17, np.float32); vit[0] = health; vit[8:14] = keys
    return {"rays": {"distance": dist, "hit_type": hit}, "visible_entities": {"features": feats, "entity_type": types, "mask": mask},
            "vitals": vit}


# ------------------------------------------------------------------ features and memory
def test_feature_shapes():
    f = Features()(fake_obs())
    assert f["ray"].shape == (7, NUM_RAYS) and f["ent"].shape == (32, 17) and f["mask"].shape == (32,)
    assert f["scal"].shape == (N_SCALAR,) and all(np.isfinite(v).all() for v in f.values())


def test_place_signature_is_rotation_invariant():
    rng = np.random.default_rng(0)
    d = rng.random(NUM_RAYS).astype(np.float32); h = rng.integers(0, 6, NUM_RAYS)
    a = place_signature(d, h); b = place_signature(np.roll(d, 17), np.roll(h, 17))
    assert np.allclose(a, b, atol=1e-5)


def test_place_signature_differs_between_places():
    d1 = np.ones(NUM_RAYS, np.float32); d2 = np.linspace(0.1, 1, NUM_RAYS).astype(np.float32)
    h = np.ones(NUM_RAYS, np.int64)
    assert np.abs(place_signature(d1, h) - place_signature(d2, h)).sum() > 0.1


def test_memory_novelty_drops_on_revisit():
    f = Features(); o = fake_obs(dist=np.linspace(0.2, 1, NUM_RAYS).astype(np.float32))
    first = f(o)["scal"][-10]; f(o); f(o); later = f(o)["scal"][-10]
    assert later < first


def test_memory_is_bounded():
    f = Features(); rng = np.random.default_rng(1)
    for _ in range(MEM_CAP * 3):
        f(fake_obs(dist=rng.random(NUM_RAYS).astype(np.float32)))
    assert f.mem.shape[0] == MEM_CAP


def test_reset_clears_memory():
    f = Features(); f(fake_obs()); f(fake_obs()); f.reset()
    assert f.n_mem == 0 and f.t == 0 and f.locked_seen == 0


def test_locked_door_event_memory():
    f = Features(); hit = np.zeros(NUM_RAYS, np.int64); hit[5] = 3
    f(fake_obs(hit=hit)); s = f(fake_obs())["scal"]
    assert f.locked_seen == 1.0 and f.since_locked == 1 and s[-6] == 1.0


def test_damage_memory():
    f = Features(); f(fake_obs(health=100)); f(fake_obs(health=60))
    assert f.dmg_recent > 0.5


def test_keys_reach_features():
    s = Features()(fake_obs(keys=(1, 0, 0, 0, 0, 0)))["scal"]
    assert s[15] == 1.0                                  # vitals: 2 + 8 weapon + 1 ammo + 4 ammo types -> key flags start at 15


# ------------------------------------------------------------------ actions
def test_action_table():
    assert N_ACT == len(ACTIONS) == 15 and all(all(0 <= b < 9 for b in a) for a in ACTIONS)


def test_use_is_pulsed():
    f = Features(); use_action = next(i for i, a in enumerate(ACTIONS) if a == (USE,))
    presses = [f.button_vector(use_action)[USE] for _ in range(6)]
    assert presses == [1, 0, 1, 0, 1, 0]


def test_button_vector_valid():
    f = Features()
    for a in range(N_ACT):
        v = f.button_vector(a)
        assert v.shape == (9,) and set(np.unique(v)) <= {0, 1}


# ------------------------------------------------------------------ network
def test_net_step_and_sequence_agree():
    torch.manual_seed(0); net = EgoMemNet(64).eval(); fe = Features()
    obs = [fake_obs(dist=np.full(NUM_RAYS, 0.1 * (i + 1), np.float32), n_ent=i % 3) for i in range(5)]
    feats = [fe(o) for o in obs]
    h = torch.zeros(1, 64); step_logits = []
    with torch.no_grad():
        for f in feats:
            lg, _, h = net.step({k: torch.from_numpy(v)[None] for k, v in f.items()}, h); step_logits.append(lg[0])
        x = {k: torch.from_numpy(np.stack([f[k] for f in feats]))[:, None] for k in feats[0]}
        seq_logits, _ = net.sequence(x, torch.zeros(1, 64), torch.zeros(5, 1))
    assert torch.allclose(torch.stack(step_logits), seq_logits[:, 0], atol=1e-5)


def test_entity_pooling_ignores_padding():
    torch.manual_seed(0); net = EgoMemNet(64).eval(); f = Features()(fake_obs(n_ent=2))
    x = {k: torch.from_numpy(v)[None] for k, v in f.items()}
    y = {k: v.clone() for k, v in x.items()}; y["ent"][0, 10:] = 123.0          # garbage in masked slots
    with torch.no_grad():
        assert torch.allclose(net.step(x, torch.zeros(1, 64))[0], net.step(y, torch.zeros(1, 64))[0], atol=1e-5)


def test_sequence_resets_hidden_at_episode_start():
    torch.manual_seed(0); net = EgoMemNet(32).eval(); f = Features()(fake_obs())
    x = {k: torch.from_numpy(np.stack([f[k]] * 2))[:, None] for k in f}
    h0 = torch.randn(1, 32)
    with torch.no_grad():
        a, _ = net.sequence(x, h0, torch.tensor([[1.0], [0.0]]))
        b, _ = net.sequence(x, torch.zeros(1, 32), torch.tensor([[1.0], [0.0]]))
    assert torch.allclose(a, b)


def test_stack():
    fe = Features(); s = stack([fe(fake_obs()), fe(fake_obs())])
    assert s["ray"].shape == (2, 7, NUM_RAYS)


# ------------------------------------------------------------------ level generator
@pytest.mark.parametrize("tier", [0, 1, 2, 3, 4])
def test_generated_levels_have_start_and_exit(tier, tmp_path):
    rows = generate(123, tier, tmp_path / "m.wad")
    flat = "".join(rows)
    assert flat.count("P") == 1 and flat.count("E") == 1 and (tmp_path / "m.wad").stat().st_size > 500


@pytest.mark.parametrize("seed", range(6))
def test_locked_levels_are_solvable(seed, tmp_path):
    rows = generate(seed, 3, tmp_path / "m.wad")
    g = [list(r) for r in rows]
    start = next((x, y) for y, r in enumerate(g) for x, c in enumerate(r) if c == "P")
    exit_ = next((x, y) for y, r in enumerate(g) for x, c in enumerate(r) if c == "E")
    keys = {c for r in g for c in r if c in "rby"}
    reach = _bfs(g, start, lambda c: c not in "#RBY")
    for k in keys:                                              # every key reachable without crossing locks
        kpos = next((x, y) for y, r in enumerate(g) for x, c in enumerate(r) if c == k)
        assert kpos in reach
    full = _bfs(g, start, lambda c: c != "#")
    assert exit_ in full


def test_udmf_is_well_formed():
    rng = random.Random(0); g, rooms = layout(rng, 3); g = populate(rng, g, rooms, 3)
    text = to_udmf(g, rng)
    assert text.startswith('namespace = "zdoom";') and text.count("{") == text.count("}")
    assert "special = 243" in text and "type = 1;" in text


def test_generator_is_deterministic(tmp_path):
    a = generate(7, 2, tmp_path / "a.wad"); b = generate(7, 2, tmp_path / "b.wad")
    assert a == b and (tmp_path / "a.wad").read_bytes() == (tmp_path / "b.wad").read_bytes()


# ------------------------------------------------------------------ integration with the official wrapper
def test_generated_level_runs_in_official_env(tmp_path):
    from doom_env import make_env
    generate(5, 3, tmp_path / "m.wad")
    env = make_env(str(tmp_path / "m.wad"), "MAP01", episode_ticks=200, render_buffers=False)
    obs, _ = env.reset(seed=0); f = Features()
    for _ in range(20):
        obs, _, te, tr, _ = env.step(f.button_vector(1)); f(obs)
        if te or tr:
            break
    assert env.mapinfo.num_sectors >= 2
    env.close()


def test_submission_agent_contract():
    from doom_env import make_env
    sys.path.insert(0, str(ROOT / "submission"))
    import agent as agent_mod
    env = make_env(str(ROOT / "maps" / "two_rooms.wad"), "MAP01", episode_ticks=100, render_buffers=False)
    ag = agent_mod.DoomAgent(env.observation_space, env.action_space, "cpu")
    obs, _ = env.reset(seed=0); ag.reset()
    for _ in range(10):
        a = ag.act(obs)
        assert a.shape == (9,) and set(np.unique(a)) <= {0, 1}
        obs, _, te, tr, _ = env.step(a)
        if te or tr:
            break
    env.close()


def test_running_std():
    from ppo import RunningStd
    r = RunningStd(); x = np.random.default_rng(0).normal(0, 3, 10000); r.update(x)
    assert abs(r.std - 3) < 0.1
