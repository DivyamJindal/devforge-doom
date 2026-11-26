# EgoMem: a DOOM agent with episodic place memory

**Yugaantar 2026 · DevForge, Track 1: "Can It Play DOOM?"** · built on the official
[`devforge-doom-template`](https://github.com/Reinforce-SST/devforge-doom-template).

EgoMem plays unseen DOOM levels zero-shot from the official ego-centric observations (rays, visible entities,
vitals). It was trained with **our own recurrent PPO** on an **endless stream of procedurally generated levels** that
mirror the four hidden tiers, so it cannot memorise any layout.

```
                 ┌─────────────────────────── observation (official doom_env) ───────────────────────────┐
  64 rays ───────► circular 1-D CNN (ring has no seam) ─┐
  32 entities ───► per-entity MLP → attention + max pool ┼─► fuse ─► GRU (256) ─► 15 macro-actions + value
  vitals, keys ──► MLP ─────────────────────────────────┤
  memory ────────► episodic place memory + event memory ┘
                   (have I been here? locked door seen? how long ago? keys held? hurt recently?)
```

## Why this design

| Problem in the brief | What EgoMem does |
|---|---|
| **Unseen maps** (no memorising) | Every training episode is a fresh level from our generator (`training/mapgen.py`), across 4 tiers: combat arenas, multi-room mazes with doors and dead ends, keycard → locked door → exit chains, and gauntlets. Real Freedoom 2 maps are mixed in, with 6 held out for evaluation. |
| **Partial observability, spatial memory** | *Episodic place memory*: each step, a rotation-invariant signature of the surroundings (FFT magnitude of the 64-ray ring plus a hit-type histogram) is compared with every place visited this episode. The novelty and familiarity signals tell the agent "I've been here" without any coordinates. We first tried velocity dead-reckoning for a metric map; it drifted ~800 map units in 300 steps, so we dropped it. |
| **Long horizon (keycards, locked doors, backtracking)** | *Event memory* (locked door seen and how long ago, door seen, keys held from vitals) feeds a GRU. Shaped rewards make keycards and opened locked doors worth far more than wandering. |
| **Combat and survival** | Permutation-invariant attention over visible entities (any number of enemies), damage-recency signal, and kill/damage reward terms. |
| **Variable enemy count** | Masked attention pooling + max pooling over up to 32 entity slots. |
| **Rotation** | Circular convolutions over the ray ring, so an obstacle "behind-left" is encoded the same way wherever it sits on the ring. |
| **USE semantics** | USE fires on the button-down edge, so our action encoder pulses it (held USE is released every other step) and doors don't bounce. |

**Pure RL.** The shipped policy is trained only with our recurrent PPO from random initialisation (with warm starts from our own earlier PPO runs). There is no imitation of scripted behaviour.

All decisions come from the trained network. There is no pathfinding, no hand-written rules and no aimbot; the agent
samples from its learned action distribution with a fixed seed.

## Training at scale

- **Procedural levels** (`training/mapgen.py`): random BSP-style rooms + corridors → UDMF maps in the same format as
  the official test fixtures (doors with special 12, locked doors with special 13 and ZDoom lock numbers, exits with
  special 243, keycards, monsters, health, ammo, weapons, armour). Every tier-3/4 level is checked to be solvable:
  the key is reachable without passing its lock, and the exit is behind the lock.
- **Recurrent PPO from scratch** (`training/ppo.py`): GAE, BPTT over 64-step segments, return-scaled rewards,
  entropy annealing.
- **Asynchronous rollouts**: each worker process runs 6 games with a shared-memory copy of the policy and returns whole
  trajectories. Level loading is double-buffered in a background thread, so finishing an episode never stalls
  training. The GPU learner updates while the next rollout is collected.
- **Modal** (`training/modal_train.py`): several configurations train in parallel, each on a 36-core box with a GPU
  learner, streaming checkpoints to a volume.
- **Reward shaping from ground truth** (training only; the agent never sees it): new 128-unit areas, kills, pickups,
  keycards (+4), doors (+1), locked doors (+3), exit (+15 plus time/health bonus), minus damage, death and time.

## Results (held-out maps, `training/evaluate.py`)

The shipped weights come from **pure RL** (our recurrent PPO, ~83M environment steps, run `e`: warm-started from run
`d`, curriculum with an easy tier-0 on-ramp). Scored by the exact submission agent on levels it never saw: fresh generator
seeds outside the training range, plus 6 Freedoom 2 maps excluded from training. Sampling temperature 0.75.

| held-out set | n | exit reached | keycards | doors opened | locked doors | areas explored | kills | died |
|---|---|---|---|---|---|---|---|---|
| Tier 1, combat | 8 | **75%** | n/a | n/a | n/a | 19 | 3.9 | 12% |
| Tier 2, multi-room | 8 | 38% | n/a | 2.5 | n/a | 47 | 3.4 | 25% |
| Tier 3, key → lock → exit | 8 | 25% | **0.50** | 1.9 | **0.38** | 40 | 2.5 | 12% |
| Tier 4, gauntlet | 8 | 0% | 0.50 | 1.1 | 0.12 | 35 | 4.2 | 50% |
| Freedoom 2 (real maps) | 6 | 0% | 0 | 1.0 | 0 | 28 | 2.8 | 100% |

Learning curve on the same held-out suite (tier-weighted proxy of the scorer's categories):

| checkpoint | steps | proxy |
|---|---|---|
| random policy | 0 | ≈ -3 |
| run `d` | 4M | -1.6 |
| run `e` | 83M | **30.2** |

Latency: 3-7 ms per decision on one CPU thread (budget 40 ms); the official validator passes, and the agent reaches the
exit of the template test map in 66 steps.

**What did not work, and why it matters.** Lock-step vectorised training was bottlenecked at ~1k steps/s by level loading;
asynchronous workers with double-buffered level boot raised it to ~5k steps/s per run. Velocity dead-reckoning for a
metric map drifted too much to be useful, which is why the memory is episodic and place-based. A higher sampling
temperature (1.0) and a lower one (0.5) both scored slightly worse than 0.75.

## Run it

```bash
uv sync
uv run python validate.py                       # official checks (passes)
uv run python training/evaluate.py              # held-out scorecard with the submission agent
uv run python training/ppo.py --workers 8 --envs-per-worker 4 --steps 2e6 --out runs/local     # train locally
modal run --detach training/modal_train.py --runs "a:default,b:explore"                        # train on Modal
uv run python training/mapgen.py --tier 3 --n 4 --out /tmp/maps                                 # preview levels
```

## Repository layout

```
submission/agent.py      DoomAgent (official entry point)
submission/policy.py     features, episodic memory, EgoMemNet (shared by training and evaluation)
submission/weights/      egomem.pt (Git LFS)
training/mapgen.py       procedural level generator (4 tiers)
training/ppo.py          recurrent PPO, async workers, reward shaping
training/modal_train.py  parallel cloud training
training/evaluate.py     held-out scorecard
env/, maps/, docs/, validate.py   official template (unchanged)
```
