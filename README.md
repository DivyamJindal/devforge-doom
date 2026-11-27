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
  memory ────────► episodic place memory + event memory ┤
  RGB + depth ───► small CNN (30×40, sees exit pads/switches) ┘
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
| **Exits are invisible to rays** | A small CNN on a 30×40 RGB + depth frame (vision branch). Exit pads and exit switches never appear in rays or entities, so the agent previously found exits only by stumbling onto them. Adding vision raised gauntlet exits from 25% to 50% on held-out maps. |
| **USE semantics** | USE fires on the button-down edge, so our action encoder pulses it (held USE is released every other step) and doors don't bounce. |

**Pure RL ships.** The shipped policy is trained only with our recurrent PPO. Following the organisers' guidance
("Behavioral Cloning / Imitation Learning → PPO fine-tuning") we also tried DAgger from a training-only teacher
(`training/expert.py`, which knows the generated map and is never imported by `submission/`). Applied on top of our PPO agent,
the imitating student scored *lower* (tier-3 keycards 0.23-0.39 per level vs 0.56), so the shipped weights are pure PPO.

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

The shipped weights come from RL (our recurrent PPO): run `v3`, which is run `h` (88M + 17M steps of ray/entity training)
with the **vision branch** grafted on (zero-initialised, so it starts identical) and trained 39M more steps with pixels
using the `polish` preset (tier 3-4 heavy level mix, lr 3e-4) plus the curiosity bonus (`v1` was the same graft, an
independent run). Scored by the exact submission agent on levels it never saw: fresh generator
seeds outside the training range, plus 6 Freedoom 2 maps excluded from training. Sampling temperature 0.75.

| held-out set | n | exit reached | keycards | doors opened | locked doors | areas explored | kills | died |
|---|---|---|---|---|---|---|---|---|
| Tier 1, combat | 24 | 54% | n/a | n/a | n/a | 22 | 4.3 | 17% |
| Tier 2, multi-room | 24 | **50%** | n/a | 2.7 | n/a | 46 | 3.5 | 12% |
| Tier 3, key → lock → exit | 24 | 33% | **0.71** | 2.6 | **0.50** | 53 | 3.8 | 12% |
| Tier 4, gauntlet | 24 | 33% | **0.71** | 2.1 | **0.54** | 46 | 6.8 | 25% |
| Freedoom 2 (real maps) | 6 | 0% | 0 | 1.5 | 0 | 26 | 3.2 | 83% |

Re-check on a **fresh block of seeds never used for any decision** (`--seed0 950000000`, 32 levels per tier, paired):
v3 scores **41.8** vs v1 35.3 and is ahead on every tier (exits 56/31/28/25% vs 47/22/25/19%). Of the levels v3 does not
finish, about 40% end in death and 60% in a timeout, so both survival and navigation still have headroom.

Learning curve on the same held-out suite (tier-weighted proxy of the scorer's categories; 12 maps per tier, noise about ±4, so ship decisions use 24 per tier):

| checkpoint | steps | proxy |
|---|---|---|
| random policy | 0 | ≈ -3 |
| run `d` | 4M | -1.6 |
| run `e` | 88M | 32.3 |
| run `h` (`e` polished on tiers 3-4) | 88M + 17M | 40.0 |
| run `v1` (`h` + vision branch) | +36M | 43.0 (41.2 on 24/tier) |
| **run `v3` (`h` + vision branch, shipped)** | **+39M** | **45.3 (45.5 on 24/tier)** |

Latency: 2-5 ms per decision on one CPU thread (budget 40 ms); the official validator passes, and the agent reaches the
exit of the template test map in 59 steps.

**What did not work, and why it matters.** Lock-step vectorised training was bottlenecked at ~1k steps/s by level loading;
asynchronous workers with double-buffered level boot raised it to ~5k steps/s per run. Velocity dead-reckoning for a
metric map drifted too much to be useful, which is why the memory is episodic and place-based. A higher sampling
temperature (1.0) and a lower one (0.5) both scored slightly worse than 0.75.
Adding observation-based action masks and a "stuck for 200 tics ends the episode" rule to an already-trained policy
(both borrowed from another team) cut its training-time exit rate from about 30% to 6% on the first update and the run
never recovered (held-out 39 vs 45): the policy was trained without them, so they are off by default now
(`--masks`, `--shape stuck_tics=0`).

**Built and training now (not in the shipped weights yet):** an ego-centric occupancy map (`--omap`). Heading is
dead-reckoned from the agent's own turn presses with DOOM's exact turn model (5 slow tics at 1.76 deg, then 3.52 deg/tic,
checked against ground truth in `tests/`), position from the observed velocity (fit on ground truth, about 0.6 map
units of error per step). Rays paint free space, walls and doors into a world grid; a 21x21 crop rotated to the agent's
heading feeds a small CNN, zero-initialised so it starts identical to v3. Also training-only path-distance shaping
(`--shape geo=0.1`: reward for getting closer, along the level's corridors, to the next missing key or the exit) and
levels whose exit is a wall switch (`--switch 0.25`), as in real DOOM maps. These ship only if they beat v3 on held-out
levels.

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
