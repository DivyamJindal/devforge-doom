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

The shipped weights come from RL (our recurrent PPO): run `om2` = run `v3` (ray/entity network + vision branch, see the
curve below) with the **ego occupancy map** grafted on (zero-initialised, so it starts identical to v3) and trained 35M more
steps with training-only path-distance shaping and 25% wall-switch exits, then 11M more at a lower learning rate (run `om3`, shipped). Scored by the exact submission agent on levels
it never saw: fresh generator seeds outside the training range, plus 6 Freedoom 2 maps excluded from training.
Sampling temperature 0.75.

| held-out set | n | exit reached | keycards | doors opened | locked doors | areas explored | kills | died |
|---|---|---|---|---|---|---|---|---|
| Tier 1, combat | 24 | 62% | n/a | n/a | n/a | 17 | 3.2 | 29% |
| Tier 2, multi-room | 24 | **88%** | n/a | 3.0 | n/a | 42 | 3.1 | 8% |
| Tier 3, key → lock → exit | 24 | **62%** | 0.79 | 3.2 | **0.75** | 53 | 3.2 | 12% |
| Tier 4, gauntlet | 24 | **58%** | **0.88** | 2.5 | 0.67 | 56 | 7.5 | 29% |
| Freedoom 2 (real maps) | 6 | 0% | 0 | 1.2 | 0 | 21 | 2.5 | 67% |

Re-check on a **fresh block of seeds never used for any decision** (`--seed0 950000000`, 32 levels per tier, paired):
om3 **50.2**, om2@35M 47.5, om2@25M 42.4, v3 41.8, v1 35.3. Exits per tier: om3 66/81/69/34%, om2@35M 59/78/56/31%, om2@25M 62/75/50/22%, v3 56/31/28/25%, v1 47/22/25/19%. `training/compare.py` pairs the two
runs level by level and bootstraps the difference: om2@35M beats v3 by **+5.8 (90% CI +0.2 .. +11.6)** and om2@25M by
+5.1 (+1.1 .. +9.3), so the gains are not seed luck. om3 vs om2@35M: +2.7 (-3.2 .. +8.3), ahead on both suites and on every generated tier's exit rate, so it ships, but that step on its own is within noise. The occupancy map
more than doubles multi-room exits; the gauntlet (dense ambushes) is where it still dies most. Of the levels v3 did not
finish, about 40% end in death and 60% in a timeout, so both survival and navigation still have headroom.

Learning curve on the same held-out suite (tier-weighted proxy of the scorer's categories; 12 maps per tier, noise about ±4, so ship decisions use 24 per tier):

| checkpoint | steps | proxy |
|---|---|---|
| random policy | 0 | ≈ -3 |
| run `d` | 4M | -1.6 |
| run `e` | 88M | 32.3 |
| run `h` (`e` polished on tiers 3-4) | 88M + 17M | 40.0 |
| run `v1` (`h` + vision branch) | +36M | 43.0 (41.2 on 24/tier) |
| run `v3` (`h` + vision branch) | +39M | 45.3 (45.5 on 24/tier) |
| run `om2` @ 25M (`v3` + occupancy map + path shaping) | +26M | 45.0 on 24/tier |
| run `om2` @ 35M | +35M | 53.3 on 24/tier, exits 79/83/58/42% |
| **run `om3` (`om2`@35M + 11M at lr 1.5e-4, shipped)** | **+11M** | **56.7 on 24/tier, exits 62/88/62/58%** |

Latency: 2-5 ms per decision on one CPU thread (budget 40 ms); the official validator passes, and the agent reaches the
exit of the template test map in 59 steps.

**Where it still fails** (`training/diagnose.py` on the fresh-seed run): of all held-out levels (shipped om3), 60% end at the exit,
31% in death, 8% lost (explored a lot, never reached the exit) and 1% stuck. Death dominates the gauntlet (47%),
so survival, not navigation, is the next thing to train.

**Tooling for honest ship decisions:** `training/evaluate.py` (held-out levels, `--seed0` for a fresh block),
`training/compare.py` (paired per-level bootstrap CI between two checkpoints), `training/diagnose.py` (failure
causes and worst levels to replay with `training/watch.py`), `training/ship.py` (installs a checkpoint only if it loads
strictly in the submission agent and the official validator + tests pass; restores the old weights otherwise).

**What did not work, and why it matters.** Lock-step vectorised training was bottlenecked at ~1k steps/s by level loading;
asynchronous workers with double-buffered level boot raised it to ~5k steps/s per run. Velocity dead-reckoning for a
metric map drifted too much to be useful, which is why the memory is episodic and place-based. A higher sampling
temperature (1.0) and a lower one (0.5) both scored slightly worse than 0.75.
Adding observation-based action masks and a "stuck for 200 tics ends the episode" rule to an already-trained policy
(both borrowed from another team) cut its training-time exit rate on multi-room levels from about 30% to 6% straight away, and a run trained that
way (`v6`) ended at a held-out 39 vs v3's 45: the policy was trained without them, so they are off by default now
(`--masks`, `--shape stuck_tics=0`).

**Ego-centric occupancy map (shipped in `om2`):** (`--omap`). Heading is
dead-reckoned from the agent's own turn presses with DOOM's exact turn model (5 slow tics at 1.76 deg, then 3.52 deg/tic,
checked against ground truth in `tests/`), position from the observed velocity (fit on ground truth, about 0.6 map
units of error per step). Rays paint free space, walls and doors into a world grid; a 21x21 crop rotated to the agent's
heading feeds a small CNN, zero-initialised so it starts identical to v3. Also training-only path-distance shaping
(`--shape geo=0.1`: reward for getting closer, along the level's corridors, to the next missing key or the exit) and
levels whose exit is a wall switch (`--switch 0.25`), as in real DOOM maps. They shipped because they beat v3 on two held-out
suites. Path shaping without the map (run `geo2`) scored lower (42.5), so the map is what makes it work.

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
