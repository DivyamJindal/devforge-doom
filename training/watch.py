"""Watch the submission agent play: records what it sees to an animated GIF (and prints what happened).

  python training/watch.py --tier 3 --seed 5 --out agent.gif        # a fresh generated level of that tier
  python training/watch.py --wad maps/two_rooms.wad --out demo.gif   # any WAD
"""
import argparse, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training")); sys.path.insert(0, str(ROOT / "submission"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=3); ap.add_argument("--seed", type=int, default=900_000_005)
    ap.add_argument("--wad", default=None); ap.add_argument("--map", default="MAP01"); ap.add_argument("--out", default="agent.gif")
    ap.add_argument("--every", type=int, default=2, help="keep every Nth frame")
    a = ap.parse_args()
    from doom_env import make_env
    from ppo import Shaper, TIER_TICKS
    import agent as agent_mod
    if a.wad:
        wad, ticks = a.wad, 4200
    else:
        from mapgen import generate
        wad = "/tmp/egomem_watch.wad"; rows = generate(a.seed, a.tier, wad); ticks = TIER_TICKS[a.tier]
        print("\n".join(r for r in rows if set(r) != {"#"}))
    env = make_env(wad, a.map, episode_ticks=ticks, render_buffers=True)
    ag = agent_mod.DoomAgent(env.observation_space, env.action_space, "cpu")
    obs, _ = env.reset(seed=0); ag.reset(); sh = Shaper(env); frames = []
    while True:
        frames.append(Image.fromarray(np.asarray(obs["rgb"])).resize((320, 240), Image.NEAREST))
        obs, _, te, tr, _ = env.step(ag.act(obs)); ps = env.privileged_state(); sh(ps)
        if te or tr:
            break
    frames = frames[:: a.every]
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=1000 * 4 * a.every // 35, loop=0)
    print(f"\nresult: exit {'REACHED' if ps['exit_reached'] else 'no'} | keys {int(np.sum(ps['keys']))} | doors opened {sh.n_doors} "
          f"| locked doors {sh.n_locks} | kills {ps['kills']} | areas explored {len(sh.cells)} | died {ps['dead']} | {ps['tic'] / 35:.0f}s game time")
    print(f"saved {a.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
