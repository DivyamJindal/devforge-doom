"""Why does the agent fail? Breaks the non-exits of an `evaluate.py --json` run down by tier and cause.

  python training/diagnose.py eval.json

Causes: died (combat / hazards), timed out after exploring a lot (could not find or reach the exit: navigation),
timed out after exploring little (stuck or looping: movement). The worst levels are listed so they can be replayed
with `training/watch.py --tier T --seed 900000000+N`.
"""
import argparse
import json
from collections import Counter

LOW_CELLS = 15          # 128-unit areas; fewer than this in a whole episode means the agent barely moved


def cause(m) -> str:
    if m["exit"]:
        return "exit"
    if m["dead"]:
        return "died"
    return "lost" if m["cells"] >= LOW_CELLS else "stuck"


def summarise(maps):
    by_src = {}
    for m in maps:
        by_src.setdefault(m["src"], Counter())[cause(m)] += 1
    return by_src


def worst(maps, n=5):
    fails = [m for m in maps if not m["exit"]]
    return sorted(fails, key=lambda m: m["proxy"])[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json"); ap.add_argument("--worst", type=int, default=5)
    a = ap.parse_args()
    maps = json.load(open(a.json))["maps"]
    print(f"{'set':9s} {'n':>3s} {'exit':>5s} {'died':>5s} {'lost':>5s} {'stuck':>5s}")
    total = Counter()
    for src, c in sorted(summarise(maps).items()):
        n = sum(c.values()); total += c
        print(f"{src:9s} {n:3d} {c['exit'] / n:5.0%} {c['died'] / n:5.0%} {c['lost'] / n:5.0%} {c['stuck'] / n:5.0%}")
    n = sum(total.values())
    print(f"{'all':9s} {n:3d} {total['exit'] / n:5.0%} {total['died'] / n:5.0%} {total['lost'] / n:5.0%} {total['stuck'] / n:5.0%}")
    print("\nworst levels (replay with training/watch.py):")
    for m in worst(maps, a.worst):
        print(f"  {m['src']:9s} map {m['map']:>6s}  {cause(m):5s}  proxy {m['proxy']:6.1f}  areas {m['cells']:3d}  kills {m['kills']}")


if __name__ == "__main__":
    main()
