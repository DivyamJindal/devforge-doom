"""Paired comparison of two checkpoints scored on the SAME held-out levels (two `evaluate.py --json` files).

  python training/evaluate.py --weights a.pt --per-tier 32 --seed0 950000000 --temp 0.75 --json a.json
  python training/evaluate.py --weights b.pt --per-tier 32 --seed0 950000000 --temp 0.75 --json b.json
  python training/compare.py a.json b.json

Levels differ far more from each other than checkpoints do, so we compare per level (paired) and bootstrap the
tier-weighted difference. "B better" is only claimed when the 90% interval excludes zero.
"""
import argparse
import json
import random

TIER_W = {"gen1": 1, "gen2": 2, "gen3": 3, "gen4": 4, "freedoom": 4}


def load(path):
    return {(m["src"], m["map"]): m for m in json.load(open(path))["maps"]}


def weighted(diffs_by_src):
    """Tier-weighted mean of per-set mean differences (same weighting as evaluate.py's proxy score)."""
    num = sum(TIER_W[s] * sum(d) / len(d) for s, d in diffs_by_src.items() if d)
    den = sum(TIER_W[s] for s, d in diffs_by_src.items() if d)
    return num / den


def paired(a, b, key="proxy"):
    shared = sorted(set(a) & set(b))
    by_src = {}
    for k in shared:
        by_src.setdefault(k[0], []).append(b[k][key] - a[k][key])
    return by_src


def bootstrap(by_src, n=2000, seed=0):
    rng = random.Random(seed); out = []
    for _ in range(n):
        out.append(weighted({s: [rng.choice(d) for _ in d] for s, d in by_src.items()}))
    out.sort()
    return out[int(0.05 * n)], out[int(0.95 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    by_src = paired(a, b)
    if not by_src:
        raise SystemExit("no shared levels: score both checkpoints with the same --per-tier/--seed0")
    print(f"{'set':9s} {'n':>3s} {'exit A':>7s} {'exit B':>7s} {'proxy diff':>11s}")
    for s in sorted(by_src):
        ks = [k for k in a if k[0] == s and k in b]
        ea = sum(a[k]["exit"] for k in ks) / len(ks); eb = sum(b[k]["exit"] for k in ks) / len(ks)
        print(f"{s:9s} {len(ks):3d} {ea:7.2f} {eb:7.2f} {sum(by_src[s]) / len(by_src[s]):+11.1f}")
    d = weighted(by_src); lo, hi = bootstrap(by_src, args.boot)
    verdict = "B better" if lo > 0 else "A better" if hi < 0 else "no clear difference"
    print(f"tier-weighted proxy difference B - A: {d:+.1f}  (90% CI {lo:+.1f} .. {hi:+.1f})  -> {verdict}")


if __name__ == "__main__":
    main()
