"""Ship a checkpoint as the submission, but only if every gate passes. Does not commit; prints what to commit.

  python training/ship.py runs/om2/egomem_35M.pt

Gates: the checkpoint loads in the exact submission agent, the official validator passes, the test suite passes.
It also refreshes the plain-git copy of the weights (used when a clone has no Git LFS). On any failure the previous
weights are restored, so a bad checkpoint can never be left in submission/.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "submission" / "weights"
TARGETS = (WEIGHTS / "egomem.pt", WEIGHTS / "egomem_nolfs.raw")


def check_checkpoint(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "state" not in ck or "cfg" not in ck:
        raise SystemExit(f"{path}: not an EgoMem checkpoint (needs 'state' and 'cfg')")
    sys.path.insert(0, str(ROOT / "submission"))
    from policy import EgoMemNet
    EgoMemNet(**{k: v for k, v in ck["cfg"].items() if k != "masks"}).load_state_dict(ck["state"])  # strict
    return ck


def run(cmd) -> bool:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = r.stdout + r.stderr
    print(out.strip().splitlines()[-1] if out.strip() else "")
    return r.returncode == 0 and "FAILED" not in out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    a = ap.parse_args()
    ck = check_checkpoint(a.checkpoint)
    print(f"checkpoint ok: cfg {ck['cfg']}, {ck.get('steps', 0) / 1e6:.1f}M steps")
    backup = {t: t.read_bytes() for t in TARGETS if t.exists()}
    for t in TARGETS:
        shutil.copyfile(a.checkpoint, t)
    py = sys.executable
    ok = run([py, "validate.py"]) and run([py, "-m", "pytest", "-q", "tests/test_egomem.py"])
    if not ok:
        for t, data in backup.items():
            t.write_bytes(data)
        raise SystemExit("gate failed: previous weights restored")
    print("shipped. now: update the README / brief tables, then\n"
          "  git add submission/weights && git commit && git push")


if __name__ == "__main__":
    main()
