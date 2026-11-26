"""Massively parallel training on Modal: each run = one container with a GPU learner and ~36 CPU cores of Doom.

  modal run training/modal_train.py --runs "a:default,b:explore"         # launches runs in parallel (detached-safe)
  modal volume get egomem-runs a/egomem.pt submission/weights/egomem.pt  # pull a checkpoint any time

Checkpoints and stats.json stream to the Modal volume `egomem-runs` while training.
"""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parent.parent
image = (modal.Image.debian_slim(python_version="3.13")
         .apt_install("libgl1", "libglib2.0-0", "libsdl2-2.0-0", "libopenal1")
         .pip_install("vizdoom==1.3.3", "gymnasium==1.4.0", "numpy", "torch")
         .add_local_dir(ROOT / "env", "/repo/env", copy=True, ignore=["**/__pycache__"])
         .run_commands("pip install /repo/env")
         .add_local_dir(ROOT / "submission", "/repo/submission", ignore=["weights/*.pt", "**/__pycache__"])
         .add_local_dir(ROOT / "training", "/repo/training", ignore=["**/__pycache__"]))
vol = modal.Volume.from_name("egomem-runs", create_if_missing=True)
app = modal.App("egomem-doom", image=image)

PRESETS = {
    "default": [],
    "explore": ["--ent", "0.03", "--mix", "gen1:0.15,gen2:0.3,gen3:0.3,gen4:0.15,freedoom:0.1"],
    "progress": ["--mix", "gen1:0.15,gen2:0.2,gen3:0.4,gen4:0.15,freedoom:0.1", "--gamma", "0.997"],
    "big": ["--hidden", "384"],
    "fast": ["--lr", "7e-4", "--ent", "0.01", "--ent-end", "0.002", "--epochs", "4", "--minibatches", "4",
             "--mix", "gen0:0.3,gen1:0.1,gen2:0.2,gen3:0.25,gen4:0.05,freedoom:0.1"],
    "smoke": ["--steps", "300000"],
    "polish": ["--lr", "3e-4", "--ent", "0.006", "--ent-end", "0.002", "--epochs", "4", "--minibatches", "4",
               "--mix", "gen0:0.05,gen1:0.15,gen2:0.2,gen3:0.35,gen4:0.2,freedoom:0.05"],
    "survive": ["--lr", "7e-4", "--ent", "0.01", "--ent-end", "0.002", "--epochs", "4", "--minibatches", "4",
                "--mix", "gen0:0.3,gen1:0.1,gen2:0.2,gen3:0.25,gen4:0.05,freedoom:0.1", "--shape", "death=8,dmg=0.02,cell=0.08"],
}


@app.function(gpu=["A10G", "L4", "T4", "L40S"], cpu=36, memory=65536, timeout=24 * 3600, volumes={"/vol": vol})
def train(name: str, preset: str, steps: float, workers: int, extra: list[str]):
    import subprocess, sys, threading, time
    out = f"/vol/{name}"
    extra = [x.replace("VOL:", "/vol/") for x in extra]          # e.g. --init VOL:d/egomem.pt
    cmd = [sys.executable, "/repo/training/ppo.py", "--workers", str(workers), "--envs-per-worker", "6", "--steps", str(steps),
           "--out", out, "--save-every", "20", "--seed", str(abs(hash(name)) % 1000), *PRESETS[preset], *extra]
    stop = False

    def committer():
        while not stop:
            time.sleep(120)
            try: vol.commit()
            except Exception: pass
    threading.Thread(target=committer, daemon=True).start()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="/repo")
    log = open(f"{out}.log", "a") if Path("/vol").exists() else None
    for line in p.stdout:
        print(f"[{name}] {line}", end="", flush=True)
        if log: log.write(line); log.flush()
    p.wait(); stop = True; vol.commit()
    return p.returncode


@app.local_entrypoint()
def main(runs: str = "a:default", steps: float = 3e8, workers: int = 32, extra: str = ""):
    calls = []
    for spec in runs.split(","):
        name, preset = spec.split(":")
        calls.append((name, train.spawn(name, preset, steps, workers, extra.split())))
        print("launched", name, preset)
    for name, c in calls:
        print(name, "exit code", c.get())
