"""Run anytime during training: python3 check_progress.py"""
import glob, os, numpy as np

print("\n── Checkpoint status ──────────────────────────────")
zips = sorted(glob.glob("results/ppo_model_*.zip"))
if zips:
    for z in zips:
        size = os.path.getsize(z) / 1024
        mtime = os.path.getmtime(z)
        import time
        age = (time.time() - mtime) / 60
        print(f"  {z:<45} {size:>7.1f} KB  ({age:.0f} min ago)")
else:
    print("  No models saved yet — training still in phase 1")

print("\n── Latest checkpoints ─────────────────────────────")
ckpts = sorted(glob.glob("results/checkpoints/*/*.zip"), key=os.path.getmtime)
if ckpts:
    for c in ckpts[-5:]:
        print(f"  {c}")
else:
    print("  No checkpoints yet")

print("\n── Log dirs ────────────────────────────────────────")
logs = sorted(glob.glob("logs/PPO_*/"))
for l in logs:
    files = glob.glob(l + "events.*")
    if files:
        sz = os.path.getsize(files[0]) / 1024
        print(f"  {l:<40} {sz:>7.1f} KB")
print()
