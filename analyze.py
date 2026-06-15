# analyze.py — characterize one recording for slides.
# Usage: python analyze.py recordings/<timestamp>
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

run = Path(sys.argv[1])
cfg = json.loads((run / "run.json").read_text())
feature, thr = cfg["feature"], cfg["threshold"]

# --- per-IMU effective rate (from host timestamps) ---
print(f"run: {run.name}   feature={feature}  threshold={thr}  refractory={cfg['refractory_s']}s")
imus = {}
for p in sorted(run.glob("imu_*.csv")):
    df = pd.read_csv(p)
    imus[p.stem.replace("imu_", "")] = df
    dt = np.diff(df["host_t_us"].to_numpy()) / 1e6           # inter-sample, seconds
    hz = 1.0 / dt
    print(f"  {p.stem:14s} n={len(df):6d}  rate={hz.mean():5.1f} ± {hz.std():4.1f} Hz  "
          f"max gap={dt.max()*1e3:5.1f} ms")

# --- trigger behaviour ---
ev = pd.read_csv(run / "events.csv")
print(f"  pulses: {len(ev)}")
if len(ev) > 1:
    iti = np.diff(ev["host_t_us"].to_numpy()) / 1e6
    print(f"  inter-pulse: min={iti.min():.2f}s (refractory={cfg['refractory_s']}s — "
          f"{'OK' if iti.min() >= cfg['refractory_s'] - 0.05 else 'VIOLATED'})")

# --- figure: source feature over time, threshold, fire markers ---
src = imus[cfg["source_role"]]
t = (src["host_t_us"] - src["host_t_us"].iloc[0]) / 1e6
mag = np.linalg.norm(src[["ax", "ay", "az"]].to_numpy(), axis=1) if feature == "acc_magnitude" \
    else np.linalg.norm(src[["gx", "gy", "gz"]].to_numpy(), axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [3, 1]})
ax1.plot(t, mag, lw=0.8, color="#1f6f80")
ax1.axhline(thr, color="#d08a2e", ls="--", lw=1, label=f"threshold {thr}")
for et in ev["host_t_us"]:
    ax1.axvline((et - src["host_t_us"].iloc[0]) / 1e6, color="#d04a6a", lw=1, alpha=.7)
ax1.set(xlabel="time (s)", ylabel=feature, title=f"{cfg['source_role']} — fires marked")
ax1.legend(fontsize=8, loc="upper right")
ax2.hist(mag, bins=40, orientation="horizontal", color="#1f6f80", alpha=.8)
ax2.axhline(thr, color="#d08a2e", ls="--", lw=1)
ax2.set(xlabel="count", title="rest vs reach")
fig.tight_layout()
out = run / "characterization.png"
fig.savefig(out, dpi=150)
print(f"  figure -> {out}")