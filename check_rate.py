#!/usr/bin/env python3
"""Step 1 acceptance check. Run on a recording directory.

    python check_rate.py recordings/20260615_143020

PASS means the stream is good enough to feed a 12-sample / 200 ms window.
All timing is measured on sensor_t_us (device clock), never host_t_us.
"""
import sys, glob, os
import numpy as np
import pandas as pd

RATE_HZ      = 60.0
DT_NOMINAL   = 1e6 / RATE_HZ      # 16666.7 us
MAX_GAP_US   = 150_000            # config.json max_gap_ms
SYNC_TOL_US  = 8_333              # config.json sensor_sync, half a sample
MIN_RATE     = 59.0
MIN_DUR_S    = 300.0


def unwrap_us(t):
    """Undo the 32-bit microsecond counter wrap (~71.6 min)."""
    t = np.asarray(t, dtype=np.int64)
    return t + (np.cumsum(np.diff(t, prepend=t[0]) < -(2**31)) * 2**32)


def check(run_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, "imu_*.csv")))
    if not paths:
        print(f"no imu_*.csv in {run_dir}");  return False

    rows, ok = [], True
    for p in paths:
        role = os.path.basename(p)[4:-4]
        df = pd.read_csv(p).dropna()
        if len(df) < 100:
            print(f"  {role}: only {len(df)} samples");  ok = False;  continue

        t   = unwrap_us(df.sensor_t_us.to_numpy())
        dt  = np.diff(t)
        dur = (t[-1] - t[0]) / 1e6
        hz  = len(t) / dur

        n_gap  = int((dt > MAX_GAP_US).sum())
        n_slow = int((dt > 2 * DT_NOMINAL).sum())
        n_back = int((dt <= 0).sum())

        pass_role = (hz >= MIN_RATE and n_gap == 0 and n_back == 0
                     and n_slow / len(dt) < 1e-3)
        ok &= pass_role
        rows.append((role, len(t), dur, hz, np.median(dt) / 1000,
                     dt.max() / 1000, n_gap, n_slow, n_back, pass_role))

    print(f"\n{run_dir}")
    print(f"{'role':11s} {'n':>7} {'dur_s':>7} {'Hz':>6} {'dt_med':>7} "
          f"{'dt_max':>9} {'gaps':>5} {'slow':>5} {'back':>5}  ")
    for r in rows:
        print(f"{r[0]:11s} {r[1]:7d} {r[2]:7.1f} {r[3]:6.2f} {r[4]:7.2f} "
              f"{r[5]:9.1f} {r[6]:5d} {r[7]:5d} {r[8]:5d}  "
              f"{'PASS' if r[9] else 'FAIL'}")

    # all sensors must stream for the same span — a sensor that dies early
    # still looks like 60 Hz over its own shortened lifetime
    if len(rows) > 1:
        durs = [r[2] for r in rows]
        if max(durs) - min(durs) > 2.0:
            print(f"  ! sensors stopped at different times "
                  f"(spread {max(durs)-min(durs):.1f}s) — one dropped out early")
            ok = False

    # duration gate
    if rows and min(r[2] for r in rows) < MIN_DUR_S:
        print(f"  ! session shorter than {MIN_DUR_S:.0f}s "
              f"— rate collapse can appear late, record longer")
        ok = False

    # cross-sensor alignment feasibility: is the offset between device
    # clocks stable enough to align to within half a sample?
    if len(paths) > 1:
        offs = {}
        for p in paths:
            role = os.path.basename(p)[4:-4]
            df = pd.read_csv(p).dropna()
            h = df.host_t_us.to_numpy().astype(np.int64)
            s = unwrap_us(df.sensor_t_us.to_numpy())
            o = h - s
            # rolling minimum = least-delayed packet, rejects BLE queueing
            k = int(10 * RATE_HZ)
            mins = np.array([o[i:i + k].min() for i in range(0, len(o) - k, k)])
            offs[role] = mins
        n = min(len(v) for v in offs.values())
        base = list(offs)[0]
        print()
        if n < 2:
            print("  offset drift: not enough clean data to evaluate")
            return False
        for role, v in offs.items():
            if role == base:
                continue
            d = v[:n] - offs[base][:n]
            drift = d.max() - d.min()
            print(f"  offset {role} - {base}: drift over session = "
                  f"{drift/1000:7.2f} ms  "
                  f"{'PASS' if drift < SYNC_TOL_US else 'FAIL (> half a sample)'}")
            ok &= drift < SYNC_TOL_US

    print(f"\n  => {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [check(d) for d in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)