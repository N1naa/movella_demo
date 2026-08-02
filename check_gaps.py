#!/usr/bin/env python3
"""Step 2 acceptance check: did the event loop stall, and does the clock survive the 32-bit wrap?

    python check_gaps.py recordings/20260615_143020
    python check_gaps.py recordings/*                 # several at once

check_rate.py asks "is the stream fast enough on average". This asks the two questions Step 2 is
actually about:

  host_t_us   - stamped in the BLE callback, so a gap here is the HOST stalling. A simultaneous
                gap on all three sensors at the same host timestamp is an event-loop block, not
                radio: the radio cannot stall three independent links in lockstep.
  sensor_t_us - stamped on the device, so a gap here is a genuinely lost packet.

PASS requires, per sensor: no host gap > 500 ms, no sensor gap > 150 ms (config.json max_gap_ms),
and unwrapped sensor_t_us strictly increasing. Truncated/NaN rows are reported and fail too - the
tail row of recordings/20260615_140345 is all-NaN from a writerow interrupted at close, which is
exactly what the queued recorder is meant to make impossible.
"""
import sys
import glob
import os

import numpy as np
import pandas as pd

from check_rate import unwrap_us          # single source of truth for the wrap rule

HOST_GAP_US = 500_000        # a host stall this long stalls every stream at once
SENSOR_GAP_US = 150_000      # config.json max_gap_ms
COINCIDENT_US = 100_000      # how close two sensors' host gaps must be to count as the same stall
WRAP = 2 ** 32


def analyse(path):
    """Per-sensor timing facts for one imu_*.csv."""
    role = os.path.basename(path)[4:-4]
    raw = pd.read_csv(path)
    df = raw.dropna()
    n_bad = len(raw) - len(df)               # NaN / truncated rows, e.g. a half-written tail
    if len(df) < 2:
        return dict(role=role, n=len(df), n_bad=n_bad, empty=True)

    host = df.host_t_us.to_numpy(np.int64)
    raw_s = df.sensor_t_us.to_numpy(np.int64)
    sensor = unwrap_us(raw_s)

    dh = np.diff(host)
    ds = np.diff(sensor)
    host_bad = np.flatnonzero(dh > HOST_GAP_US)
    sensor_bad = np.flatnonzero(ds > SENSOR_GAP_US)
    wraps = np.flatnonzero(np.diff(raw_s) < -(WRAP // 2))     # where the counter rolled over

    return dict(
        role=role, n=len(df), n_bad=n_bad, empty=False,
        dur_s=(sensor[-1] - sensor[0]) / 1e6,
        host_max_us=int(dh.max()), sensor_max_us=int(ds.max()),
        host_bad=[(int(host[i]), int(dh[i])) for i in host_bad],
        sensor_bad=[(int(sensor[i]), int(ds[i])) for i in sensor_bad],
        monotonic=bool((ds > 0).all()),
        n_backwards=int((ds <= 0).sum()),
        # (raw counter just before the roll, dt across the boundary, minutes into the recording)
        wraps=[(int(raw_s[i]), int(sensor[i + 1] - sensor[i]),
                (sensor[i] - sensor[0]) / 60e6) for i in wraps],
    )


def check(run_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, "imu_*.csv")))
    if not paths:
        print(f"\n{run_dir}\n  no imu_*.csv here")
        return False

    rows = [analyse(p) for p in paths]
    print(f"\n{run_dir}")
    print(f"  {'role':11s} {'n':>7} {'dur_s':>7} {'host_max':>10} {'sensor_max':>11} "
          f"{'host>500ms':>10} {'sens>150ms':>10} {'monotonic':>10} {'bad_rows':>9}")

    ok = True
    for r in rows:
        if r["empty"]:
            print(f"  {r['role']:11s} {r['n']:7d}   (too few rows to analyse)")
            ok = False
            continue
        good = (not r["host_bad"]) and (not r["sensor_bad"]) and r["monotonic"] and not r["n_bad"]
        ok &= good
        print(f"  {r['role']:11s} {r['n']:7d} {r['dur_s']:7.1f} "
              f"{r['host_max_us'] / 1000:9.1f}ms {r['sensor_max_us'] / 1000:10.1f}ms "
              f"{len(r['host_bad']):10d} {len(r['sensor_bad']):10d} "
              f"{str(r['monotonic']):>10s} {r['n_bad']:9d}  {'PASS' if good else 'FAIL'}")

    # detail lines only where something is wrong, so a clean run stays one table
    for r in rows:
        if r["empty"]:
            continue
        for t, d in r["host_bad"][:10]:
            print(f"    ! {r['role']}: host gap {d / 1000:.0f} ms at host_t_us={t}")
        if len(r["host_bad"]) > 10:
            print(f"    ! {r['role']}: ... and {len(r['host_bad']) - 10} more host gaps")
        for t, d in r["sensor_bad"][:10]:
            print(f"    ! {r['role']}: sensor gap {d / 1000:.0f} ms at sensor_t_us={t}")
        if len(r["sensor_bad"]) > 10:
            print(f"    ! {r['role']}: ... and {len(r['sensor_bad']) - 10} more sensor gaps")
        if r["n_backwards"]:
            print(f"    ! {r['role']}: {r['n_backwards']} non-increasing step(s) after unwrap")
        if r["n_bad"]:
            print(f"    ! {r['role']}: {r['n_bad']} NaN/truncated row(s) - a partial write")

    # A host gap that hits every sensor at the same moment is the event-loop signature: the radio
    # cannot stall three independent links in lockstep. The stamps are not bit-identical (each
    # callback runs a few ms apart), so co-occurrence is judged within COINCIDENT_US.
    live = [r for r in rows if not r["empty"] and r["host_bad"]]
    if len(live) == len(rows) > 1:
        first, others = live[0], live[1:]
        together = [t for t, _ in first["host_bad"]
                    if all(any(abs(t - t2) <= COINCIDENT_US for t2, _ in o["host_bad"])
                           for o in others)]
        if together:
            print(f"    !! {len(together)}/{len(first['host_bad'])} host gaps hit ALL "
                  f"{len(rows)} sensors within {COINCIDENT_US / 1000:.0f} ms of each other "
                  f"-> event-loop block, not radio. First at host_t_us={together[0]}")

    # the 32-bit wrap: report it whether or not it is a problem
    for r in rows:
        if not r["empty"] and r["wraps"]:
            for t, d, at_min in r["wraps"]:
                print(f"    wrap {r['role']}: counter rolled at sensor_t_us={t}, "
                      f"{at_min:.1f} min into the recording; dt across the boundary = "
                      f"{d / 1000:.2f} ms {'OK' if 0 < d <= SENSOR_GAP_US else 'BAD'}")
    if not any(r.get("wraps") for r in rows):
        longest = max((r["dur_s"] for r in rows if not r["empty"]), default=0)
        print(f"    (no 32-bit wrap in {longest / 60:.1f} min; the counter wraps at 71.6 min)")

    print(f"\n  => {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    results = [check(d) for d in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)
