"""Replay-based tests for inference.Aligner. No hardware, no bleak.

    python test_aligner.py                  # self-contained runner
    pytest test_aligner.py                  # if pytest is installed

Feeds recorded rows in host_t_us order, exactly as the BLE callbacks would deliver them, and
checks that every window the Aligner emits is complete, aligned and fresh.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent     # tests/ sits one level under movella_demo
sys.path.insert(0, str(HERE))

import features
import inference
from inference import Aligner, ROLE_TO_SENSOR, SENSOR_TO_ROLE, N_CHANNELS

CLEAN = HERE / "recordings" / "20260615_143020"
DEGRADED = HERE / "recordings" / "20260615_141413"
ROLES = ("forearm", "upper_arm", "torso")          # roles[0] is the offset reference
WIN_N = 12 + features.ANGLE_LEAD                   # 17
RATE = 60.0
YIELD = {}          # run -> fraction of samples that produced a window


@dataclass
class Row:
    """Duck-types placement_scan_movella.Sample without importing it (that module needs bleak)."""
    sensor_id: str
    sensor_t_us: int
    host_t_us: int
    quat: np.ndarray
    acc: np.ndarray
    gyr: np.ndarray
    role: str


def load_run(run_dir):
    """Every row of every imu_*.csv, in host_t_us order - the order BLE delivers them."""
    rows = []
    for role in ROLES:
        df = pd.read_csv(run_dir / f"imu_{role}.csv").dropna()
        q = df[["qw", "qx", "qy", "qz"]].to_numpy(float)
        a = df[["ax", "ay", "az"]].to_numpy(float)
        g = df[["gx", "gy", "gz"]].to_numpy(float)
        h = df["host_t_us"].to_numpy(np.int64)
        s = df["sensor_t_us"].to_numpy(np.int64)
        rows += [Row(role, int(s[i]), int(h[i]), q[i], a[i], g[i], role) for i in range(len(df))]
    rows.sort(key=lambda r: r.host_t_us)
    return rows


def check_window_is_aligned(al, k, w):
    """Rebuild the window straight from the ring buffers and the shifts: proves every row of the
    block is the SAME aligned index on all three sensors. Must run while the window is still in
    the buffers (they hold 4*win_n samples), so replay() calls it inline."""
    want = np.empty_like(w)
    for si, sensor in enumerate(features.SENSORS):
        role = SENSOR_TO_ROLE[sensor]
        rows, sh = al.state[role].rows, al.shift[role]
        for ti, a in enumerate(range(k - WIN_N + 1, k + 1)):
            want[ti, si] = rows[a - sh]          # KeyError here = a partial/padded window
    assert np.array_equal(w, want), "window rows are not the sensors' same aligned index"


def replay(rows, verify=True, **kw):
    """Push every row, collecting (row_number, end_index, window) for each non-None window."""
    al = Aligner(ROLES, WIN_N, rate_hz=RATE, **kw)
    out, gaps = [], 0
    for i, r in enumerate(rows):
        al.push(r)
        if al.gap_detected():
            gaps += 1
        w = al.window()
        if w is not None:
            if verify:
                check_window_is_aligned(al, al.last_end_index, w)
            out.append((i, al.last_end_index, w))
    return al, out, gaps


def test_clean_run():
    rows = load_run(CLEAN)
    al, got, gaps = replay(rows)
    assert got, "no windows produced on the clean recording"

    for _, k, w in got:
        assert w.shape == (WIN_N, len(features.SENSORS), N_CHANNELS), w.shape
        assert np.isfinite(w).all(), "non-finite value in a window"

    # window() is called after EVERY push, but three sensors push per aligned index, so the same
    # k is legitimately returned until the next index completes on all three. The invariant is
    # that the end index never skips and never goes backwards: one new index per new sample.
    ends = np.array([k for _, k, _ in got])
    steps = np.diff(ends)
    assert set(steps.tolist()) <= {0, 1}, f"end index jumped by {sorted(set(steps.tolist()))}"
    uniq = np.unique(ends)
    assert (np.diff(uniq) == 1).all(), "an end index was skipped"
    assert len(uniq) == ends[-1] - ends[0] + 1

    # the offset trace must be STABLE: a shift that flips mid-session moves every subsequent
    # window by one sample on that sensor, which is a discontinuity, not a resync.
    traces = {r: sorted({log[r] for _, log, _ in al.offset_log}) for r in ROLES}
    assert all(len(v) == 1 for v in traces.values()), f"a shift changed mid-session: {traces}"

    frac = len(got) / len(rows)
    print(f"  clean     {len(rows):6d} samples -> {len(got):6d} windows  ({frac:6.1%}), "
          f"{gaps} gap(s), shifts {al.shift} stable over "
          f"{len(al.offset_log)} recomputation(s)")
    assert frac > 0.25, f"only {frac:.1%} of samples yielded a window"
    YIELD["clean"] = frac


def test_injected_gap():
    """Silence the torso for 2 s. window() must go None and stay None; one gap event on resume."""
    rows = load_run(CLEAN)
    t0 = rows[0].host_t_us + 20_000_000              # 20 s in, well past filter/offset warm-up
    t1 = t0 + 2_000_000
    kept = [r for r in rows if not (r.role == "torso" and t0 <= r.host_t_us < t1)]

    al = Aligner(ROLES, WIN_N, rate_hz=RATE)
    stale_us = al.stale_us
    n_gaps, during, resumed_ok = 0, [], False
    for r in kept:
        al.push(r)
        if al.gap_detected():
            n_gaps += 1
        w = al.window()
        # once the torso has been quiet for longer than stale_ms, nothing may come out
        if t0 + stale_us <= r.host_t_us < t1:
            during.append(w)
        if r.host_t_us >= t1 and w is not None:
            resumed_ok = True

    assert during, "the gap window was never exercised"
    assert all(w is None for w in during), \
        f"{sum(w is not None for w in during)}/{len(during)} windows returned during the outage"
    assert n_gaps == 1, f"gap_detected() fired {n_gaps} times, expected exactly 1"
    assert resumed_ok, "never recovered after the gap"
    print(f"  gap       2 s torso outage -> {len(during)} calls, all None; "
          f"gap_detected fired {n_gaps}x; recovered afterwards")


def test_degraded_run():
    rows = load_run(DEGRADED)
    al, got, gaps = replay(rows)
    frac = len(got) / len(rows)
    print(f"  degraded  {len(rows):6d} samples -> {len(got):6d} windows  ({frac:6.1%}), "
          f"{gaps} gap(s)")
    for _, k, w in got:
        assert np.isfinite(w).all()
    assert frac < 0.05, f"{frac:.1%} of samples yielded a window on a stream that cannot support it"
    YIELD["degraded"] = frac


def test_hop_gating():
    """window() can return the same k up to three times (one per sensor push). The tick loop's
    hop guard must turn that into exactly one scored window per HOP_SAMPLES."""
    rows = load_run(CLEAN)
    al = Aligner(ROLES, WIN_N, rate_hz=RATE)
    scored, last_scored = [], -10 ** 9
    for r in rows:
        al.push(r)
        w = al.window()
        k = al.last_end_index
        if w is not None and k >= last_scored + inference.HOP_SAMPLES:
            scored.append(k)
            last_scored = k
    steps = np.diff(scored)
    assert (steps == inference.HOP_SAMPLES).all(), f"scored strides {sorted(set(steps.tolist()))}"
    print(f"  hop gate  {len(scored)} scored windows, stride exactly {inference.HOP_SAMPLES} "
          f"(from {len(rows)} pushes)")


def test_shift_change_raises_gap():
    """A shift that legitimately moves is a one-sample step in the aligned signal, so it must
    raise the same flag as a data gap - the filters need the same reset."""
    rows = load_run(CLEAN)
    t_jump = rows[0].host_t_us + 30_000_000
    bumped = [Row(r.sensor_id, r.sensor_t_us,
                  r.host_t_us + (20_000 if r.role == "torso" and r.host_t_us >= t_jump else 0),
                  r.quat, r.acc, r.gyr, r.role)
              for r in rows]                    # +20 ms on torso = 1.2 samples, past the deadband

    al = Aligner(ROLES, WIN_N, rate_hz=RATE)
    fired_after_jump = 0
    for r in sorted(bumped, key=lambda x: x.host_t_us):
        al.push(r)
        if al.gap_detected() and r.host_t_us >= t_jump:
            fired_after_jump += 1
    shifts = sorted({log["torso"] for _, log, _ in al.offset_log})
    assert len(shifts) > 1, f"torso shift never moved despite a 20 ms jump: {shifts}"
    assert fired_after_jump >= 1, "the shift changed without raising the gap flag"
    print(f"  shift     torso shift moved {shifts} -> gap flag raised {fired_after_jump}x")


def test_stale_now_us():
    """window(now_us) must go stale on the caller's clock even when no sample ever arrives again,
    and must reject a clock from a different domain rather than silently never going stale."""
    rows = load_run(CLEAN)
    al = Aligner(ROLES, WIN_N, rate_hz=RATE)
    for r in rows[:3000]:
        al.push(r)
    last_host = al.now_us
    assert al.window() is not None, "expected a window before going quiet"
    assert al.window(last_host + 100_000) is not None, "100 ms quiet is not stale"
    assert al.window(last_host + 500_000) is None, "500 ms quiet must be stale"

    try:                                          # a monotonic clock is a different epoch
        al.window(12_345_678)
    except ValueError as e:
        assert "clock domain" in str(e), e
    else:
        raise AssertionError("a monotonic-domain now_us was accepted silently")
    print("  staleness now_us drives staleness with no new samples; wrong clock domain raises")


def test_index_slope():
    """Indexing on the measured TICK_US gives a slope of exactly 1.0, so a stream with no dropped
    packets gets no phantom holes. At 1e6/60 the 20 ppm error inserted one every ~50 000 samples;
    a real recording is far too short to show that, so it is checked on a synthetic hour."""
    al = Aligner(ROLES, WIN_N, rate_hz=RATE)
    for r in load_run(CLEAN):
        al.push(r)
    for role in ROLES:
        idx = np.array(sorted(al.state[role].rows))           # only the tail is still buffered
        assert (np.diff(idx) == 1).all(), f"clean/{role}: index skipped"

    n = 60 * 60 * 60                                          # one hour of perfect 16667 us ticks
    t = np.arange(n, dtype=np.int64) * inference.TICK_US
    al = Aligner(ROLES, WIN_N, rate_hz=RATE)
    seen = []
    st = al.state["forearm"]
    for i in range(n):
        al.push(Row("forearm", int(t[i]), int(t[i]), np.zeros(4), np.zeros(3), np.zeros(3),
                    "forearm"))
        seen.append(st.last_idx)
    steps = np.diff(seen)
    assert (steps == 1).all(), f"phantom hole: index stepped by {sorted(set(steps.tolist()))}"
    assert seen[-1] == n - 1, f"index drifted: {seen[-1]} != {n - 1} after an hour"
    print(f"  index     slope exactly 1.0 on the clean run and over {n} synthetic ticks (1 h), "
          f"no phantom holes")


def test_no_hardware_imports():
    """Requirement 5: the module must be importable with no BLE/GPIO stack present."""
    for mod in ("bleak", "aiohttp", "lgpio"):
        assert mod not in sys.modules, f"importing inference pulled in {mod}"
    Aligner(ROLES, WIN_N)                          # constructible with no hardware


if __name__ == "__main__":
    print("test_aligner")
    test_no_hardware_imports()
    print("  imports   no bleak / aiohttp / lgpio; Aligner constructible")
    test_clean_run()
    test_injected_gap()
    test_degraded_run()
    test_hop_gating()
    test_shift_change_raises_gap()
    test_stale_now_us()
    test_index_slope()
    print(f"\n  window yield: clean {YIELD['clean']:.1%}, degraded {YIELD['degraded']:.1%}")
    print("ALL PASS")
