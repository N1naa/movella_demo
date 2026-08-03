"""Unit tests for inference.Decision.

    python tests/test_decision.py            # self-contained runner
    pytest tests/test_decision.py

The reference is vns/mvmt_det/src/smoothing.py::causal_moving_average +
src/events.py::rearm_triggers. Where the brief and those files disagreed, the files won; the
three places that happened are marked DEVIATION below and reported in the summary.

The strongest test here is not any single hand-built case: it is test_matches_reference, which
runs random probability sequences (with gaps) through Decision and through the offline pair, and
requires the fired arrays to be identical element for element.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
MVMT_DET = Path("/Users/ninabodenstab/Desktop/Documents personnels/Nina Bodenstab/NB_University/"
                "EPFL/Master/master_thesis/analysis.nosync/stroke/vns/mvmt_det")
sys.path.insert(0, str(MVMT_DET))

from inference import Decision

CFG = dict(smoothing_k=3, threshold=0.68, stim_ms=1000, lockout_ms=0, hop_ms=100, max_gap_ms=150)
STEP_US = 100_000


def run(ps, cfg=None, t0=0):
    """Feed a probability sequence at the nominal hop. Returns (fires, stim mask, p_smooth trace)."""
    d = Decision(cfg or CFG)
    fires, stim, smooth = [], [], []
    for i, p in enumerate(ps):
        fires.append(d.update(float(p), t0 + i * STEP_US))
        stim.append(d.stimulating)
        smooth.append(d.p_smooth)
    return np.array(fires), np.array(stim), np.array(smooth)


def test_partial_smoothing_at_segment_start():
    """The first window averages 1 value, the second 2, the third 3 - no zero padding, no waiting."""
    _, _, sm = run([0.3, 0.9, 0.6, 0.0])
    assert np.isclose(sm[0], 0.3), sm[0]
    assert np.isclose(sm[1], (0.3 + 0.9) / 2), sm[1]
    assert np.isclose(sm[2], (0.3 + 0.9 + 0.6) / 3), sm[2]
    assert np.isclose(sm[3], (0.9 + 0.6 + 0.0) / 3), sm[3]
    d = Decision(CFG)
    assert d.p_smooth is None, "p_smooth before the first update must be None"


def test_constant_high_fires_once():
    """DEVIATION from the brief, which expects 'once per second' for a constant p = 0.9.

    The rule re-arms only once p_smooth < threshold, and a constant 0.9 never goes below it, so
    the reference fires exactly ONCE and then waits forever. Confirmed directly against
    events.rearm_triggers on a constant-True array: 1 trigger, 10 windows stimulated.
    """
    fires, stim, _ = run([0.9] * 100)
    assert fires.sum() == 1, f"expected 1 trigger, got {fires.sum()}"
    assert np.flatnonzero(fires)[0] == 0
    assert stim[:10].all() and not stim[10:].any(), "stimulation must cover exactly 10 windows"


def test_constant_low_never_fires():
    fires, stim, _ = run([0.5] * 100)
    assert fires.sum() == 0 and not stim.any()


def test_single_spike_is_suppressed():
    """DEVIATION from the brief, which expects a lone 0.9 in a 0.1 field to fire once.

    With k = 3 it cannot: the best p_smooth reachable is (0.1 + 0.1 + 0.9) / 3 = 0.367, far under
    0.68. Suppressing exactly this single-window flicker is what smoothing.py says the smoother is
    for, so 0 triggers is the correct answer, not a bug.
    """
    ps = [0.1] * 20
    ps[10] = 0.9
    fires, _, sm = run(ps)
    assert np.isclose(sm.max(), (0.1 + 0.1 + 0.9) / 3), sm.max()
    assert fires.sum() == 0, "a single-window spike must not survive k=3 smoothing"

    ps = [0.1] * 20                       # three in a row does clear the threshold
    ps[10:13] = [0.9, 0.9, 0.9]
    fires, _, _ = run(ps)
    assert fires.sum() == 1, f"a 3-window burst should fire once, got {fires.sum()}"


def test_rise_fall_rise_fires_twice():
    ps = [0.1] * 5 + [0.9] * 10 + [0.1] * 15 + [0.9] * 10 + [0.1] * 5
    fires, _, _ = run(ps)
    assert fires.sum() == 2, f"expected 2 triggers, got {fires.sum()}"


def test_exactly_at_threshold_fires():
    """>= is inclusive: a sustained p_smooth of exactly 0.68 must fire."""
    fires, _, sm = run([0.68] * 20)
    assert np.isclose(sm[0], 0.68)
    assert fires.sum() == 1 and fires[0]


def test_exactly_at_threshold_does_not_rearm():
    """< is strict: p_smooth == threshold while disarmed must NOT re-arm, so no second trigger."""
    ps = [0.68] * 40                       # fires at window 0, then sits exactly on the threshold
    fires, _, sm = run(ps)
    assert np.allclose(sm[2:], 0.68)
    assert fires.sum() == 1, "p_smooth == threshold re-armed; the comparison must be strict <"

    ps = [0.68] * 12 + [0.6799] * 1 + [0.68] * 30   # a dip strictly below does re-arm
    fires, _, _ = run(ps)
    assert fires.sum() == 2, f"a value strictly below threshold must re-arm, got {fires.sum()}"


def test_no_rearm_during_stimulation():
    """DEVIATION from the brief, which says a dip during the 1 s burst must re-arm.

    rearm_triggers skips the index past the stimulation and lockout, so the windows inside a burst
    are never examined and cannot re-arm. Verified against it directly: high, one-window dip inside
    the burst, then high again -> 1 trigger, not 2.
    """
    ps = [0.9] * 3 + [0.0] * 2 + [0.9] * 40    # the dip lands inside the burst that starts at 0
    fires, _, _ = run(ps)
    assert fires.sum() == 1, f"a dip inside the stimulation re-armed; expected 1, got {fires.sum()}"


def test_minimum_inter_trigger_spacing():
    """DEVIATION from the brief's '1 s minimum'. Re-arming costs one window: the reference fires
    at i, resumes at i + n_stim + n_lock, and can only re-arm there, so the earliest next trigger
    is i + 11 windows = 1.1 s.

    Measured on the arm/fire logic alone (k = 1), with the dip landing exactly on the first window
    the loop examines after the burst. With k = 3 the smoother adds its own recovery on top: one
    raw zero only drags p_smooth to 0.667 and three windows are needed to climb back, so the real
    pipeline spaces those triggers 14 windows apart. 1.1 s is the floor, not the typical value.
    """
    cfg1 = {**CFG, "smoothing_k": 1}
    ps = [1.0] * 10 + [0.0] + [1.0] * 30       # dip on window 10, the first one examined post-burst
    idx = np.flatnonzero(run(ps, cfg1)[0])
    assert len(idx) == 2 and idx[1] - idx[0] == 11, f"k=1 fire indices {idx}, expected spacing 11"

    idx3 = np.flatnonzero(run([1.0] * 11 + [0.0] + [1.0] * 30)[0])
    assert idx3[1] - idx3[0] == 14, f"k=3 spacing {idx3[1] - idx3[0]}, expected 14"


def test_gap_resets_segment():
    """A gap > max_gap_ms starts a new segment: the smoother restarts partial and the arm state
    is cleared, exactly as causal_moving_average / rearm_triggers segment the offline arrays."""
    d = Decision(CFG)
    for i in range(5):
        d.update(0.9, i * STEP_US)
    assert not d.armed and d.stimulating
    d.update(0.3, 5 * STEP_US + 400_000)       # 500 ms gap
    assert d.armed and not d.stimulating, "a gap must clear arm and stimulation state"
    assert np.isclose(d.p_smooth, 0.3), "the smoother must restart from one value, not carry over"


def test_reset_mid_stimulation():
    d = Decision(CFG)
    for i in range(4):
        d.update(0.9, i * STEP_US)
    assert d.stimulating and not d.armed and d.p_smooth is not None
    d.reset()
    assert d.armed and not d.stimulating and d.p_smooth is None
    assert d.update(0.9, 99 * STEP_US) is True, "after reset the next high window must fire"


def test_matches_reference():
    """The real check: identical fired arrays against causal_moving_average + rearm_triggers on
    random sequences, including ones with segment gaps."""
    import pandas as pd
    from src import events, smoothing

    rng = np.random.default_rng(0)
    for trial in range(30):
        n = 400
        p = np.clip(rng.beta(2, 2, n) + rng.normal(0, 0.25, n), 0, 1)
        step = np.full(n, 100)
        if trial % 3:                                    # sprinkle segment-breaking gaps
            step[rng.choice(n, 5, replace=False)] = 900
        t_us = np.cumsum(step) * 1000
        t_end = np.datetime64("2026-01-01") + t_us.astype("timedelta64[us]")

        sm = smoothing.causal_moving_average(t_end, p, CFG["smoothing_k"], CFG["max_gap_ms"])
        ref = events.rearm_triggers(t_end, sm >= CFG["threshold"], stim_ms=CFG["stim_ms"],
                                    lockout_ms=CFG["lockout_ms"], step_ms=CFG["hop_ms"],
                                    max_gap_ms=CFG["max_gap_ms"])
        d = Decision(CFG)
        mine_stim, mine_smooth = [], []
        for pi, ti in zip(p, t_us):
            d.update(float(pi), int(ti))
            mine_stim.append(int(d.stimulating))
            mine_smooth.append(d.p_smooth)
        assert np.allclose(mine_smooth, sm), f"trial {trial}: smoother differs from the reference"
        assert np.array_equal(mine_stim, ref), f"trial {trial}: fired array differs from rearm_triggers"
    print(f"  reference  30 random sequences (with gaps): smoother and fired array both identical")


def test_lhoste_matches_reference():
    """The same check for the Lhoste baseline row: identical smoothed score, adaptive threshold and
    stimulation mask against baselines.lhoste_predict + rearm_triggers, on random forearm-speed
    sequences with gaps. The stimulation MASK is compared rather than the fired onsets because a
    burst starting on the first window of a new segment leaves the mask at 1 across the seam."""
    from src import baselines, events

    from inference import Lhoste, TICK_US

    rng = np.random.default_rng(1)
    for trial in range(30):
        n = 400
        speed = np.abs(rng.normal(0.02, 0.02, n))         # rad/s: rest, plus a few bursts
        for _ in range(6):
            s = rng.integers(0, n - 30)
            speed[s:s + rng.integers(5, 25)] += rng.uniform(0.5, 4.0)
        idx = np.arange(n) * 6                            # aligned sample index, one hop apart
        if trial % 2:
            for _ in range(3):
                idx[rng.integers(50, n):] += 60           # a 1 s hole -> both sides must segment
        t_us = idx * TICK_US
        t_end = np.array(t_us * 1000, dtype="datetime64[ns]")

        y, v_ref, thr_ref = baselines.lhoste_predict(t_end, speed, max_gap_ms=CFG["max_gap_ms"],
                                                     max_rate=np.pi / 2)
        ref = events.rearm_triggers(t_end, y.astype(bool), stim_ms=CFG["stim_ms"],
                                    lockout_ms=CFG["lockout_ms"], step_ms=CFG["hop_ms"],
                                    max_gap_ms=CFG["max_gap_ms"])
        lh = Lhoste(CFG)
        mine_v, mine_thr, mine_stim = [], [], []
        for si, ti in zip(speed, t_us):
            lh.update(float(si), int(ti))
            mine_v.append(lh.score)
            mine_thr.append(lh.threshold)
            mine_stim.append(int(lh.decision.stimulating))
        assert np.allclose(mine_v, v_ref), f"trial {trial}: smoothed score differs"
        assert np.allclose(mine_thr, thr_ref), f"trial {trial}: adaptive threshold differs"
        assert np.array_equal(mine_stim, ref), f"trial {trial}: stimulation mask differs"
    print("  lhoste     30 random sequences (with gaps): score, threshold and stim mask identical")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    print("test_decision")
    for fn in TESTS:
        fn()
        print(f"  {'PASS':4s}  {fn.__name__}")
    print("ALL PASS")
