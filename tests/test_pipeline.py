"""Headless test of block1_pipeline.Block1Pipeline - the whole Block 1 chain, no hardware.

    python tests/test_pipeline.py

Drives synthetic 60 Hz samples from three sensors into .push() and calls .tick() on a 100 ms
schedule, exactly as web_demo's block1_task does. This is what stands in for the mock session on a
machine with no DOTs: it cannot tell you whether p tracks real movement, but it does prove the
loop runs clean and that the hop gate holds.
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import features
from block1_pipeline import Block1Pipeline
from inference import TICK_US

MODEL_DIR = HERE.parent / "jsons"       # the bundle the Pi actually loads
ROLES = ("forearm", "upper_arm", "torso")
FS = 60.0


@dataclass
class Row:
    sensor_id: str
    sensor_t_us: int
    host_t_us: int
    quat: np.ndarray
    acc: np.ndarray
    gyr: np.ndarray
    role: str


def synth(n, moving_from=None, seed=0):
    """n samples per sensor. From `moving_from` the signal gets large, to move p off the floor."""
    rng = np.random.default_rng(seed)
    out = {}
    for si, role in enumerate(ROLES):
        t = np.arange(n) * TICK_US + si * 137                 # a few us of inter-sensor skew
        amp = np.where(np.arange(n) >= (moving_from if moving_from is not None else n), 1.0, 0.02)
        ang = 2 * np.pi * 1.3 * np.arange(n) / FS
        q = np.stack([np.cos(ang * amp / 2), np.sin(ang * amp / 2),
                      np.zeros(n), np.zeros(n)], axis=1)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        gyr = (120 * amp)[:, None] * np.stack([np.sin(ang), 0.6 * np.cos(ang),
                                               0.3 * np.sin(2 * ang)], axis=1)
        acc = np.stack([np.zeros(n), np.zeros(n), np.full(n, 9.81)], axis=1) + \
            (6 * amp)[:, None] * rng.normal(0, 1, (n, 3))
        out[role] = (t, q, acc, gyr)
    return out


def stream(pipe, data, n, t0_host=1_700_000_000_000_000, tick_every=6):
    """Push samples in host order; tick every `tick_every` samples (100 ms at 60 Hz)."""
    scored = []
    for i in range(n):
        for role in ROLES:
            t, q, acc, gyr = data[role]
            pipe.push(Row(role, int(t[i]), int(t0_host + t[i]), q[i], acc[i], gyr[i], role))
        if i % tick_every == 0:
            out = pipe.tick(t0_host + int(t[i]))
            if out is not None:
                scored.append(out)
    return scored


def test_runs_clean_and_hop_gated():
    pipe = Block1Pipeline(MODEL_DIR, ROLES)
    n = int(120 * FS)                                          # 2 minutes
    scored = stream(pipe, synth(n, moving_from=n // 2), n)

    assert scored, "no window was ever scored"
    ks = np.array([s["k"] for s in scored])
    steps = np.diff(ks)
    assert (steps == pipe.hop).all(), \
        f"scored stride {sorted(set(steps.tolist()))}, expected exactly {pipe.hop}"
    for s in scored:
        assert 0.0 <= s["p"] <= 1.0 and 0.0 <= s["p_smooth"] <= 1.0
        assert isinstance(s["armed"], bool) and isinstance(s["stimulating"], bool)
    print(f"  loop      {n} samples/sensor -> {len(scored)} scored windows, "
          f"stride exactly {pipe.hop} aligned samples, no exceptions")

    p = np.array([s["p"] for s in scored])
    half = len(p) // 2
    print(f"  p         rest mean {p[:half].mean():.3f}   'moving' mean {p[half:].mean():.3f} "
          f"(synthetic, so this is a smoke test, not evidence p tracks real movement)")


def test_gap_resets_and_recovers():
    pipe = Block1Pipeline(MODEL_DIR, ROLES)
    n = int(30 * FS)
    data = synth(n, moving_from=0)
    before = stream(pipe, data, n)
    assert before, "nothing scored before the gap"

    data2 = synth(n, moving_from=0, seed=1)                     # a 2 s hole, then a fresh stream
    t0 = 1_700_000_000_000_000 + int(n * TICK_US) + 2_000_000
    for role in ROLES:
        t, q, a, g = data2[role]
        data2[role] = (t + int(n * TICK_US) + 2_000_000, q, a, g)
    after = stream(pipe, data2, n, t0_host=t0 - int(data2[ROLES[0]][0][0]))
    assert after, "never recovered after the gap"
    ks = np.array([s["k"] for s in after])
    assert (np.diff(ks) == pipe.hop).all(), "stride broken after the gap"
    print(f"  gap       2 s hole -> reset, then {len(after)} scored windows, stride still "
          f"{pipe.hop}")


def test_benchmark():
    pipe = Block1Pipeline(MODEL_DIR, ROLES)
    us = pipe.benchmark(1000)
    print(f"  bench     extract + Block1.p = {us:.0f} us/call  (THIS MACHINE, not the Pi; "
          f"the 100 ms tick budget is {100_000 / us:.0f}x this)")
    assert us < 100_000, "a single window must not take longer than the tick"


if __name__ == "__main__":
    print("test_pipeline")
    test_runs_clean_and_hop_gated()
    test_gap_resets_and_recovers()
    test_benchmark()
    print("ALL PASS")
