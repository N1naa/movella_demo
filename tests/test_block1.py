"""Unit + parity tests for inference.Block1.

    python tests/test_block1.py
    python tests/test_block1.py --subject GG051 --max-samples 20000

The parity test is the real one: it reuses replay.py to build the ONLINE feature matrix (streamed
through Preprocessor + extract) and the matched OFFLINE matrix (build_windows), scores the online
rows one at a time through Block1.p, scores the offline matrix with the same booster the way the
offline pipeline does, and requires max |delta| < 1e-9.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import features
from inference import Block1

import replay                                   # reused, not reimplemented

MODEL_DIR = replay.MVMT_DET / "frozen_jsons_deploy"
SCRATCH = Path("/private/tmp/claude-501/-Users-ninabodenstab-Desktop-Documents-personnels-Nina-"
               "Bodenstab-NB-University-EPFL-Master-master-thesis-analysis-nosync-stroke/"
               "ea327cdc-09dc-4cde-8dbe-cce5649c5600/scratchpad")
TOL = 1e-9


def _bundle_with_swapped_names(dst):
    """Copy the real bundle but swap two feature names in config.json."""
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(MODEL_DIR / "booster.json", dst / "booster.json")
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    names = list(cfg["feature_names"])
    names[3], names[4] = names[4], names[3]
    cfg["feature_names"] = names
    (dst / "config.json").write_text(json.dumps(cfg))
    return dst, names


def test_reordered_feature_names_raises():
    dst, names = _bundle_with_swapped_names(SCRATCH / "bundle_swapped")
    try:
        Block1(dst)
    except ValueError as e:
        assert "index 3" in str(e), f"error should name the first mismatching index: {e}"
        assert names[3] in str(e) and features.SET85[3] in str(e), e
        print(f"  reorder   raises and names index 3: {str(e)[:88]}...")
    else:
        raise AssertionError("a reordered feature_names was accepted")


def test_p_returns_float_in_range():
    b = Block1(MODEL_DIR)
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = b.p(rng.normal(0, 3, 85))
        assert type(v) is float, f"p() returned {type(v)}, expected a Python float"
        assert 0.0 <= v <= 1.0, v
    print("  range     p() returns a Python float in [0, 1]")


def test_bad_input_raises():
    b = Block1(MODEL_DIR)
    for bad, why in ((np.zeros(84), "too short"), (np.zeros(86), "too long")):
        try:
            b.p(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{why} input was accepted")
    for bad, why in ((np.nan, "NaN"), (np.inf, "inf")):
        v = np.zeros(85)
        v[7] = bad
        try:
            b.p(v)
        except ValueError as e:
            assert "7" in str(e), e
        else:
            raise AssertionError(f"{why} input was accepted")
    print("  guards    wrong length and non-finite both raise before reaching the booster")


def test_parity_against_offline(subject, max_samples):
    """Online extract -> Block1.p  vs  offline build_windows -> booster.predict(DMatrix)."""
    import xgboost as xgb

    participants = pd.read_csv(replay.MVMT_DET / "configs" / "participants.csv")
    sensor_map = pd.read_csv(replay.MVMT_DET / "configs" / "sensor_map.csv")
    label_map = pd.read_csv(replay.MVMT_DET / "configs" / "label_map.csv")
    config = json.loads((MODEL_DIR / "config.json").read_text())

    prep = replay.prepare_subject(participants, sensor_map, label_map, subject, max_samples)
    args = argparse.Namespace(top=0)
    ends, ON, OFF, sanity = replay.compare(subject, prep, config, args)
    assert sanity < TOL, "replay's own alignment check failed; fix that before reading parity"

    b = Block1(MODEL_DIR)
    p_online = np.array([b.p(row) for row in ON])          # one row at a time, as online does

    booster = xgb.Booster()                                # the offline call, on the offline matrix
    booster.load_model(str(MODEL_DIR / "booster.json"))
    p_offline = booster.predict(xgb.DMatrix(OFF))

    d = np.abs(p_online - p_offline)
    print(f"\n  parity    {subject}: {len(ON)} windows, max |p_online - p_offline| = {d.max():.3e} "
          f"({'PASS' if d.max() < TOL else 'FAIL'})")
    print(f"            p range [{p_offline.min():.4f}, {p_offline.max():.4f}], "
          f"{(p_offline >= config['threshold']).mean():.1%} of windows over the 0.68 threshold")
    assert d.max() < TOL, f"max |delta| = {d.max():.3e}, expected < {TOL}"
    return ON


def test_timing(ON=None):
    b = Block1(MODEL_DIR)
    rng = np.random.default_rng(1)
    X = ON if ON is not None and len(ON) >= 1000 else rng.normal(0, 3, (1000, 85))
    for i in range(100):
        b.p(X[i % len(X)])                                  # warm up
    t0 = time.perf_counter()
    for i in range(1000):
        b.p(X[i % len(X)])
    us = (time.perf_counter() - t0) / 1000 * 1e6
    print(f"  timing    {us:.1f} us/call over 1000 calls "
          f"({'real features' if ON is not None else 'random features'}); "
          f"the tick loop has 100 ms per window")
    return us


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="GG045")
    ap.add_argument("--max-samples", type=int, default=15000)
    a = ap.parse_args()

    print("test_block1")
    test_reordered_feature_names_raises()
    test_p_returns_float_in_range()
    test_bad_input_raises()
    ON = test_parity_against_offline(a.subject, a.max_samples)
    test_timing(ON)
    print("ALL PASS")
