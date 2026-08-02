"""Acceptance: reproduce config.json's event_level block by driving inference.Decision with the
saved LOSO out-of-fold sequences, ONE SUBJECT AT A TIME with reset() between subjects.

    python tests/acceptance_decision.py

WHICH CACHE. export_bundle's default operating point is precision_targeted_XGBoost_set85.pkl, but
the deploy bundle was built from op_deploy_set85.pkl (notebooks/04_final.ipynb) and only that one
reproduces the published numbers - the default gives 8.09 / 0.713 / 0.966 instead of
8.14 / 0.710 / 0.963.

WHAT IS AND IS NOT REPLAYED HERE. train_one_subject_and_loso.loso_smoothed stores
    preds[subj] = (t_end, y_test, causal_moving_average(p_raw, k, max_gap_ms))
so the cached score is ALREADY smoothed with k = 3; the raw per-window probability is not saved
anywhere. Feeding it to a k = 3 Decision would smooth twice, so Decision runs here with k = 1 and
this file validates the arm / fire / re-arm layer against real data. The smoother itself is
validated against causal_moving_average in test_decision.py::test_matches_reference, which also
checks the two composed - on synthetic sequences, the only place both halves can be driven.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
MVMT_DET = Path("/Users/ninabodenstab/Desktop/Documents personnels/Nina Bodenstab/NB_University/"
                "EPFL/Master/master_thesis/analysis.nosync/stroke/vns/mvmt_det")
sys.path.insert(0, str(MVMT_DET))

import json
from inference import Decision
from src import events

CACHE = MVMT_DET / "thesis" / "cache_final" / "op_deploy_set85.pkl"
CONFIG = json.loads((MVMT_DET / "frozen_jsons_deploy" / "config.json").read_text())
STEP_MS, MAX_GAP_MS, MIN_EVENT_MS, MIN_OVERLAP_MS = 100, 150, 200, 300


def main():
    thr = CONFIG["threshold"]
    want = CONFIG["expected_performance"]["event_level"]
    # k=1: the cached score is already smoothed (see the module docstring)
    cfg = dict(smoothing_k=1, threshold=thr, stim_ms=CONFIG["stim_ms"],
               lockout_ms=CONFIG["lockout_ms"], hop_ms=STEP_MS, max_gap_ms=MAX_GAP_MS)

    per_fold, preds, _ = pd.read_pickle(CACHE)
    rows, mismatched = [], []
    for subj in sorted(preds):
        t_end, y_true, score = preds[subj]
        # The cache is not time-ordered (build_windows concatenates per-section tables), and
        # rearm_triggers / causal_moving_average both argsort internally. A live stream arrives in
        # time order by construction, so sort once here and let every path see that same order.
        t = np.asarray(t_end, dtype="datetime64[ns]")
        order = np.argsort(t, kind="stable")
        t, y_true, score = t[order], np.asarray(y_true, int)[order], np.asarray(score, float)[order]
        t_us = t.astype("datetime64[us]").astype(np.int64)

        d = Decision(cfg)                       # ONE SUBJECT AT A TIME - never carry state across
        fired = np.empty(len(score), int)
        for i, (p, tu) in enumerate(zip(score, t_us)):
            d.update(float(p), int(tu))
            fired[i] = d.stimulating

        ref = events.rearm_triggers(t, score >= thr,
                                    stim_ms=cfg["stim_ms"], lockout_ms=cfg["lockout_ms"],
                                    step_ms=STEP_MS, max_gap_ms=MAX_GAP_MS)
        if not np.array_equal(fired, ref):
            mismatched.append((subj, int(np.abs(fired - ref).sum())))

        m = events._subject_event_metrics(t, y_true, fired, STEP_MS, MAX_GAP_MS,
                                          MIN_EVENT_MS, onset_ms=None,
                                          min_overlap_ms=MIN_OVERLAP_MS)
        minutes = len(y_true) * STEP_MS / 60000.0
        rows.append({"subject": subj, "minutes": minutes,
                     "n_movements": m["n_true_events"], "n_stim": m["n_pred_events"],
                     "stim_per_min": m["n_pred_events"] / minutes,
                     "precision": m["event_precision"], "sensitivity": m["detection_rate"],
                     "median_latency_ms": m["median_latency_ms"]})

    ev = pd.DataFrame(rows)
    print(f"cache: {CACHE.name}   threshold={thr}  stim_ms={cfg['stim_ms']}  "
          f"lockout_ms={cfg['lockout_ms']}  n_subjects={len(ev)}\n")
    print(ev.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Compare the way export_online.py wrote the config: stim_per_min is stored rounded to 2
    # decimals, precision and sensitivity to 3 (src/export_online.py:299-306). Comparing the raw
    # mean against the rounded literal would fail on the rounding alone.
    print(f"\n{'metric':16s} {'Decision':>10s} {'rounded':>9s} {'config.json':>12s}")
    ok = True
    for key, nd in (("stim_per_min", 2), ("precision", 3), ("sensitivity", 3)):
        got, exp = ev[key].mean(), want[key]
        good = round(float(got), nd) == exp
        ok &= good
        print(f"{key:16s} {got:10.4f} {round(float(got), nd):9.3f} {exp:12.3f}  "
              f"{'OK' if good else 'MISMATCH'}")

    if mismatched:
        print(f"\n  !! fired array differs from events.rearm_triggers for {mismatched} "
              f"(subject, n_windows) - the bug is in Decision, do not tune to match")
        ok = False
    else:
        print(f"\n  fired array identical to events.rearm_triggers for all {len(ev)} subjects")
    print(f"\n=> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
