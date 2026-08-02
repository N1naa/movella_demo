"""Train/serve parity harness for the online movement-detection extractor.

Replays a real GripGlove subject through the ONLINE path (features.Preprocessor.push +
features.extract) and compares every one of the 85 features against the OFFLINE path
(vns/mvmt_det/src/windows_features.build_windows), window by window.

READ-ONLY: this script imports the offline pipeline and features.py; it modifies neither.

Run:  poetry run python replay.py                       # 3 default subjects
      poetry run python replay.py --subjects GG045 GG051 GG080 --max-samples 60000
      poetry run python replay.py --subjects GG045 --full  # whole recording, slow

------------------------------------------------------------------------------------------
WINDOW ALIGNMENT  (step 5 - worked out from build_windows, src/windows_features.py:472-486)
------------------------------------------------------------------------------------------
build_windows slides `for start in range(i0, i1 - win + 1, step)` and sets `end = start + win`,
so with sections=None (i0=0, i1=n) row r is the window over samples [r*step, r*step + win).
It records `t_ends.append(t_ref[end - 1])` - the timestamp of the LAST sample INSIDE the window,
i.e. `end` is EXCLUSIVE and t_end sits at index end-1.

Row index is NOT a reliable proxy for r: a window is silently skipped (no row emitted) when it
straddles a data seam or an exclusion interval. So each offline row's end index is recovered from
its timestamp instead:

    end_offline = searchsorted(t_ref, row.t_end) + 1          # exclusive

Online, extract(D[e - (WIN + ANGLE_LEAD) : e]) scores the LAST WIN samples of that slice, i.e.
[e - WIN, e) -> the same exclusive end index `e`.

The reference table is therefore built at step = 1 SAMPLE, not step = hop. At the deployed hop the
offline end indices are {12, 18, 24, ...} while the online sweep `range(WIN + ANGLE_LEAD, N, HOP)`
gives {17, 23, 29, ...} - congruent to 5 (mod 6), sharing NO end index with the offline grid, so
every window would silently be compared against a neighbour 1-5 samples away. A step-1 reference
makes every end index available and removes the trap; it costs ~1.1 ms per offline window.

TWO alignments, because the online extractor deliberately lags one group (features.py: LAG = 2):

  * the 73 non-angle features are compared at the SAME end index e;
  * the 12 elev_*/flex_* ones at e - LAG. Offline smooths elevation/flexion with a CENTRED
    uniform_filter1d(size=5), so it reads 2 samples past the window end - data that does not exist
    live. extract() takes that as a lag on those channels only (config.json's notes prescribe
    exactly this), so their correct offline counterpart is the window ending LAG samples earlier.
    The unlagged comparison is also reported, to show what the lag is worth.
------------------------------------------------------------------------------------------
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MVMT_DET = Path("/Users/ninabodenstab/Desktop/Documents personnels/Nina Bodenstab/NB_University/"
                "EPFL/Master/master_thesis/analysis.nosync/stroke/vns/mvmt_det")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MVMT_DET))

import features as F                                   # the online path under test
from src import preprocessing, train_one_subject_and_loso as tl, windows_features as wf

FS = 60.0
WIN, HOP = 12, 6                        # config.json: window_samples / hop_samples
PAD = F.ANGLE_LEAD                      # 5 - extra history extract() needs for the angle path
WARMUP_WINDOWS = 30                     # excluded from the headline number, reported separately
TOL = 1e-9

DEFAULT_SUBJECTS = ["GG045", "GG051", "GG080"]
BOOSTER = MVMT_DET / "frozen_jsons_deploy" / "booster.json"
CONFIG = MVMT_DET / "frozen_jsons_deploy" / "config.json"

# Feature groups the task flags as likely failure modes, so a mismatch can be named immediately.
GROUPS = {
    "jerk / ldlj (raw vs filtered acc_free)": lambda f: f.endswith(("_jerk_rms", "_ldlj")),
    "ori_*_speed (Euler convention / wrap)": lambda f: "_ori_" in f,
    "pron_speed (raw vs band-passed gyr_x)": lambda f: f == "forearm_pron_speed_mean",
    "elev_* / flex_* (angle definition)": lambda f: f.startswith(("elev_", "flex_")),
    "vqf_speed (movement_signal)": lambda f: f.endswith("_vqf_speed"),
}


def gyr_perp_speed(df):
    """The deployed movement signal, config['movement_signal'] == 'gyr_perp':
    |omega| perpendicular to the sensor long axis (x), rad/s, on the RAW gyro."""
    return np.deg2rad(np.hypot(df["Gyr_Y"].to_numpy(float), df["Gyr_Z"].to_numpy(float)))


def prepare_subject(participants, sensor_map, label_map, pid, max_samples, align_first=True):
    """Offline-pipeline frames for one subject, plus the RAW channels the online path replays.

    Mirrors build_windows_for_subject (src/train_one_subject_and_loso.py:37) up to the point where
    it calls build_windows, with three deliberate simplifications, all so that the offline row grid
    is a plain arithmetic lattice and the two paths see the SAME samples:

      * sections=None and no exclusion intervals -> no windows skipped, no per-section ranges;
      * no cut_imu -> the filters warm up from sample 0 in both paths, as offline does;
      * vqf_speed is the deployed gyr_perp, not vqf_axis_speed (9-axis VQF, needs the magnetometer).
        The VQF variant is compared separately at the end - it is a bundle-config choice, not a
        formula, and config.json records which one this bundle ships.

    `align_first` (default) runs align_sensors BEFORE filter_imu. Production does the opposite
    (filter, then align, src/train_one_subject_and_loso.py:94), and on subjects where alignment
    drops a sample that ordering makes the comparison meaningless: the offline filters are seeded
    from a sample align_sensors then deletes, which the online stream - and the real device - never
    sees. The two filter states then never reconcile. --pipeline-order reproduces that and measures
    the resulting skew; it is a property of the offline pipeline, not of the extractor.
    """
    DATA = preprocessing.load_subject(participants, pid)
    side = DATA["metadata"]["impaired_side"]
    if side not in ("R", "L"):
        raise ValueError(f"{pid}: impaired_side is {side!r}")
    imu_data, _ = tl._filter_rename_sensors(DATA, sensor_map, label_map)

    fa = f"{'left' if side == 'L' else 'right'}_forearm"
    ua = f"{'left' if side == 'L' else 'right'}_upperarm"
    roles = {"forearm": fa, "upperarm": ua, "chest": "chest"}
    raw_of = {"right_forearm": "a2", "right_upperarm": "a3",
              "left_forearm": "b2", "left_upperarm": "b3", "chest": "b4"}

    prepared, vqf9 = {}, {}
    for name in roles.values():
        df = imu_data[name].copy()
        if max_samples:
            df = df.iloc[:max_samples].copy()
        # RAW mirrors of the two channel groups filter_imu overwrites. filter_imu only touches
        # Acc_*, Acc_Gravity_Free_* and Gyr_*, so these ride through untouched; Quat_* is never
        # filtered, so it is already raw. This is what the online Preprocessor replays.
        for c in ("Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"):
            df[f"RAW_{c}"] = df[c].to_numpy(float)
        df["vqf_speed"] = gyr_perp_speed(df)            # deployed movement signal (mag-free)
        # the 9-axis VQF column the thesis pipeline attaches, kept aside for the closing check
        raw = DATA["imu"][raw_of[name]]
        if max_samples:
            raw = raw.iloc[:max_samples]
        vqf9[name] = wf.vqf_axis_speed(raw)
        df["y"] = 0                                     # build_windows needs it; unused by features
        prepared[name] = df

    if align_first:
        prepared, n_dropped = preprocessing.align_sensors(prepared, ref_sensor=fa)
        imu_cut = preprocessing.filter_imu(prepared)
    else:
        imu_f = preprocessing.filter_imu(prepared)                   # production order
        imu_cut, n_dropped = preprocessing.align_sensors(imu_f, ref_sensor=fa)
    preprocessing.assert_time_aligned(imu_cut)
    return dict(imu=imu_cut, side=side, roles=roles, n_dropped=n_dropped, align_first=align_first,
                vqf9={r: vqf9[n] for r, n in roles.items()}, participant=pid)


def run_online(imu, roles, config):
    """Stream every sample through one Preprocessor per sensor -> D of shape (N, 3, 14)."""
    pres = {r: F.Preprocessor.from_config(config) for r in F.SENSORS}
    cols_q = ["Quat_W", "Quat_X", "Quat_Y", "Quat_Z"]
    cols_a = ["RAW_Acc_X", "RAW_Acc_Y", "RAW_Acc_Z"]
    cols_g = ["RAW_Gyr_X", "RAW_Gyr_Y", "RAW_Gyr_Z"]
    per_sensor = {r: (imu[roles[r]][cols_q].to_numpy(float),
                      imu[roles[r]][cols_a].to_numpy(float),
                      imu[roles[r]][cols_g].to_numpy(float)) for r in F.SENSORS}
    n = len(imu[roles["forearm"]])
    D = np.empty((n, len(F.SENSORS), len(F.DERIVED_CHANNELS)))
    for si, r in enumerate(F.SENSORS):
        q, a, g = per_sensor[r]
        pre = pres[r]
        for i in range(n):
            D[i, si] = pre.push(q[i], a[i], g[i])
    return D


ANGLE = [f for f in wf.SET85 if f.startswith(("elev_", "flex_"))]
ANGLE_IX = [wf.SET85.index(f) for f in ANGLE]
OTHER_IX = [i for i in range(85) if i not in ANGLE_IX]


def compare(pid, prep, config, args):
    imu, roles = prep["imu"], prep["roles"]
    fa_df = imu[roles["forearm"]]
    t_ref = fa_df["Absolute Time"].values
    n = len(fa_df)

    # step = 1 sample -> every end index is available (see the module docstring)
    off = wf.build_windows(imu, paretic_side=prep["side"], win_ms=1000 * WIN / FS,
                           step_ms=1000 / FS, fs=FS, sections=None)
    end_off = np.searchsorted(t_ref, off["t_end"].values) + 1        # exclusive end sample index
    OFF_by_end = dict(zip(end_off.tolist(), range(len(off))))
    OFF_ALL = off[wf.SET85].values

    D = run_online(imu, roles, config)
    ends = [e for e in range(WIN + PAD, n + 1, HOP)
            if e in OFF_by_end and (e - F.LAG) in OFF_by_end]
    ON = np.array([F.extract(D[e - WIN - PAD:e], fs=FS) for e in ends])

    OFF = np.empty_like(ON)                                          # the correct counterpart row
    same = OFF_ALL[[OFF_by_end[e] for e in ends]]                    # offline at e
    lagd = OFF_ALL[[OFF_by_end[e - F.LAG] for e in ends]]            # offline at e - LAG
    OFF[:, OTHER_IX] = same[:, OTHER_IX]
    OFF[:, ANGLE_IX] = lagd[:, ANGLE_IX]

    print(f"\n{'=' * 78}\n{pid}  ({prep['side']}-paretic)   {n} samples, "
          f"{len(off)} offline windows (step=1), {len(ends)} compared, "
          f"align dropped {prep['n_dropped']}")
    if prep["n_dropped"] and not prep["align_first"]:
        print("  WARNING: production order (filter -> align) AND align_sensors dropped a sample: the "
              "offline filters were seeded from a row the online stream never sees. Any mismatch "
              "below is that skew, not the extractor. Re-run without --pipeline-order.")
    elif prep["n_dropped"]:
        print(f"  note: align_sensors dropped {prep['n_dropped']} sample(s) BEFORE filtering, so both "
              f"paths still see an identical series")

    # sanity check demanded by the task: if this one is off, the alignment is wrong, not the formulas
    j = wf.SET85.index("forearm_acc_mag_mean")
    sanity = np.abs(ON[:, j] - OFF[:, j]).max()
    print(f"  alignment sanity  forearm_acc_mag_mean  max|delta| = {sanity:.3e}  "
          f"{'OK' if sanity < 1e-9 else 'FAIL -> alignment, not formulas'}")

    # what the deliberate LAG buys: the same 12 features against the UNLAGGED offline row
    unlag = np.abs(ON[WARMUP_WINDOWS:, ANGLE_IX] - same[WARMUP_WINDOWS:, ANGLE_IX]).max()
    lagged = np.abs(ON[WARMUP_WINDOWS:, ANGLE_IX] - lagd[WARMUP_WINDOWS:, ANGLE_IX]).max()
    print(f"  angle group (12)  vs offline @e-LAG {lagged:.3e}   vs offline @e {unlag:.3e} "
          f"(the {F.LAG}-sample lag features.py takes by design)")

    return ends, ON, OFF, sanity


def per_feature_table(ON, OFF, skip):
    """(name, max abs diff, max rel diff) for all 85, sorted by abs diff descending."""
    on, offl = ON[skip:], OFF[skip:]
    d = np.abs(on - offl)
    denom = np.maximum(np.abs(offl).max(axis=0), 1e-12)
    return sorted(zip(wf.SET85, d.max(axis=0), d.max(axis=0) / denom),
                  key=lambda r: -r[1])


def print_table(rows, title, top=15):
    n_ok = sum(1 for _, a, _ in rows if a < TOL)
    print(f"\n  {title}   {n_ok}/85 under {TOL:g}")
    print(f"    {'feature':28s} {'max|delta|':>12s} {'max rel':>12s}")
    for name, a, rel in rows[:top]:
        flag = "" if a < TOL else "   <-- MISMATCH"
        print(f"    {name:28s} {a:12.3e} {rel:12.3e}{flag}")
    return n_ok


def name_groups(rows):
    """Which of the flagged failure-mode groups a mismatch falls into."""
    bad = {name for name, a, _ in rows if a >= TOL}
    if not bad:
        return []
    hits = [(label, sorted(f for f in bad if pred(f))) for label, pred in GROUPS.items()]
    hits = [(label, fs) for label, fs in hits if fs]
    other = sorted(bad - {f for _, fs in hits for f in fs})
    if other:
        hits.append(("ungrouped", other))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--max-samples", type=int, default=60000,
                    help="cap samples per subject (0 / --full = whole recording)")
    ap.add_argument("--full", action="store_true", help="replay the whole recording")
    ap.add_argument("--top", type=int, default=15, help="rows of the per-feature table to print")
    ap.add_argument("--pipeline-order", action="store_true",
                    help="filter BEFORE aligning, as production does - measures the train/serve skew "
                         "a dropped alignment sample causes (see prepare_subject)")
    args = ap.parse_args()
    max_samples = 0 if args.full else args.max_samples

    config = json.loads(CONFIG.read_text())
    assert config["feature_names"] == list(F.SET85) == list(wf.SET85), "feature order drift"
    assert (config["window_samples"], config["hop_samples"]) == (WIN, HOP)
    participants = pd.read_csv(MVMT_DET / "configs" / "participants.csv")
    sensor_map = pd.read_csv(MVMT_DET / "configs" / "sensor_map.csv")
    label_map = pd.read_csv(MVMT_DET / "configs" / "label_map.csv")

    results, all_pass = [], True
    for pid in args.subjects:
        prep = prepare_subject(participants, sensor_map, label_map, pid, max_samples,
                               align_first=not args.pipeline_order)
        ends, ON, OFF, sanity = compare(pid, prep, config, args)

        rows = per_feature_table(ON, OFF, WARMUP_WINDOWS)
        n_ok = print_table(rows, f"windows {WARMUP_WINDOWS}..{len(ends)} (headline)", args.top)
        warm = per_feature_table(ON[:WARMUP_WINDOWS], OFF[:WARMUP_WINDOWS], 0)
        n_ok_warm = sum(1 for _, a, _ in warm if a < TOL)
        print(f"\n  warm-up (first {WARMUP_WINDOWS} windows)   {n_ok_warm}/85 under {TOL:g}, "
              f"worst {warm[0][0]} {warm[0][1]:.3e}")

        for label, fs_ in name_groups(rows):
            print(f"    GROUP MISMATCH - {label}: {', '.join(fs_)}")
        ok = n_ok == 85 and sanity < TOL
        all_pass &= ok
        results.append((pid, len(ends), n_ok, rows[0][1], ok, ON, OFF))

        # the one deliberate offline deviation, reported rather than hidden
        v9 = prep["vqf9"]
        d9 = max(abs(np.nanmean(v9[r][max(0, e - WIN):e]) - OFF[k, wf.SET85.index(f"{r}_vqf_speed")])
                 for r in F.SENSORS for k, e in list(enumerate(ends))[WARMUP_WINDOWS::997])
        print(f"\n  aside: same features under the 9-axis VQF movement signal would differ by "
              f"up to {d9:.3e} (rad/s) - config choice, not a formula difference")

    print(f"\n{'=' * 78}\nSUMMARY")
    print(f"  {'subject':10s} {'windows':>8s} {'pass/85':>8s} {'worst |delta|':>14s}  verdict")
    for pid, nw, n_ok, worst, ok, *_ in results:
        print(f"  {pid:10s} {nw:8d} {n_ok:8d} {worst:14.3e}  {'PASS' if ok else 'FAIL'}")

    if not all_pass:
        print("\nFAIL - at least one feature group mismatches. Reported above; features.py NOT "
              "modified. Stopping before the booster check.")
        return 1

    print("\nAll 85 features match on every subject. Scoring both matrices with the frozen booster.")
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(BOOSTER))
    print(f"  {'subject':10s} {'max|p_on - p_off|':>18s}  {'p range':>22s}")
    for pid, _, _, _, _, ON, OFF in results:
        p_on = booster.predict(xgb.DMatrix(ON))
        p_off = booster.predict(xgb.DMatrix(OFF))
        print(f"  {pid:10s} {np.abs(p_on - p_off).max():18.3e}  "
              f"[{p_off.min():.4f}, {p_off.max():.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
