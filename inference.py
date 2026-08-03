"""Online inference blocks.

`Aligner` is the synchroniser between the three independent BLE streams and the feature
extractor: it turns three unsynchronised Sample streams into one (win_n, 3, 10) block whose
rows are the same instant on every sensor. It does no filtering and computes no features -
that is features.Preprocessor / features.extract, downstream.

No bleak / aiohttp / lgpio here: Aligner is constructible and testable with no hardware.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import features
from features import Preprocessor, extract, SENSORS, ANGLE_LEAD, SET85
from check_rate import unwrap_us          # single source of truth for the 32-bit counter wrap

if TYPE_CHECKING:                         # placement_scan_movella imports bleak at module level;
    from placement_scan_movella import Sample   # Aligner only duck-types a Sample, so type-only

# ---------------------------------------------------------------------------------------------
# THE role mapping. The repo's BLE roles and the model's sensor names differ; this is the one
# place they meet. features.SENSORS fixes the sensor axis order of every window.
# ---------------------------------------------------------------------------------------------
ROLE_TO_SENSOR = {"forearm": "forearm", "upper_arm": "upperarm", "torso": "chest"}
SENSOR_TO_ROLE = {v: k for k, v in ROLE_TO_SENSOR.items()}

# quat(4) + acc(3) + gyr(3) - config.json's channels_per_sensor order, i.e. exactly what
# features.Preprocessor.push() takes.
N_CHANNELS = 10
HOP_SAMPLES = 6               # config.json hop_samples - the tick loop's scoring stride
TICK_US = 16667               # measured device tick; NOT 1e6/60. Indexing on the real tick makes
                              # the index slope exactly 1.0, so it can never skip; at 1e6/60 the
                              # 20 ppm error inserted a phantom hole every ~14 min per sensor.
OFFSET_WINDOW_S = 10.0        # rolling min(host - sensor) horizon
OFFSET_REFRESH_S = 30.0       # how often the integer offsets are recomputed
# The index is relative to each sensor's own first sample, so before an offset is adopted all
# three sensors sit at index 0 and look spuriously aligned. window() therefore returns None until
# adoption, and adoption waits for enough packets for the rolling minimum to have rejected BLE
# queueing delay - a min over one bursty packet is not an offset estimate.
OFFSET_WARMUP_S = 1.0
# Sensors free-run, so the true offset is rarely a whole number of samples: measured 0.44 samples
# on the clean recording. Re-rounding that every 30 s makes the integer chatter between two
# neighbours, and a mid-stream flip shifts every later row of that sensor by one sample - a fake
# 16.7 ms step in the features, not a resync. So an adopted shift is HELD until the estimate
# moves more than this far from it. The measured drift (< 0.3 ms/min) needs ~40 min to cover
# that, and every estimate is logged either way, so genuine drift is still visible.
OFFSET_DEADBAND = 0.75        # samples


def compute_features(sample, feature_name):
    if feature_name == "gyro_magnitude":
        return float(np.linalg.norm(sample.gyr)) # deg/s
    elif feature_name == "acc_magnitude":
        return float(np.linalg.norm(sample.acc)) # m/s²
    else:
        raise ValueError(f"Unknown feature: {feature_name}")


class _SensorState:
    """Ring buffer + clock bookkeeping for one sensor."""

    def __init__(self, cap):
        self.cap = cap
        self.rows = {}            # sample index (own clock, unshifted) -> (10,) float row
        self.prev_raw = None      # last RAW sensor_t_us, for wrap detection
        self.wraps = 0            # how many times the 32-bit counter has wrapped
        self.t0 = None            # first unwrapped sensor_t_us; the index is relative to it
        self.last_idx = None
        self.last_host_us = None
        self.offsets = deque()    # (host_t_us, host_t_us - (t_us - t0)) over OFFSET_WINDOW_S


class Aligner:      # ring buffers + offsets -> (17, 3, 10) or None
    """Synchronise the sensor streams onto one integer sample index.

    The device clocks tick at exactly 16667 us and their offsets drift < 0.3 ms/min, so aligning
    them is ONE CONSTANT INTEGER SAMPLE SHIFT per sensor - no interpolation, no resampling, no
    drift model. host_t_us is bursty and is used for nothing except estimating that shift (via a
    rolling minimum, which picks the least-delayed packet) and judging staleness.

        aligner = Aligner(("forearm", "upper_arm", "torso"), win_n=12 + features.ANGLE_LEAD)
        ...
        # BLE callbacks
        aligner.push(sample)

        # tick loop, every 100 ms
        if aligner.gap_detected():                    # data gap OR an adopted shift change
            for p in preprocessors.values(): p.reset()
            decision.reset()
            last_scored = -10**9
        w = aligner.window(time.time_ns() // 1000)    # (win_n, 3, 10) or None
        k = aligner.last_end_index
        if w is not None and k >= last_scored + HOP_SAMPLES:
            ...score...                               # window() can repeat the same k, so gate
            last_scored = k                           # on the hop or the same window is scored
                                                      # up to three times
    """

    def __init__(self, roles, win_n, rate_hz=60.0, max_gap_ms=150, stale_ms=300):
        roles = list(roles)
        unknown = [r for r in roles if r not in ROLE_TO_SENSOR]
        if unknown:
            raise ValueError(f"unknown role(s) {unknown}; expected {list(ROLE_TO_SENSOR)}")
        if {ROLE_TO_SENSOR[r] for r in roles} != set(features.SENSORS):
            raise ValueError(f"roles {roles} do not cover features.SENSORS {features.SENSORS}")
        self.roles = roles
        self.ref_role = roles[0]                 # offsets are expressed relative to this one
        self.win_n = int(win_n)
        self.rate_hz = float(rate_hz)            # nominal only; TICK_US is what indexing uses
        self.max_gap_n = int(round(max_gap_ms * 1000.0 / TICK_US))
        self.stale_us = int(stale_ms * 1000)
        self.cap = 4 * self.win_n
        self.offset_log = []                     # [(host_t_us, {role: shift})] - drift audit trail
        self.shift = {r: 0 for r in self.roles}
        self._adopted = False                    # has a shift ever been adopted?
        self._last_offset_us = None
        self.reset()

    # ------------------------------------------------------------------ ingest
    def push(self, sample: "Sample") -> None:
        """Take one Sample. Ignores a duplicate index or one older than the buffer."""
        st = self.state[sample.role]
        raw = int(sample.sensor_t_us)
        # 32-bit wrap: hand the (previous, current) pair to check_rate.unwrap_us rather than
        # writing a second wrap rule; if it bumped the second element, the counter wrapped.
        if st.prev_raw is not None and int(unwrap_us([st.prev_raw, raw])[1]) != raw:
            st.wraps += 1
        st.prev_raw = raw
        t_us = raw + st.wraps * 2 ** 32
        if st.t0 is None:
            st.t0 = t_us
        idx = int(round((t_us - st.t0) / TICK_US))   # slope exactly 1.0 -> never skips

        host = int(sample.host_t_us)
        st.last_host_us = host if st.last_host_us is None else max(st.last_host_us, host)
        self.now_us = max(self.now_us, host)     # "now" = the newest host stamp seen anywhere

        if st.last_idx is not None and idx - st.last_idx > self.max_gap_n:
            self._gap = True                     # cleared by gap_detected()

        # Offset trace - the host clock enters here and in the staleness test, nowhere else.
        # Stored against the INDEX-relative time (t_us - t0), so each sensor's own clock origin
        # falls out and (min_r - min_ref) / TICK_US is directly the shift in samples.
        st.offsets.append((host, host - (t_us - st.t0)))
        horizon = host - int(OFFSET_WINDOW_S * 1e6)
        while st.offsets and st.offsets[0][0] < horizon:
            st.offsets.popleft()
        if (self._last_offset_us is None
                or self.now_us - self._last_offset_us >= OFFSET_REFRESH_S * 1e6):
            self._recompute_offsets()

        if idx in st.rows or (st.rows and idx < min(st.rows)):
            return                               # already present, or older than the buffer
        st.rows[idx] = np.concatenate([np.asarray(sample.quat, float),
                                       np.asarray(sample.acc, float),
                                       np.asarray(sample.gyr, float)])
        st.last_idx = idx if st.last_idx is None else max(st.last_idx, idx)
        while len(st.rows) > self.cap:
            st.rows.pop(min(st.rows))

    def _recompute_offsets(self):
        """Integer sample shift per sensor, relative to roles[0], from the rolling min offset."""
        warm = {r: st for r, st in self.state.items()
                if st.offsets and st.offsets[-1][0] - st.offsets[0][0] >= OFFSET_WARMUP_S * 1e6}
        if len(warm) < len(self.roles):
            return                               # not every sensor has streamed long enough yet
        mins = {r: min(o for _, o in st.offsets) for r, st in warm.items()}
        base = mins[self.ref_role]
        exact = {r: (mins[r] - base) / TICK_US for r in self.roles}
        for r in self.roles:
            if not self._adopted or abs(exact[r] - self.shift[r]) > OFFSET_DEADBAND:
                new = int(round(exact[r]))
                # A shift that actually moves is a one-sample step in the aligned signal: the
                # Preprocessors' filter states no longer correspond to the samples now arriving.
                # Same remedy as a data gap, so raise the same flag - the deadband makes this
                # rare, the reset makes it correct when it does happen.
                if self._adopted and new != self.shift[r]:
                    self._gap = True
                self.shift[r] = new
        self._adopted = True
        # log the exact estimate too - that is the trace drift is actually visible in
        self.offset_log.append((self.now_us, dict(self.shift),
                                {r: round(exact[r], 3) for r in self.roles}))
        self._last_offset_us = self.now_us

    # ------------------------------------------------------------------ output
    def window(self, now_us=None):
        """(win_n, 3, 10) float array ending at the newest index all sensors share, else None.

        Sensor axis is features.SENSORS order; channel axis is quat(4) + acc(3) + gyr(3).
        Never partial and never zero-padded: if any sensor is missing any of the win_n indices,
        or any sensor has gone quiet for longer than stale_ms, the answer is None.

        `now_us` is the caller's clock, for staleness. A TIMER-DRIVEN tick loop must pass it:
        without it "now" is the newest host stamp seen, which stops advancing when every sensor
        goes quiet, so a total stream death would never look stale. It MUST be the same clock
        as Sample.host_t_us, which placement_scan_movella sets from time.time_ns() // 1000 -
        so pass time.time_ns() // 1000, NOT time.monotonic_ns() // 1000. Monotonic is
        boot-relative; mixing the two makes every difference negative and staleness silently
        unreachable, which is why the mismatch raises below instead of passing quietly.
        """
        if now_us is not None:
            now_us = int(now_us)
            if self.now_us and now_us < self.now_us - self.stale_us - 60_000_000:
                raise ValueError(
                    f"now_us={now_us} is {(self.now_us - now_us) / 1e6:.0f} s behind the newest "
                    f"host stamp ({self.now_us}): different clock domains. Sample.host_t_us is "
                    f"time.time_ns()//1000 (epoch), so pass that, not time.monotonic_ns()//1000.")
            self.now_us = max(self.now_us, now_us)
        if not self._adopted:
            return None                          # no offsets yet -> no defined alignment
        for st in self.state.values():
            if st.last_host_us is None or self.now_us - st.last_host_us > self.stale_us:
                return None
        aligned = [{i + self.shift[r] for i in self.state[r].rows} for r in self.roles]
        common = set.intersection(*aligned)
        if not common:
            return None
        k = max(common)
        need = list(range(k - self.win_n + 1, k + 1))
        if not all(a in common for a in need):
            return None
        out = np.empty((self.win_n, len(features.SENSORS), N_CHANNELS))
        for si, sensor in enumerate(features.SENSORS):
            role = SENSOR_TO_ROLE[sensor]
            rows, sh = self.state[role].rows, self.shift[role]
            for ti, a in enumerate(need):
                out[ti, si] = rows[a - sh]
        self.last_end_index = k
        return out

    def gap_detected(self) -> bool:
        """True once per gap (any sensor skipping more than max_gap_ms of samples), then clears."""
        gap, self._gap = self._gap, False
        return gap

    def reset(self) -> None:
        """Drop all buffered samples and the gap flag. The clock shifts are a property of the
        hardware, not of the stream, so they survive - only the data is dropped."""
        self.state = {r: _SensorState(self.cap) for r in self.roles}
        self._gap = False
        self.now_us = 0
        self.last_end_index = None


# ------------------ Movement Detection Block ------------------
class Block1:  # booster + preallocated buffer -> p
    """85 features in, one P(MOVE) out. No smoothing, no threshold, no state - Decision's job.

    Scoring goes through booster.inplace_predict on a preallocated row buffer, NOT a preallocated
    DMatrix. xgb.DMatrix COPIES its input at construction, so a DMatrix built once and re-read
    after overwriting the buffer keeps returning the prediction for whatever the buffer held at
    construction time - with a zeroed buffer, p = 0.001321 for every window, forever. That is a
    silent wrong answer of exactly the kind the feature-order assertion below exists to prevent.
    inplace_predict reads the live buffer, allocates nothing per call, and is bit-identical to
    booster.predict(xgb.DMatrix(x)) - verified to 0.0 over 5000 real windows.

    The scaler is deliberately not applied: config.json says scaler_applied false, so the booster
    was fit on raw feature values and its split thresholds are in raw units.
    """

    def __init__(self, model_dir, nthread=1):
        import xgboost as xgb           # lazy: keeps inference importable without xgboost

        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())
        names = list(cfg["feature_names"])
        # booster.json carries no feature names, so this is the ONLY thing standing between a
        # reordered extractor and confident, wrong, perfectly valid-looking probabilities.
        if names != list(features.SET85):
            n = min(len(names), len(features.SET85))
            i = next((j for j in range(n) if names[j] != features.SET85[j]), n)
            raise ValueError(
                f"feature order mismatch at index {i}: config.json has "
                f"{names[i] if i < len(names) else '<end>'!r}, features.SET85 has "
                f"{features.SET85[i] if i < len(features.SET85) else '<end>'!r} "
                f"(lengths {len(names)} vs {len(features.SET85)}). booster.json is indexed by "
                f"this order - scoring it would produce wrong probabilities that still look valid.")
        if cfg.get("scaler_applied", False):
            raise ValueError("config.json has scaler_applied true; this class scores raw features. "
                             "Re-export the bundle without --scaled, or apply the scaler upstream.")

        self.booster = xgb.Booster()
        self.booster.load_model(str(model_dir / "booster.json"))
        if self.booster.num_features() != len(names):
            raise ValueError(f"booster expects {self.booster.num_features()} features, "
                             f"config.json lists {len(names)}")
        # One row at a time: the thread pool costs more than the trees do (201 -> 90 us/call here),
        # and on the Pi those cores are also carrying three BLE links.
        self.booster.set_param({"nthread": int(nthread)})
        self.n_features = len(names)
        self._buf = np.empty((1, self.n_features))       # overwritten in place on every call

    def p(self, feats) -> float:
        """(85,) features -> P(MOVE) as a Python float."""
        f = np.asarray(feats, dtype=float).ravel()
        if f.size != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {f.size}")
        if not np.isfinite(f).all():
            bad = np.flatnonzero(~np.isfinite(f))
            raise ValueError(f"non-finite feature(s) at index {bad.tolist()[:5]} "
                             f"({len(bad)} total): {[features.SET85[i] for i in bad[:5]]}")
        self._buf[0] = f
        return float(self.booster.inplace_predict(self._buf)[0])

# ------------------ Decision Block ------------------

class Decision: # k=3 smoother + arm/fire/re-arm -> bool
    """config: smoothing_k=3, threshold=0.68, stim_ms=1000, lockout_ms=0.

    Streaming form of src/smoothing.py::causal_moving_average followed by
    src/events.py::rearm_triggers. Those two produced the offline event-level numbers, so this is
    a translation of them, not a reimplementation - tests/test_decision.py::test_matches_reference
    requires an identical fired array on random sequences.

    Three behaviours are easy to assume wrongly, and all three follow rearm_triggers:

      * A signal that stays above threshold fires ONCE. Re-arming needs p_smooth to fall back
        below, so a constant 0.9 does not fire once per second - it fires at the first window and
        then waits.
      * A dip DURING the stimulation does not re-arm. rearm_triggers advances its index past the
        burst and lockout, so those windows are never examined; `armed` and `stimulating` are
        separate flags but the arm state simply does not move while the burst is running.
      * The minimum spacing is n_stim + n_lock + 1 windows, not n_stim: re-arming consumes the
        first window after the burst, so the earliest re-fire is 1.1 s, not 1.0 s.

    Comparisons are `>= threshold` to fire and `< threshold` to re-arm, both as written.
    A gap longer than max_gap_ms starts a new segment - the smoother restarts partial (1 value,
    then 2, then k) and the arm state resets, exactly as the offline pair segment their arrays.
    """

    def __init__(self, cfg):
        self.k = int(cfg["smoothing_k"])
        self.threshold = float(cfg["threshold"])
        self.step_ms = float(cfg["hop_ms"])
        self.max_gap_us = int(float(cfg["max_gap_ms"]) * 1000)
        self.n_stim = max(1, int(round(float(cfg["stim_ms"]) / self.step_ms)))
        self.n_lock = int(round(float(cfg["lockout_ms"]) / self.step_ms))
        self.reset()

    def update(self, p: float, t_us: int) -> bool:
        """One window probability -> True only on the window that STARTS a stimulation."""
        if self._last_t_us is not None and t_us - self._last_t_us > self.max_gap_us:
            self.reset()                       # new segment
        self._last_t_us = t_us

        self._buf.append(float(p))             # fewer than k at a segment start, by construction
        self._p_smooth = sum(self._buf) / len(self._buf)
        above = self._p_smooth >= self.threshold

        if self._skip > 0:                     # inside the stimulation or the lockout
            self._in_stim = self._stim_left > 0
            self._stim_left = max(0, self._stim_left - 1)
            self._skip -= 1
            return False

        if self._armed and above:
            self._armed = False
            self._skip = self.n_stim + self.n_lock - 1   # this window consumed the first one
            self._stim_left = self.n_stim - 1
            self._in_stim = True
            return True

        if not self._armed and not above:      # strict <: exactly at threshold does not re-arm
            self._armed = True
        self._in_stim = False
        return False

    def reset(self) -> None:                   # on any gap > max_gap_ms
        self._buf = deque(maxlen=self.k)
        self._p_smooth = None
        self._armed = True
        self._skip = 0
        self._stim_left = 0
        self._in_stim = False
        self._last_t_us = None

    @property
    def p_smooth(self):
        """The smoothed probability of the last window, or None before the first update."""
        return self._p_smooth

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def stimulating(self) -> bool:
        """True while the last window processed lies inside a stimulation burst."""
        return self._in_stim


# ------------------ Literature baseline (display only) ------------------

LHOSTE_MAX_RATE = np.pi / 2      # their "maximum rate", rad/s - the normalisation constant
LHOSTE_K = 3                     # "average the last three estimations"
LHOSTE_THR_WINDOW_S = 10.0       # the dynamic threshold looks back this far
LHOSTE_THR_PCT = 50.0            # 50th percentile of that window
LHOSTE_THR_LO, LHOSTE_THR_HI = 0.30, 0.60    # clipped to [30%, 60%] of max normalised speed


class Lhoste:  # forearm speed -> normalise -> smooth -> adaptive threshold -> arm/fire/re-arm
    """The SmartVNS movement detector of Lhoste et al. (2025), streaming, for side-by-side display.

    Unsupervised and calibration-free: no model, no training, one forearm signal. It is the
    baseline the model was scored against offline (src/baselines.py, event-level config), run here
    on the SAME windows, the same sample clock and the same stimulation logic as Block1 + Decision,
    so the two rows in the UI differ only in how the score is produced.

      1. speed  = window-mean angular speed of the forearm long axis, rad/s. That is exactly
                  SET85's `forearm_vqf_speed`, already computed for the model, so nothing extra is
                  extracted here (a 200 ms trailing mean every 100 ms - their step 1, verbatim).
      2. score  = clip(speed / (pi/2), 0, 1)
      3. smooth = mean of the last k=3 scores
      4. thr    = 50th percentile of the last 10 s of `smooth`, clipped to [0.30, 0.60];
                  fire where smooth exceeds thr, one 1 s burst, re-armed once it falls back below.

    Step 4's arm/burst/re-arm half is delegated to a Decision with k=1 and a threshold of 0, fed
    +/-1 for above/below - that is the same code the model path uses, so the two cannot drift
    apart. It is fed the DECISION and not the margin (smooth - thr) because Decision fires on
    `>=` and Lhoste's rule is a strict `>`: a margin of exactly 0 is not the rare tie it looks
    like, since the trailing median of a slowly-varying score frequently IS the current score.

    Nothing in this class touches the TTL. It is a reference reading, not a second trigger.
    """

    def __init__(self, cfg):
        self.max_gap_us = int(float(cfg["max_gap_ms"]) * 1000)
        # k=1 so Decision does no smoothing of its own (step 3 already did it); the threshold
        # comparison happens here instead, because Decision cannot hold a time-varying one.
        self.decision = Decision({**cfg, "smoothing_k": 1, "threshold": 0.0})
        self.reset()

    def update(self, speed: float, t_us: int) -> bool:
        """One window's forearm long-axis speed (rad/s) -> True on the window that starts a burst."""
        if self._last_t_us is not None and t_us - self._last_t_us > self.max_gap_us:
            self.reset()                       # new segment: smoother, history and arm state restart
        self._last_t_us = t_us

        self._buf.append(float(np.clip(speed / LHOSTE_MAX_RATE, 0.0, 1.0)))     # steps 1-2
        self.score = sum(self._buf) / len(self._buf)                            # step 3

        self._hist.append((t_us, self.score))                                   # step 4
        horizon = t_us - int(LHOSTE_THR_WINDOW_S * 1e6)
        while self._hist and self._hist[0][0] < horizon:
            self._hist.popleft()
        pct = float(np.percentile([v for _, v in self._hist], LHOSTE_THR_PCT))
        self.threshold = float(np.clip(pct, LHOSTE_THR_LO, LHOSTE_THR_HI))

        self.fired = self.decision.update(1.0 if self.score > self.threshold else -1.0, t_us)
        return self.fired

    def reset(self) -> None:
        self.decision.reset()
        self._buf = deque(maxlen=LHOSTE_K)
        self._hist = deque()                   # (t_us, score) over the last LHOSTE_THR_WINDOW_S
        self._last_t_us = None
        self.score = None                      # None until the first window is scored
        self.threshold = None
        self.fired = False
