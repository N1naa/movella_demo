"""Online inference blocks.

`Aligner` is the synchroniser between the three independent BLE streams and the feature
extractor: it turns three unsynchronised Sample streams into one (win_n, 3, 10) block whose
rows are the same instant on every sensor. It does no filtering and computes no features -
that is features.Preprocessor / features.extract, downstream.

No bleak / aiohttp / lgpio here: Aligner is constructible and testable with no hardware.
"""
from __future__ import annotations

from collections import deque
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
class Block1:  # booster + preallocated DMatrix -> p
    def __init__(self, model_dir):
        self.booster = xgb.Booster(); self.booster.load_model(...)
        assert cfg["feature_names"] == features.SET85
        self._buf = np.zeros((1, 85), dtype=np.float32)
        self._dm  = xgb.DMatrix(self._buf)      # allocate once
    def p(self, feats) -> float: ...

# ------------------ Decision Block ------------------

class Decision: # k=3 smoother + arm/fire/re-arm -> bool
    """config: smoothing_k=3, threshold=0.68, stim_ms=1000, lockout_ms=0."""
    def update(self, p, t_us) -> bool:   # True on a rising trigger
        ...
    def reset(self): ...                 # on any gap > max_gap_ms
