"""The Block 1 online chain in one place: Aligner -> Preprocessor -> extract -> Block1 -> Decision.

Kept out of web_demo.py so it can be driven with no bleak, no aiohttp and no GPIO - which is the
only way the wiring gets tested off the Pi. web_demo's block1_task is a timer around .tick().

WHY THE PREPROCESSORS RUN HERE AND NOT IN THE BLE CALLBACK
Aligner buffers RAW channels - it stores concatenate([quat, acc, gyr]), 10 wide, and window()
returns (win_n, 3, 10). extract() needs the 14 DERIVED channels. Preprocessor is stateful, so it
must see each sample exactly ONCE and IN ORDER; feeding it from the BLE callback would feed it the
unaligned stream, and re-running it over the overlapping windows Aligner emits would corrupt its
filter state. So the Preprocessors run on the ALIGNED stream: each tick pushes only the aligned
indices that are new since the previous tick, straight out of the window Aligner just returned.
At a 100 ms tick and a 60 Hz stream that is ~6 new rows out of the 17 in the window, so they are
always still in it. If more than win_n indices went by (a stall longer than ~283 ms) the missing
rows cannot be reconstructed, and the filters are reset instead - same remedy as a gap.
"""
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

import features
from inference import Aligner, Block1, Decision, Lhoste, TICK_US

# The Lhoste baseline's whole input: the forearm long-axis angular speed, already one of the 85.
FOREARM_SPEED = features.SET85.index("forearm_vqf_speed")


class Block1Pipeline:
    """Feed it samples from the BLE callback; call tick() on a fixed timer.

        pipe = Block1Pipeline(model_dir, ("forearm", "upper_arm", "torso"))
        ...
        pipe.push(sample)                      # BLE callback - cheap, no inference
        out = pipe.tick(time.time_ns() // 1000)   # 100 ms timer; dict when scored, else None
    """

    def __init__(self, model_dir, roles):
        self.cfg = json.loads((Path(model_dir) / "config.json").read_text())
        self.win_n = int(self.cfg["window_samples"]) + features.ANGLE_LEAD
        self.hop = int(self.cfg["hop_samples"])
        self.aligner = Aligner(roles, self.win_n, rate_hz=self.cfg["rate_hz"],
                               max_gap_ms=self.cfg["max_gap_ms"])
        self.block1 = Block1(model_dir)
        self.decision = Decision(self.cfg)
        # Unsupervised literature baseline, scored on the same windows. Display/logging only - it
        # never reaches the TTL; web_demo fires on `fired` from Decision alone.
        self.lhoste = Lhoste(self.cfg)
        self.pre = {s: features.Preprocessor.from_config(self.cfg) for s in features.SENSORS}
        self.n_scored = self.n_fired = self.n_lhoste_fired = 0
        self.n_gap = self.n_desync = 0        # real data gaps / times we fell behind the stream
        self.t_pre = self.t_extract = self.t_predict = self.t_decide = 0.0
        self.reset()

    # ------------------------------------------------------------------ ingest
    def push(self, sample):
        """From the BLE notify path. Alignment bookkeeping only - no filtering, no inference."""
        self.aligner.push(sample)

    def reset(self):
        """Discontinuity: filters, smoother, arm state and the derived buffer all restart."""
        for p in self.pre.values():
            p.reset()
        self.decision.reset()
        self.lhoste.reset()
        self._buf = deque(maxlen=self.win_n)
        self._last_k = None
        self._last_scored = -10 ** 9

    # ------------------------------------------------------------------ tick
    def tick(self, now_us):
        """One timer tick. Scores EVERY hop-aligned window that has become available since the
        last tick, in order, and returns them ALL as a list (empty if none were ready).

        The caller must handle every entry, not just the newest: a late tick scores two or three
        windows, and dropping the earlier ones would lose their CSV rows and - worse - swallow a
        `fired` that should have driven the TTL.

        Scoring is driven by the SAMPLE INDEX, never by the tick. A late tick therefore yields
        two or three windows instead of one, and consecutive scored windows stay exactly `hop`
        apart. That is what keeps Decision's timestamps hop-spaced: score once per tick instead,
        and a slow tick makes k jump by >= 9, dt = 9*16667 = 150003 us crosses Decision's
        150000 us gap threshold by 3 us, and the smoother silently resets mid-session. That is
        the bug that turned 8 stim/min into 100.
        """
        if self.aligner.gap_detected():          # real gap: samples missing, or a clock-shift step
            self.n_gap += 1
            self.reset()
        w = self.aligner.window(now_us)
        if w is None:
            return []
        return [self._score(idx, derived)
                for idx, derived in self._advance(w, self.aligner.last_end_index)]

    def _advance(self, w, k):
        """Push the aligned rows new since the last tick; return [(index, window)] to score.

        Nothing here looks at wall-clock time: a window spanning more aligned indices than the
        nominal hop just means the tick was slow, and no sample was missing.
        """
        if self._last_k is not None and k - self._last_k > self.win_n:
            # More rows went by than the window can carry, so the ones in between are gone and
            # the filters cannot be continued across them. This is falling behind the stream, NOT
            # a data gap; the absolute-deadline scheduler should keep it at zero.
            self.n_desync += 1
            self.reset()
        n_new = self.win_n if self._last_k is None else min(k - self._last_k, self.win_n)
        first = k - n_new + 1
        ready, last = [], self._last_scored
        t0 = time.perf_counter()
        for j in range(n_new):
            row = w[self.win_n - n_new + j]      # oldest new row first, order matters
            self._buf.append(np.stack([
                self.pre[s].push(row[i, 0:4], row[i, 4:7], row[i, 7:10])
                for i, s in enumerate(features.SENSORS)]))
            idx = first + j
            if len(self._buf) == self.win_n and idx >= last + self.hop:
                ready.append((idx, np.stack(self._buf)))
                last = idx
        self.t_pre += time.perf_counter() - t0
        self._last_k = k
        return ready

    def _score(self, idx, derived):
        t0 = time.perf_counter()
        feats = features.extract(derived)
        t1 = time.perf_counter()
        p = self.block1.p(feats)
        t2 = time.perf_counter()
        # Decision runs on the SAMPLE clock. idx is the aligned sample index and consecutive
        # scored windows are exactly hop apart, so dt is always 100002 us - well under the
        # 150000 us gap threshold, whatever the tick loop is doing.
        fired = self.decision.update(p, idx * TICK_US)
        # The baseline reads one feature out of the vector just extracted, so it costs no extra
        # signal processing; it is timed with `decide` because that is the stage it belongs to.
        lh_fired = self.lhoste.update(feats[FOREARM_SPEED], idx * TICK_US)
        t3 = time.perf_counter()
        self.t_extract += t1 - t0
        self.t_predict += t2 - t1
        self.t_decide += t3 - t2
        self._last_scored = idx
        self.n_scored += 1
        self.n_fired += bool(fired)
        self.n_lhoste_fired += bool(lh_fired)
        # us_* are the SAME measurements timings() averages, emitted per window instead of only
        # as a lifetime mean - a mean cannot show a tail, and these accumulators are never reset,
        # so late-session degradation is invisible in timings(). Preprocessing is deliberately
        # absent: it runs once per _advance() batch over all the new rows, so it has no per-window
        # value to report; timings()["preprocess"] remains the way to read it.
        return {"k": idx, "sensor_t_us": self.aligner.sensor_t_us_at(idx),
                "p": p, "p_smooth": self.decision.p_smooth,
                "armed": self.decision.armed, "stimulating": self.decision.stimulating,
                "fired": bool(fired),
                "lh": self.lhoste.score, "lh_thr": self.lhoste.threshold,
                "lh_fired": bool(lh_fired),
                "us_extract": (t1 - t0) * 1e6, "us_predict": (t2 - t1) * 1e6,
                "us_decide": (t3 - t2) * 1e6}

    def timings(self):
        """Mean us per scored window, per stage. Read these off the Pi, not a laptop."""
        n = max(1, self.n_scored)
        return {"preprocess": self.t_pre / n * 1e6, "extract": self.t_extract / n * 1e6,
                "predict": self.t_predict / n * 1e6, "decide": self.t_decide / n * 1e6}

    # ------------------------------------------------------------------ startup benchmark
    def benchmark(self, n=1000):
        """us/call for extract + Block1.p. MUST be read off the Pi - a laptop number is not it."""
        rng = np.random.default_rng(0)
        w = rng.normal(0, 1, (self.win_n, len(features.SENSORS), len(features.DERIVED_CHANNELS)))
        e = w[:, :, 11:14]                       # the long-axis channels are unit vectors
        w[:, :, 11:14] = e / np.linalg.norm(e, axis=2, keepdims=True)
        for _ in range(20):
            self.block1.p(features.extract(w))   # warm up
        t0 = time.perf_counter()
        for _ in range(n):
            self.block1.p(features.extract(w))
        return (time.perf_counter() - t0) / n * 1e6
