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
from inference import Aligner, Block1, Decision, TICK_US


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
        self.pre = {s: features.Preprocessor.from_config(self.cfg) for s in features.SENSORS}
        self.n_scored = 0
        self.n_fired = 0
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
        self._buf = deque(maxlen=self.win_n)
        self._last_k = None
        self._last_scored = -10 ** 9

    # ------------------------------------------------------------------ tick
    def tick(self, now_us):
        """One timer tick. Returns a dict on a SCORED window, else None."""
        if self.aligner.gap_detected():          # data gap, or an adopted clock-shift change
            self.reset()
        w = self.aligner.window(now_us)
        if w is None:
            return None
        k = self.aligner.last_end_index
        derived = self._feed(w, k)
        if derived is None or k < self._last_scored + self.hop:
            return None                          # same k as last time, or still warming up

        p = self.block1.p(features.extract(derived))
        # Decision is timed on the SAMPLE clock, not the tick clock: k is the aligned sample index,
        # so consecutive scored windows are exactly hop*TICK_US = 100 ms apart. Timing it on the
        # wall clock would let one late tick (>150 ms) look like a gap and reset the smoother.
        fired = self.decision.update(p, k * TICK_US)
        self._last_scored = k
        self.n_scored += 1
        self.n_fired += bool(fired)
        return {"k": k, "p": p, "p_smooth": self.decision.p_smooth,
                "armed": self.decision.armed, "stimulating": self.decision.stimulating,
                "fired": bool(fired)}

    def _feed(self, w, k):
        """Push the aligned rows that are new since the last tick through the Preprocessors."""
        if self._last_k is not None and k - self._last_k > self.win_n:
            self.reset()                         # too many missed rows to rebuild from this window
        n_new = self.win_n if self._last_k is None else min(k - self._last_k, self.win_n)
        for row in w[self.win_n - n_new:]:       # oldest new row first, order matters
            self._buf.append(np.stack([
                self.pre[s].push(row[i, 0:4], row[i, 4:7], row[i, 7:10])
                for i, s in enumerate(features.SENSORS)]))
        self._last_k = k
        return np.stack(self._buf) if len(self._buf) == self.win_n else None

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
