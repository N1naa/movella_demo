# recorder.py — per-IMU CSVs + trigger event log + a run.json config snapshot.
# Inactive until start(); write_sample / write_event are cheap no-ops when idle,
# so it's safe to wire as on_sample for every source and gate it on the session.
import asyncio, csv, json
from collections import deque
from pathlib import Path
from datetime import datetime

DRAIN_PERIOD_S = 0.2      # how often the drain task empties the queue


class Recorder:
    def __init__(self, root="recordings"):
        self.root = Path(root)
        self.active = False
        self.run_dir = None
        self._imu = {}            # role -> (file, writer)
        self._ev_f = None
        self._ev_w = None
        self._q = deque()         # (role, row) queued by the BLE path, written by _drain
        self._task = None         # the drain task, if there was a loop to start it on

    def start(self, roles, config):
        if self.active:
            self.stop()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.root / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for role in roles:
            f = open(self.run_dir / f"imu_{role}.csv", "w", newline="")
            w = csv.writer(f)
            w.writerow(["host_t_us", "sensor_t_us", "qw", "qx", "qy", "qz",
                        "ax", "ay", "az", "gx", "gy", "gz"])
            self._imu[role] = (f, w)
        self._ev_f = open(self.run_dir / "events.csv", "w", newline="")
        self._ev_w = csv.writer(self._ev_f)
        self._ev_w.writerow(["pulse_n", "host_t_us", "sensor_t_us",
                             "feature", "value", "threshold", "refractory_s"])
        (self.run_dir / "run.json").write_text(json.dumps({**config, "started": ts}, indent=2))
        self._q.clear()
        self.active = True
        try:
            self._task = asyncio.get_running_loop().create_task(self._drain())
        except RuntimeError:      # no event loop (tests, sync callers): drained by stop() instead
            self._task = None
        print(f"  [recorder] recording -> {self.run_dir}")
        return self.run_dir

    def write_sample(self, s):
        """Called from the BLE notify path for every sample — keep it cheap.
        Queue only: the csv.writerow and the file I/O happen in _drain, off this path."""
        if not self.active:
            return
        if s.role not in self._imu:
            return
        q, a, g = s.quat, s.acc, s.gyr # unpack the samples for readability
        self._q.append((s.role, [s.host_t_us, s.sensor_t_us,
                                 q[0], q[1], q[2], q[3], a[0], a[1], a[2], g[0], g[1], g[2]])) # following the Xsens datasheet

    def _drain_once(self):
        """Write every queued row. Rows leave the queue only once fully written, so a row is
        either absent or complete - never the truncated half-row a writerow interrupted at
        close would leave."""
        wrote = False
        while self._q:
            role, row = self._q.popleft()
            entry = self._imu.get(role)
            if entry is not None:
                entry[1].writerow(row)
                wrote = True
        if wrote:
            for f, _ in self._imu.values():
                f.flush()         # bounds what a crash can lose to one drain period

    async def _drain(self):
        while True:
            self._drain_once()
            await asyncio.sleep(DRAIN_PERIOD_S)

    def write_event(self, pulse_n, host_t_us, sensor_t_us, feature, value, threshold, refractory_s):
        if not self.active or self._ev_w is None:
            return
        self._ev_w.writerow([pulse_n, host_t_us, sensor_t_us,
                             feature, f"{value:.4f}", threshold, refractory_s])
        self._ev_f.flush()        # events are rare; flush so they survive a crash

    def stop(self):
        if not self.active:
            return
        self.active = False       # no new rows queue while we are shutting down
        if self._task is not None:
            self._task.cancel()   # stop() never awaits, so the task cannot interleave with it
            self._task = None
        self._drain_once()        # full drain BEFORE closing - no lost tail rows
        for f, _ in self._imu.values():
            f.flush(); f.close()
        self._imu = {}
        if self._ev_f:
            self._ev_f.flush(); self._ev_f.close()
            self._ev_f = self._ev_w = None
        print("  [recorder] stopped")