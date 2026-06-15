# recorder.py — per-IMU CSVs + trigger event log + a run.json config snapshot.
# Inactive until start(); write_sample / write_event are cheap no-ops when idle,
# so it's safe to wire as on_sample for every source and gate it on the session.
import csv, json
from pathlib import Path
from datetime import datetime


class Recorder:
    def __init__(self, root="recordings"):
        self.root = Path(root)
        self.active = False
        self.run_dir = None
        self._imu = {}            # role -> (file, writer)
        self._ev_f = None
        self._ev_w = None

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
        self.active = True
        print(f"  [recorder] recording -> {self.run_dir}")
        return self.run_dir

    def write_sample(self, s):
        """Called from the BLE notify path for every sample — keep it cheap."""
        if not self.active:
            return
        entry = self._imu.get(s.role)
        if entry is None:
            return
        q, a, g = s.quat, s.acc, s.gyr # unpack the samples for readability
        entry[1].writerow([s.host_t_us, s.sensor_t_us,
                           q[0], q[1], q[2], q[3], a[0], a[1], a[2], g[0], g[1], g[2]]) # following the Xsens datasheet

    def write_event(self, pulse_n, host_t_us, sensor_t_us, feature, value, threshold, refractory_s):
        if not self.active or self._ev_w is None:
            return
        self._ev_w.writerow([pulse_n, host_t_us, sensor_t_us,
                             feature, f"{value:.4f}", threshold, refractory_s])
        self._ev_f.flush()        # events are rare; flush so they survive a crash

    def stop(self):
        if not self.active:
            return
        for f, _ in self._imu.values():
            f.flush(); f.close()
        self._imu = {}
        if self._ev_f:
            self._ev_f.flush(); self._ev_f.close()
            self._ev_f = self._ev_w = None
        self.active = False
        print("  [recorder] stopped")