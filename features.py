import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d
from scipy.signal import sosfilt


# ------------------------- ONLINE feature extractor -------------------------
# Two stages, matching the frozen bundle (vns/mvmt_det/frozen_jsons_deploy/):
#
#   Preprocessor.push(quat, acc, gyr)  raw BLE sample -> DERIVED_CHANNELS  (stateful: filters + rates)
#   extract(window)                    (17, 3, 14)    -> SET85             (pure)
#
# 17 = the 12 scored samples + ANGLE_LEAD run-up for the elevation/flexion path; the ring buffer
# holds 17, the hop is still 6. Every one of the 85 is then bit-identical to the offline value.
# ----------------------------------------------------------------------------

# Univariate feature TYPES, applied per IMU (and the stat-subset to elevation/flexion speed).
# Per-axis names are "{acc|gyr}_{x|y|z}_rms" = RMS of ONE signed axis component (acc axes are gravity-free).
UNIVARIATE_FEATURES = [  # for each imu, and (stat subset) for elevation speed and flexion speed
    "acc_x_rms", "acc_y_rms", "acc_z_rms", "gyr_x_rms", "gyr_y_rms", "gyr_z_rms", "acc_mag_rms", "gyr_mag_rms",
    "acc_mag_std", "gyr_mag_std", "acc_mag_mean", "gyr_mag_mean", "acc_mag_max", "gyr_mag_max", "acc_mag_min", "gyr_mag_min",
    "ori_roll_speed", "ori_pitch_speed", "ori_yaw_speed", "jerk_rms", "ldlj", "acc_zcr",
    "vqf_speed",   # window-mean of the per-sample long-axis angular speed (in space), one per IMU
]

SENSORS = ("forearm", "upperarm", "chest")
# Generic stats applied to the two derived speed signals (reproduces elev_speed_mean/max etc.).
SPEED_STATS = ("mean", "max", "min", "std", "rms")

SET85 = [
    # per-axis RMS (18) - RMS of one signed axis component (frame/mounting-orientation dependent)
    "forearm_acc_x_rms", "forearm_acc_y_rms", "forearm_acc_z_rms",
    "forearm_gyr_x_rms", "forearm_gyr_y_rms", "forearm_gyr_z_rms",
    "upperarm_acc_x_rms", "upperarm_acc_y_rms", "upperarm_acc_z_rms",
    "upperarm_gyr_x_rms", "upperarm_gyr_y_rms", "upperarm_gyr_z_rms",
    "chest_acc_x_rms", "chest_acc_y_rms", "chest_acc_z_rms",
    "chest_gyr_x_rms", "chest_gyr_y_rms", "chest_gyr_z_rms",
    # magnitude stats (30) - |acc| & |gyr| x {rms,std,mean,max,min} per sensor (frame-invariant)
    "forearm_acc_mag_rms", "forearm_gyr_mag_rms", "forearm_acc_mag_std", "forearm_gyr_mag_std",
    "forearm_acc_mag_mean", "forearm_gyr_mag_mean", "forearm_acc_mag_max", "forearm_gyr_mag_max",
    "forearm_acc_mag_min", "forearm_gyr_mag_min",
    "upperarm_acc_mag_rms", "upperarm_gyr_mag_rms", "upperarm_acc_mag_std", "upperarm_gyr_mag_std",
    "upperarm_acc_mag_mean", "upperarm_gyr_mag_mean", "upperarm_acc_mag_max", "upperarm_gyr_mag_max",
    "upperarm_acc_mag_min", "upperarm_gyr_mag_min",
    "chest_acc_mag_rms", "chest_gyr_mag_rms", "chest_acc_mag_std", "chest_gyr_mag_std",
    "chest_acc_mag_mean", "chest_gyr_mag_mean", "chest_acc_mag_max", "chest_gyr_mag_max",
    "chest_acc_mag_min", "chest_gyr_mag_min",
    # orientation speed (9) - window-mean |roll/pitch/yaw angular rate| (deg/s) per sensor
    "forearm_ori_roll_speed", "forearm_ori_pitch_speed", "forearm_ori_yaw_speed",
    "upperarm_ori_roll_speed", "upperarm_ori_pitch_speed", "upperarm_ori_yaw_speed",
    "chest_ori_roll_speed", "chest_ori_pitch_speed", "chest_ori_yaw_speed",
    # smoothness (6) - jerk_rms / LDLJ per sensor (SPARC is not in SET85)
    "forearm_jerk_rms", "forearm_ldlj", 
    "upperarm_jerk_rms", "upperarm_ldlj", 
    "chest_jerk_rms", "chest_ldlj", 
    # acc zero-crossing rate (3) - per sensor
    "forearm_acc_zcr", "upperarm_acc_zcr", "chest_acc_zcr",
    # joint speed (10) - elevation & flexion angular speed x {mean,max,min,std,rms}
    "elev_speed_mean", "elev_speed_max", "elev_speed_min", "elev_speed_std", "elev_speed_rms",
    "flex_speed_mean", "flex_speed_max", "flex_speed_min", "flex_speed_std", "flex_speed_rms",
    # coordination (1) - legacy zero-lag fa_ua_xcorr
    "fa_ua_xcorr",
    # special (5) - posture, pronation, SMA, ROM
    "torso_incl_mean", "forearm_pron_speed_mean", "forearm_sma", "elev_rom", "flex_rom",
    # per-sensor long-axis angular speed in space (3) - the movement-speed signal, one per IMU.
    # Named *_vqf_speed for column compatibility with the offline pipeline; this bundle computes it
    # as gyr_perp (no VQF, no magnetometer) - see config.json["movement_signal"].
    "forearm_vqf_speed", "upperarm_vqf_speed", "chest_vqf_speed",
]
assert len(SET85) == 85, f"SET85 has {len(SET85)} features, expected 85"


# ----------------------------------------------------------------------------- geometry / angle helpers
def _angle_between(v1, v2):
    """Calculate the angle in degrees between two unit vectors (row-wise)."""
    dots = np.einsum("ij,ij->i", v1, v2)    # row-wise dot product
    dots = np.clip(dots, -1.0, 1.0)       # avoid numerical issues
    return np.degrees(np.arccos(dots))

# ----------------------------------------------------------------------------- window-summary primitives
def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))

def _stat(x, st):
    """One generic 1-D statistic (used for the elevation/flexion speed signals)."""
    if st == "mean":  return float(np.mean(x))
    if st == "max":   return float(np.max(x))
    if st == "min":   return float(np.min(x))
    if st == "std":   return float(np.std(x))
    if st == "rms":   return _rms(x)
    raise ValueError(st)

def _ldlj_acc(acc_vec_win, jerk_mag_win, fs, eps=1e-12):
    """Log dimensionless jerk (acceleration-based) over one window; higher = smoother.
    Dimensionless & scale-invariant: DJ = T * integral(jerk^2 dt) / a_peak^2, LDLJ = -ln(DJ).
    Gravity is removed for a_peak by subtracting the window-mean acc vector; jerk already
    differentiates the static gravity offset away.
    """
    a_dyn = acc_vec_win - acc_vec_win.mean(axis=0)              # remove ~static gravity offset
    a_peak = np.sqrt((a_dyn ** 2).sum(axis=1)).max()
    T = len(jerk_mag_win) / fs                                  # window duration (s)
    jerk_integral = np.sum(jerk_mag_win ** 2) / fs             # integral(jerk^2 dt), dt = 1/fs
    dj = T * jerk_integral / (a_peak ** 2 + eps)
    return float(-np.log(dj + eps))


def _zcr(x):
    """Zero-crossing rate of a window about its own mean."""
    return float(np.mean(np.diff(np.sign(x - x.mean())) != 0)) if len(x) > 1 else 0.0

def _zero_lag_corr(a, b):
    """Zero-lag Pearson correlation — the legacy fa_ua_xcorr definition."""
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0

# type -> f(windowed sensor-signal dict, fs). Keys must match UNIVARIATE_FEATURES exactly.
_UNIVARIATE_FN = {
    "acc_x_rms":         lambda w, fs: _rms(w["acc_x"]),   # gravity-free axis RMS
    "acc_y_rms":         lambda w, fs: _rms(w["acc_y"]),
    "acc_z_rms":         lambda w, fs: _rms(w["acc_z"]),
    "gyr_x_rms":         lambda w, fs: _rms(w["gyr_x"]),
    "gyr_y_rms":         lambda w, fs: _rms(w["gyr_y"]),
    "gyr_z_rms":         lambda w, fs: _rms(w["gyr_z"]),
    "acc_mag_rms":       lambda w, fs: _rms(w["acc_mag"]),
    "gyr_mag_rms":       lambda w, fs: _rms(w["gyr_mag"]),
    "acc_mag_std":       lambda w, fs: float(w["acc_mag"].std()),
    "gyr_mag_std":       lambda w, fs: float(w["gyr_mag"].std()),
    "acc_mag_mean":      lambda w, fs: float(w["acc_mag"].mean()),
    "gyr_mag_mean":      lambda w, fs: float(w["gyr_mag"].mean()),
    "acc_mag_max":       lambda w, fs: float(w["acc_mag"].max()),
    "gyr_mag_max":       lambda w, fs: float(w["gyr_mag"].max()),
    "acc_mag_min":       lambda w, fs: float(w["acc_mag"].min()),
    "gyr_mag_min":       lambda w, fs: float(w["gyr_mag"].min()),
    "ori_roll_speed":    lambda w, fs: float(w["ori_roll_speed"].mean()),   # window-mean |roll rate| (deg/s)
    "ori_pitch_speed":   lambda w, fs: float(w["ori_pitch_speed"].mean()),  # window-mean |pitch rate| (deg/s)
    "ori_yaw_speed":     lambda w, fs: float(w["ori_yaw_speed"].mean()),    # window-mean |yaw rate| (deg/s)
    "jerk_rms":          lambda w, fs: _rms(w["jerk_mag"]),
    "ldlj":              lambda w, fs: _ldlj_acc(w["acc_vec"], w["jerk_mag"], fs),
    "acc_zcr":           lambda w, fs: _zcr(w["acc_mag"]),
    "vqf_speed":         lambda w, fs: float(np.nanmean(w["vqf_speed"])),   # window-mean of long-axis speed
}


def _window_features(sensor_sig, elev_speed, flex_speed, elevation_s, flexion_s, torso_incl, start, end, fs):
    """Feature dict for the causal window [start:end), keyed by SET85 names.
    `sensor_sig` maps 'forearm'/'upperarm'/'chest' -> that sensor's signal dict (online: already
    exactly one window long, so start/end are 0/n)."""
    s, e = start, end
    win = {sensor: {k: v[s:e] for k, v in sig.items()} for sensor, sig in sensor_sig.items()}
    row = {}
    for sensor in SENSORS:                                      # univariate, per IMU
        w = win[sensor]
        for t in UNIVARIATE_FEATURES:
            row[f"{sensor}_{t}"] = _UNIVARIATE_FN[t](w, fs)
    for sp, speed in (("elev_speed", elev_speed), ("flex_speed", flex_speed)):   # stats on speed signals
        ws = speed[s:e]
        for st in SPEED_STATS:
            row[f"{sp}_{st}"] = _stat(ws, st)
    fw = win["forearm"]                                         # special (posture / ROM / coordination)
    row["torso_incl_mean"] = float(torso_incl[s:e].mean())
    row["forearm_pron_speed_mean"] = float(np.abs(fw["gyr_x"]).mean())   # |gyr| about the forearm long axis
    row["forearm_sma"] = float(np.abs(fw["acc_vec"]).sum(axis=1).mean())
    row["elev_rom"] = float(np.ptp(elevation_s[s:e]))
    row["flex_rom"] = float(np.ptp(flexion_s[s:e]))
    row["fa_ua_xcorr"] = _zero_lag_corr(fw["acc_mag"], win["upperarm"]["acc_mag"])   # legacy zero-lag def
    return row


# ------------- Online Implementation -------------
# Sample-rate stage of the online path: one Preprocessor per sensor turns the raw BLE sample
# (quat, acc, gyr) into the per-sample channels the SET85 extractor slices. Everything here is
# strictly causal and stateful, so it reproduces the offline signals of _sensor_signals() sample
# for sample - including the ones offline gets "for free" from full-length arrays (the filters,
# jerk and the Euler rates all need the PREVIOUS sample, which the offline window slice reads
# from outside the window; here that history lives in this object instead of in the buffer).

GRAVITY_WORLD = np.array([0.0, 0.0, 9.81])   # world gravity, Z-up (the repo's real-IMU convention)
LONG_AXIS = np.array([1.0, 0.0, 0.0])        # sensor x = the limb long axis in all three placements

# What push() returns, in this order. This is the channel axis of the window extract() consumes,
# so it is binding between the two. acc_mag / gyr_mag are NOT stored - they are norms of channels
# already here and extract() recomputes them.
DERIVED_CHANNELS = [
    "acc_free_x", "acc_free_y", "acc_free_z",              # m/s^2, gravity removed then 20 Hz low-passed
    "gyr_x", "gyr_y", "gyr_z",                             # deg/s, 0.3-8 Hz band-passed
    "jerk_mag",                                            # |d(acc_free)/dt|, m/s^3
    "ori_roll_speed", "ori_pitch_speed", "ori_yaw_speed",  # |d(euler xyz)/dt|, deg/s, from the RAW quat
    "vqf_speed",                                           # long-axis angular speed, rad/s, from the RAW gyro
    "e_x", "e_y", "e_z",                                   # the long axis in world coords (unit vector)
]


class Preprocessor:
    """Preprocessing per sensor: holds sosfilt zi for acc_free (20 Hz LP) and gyr (0.3-8 Hz BP)."""
    def __init__(self, filters, fs=60.0):
        self.fs = float(fs)
        self.sos_acc = np.asarray(filters["acc_gravity_free"]["sos"], float)
        self.zi_unit_acc = np.asarray(filters["acc_gravity_free"]["zi_unit"], float)
        self.sos_gyr = np.asarray(filters["gyr"]["sos"], float)
        self.zi_unit_gyr = np.asarray(filters["gyr"]["zi_unit"], float)
        self.reset()

    @classmethod
    def from_config(cls, config):
        return cls(config["filters"], fs=config["rate_hz"])

    def reset(self):
        """Drop all state; call on any gap > max_gap_ms (and at session start).
        The next push() re-seeds both filters from that sample, exactly as the offline
        zi = sosfilt_zi(sos) * x[0] does at the start of a recording."""
        self.zi_acc = None
        self.zi_gyr = None
        self.prev_acc_free = None
        self.prev_eul = None

    def push(self, quat, acc, gyr) -> np.ndarray:
        """One raw sample -> the (14,)
        quat: (4,) unit quaternion [w, x, y, z]
        acc:  (3,) m/s^2, WITH gravity
        gyr:  (3,) deg/s"""
        quat = np.asarray(quat, float)
        acc = np.asarray(acc, float)
        gyr = np.asarray(gyr, float)
        rot = R.from_quat(quat[[1, 2, 3, 0]])              # scipy [x,y,z,w]

        # acc_free = acc - R(quat)^-1 @ [0,0,9.81]      BEFORE the low-pass
        acc_free = acc - rot.apply(GRAVITY_WORLD, inverse=True)

        if self.zi_acc is None:                            # first sample after a reset: steady state
            self.zi_acc = self.zi_unit_acc[:, None, :] * acc_free[None, :, None]
            self.zi_gyr = self.zi_unit_gyr[:, None, :] * gyr[None, :, None]
        # sosfilt over a length-1 signal per channel: x is (3, 1), zi is (n_sections, 3, 2)
        acc_f, self.zi_acc = sosfilt(self.sos_acc, acc_free[:, None], zi=self.zi_acc)
        gyr_f, self.zi_gyr = sosfilt(self.sos_gyr, gyr[:, None], zi=self.zi_gyr)
        acc_f, gyr_f = acc_f[:, 0], gyr_f[:, 0]

        # jerk: causal first difference of the FILTERED acc_free, 0 on the first sample of a
        jerk_mag = 0.0 if self.prev_acc_free is None else \
            float(np.linalg.norm(acc_f - self.prev_acc_free) * self.fs)
        self.prev_acc_free = acc_f

        # orientation rates from the RAW quat, wrapped to (-180, 180] 
        eul = rot.as_euler("xyz", degrees=True)
        if self.prev_eul is None:
            ori_speed = np.zeros(3)
        else:
            d = (eul - self.prev_eul + 180.0) % 360.0 - 180.0
            ori_speed = np.abs(d) * self.fs
        self.prev_eul = eul

        # movement signal, config["movement_signal"] == "gyr_perp": |omega| perpendicular to the
        vqf_speed = float(np.deg2rad(np.hypot(gyr[1], gyr[2])))

        e = rot.apply(LONG_AXIS)                           # long axis in world
        return np.concatenate([acc_f, gyr_f, [jerk_mag], ori_speed, [vqf_speed], e])


_CH = {name: i for i, name in enumerate(DERIVED_CHANNELS)}

# The elevation/flexion path is the only non-causal step offline: uniform_filter1d(size=5) is
# CENTRED, so it peeks 2 samples (33 ms) ahead. We take the lag config.json's notes prescribe -
# those channels are read LAG samples behind the scored window - and pay for it with extra HISTORY
# rather than latency: the scored window is still the newest 12 samples, the angle channels just
# describe the window 33 ms earlier. ANGLE_LEAD is how many samples of run-up that costs.
LAG = 2                    # the kernel's forward peek, taken as a lag on the angle channels
ANGLE_LEAD = LAG + 1 + 2   # + the leading difference (1) + half the size-5 kernel (2) = 5


def extract(window: np.ndarray, fs=60.0) -> np.ndarray:
    """(12 + ANGLE_LEAD, 3, n_derived) -> (85,)  — pure function, no state, no filtering.

    window[t, s, c]: t = samples, oldest first (the window ENDS at the current sample);
    s indexes SENSORS = ("forearm", "upperarm", "chest"); c indexes DERIVED_CHANNELS, i.e. what
    Preprocessor.push() returns - NOT the 10 raw channels of config['channels_per_sensor'],
    which are push()'s INPUT. Returns SET85 == config['feature_names'] order.

    The LAST 12 samples are the scored window; the ANGLE_LEAD=5 samples before it are history the
    elevation/flexion path needs (see below) and nothing else reads. So the caller's ring buffer
    holds 17 samples, not 12, and the hop is unchanged.

    Every formula is the offline one: this only rebuilds _sensor_signals' dict from the derived
    channels and hands it to _window_features.
    """
    window = np.asarray(window, float)
    n = len(window) - ANGLE_LEAD                        # the scored window: the LAST n samples
    if len(window) != 12 + ANGLE_LEAD:
        raise ValueError(f"expected {12 + ANGLE_LEAD} samples, got {len(window)}")
    sensor_sig, e_world = {}, {}
    for i, sensor in enumerate(SENSORS):
        w = window[ANGLE_LEAD:, i, :]                   # per-sensor features: scored window only
        acc = w[:, _CH["acc_free_x"]:_CH["acc_free_z"] + 1]     # already gravity-free + low-passed
        gyr = w[:, _CH["gyr_x"]:_CH["gyr_z"] + 1]               # already band-passed
        sensor_sig[sensor] = dict(
            acc_vec=acc,
            acc_x=acc[:, 0], acc_y=acc[:, 1], acc_z=acc[:, 2],
            gyr_x=gyr[:, 0], gyr_y=gyr[:, 1], gyr_z=gyr[:, 2],
            acc_mag=np.sqrt((acc ** 2).sum(axis=1)), gyr_mag=np.sqrt((gyr ** 2).sum(axis=1)),
            ori_roll_speed=w[:, _CH["ori_roll_speed"]],
            ori_pitch_speed=w[:, _CH["ori_pitch_speed"]],
            ori_yaw_speed=w[:, _CH["ori_yaw_speed"]],
            jerk_mag=w[:, _CH["jerk_mag"]],
            vqf_speed=w[:, _CH["vqf_speed"]],
        )
        # the angle path reads the FULL padded window, the scored one only for torso inclination
        e_world[sensor] = window[:, i, _CH["e_x"]:_CH["e_z"] + 1]   # long axis in world coords

    torso_incl = _angle_between(e_world["chest"][ANGLE_LEAD:],      # not smoothed -> no lag needed
                                np.broadcast_to([0.0, 0.0, 1.0], (n, 3)))

    # Elevation / flexion: same formulas as _arm_elevation_elbow_flexion, same (1,0,0) limb axis,
    # but run over the whole padded window and then read LAG samples behind the scored one. That is
    # the lag config.json's notes prescribe, and it buys exactness: offline's centred size-5 kernel
    # peeks 2 samples ahead and its leading difference reads 1 sample back, neither of which exists
    # at the live edge. Sliced this way every value is bit-identical to the offline one - it just
    # describes the window LAG samples (33 ms) earlier than the other 73 features do.
    elevation = _angle_between(e_world["chest"], e_world["upperarm"])
    flexion = _angle_between(e_world["upperarm"], e_world["forearm"])
    elevation_s = uniform_filter1d(elevation, size=5)   # exact for [2 : len-2), which covers a0-1:a1
    flexion_s = uniform_filter1d(flexion, size=5)
    elev_speed = np.abs(np.diff(elevation_s, prepend=elevation_s[0])) * fs
    flex_speed = np.abs(np.diff(flexion_s, prepend=flexion_s[0])) * fs
    a1 = len(window) - LAG                              # the angle window: n samples, ending at -LAG
    a0 = a1 - n

    row = _window_features(sensor_sig, elev_speed[a0:a1], flex_speed[a0:a1],
                           elevation_s[a0:a1], flexion_s[a0:a1], torso_incl, 0, n, fs)
    return np.array([row[name] for name in SET85], float)


    