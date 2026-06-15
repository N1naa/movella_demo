from collections import deque
from dataclasses import dataclass, field

import numpy as np
from placement_scan_movella import Sample

def compute_features(sample, feature_name):
    if feature_name == "gyro_magnitude":
        return float(np.linalg.norm(sample.gyr)) # deg/s
    elif feature_name == "acc_magnitude":
        return float(np.linalg.norm(sample.acc)) # m/s²
    else:
        raise ValueError(f"Unknown feature: {feature_name}")

# @dataclass
# class BufferPerIMU:
#     def __init__(self, maxlen: int):
#         self.maxlen = maxlen
#         self.samples = deque() # datastrucure that can be appended to and popped from left and right

#     """Keeps a rolling buffer of the last N samples for one IMU."""
#     maxlen: int
#     samples: deque[Sample] = field(default_factory=deque)

#     def add_sample(self, sample: Sample):
#         self.samples.append(sample)
#         while len(self.samples) > self.maxlen:
#             self.samples.popleft()
