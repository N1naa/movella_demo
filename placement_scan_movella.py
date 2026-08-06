# %%
# file name: placement_scan_movella.py
"""
Code for a demo with dummy detector and TTL trigger.
Uses the same web interface as web_closed_loop.py
has preassigned imu placements
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import argparse
import asyncio
from dataclasses import dataclass, field
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import json
from bleak import BleakScanner, BleakClient
from typing import AsyncIterator, Optional, Callable

DOT_NAMES = (
    "Xsens DOT",
    "Movella DOT",
)  # depending on the set dots advertise like this
ROLES = ("torso", "upper_arm", "forearm")

@dataclass
class TriggerConfigSimple:
    """The TTL output, and nothing else.

    There is deliberately no feature/threshold/source here: the stimulation decision is the
    model's (threshold on p_smooth in config.json, spaced by its stim_ms/lockout_ms), and it
    reads all three sensors, so there is no single 'trigger source' to name. refractory_s is a
    hardware-side ceiling on the pulse rate, independent of the model.
    """

    refractory_s: float = 8.0
    pin: int = 16
    pulse_ms: int = 20


@dataclass
class Placement:
    """
    Placement just keeps track of who should be where. It:
    - holds the role->device map and the TTL output settings
    - assign each dot to a role based on id
    - load itself from JSON and fail really loudly if it doesn't work
    """

    role_match: dict[str, list[str]]
    trigger: TriggerConfigSimple = field(default_factory=TriggerConfigSimple)

    def __post_init__(self):  # this is called when a dataclass is built
        unknown = [r for r in self.role_match if r not in ROLES]
        if unknown:
            raise ValueError(
                f"unknown role(s) {unknown} in the placement map; expected {list(ROLES)}"
            )

    @property
    def required_roles(self) -> list[str]:  # returns a list of strings
        # Roles
        return [r for r in ROLES if r in self.role_match]

    def match_role(self, *, addr_suffix: str) -> str | None:  # returns a string or None
        # Says which role each dot belongs to. case-insensitive
        addr_suffix = addr_suffix.lower()
        for role, suffixes in self.role_match.items():
            if addr_suffix in [s.lower() for s in suffixes]:
                return role
        return None

    @classmethod
    def load(cls, path: str | Path) -> Placement:  # returns a Placement instance
        # Load from JSON, validate, and return a Placement instance
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            role_match=data["roles"],
            trigger=TriggerConfigSimple(
                refractory_s=data["trigger"]["refractory_s"],
                pin=data["trigger"].get("pin", 16), # default value
                pulse_ms=data["trigger"].get("pulse_ms", 20),
            ),
        )


@dataclass
class Sample:
    """One IMU sample at one timestep, from one sensor."""
    sensor_id: str
    sensor_t_us: int  # sensor's internal clock (microseconds)
    host_t_us: int  # host arrival time (us since epoch)
    quat: np.ndarray  # shape (4,) — [w, x, y, z]
    acc: np.ndarray  # shape (3,) — m/s², WITH gravity
    gyr: np.ndarray  # shape (3,) — deg/s (Movella DOT Custom Mode)
    role: Optional[str] = None  # "torso" | "upper_arm" | "forearm"


class IMUSource(ABC):
    """Anything that produces Sample objects asynchronously."""

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    def samples(self) -> AsyncIterator[Sample]:
        """Yield Sample objects as they arrive. Use with `async for`."""
        ...


def suffix_of(address: str) -> str:
    """6-hex device id from a BLE address or CoreBluetooth UUID."""
    return address.replace(":", "").replace("-", "")[-6:].lower()


# %%
async def scan_and_assign(placement: Placement, timeout_s: float = 8.0):
    """
    Scan through BLE and match each dot with a role:
    Scan all available dots. go through them while checking if their suffix matches any role.
    If it does, assign it to that role, if not, scream and ignore it.
    Returns: a list of (role, address, suffix) in order of placement.required_roles
    """
    seen_devices = await BleakScanner.discover(
        timeout=timeout_s, return_adv=True
    )  # returns dict of address: (device, adv)
    by_role = {}
    seen_suffixes = []
    for device, adv in seen_devices.values():
        name = device.name or adv.local_name or ""
        if not any(n in name for n in DOT_NAMES):
            continue
        suffix = (
            device.address.replace(":", "").replace("-", "")[-6:].lower()
        )  # remove all colons and dashes and put in lewercase and keep last 6 characters
        seen_suffixes.append(suffix)
        role = placement.match_role(addr_suffix=suffix)
        if role is None:
            print(
                f"Found DOT with suffix {suffix} but it doesn't match any role in the placement config."
            )
            continue
        elif role not in placement.required_roles:
            print(
                f"There's a typo somewhere in your config files. Please address it :)."
            )
            continue
        else:
            print(f"Found DOT with suffix {suffix} assigned to role {role}.")
            by_role[role] = (device.address, suffix)
    missing = [r for r in placement.required_roles if r not in by_role]
    if missing:
        raise RuntimeError(
            f"Missing required role(s) {missing}. Found devices: {seen_suffixes}"
        )
    return [(role, *by_role[role]) for role in placement.required_roles]


# %%
class BleakMovellaSource(IMUSource):
    """One DOT, streamed through bleak + jiminghe. Custom Mode 5 only."""

    def __init__(
        self,
        address: str,
        sensor_id: Optional[str] = None,
        rate_hz: int = 60,
        role: Optional[str] = None,
        on_sample: Optional[Callable[[Sample], None]] = None,
        max_queue: int = 8,
    ):
        self.address = address
        self.sensor_id = sensor_id or suffix_of(address)
        self.rate_hz = rate_hz
        self.role = role
        self.on_sample = on_sample
        self.n_samples = 0 # added this

        self._sensor = None
        self._queue: asyncio.Queue[Optional[Sample]] = asyncio.Queue(max_queue)
        self._running = False
        self.latest: Optional[Sample] = None

    async def start(self) -> None:
        # Lazy import: only needed when we actually touch hardware.
        from movella_dot_py.core.sensor import MovellaDOTSensor
        from movella_dot_py.models.data_structures import SensorConfiguration
        from movella_dot_py.models.enums import (
            OutputRate,
            FilterProfile,
            PayloadMode,
        )

        config = SensorConfiguration(
            output_rate=getattr(OutputRate, f"RATE_{self.rate_hz}"),
            filter_profile=FilterProfile.GENERAL,
            payload_mode=PayloadMode.CUSTOM_MODE_5,
        )
        self._sensor = MovellaDOTSensor(config)
        self._sensor.client = BleakClient(self.address)

        await self._sensor.client.connect()
        self._sensor.is_connected = True
        self._sensor._device_address = self.address
        self._sensor._device_name = f"DOT-{self.sensor_id}"

        def wrapped(sender, data: bytearray):
            # host arrival time stamped as early as possible
            host_t_us = time.time_ns() // 1000
            try:
                if self._sensor.data_collector:
                    self._sensor.data_collector.parser.parse(data)
                    self._sensor.data_collector.add_data(data)
                collected = self._sensor.data_collector.data
                if not collected:
                    return
                d = collected[-1]
                # The collector never empties itself (~10k entries per sensor per run), and the
                # list it appends to is walked on every notification. Drop it now that we have
                # the newest sample; `d` is a reference, so it survives the clear.
                self._sensor.data_collector.data.clear()
                if (
                    d.quaternion is None
                    or d.acceleration is None
                    or d.angular_velocity is None
                ):
                    return
                sample = Sample(
                    sensor_id=self.sensor_id,
                    sensor_t_us=int(d.timestamp.microseconds), # check in reality if this is correct
                    host_t_us=host_t_us,
                    quat=d.quaternion.to_numpy(),
                    acc=d.acceleration.to_numpy(),
                    gyr=d.angular_velocity.to_numpy(),
                    role=self.role,
                )
                # print(type(d.timestamp), repr(d.timestamp)) # CHECK THIS - remove after verification

                self.latest = sample
                self.n_samples += 1 # added this
                # Lowest-latency path: decide + log right here.
                if self.on_sample is not None:
                    self.on_sample(sample)
                # Also offer it via the queue (drop-oldest if consumer lags).
                self._push_latest(sample)
            except Exception as e:
                print(f"  notify error: {type(e).__name__}: {e}")

        self._sensor.notification_handler = wrapped

        await self._sensor.configure_sensor()
        await self._sensor.start_measurement()
        self._running = True

    def _push_latest(self, sample: Sample) -> None:
        try:
            self._queue.put_nowait(sample)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()  # drop oldest
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(sample)
            except asyncio.QueueFull:
                pass

    async def stop(self) -> None:
        if not self._sensor:
            return
        self._running = False
        try:
            await self._sensor.stop_measurement()
        except Exception:
            pass
        try:
            await self._sensor.disconnect()
        except Exception:
            pass
        await self._queue.put(None)

    async def samples(self) -> AsyncIterator[Sample]:
        while True:
            s = await self._queue.get()
            if s is None:
                return
            yield s
