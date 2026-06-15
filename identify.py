"""List every DOT in range with its id + on-device tag, optionally blinking
each one so you can write the right id into placement.json.

  python identify.py            # list id / tag / firmware
  python identify.py --blink    # also LED-blink each DOT in turn

On Linux/Pi the id is the real MAC suffix (e.g. D4:22:CD:00:54:89 -> 005489).
On macOS bleak gives a per-host UUID, so prefer matching by TAG there: set a
tag once in the Movella app, then put that tag string in placement.json.
"""
import argparse
import asyncio

from bleak import BleakScanner, BleakClient
from placement_scan_movella import suffix_of


DOT_NAMES = ("Xsens DOT", "Movella DOT")


async def main(blink: bool):
    print("Scanning (8 s)...")
    devices = await BleakScanner.discover(timeout=8.0, return_adv=True)
    dots = []
    for _addr, (device, adv) in devices.items():
        name = device.name or adv.local_name or ""
        if any(n in name for n in DOT_NAMES):
            dots.append((device, adv.rssi))
    dots.sort(key=lambda x: -x[1])
    if not dots:
        print("No DOTs found.")
        return

    print(f"\nFound {len(dots)} DOT(s):\n")
    from movella_dot_py.core.sensor import MovellaDOTSensor
    from movella_dot_py.models.data_structures import SensorConfiguration
    from movella_dot_py.models.enums import OutputRate, FilterProfile, PayloadMode
    cfg = SensorConfiguration(OutputRate.RATE_60, FilterProfile.GENERAL,
                              PayloadMode.CUSTOM_MODE_5)

    for device, rssi in dots:
        suffix = suffix_of(device.address)
        tag, fw = "?", "?"
        try:
            s = MovellaDOTSensor(cfg)
            s.client = BleakClient(device.address)
            await s.client.connect()
            s.is_connected = True
            s._device_address = device.address
            info = await s.get_device_info()
            tag, fw = info.device_tag, info.firmware_version
            if blink:
                print(f"  blinking {suffix} ...")
                await s.identify_sensor()
                await asyncio.sleep(2.5)
            await s.client.disconnect()
        except Exception as e:
            print(f"  (could not read {suffix}: {type(e).__name__}: {e})")
        print(f"  id={suffix}  tag={tag!r}  fw={fw}  rssi={rssi}  "
              f"addr={device.address}")

    print("\nPut the id (or tag) for each sensor into placement.json roles.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--blink", action="store_true")
    args = p.parse_args()
    asyncio.run(main(args.blink))
