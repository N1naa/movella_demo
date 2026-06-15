# new_implementation/demo_closed_loop.py
import asyncio
from pathlib import Path
import placement_scan_movella as psm
import pi_ttl as ttl 
import inference as infer

CONFIG = Path(__file__).parent / "placement.json" 


async def main():
    placement = psm.Placement.load(CONFIG)
    assigned = await psm.scan_and_assign(placement) # [(role, address, suffix), ...]

    sources = {role: psm.BleakMovellaSource(address, sensor_id=suffix, role=role)
               for role, address, suffix in assigned}

    threshold = placement.trigger.threshold 
    trig = ttl.TTLTrigger(pin=16, pulse_ms=20, refractory_s=placement.trigger.refractory_s)
    feature = placement.trigger.feature      # from JSON
    source = placement.trigger.source

    try:
        for src in sources.values():
            await src.start()
        print(f"Live. Fire when feature: {feature} > {threshold:.0f}. Press ctrl+C to stop.")

        async for s in sources["forearm"].samples():
            val = infer.compute_features(s, feature)
            if val > threshold and trig.fire_if_ready(s.host_t_us):
                print(f"  FIRE  {feature}={val:5.0f}  pulses={trig.pulse_count}")
    finally:
        print("stopping...")
        for src in sources.values():
            await src.stop()
        trig.reset()
        trig.close()


if __name__ == "__main__":
    asyncio.run(main())