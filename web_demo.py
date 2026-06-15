# web_demo.py — serves index.html + /ws, wrapping the simplified closed-loop.
# Frame schema is documented at the top of index.html. Run: python web_demo.py
import asyncio, json
from pathlib import Path
import numpy as np
from aiohttp import web

import placement_scan_movella as psm
import pi_ttl as ttl
import inference as infer
import recorder as rec

HERE = Path(__file__).parent
CONFIG = HERE / "placement.json"
UNITS = {"acc_magnitude": "m/s\u00b2", "gyro_magnitude": "\u00b0/s"}


def sensor_dict(src):
    s = src.latest
    if s is None:
        return {"online": False, "n": src.n_samples, "gmag": 0.0, "amag": 0.0,
                "gyr": [0, 0, 0], "acc": [0, 0, 0], "quat": [1, 0, 0, 0]}
    return {"online": True, "n": src.n_samples,
            "gmag": float(np.linalg.norm(s.gyr)), "amag": float(np.linalg.norm(s.acc)),
            "gyr": s.gyr.tolist(), "acc": s.acc.tolist(), "quat": s.quat.tolist()}


async def broadcast(app, frame):
    if not app["clients"]:
        return
    data = json.dumps(frame)
    for ws in list(app["clients"]):
        try:
            await ws.send_str(data)
        except Exception:
            app["clients"].discard(ws)


async def trigger_loop(app):
    pl, trig, sources = app["placement"], app["trig"], app["sources"]
    feature, threshold = pl.trigger.feature, pl.trigger.threshold
    scale = 2.5 * threshold                       # full-scale for the ring/bars; tune to taste
    units = UNITS.get(feature, "")
    async for s in sources["forearm"].samples():
        val = infer.compute_features(s, feature)
        fired = bool(app["session"] and val > threshold and trig.fire_if_ready(s.host_t_us))
        if fired:
            app["recorder"].write_event(trig.pulse_count, s.host_t_us, s.sensor_t_us,
                                        feature, val, threshold, trig.refractory_s)
        remaining = max(0.0, trig.refractory_s - (s.host_t_us / 1e6 - trig.last_fire_t))
        await broadcast(app, {
            "connected": True, "gpio_mock": trig.is_mock, "session": app["session"],
            "trigger_role": pl.trigger.source, "feature": feature, "units": units,
            "threshold": threshold, "scale": scale,
            "value": val, "fired": fired, "fires": trig.pulse_count,
            "refractory_s": trig.refractory_s, "lockout_remaining": remaining,
            "sensors": {role: sensor_dict(src) for role, src in sources.items()},
        })


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app["clients"].add(ws)
    try:
        async for _ in ws:                        # no inbound messages expected
            pass
    finally:
        request.app["clients"].discard(ws)
    return ws


async def index(request):
    return web.FileResponse(HERE / "index.html")


async def session_handler(request):
    app = request.app
    body = await request.json()
    on = bool(body.get("on"))
    app["session"] = on
    if on:
        app["trig"].reset()                # fresh session: armed now, fire count cleared
        pl = app["placement"]
        rate = next(iter(app["sources"].values())).rate_hz
        app["recorder"].start(app["sources"].keys(), {
            "feature": pl.trigger.feature, "threshold": pl.trigger.threshold,
            "refractory_s": pl.trigger.refractory_s, "source_role": pl.trigger.source,
            "scale": 2.5 * pl.trigger.threshold, "rate_hz": rate,
            "gpio_mock": app["trig"].is_mock,
            "roles": {role: suf for role, _, suf in app["assigned"]},
        })
    else:
        app["recorder"].stop()
    return web.json_response({"ok": True, "session": on})


async def on_startup(app):
    placement = psm.Placement.load(CONFIG)
    assigned = await psm.scan_and_assign(placement)   # raises loudly if a DOT is missing
    app["recorder"] = rec.Recorder(HERE / "recordings")
    sources = {role: psm.BleakMovellaSource(addr, sensor_id=suf, role=role,
                                            on_sample=app["recorder"].write_sample)
               for role, addr, suf in assigned}
    for src in sources.values():
        await src.start()
    app["placement"] = placement
    app["assigned"] = assigned
    app["sources"] = sources
    app["trig"] = ttl.TTLTrigger(pin=16, pulse_ms=20, refractory_s=placement.trigger.refractory_s)
    app["clients"] = set()
    app["task"] = asyncio.create_task(trigger_loop(app))
    print("Live on http://0.0.0.0:8080  (open it in a browser)")


async def on_cleanup(app):
    app["task"].cancel()
    app["recorder"].stop()
    for src in app["sources"].values():
        await src.stop()
    app["trig"].reset()
    app["trig"].close()


def main():
    app = web.Application()
    app["clients"] = set()
    app["session"] = False
    app.add_routes([web.get("/", index), web.get("/ws", ws_handler),
                    web.post("/session", session_handler)])
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    web.run_app(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()