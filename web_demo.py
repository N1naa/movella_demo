# web_demo.py — serves index.html + /ws, wrapping the simplified closed-loop.
# Frame schema is documented at the top of index.html. Run: python web_demo.py
import asyncio, csv, json, os, time
from pathlib import Path
import numpy as np
from aiohttp import web

import placement_scan_movella as psm
import pi_ttl as ttl
import inference as infer
import recorder as rec
import block1_pipeline as b1p

HERE = Path(__file__).parent
JSONS = HERE / "jsons"
CONFIG = JSONS / "placement.json"
# The frozen bundle: jsons/{booster,config}.json, a copy of vns/mvmt_det/frozen_jsons_deploy/
# (verified byte-identical). MVMT_MODEL_DIR overrides it.
MODEL_DIR = Path(os.environ.get("MVMT_MODEL_DIR", JSONS))
ROLES = ("forearm", "upper_arm", "torso")      # ROLES[0] is the Aligner's offset reference
UNITS = {"acc_magnitude": "m/s\u00b2", "gyro_magnitude": "\u00b0/s"}


def sensor_dict(src, hz=0.0):
    s = src.latest
    if s is None:
        return {"online": False, "n": src.n_samples, "hz": round(hz, 1), "gmag": 0.0, "amag": 0.0,
                "gyr": [0, 0, 0], "acc": [0, 0, 0], "quat": [1, 0, 0, 0]}
    return {"online": True, "n": src.n_samples, "hz": round(hz, 1),
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


async def block1_task(app):
    """Score one window every hop_ms on a FIXED TIMER, never off the BLE callback.

    The gap reset, the hop gate and the derived-channel bookkeeping all live in
    Block1Pipeline.tick(); this is just the clock and the TTL. Does NOT drive the UI.
    """
    pipe, trig = app["pipeline"], app["trig"]
    period = pipe.cfg["hop_ms"] / 1000.0
    while True:
        await asyncio.sleep(period)
        # time.time_ns(), NOT monotonic: it has to be the same clock domain as Sample.host_t_us,
        # and Aligner.window() raises if the two are mixed.
        now_us = time.time_ns() // 1000
        out = pipe.tick(now_us)
        if out is None:
            continue
        app["last"] = out
        write_block1_row(app, now_us, out)
        if out["fired"] and app["session"]:
            await trig.fire_if_ready_async(now_us)
            app["recorder"].write_event(trig.pulse_count, now_us, out["k"], "p_smooth",
                                        out["p_smooth"], pipe.cfg["threshold"],
                                        trig.refractory_s)


def write_block1_row(app, now_us, out):
    """One CSV row per scored window, in the recorder's run dir, for offline review."""
    entry = app.get("b1")
    if entry is None:
        return
    f, w, n = entry
    w.writerow([now_us, out["k"], f"{out['p']:.6f}", f"{out['p_smooth']:.6f}",
                int(out["armed"]), int(out["stimulating"]), int(out["fired"])])
    if n % 10 == 0:                      # ~1 s of ticks; cheap, and this is not the BLE path
        f.flush()
    app["b1"] = (f, w, n + 1)


async def broadcast_task(app):
    """Drive the UI on a fixed timer, reading each source's .latest.
    Decoupled from the forearm queue, so a silent sensor shows as offline
    instead of freezing the whole page."""
    pl, trig, sources = app["placement"], app["trig"], app["sources"]
    feature, threshold = pl.trigger.feature, pl.trigger.threshold
    scale = 2.5 * threshold
    units = UNITS.get(feature, "")
    src = sources[pl.trigger.source]
    prev_pulses = 0
    marks = {role: (s.n_samples, time.monotonic()) for role, s in sources.items()}
    while True:
        await asyncio.sleep(1 / 30)
        latest = src.latest
        val = infer.compute_features(latest, feature) if latest is not None else 0.0
        fired = trig.pulse_count > prev_pulses          # a pulse happened since last frame
        prev_pulses = trig.pulse_count
        now_us = latest.host_t_us if latest is not None else time.time_ns() // 1000
        remaining = max(0.0, trig.refractory_s - (now_us / 1e6 - trig.last_fire_t))

        now = time.monotonic()                          # delivered rate over the last ~1 s
        hz = {}
        for role, s in sources.items():
            n0, t0 = marks[role]
            hz[role] = (s.n_samples - n0) / (now - t0) if now > t0 else 0.0
            if now - t0 >= 1.0:
                marks[role] = (s.n_samples, now)
        d = app.get("last") or {}
        await broadcast(app, {
            "connected": True, "gpio_mock": trig.is_mock, "session": app["session"],
            "trigger_role": pl.trigger.source, "feature": feature, "units": units,
            "threshold": threshold, "scale": scale,
            "value": val, "fired": fired, "fires": trig.pulse_count,
            "refractory_s": trig.refractory_s, "lockout_remaining": remaining,
            # Block 1: the model's own state, alongside the legacy acc gauge
            "p": d.get("p"), "p_smooth": d.get("p_smooth"),
            "armed": d.get("armed"), "stimulating": d.get("stimulating"),
            "p_threshold": app["pipeline"].cfg["threshold"],
            "n_scored": app["pipeline"].n_scored, "n_model_fires": app["pipeline"].n_fired,
            "sensors": {role: sensor_dict(s, hz.get(role, 0.0)) for role, s in sources.items()},
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
        cfg = app["pipeline"].cfg
        run_dir = app["recorder"].start(app["sources"].keys(), {
            "feature": pl.trigger.feature, "threshold": pl.trigger.threshold,
            "refractory_s": pl.trigger.refractory_s, "source_role": pl.trigger.source,
            "scale": 2.5 * pl.trigger.threshold, "rate_hz": rate,
            "gpio_mock": app["trig"].is_mock,
            "roles": {role: suf for role, _, suf in app["assigned"]},
            "model": {"name": cfg.get("name"), "threshold": cfg["threshold"],
                      "smoothing_k": cfg["smoothing_k"], "stim_ms": cfg["stim_ms"],
                      "lockout_ms": cfg["lockout_ms"], "us_per_call": app.get("us_per_call")},
        })
        close_block1_csv(app)              # one per session, alongside the IMU CSVs
        f = open(run_dir / "block1.csv", "w", newline="")
        w = csv.writer(f)
        w.writerow(["host_t_us", "k", "p", "p_smooth", "armed", "stimulating", "fired"])
        app["b1"] = (f, w, 0)
    else:
        app["recorder"].stop()
        close_block1_csv(app)
    return web.json_response({"ok": True, "session": on})


def close_block1_csv(app):
    entry = app.get("b1")
    if entry is not None:
        entry[0].flush()
        entry[0].close()
    app["b1"] = None


async def on_startup(app):
    placement = psm.Placement.load(CONFIG)
    assigned = await psm.scan_and_assign(placement)   # raises loudly if a DOT is missing
    app["recorder"] = rec.Recorder(HERE / "recordings")
    app["pipeline"] = pipe = b1p.Block1Pipeline(MODEL_DIR, ROLES)
    app["us_per_call"] = us = pipe.benchmark(1000)
    print(f"  block1: extract + p = {us:.0f} us/call over 1000 iterations "
          f"({100_000 / us:.0f}x under the {pipe.cfg['hop_ms']:.0f} ms tick)")

    def on_sample(s):                 # BLE notify path: record + align only, no inference here
        app["recorder"].write_sample(s)
        pipe.push(s)

    sources = {role: psm.BleakMovellaSource(addr, sensor_id=suf, role=role, on_sample=on_sample)
               for role, addr, suf in assigned}
    for src in sources.values():
        await src.start()

    # ----- rate gate to make sure the frequency is consistent -----

    n0 = {r: s.n_samples for r, s in sources.items()}
    await asyncio.sleep(5.0)
    rates = {r: (s.n_samples - n0[r]) / 5.0 for r, s in sources.items()}
    for r, hz in rates.items():
        print(f"  {r:10s} {hz:5.1f} Hz")
    bad = {r: hz for r, hz in rates.items() if hz < 55}
    if bad:
        raise RuntimeError(f"degraded stream: {bad} — reconnect all three DOTs")

    # ---------------------------------------------------------------

    app["placement"] = placement
    app["assigned"] = assigned
    app["sources"] = sources
    # refractory_s comes from placement.json and is 0 on the model path: Decision already enforces
    # the only spacing the offline validation assumes (stim_ms 1000, lockout_ms 0). Anything on top
    # would cap the rate below the validated 8.14 stim/min.
    app["trig"] = ttl.TTLTrigger(pin=placement.trigger.pin, pulse_ms=placement.trigger.pulse_ms, refractory_s=placement.trigger.refractory_s)
    app["clients"] = set()
    app["last"] = None
    app["tasks"] = [asyncio.create_task(block1_task(app)),
                    asyncio.create_task(broadcast_task(app))]
    print("Live on http://0.0.0.0:8080  (open it in a browser)")


async def on_cleanup(app):
    for t in app["tasks"]:
        t.cancel()
    app["recorder"].stop()
    close_block1_csv(app)
    for src in app["sources"].values():
        await src.stop()
    app["trig"].reset()
    app["trig"].close()


def main():
    app = web.Application()
    app["clients"] = set()
    app["session"] = False
    app["b1"] = None
    app.add_routes([web.get("/", index), web.get("/ws", ws_handler),
                    web.post("/session", session_handler)])
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    web.run_app(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()