import asyncio
import time

class LgpioGPIO:
    def __init__(self, pin):
        import lgpio
        self._lg = lgpio
        self.pin = pin
        self._h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self._h, pin, 0)  # start LOW
        self.pulses = 0
        # Epoch us of the most recent RISING edge, stamped in the line immediately before the
        # write so it is the closest observable proxy for when stimulation actually started.
        # time.time_ns(), NOT monotonic: it has to be the same clock domain as Sample.host_t_us
        # and the tick's now_us, or the differences downstream are meaningless.
        self.last_edge_us = None

    def pulse(self, ms = 20): # pulses should last 20ms usually
        self.last_edge_us = time.time_ns() // 1000
        self._lg.gpio_write(self._h, self.pin, 1)
        time.sleep(ms / 1000.0)
        self._lg.gpio_write(self._h, self.pin, 0)
        self.pulses += 1
        print(f"  [pin {self.pin}] TTL pulse #{self.pulses} ({ms} ms)")

    async def pulse_async(self, ms = 20):
        """Same pulse without blocking the event loop. time.sleep() here is a 20 ms blackout on
        all three BLE streams at once; asyncio.sleep() lets the notifications through."""
        self.last_edge_us = time.time_ns() // 1000
        self._lg.gpio_write(self._h, self.pin, 1)
        try:
            await asyncio.sleep(ms / 1000.0)
        finally:
            self._lg.gpio_write(self._h, self.pin, 0)   # never leave the pin high on cancellation
        self.pulses += 1
        print(f"  [pin {self.pin}] TTL pulse #{self.pulses} ({ms} ms)")

    def close(self):
        try:
            self._lg.gpio_write(self._h, self.pin, 0)
            self._lg.gpiochip_close(self._h)
        except Exception:
            pass

class MockGPIO:
    """Stand-in when there's no real GPIO. Records pulses, prints them."""
    def __init__(self, pin: int):
        self.pin = pin
        self.pulses = 0
        self.last_edge_us = None      # same contract as LgpioGPIO, so the log looks identical

    def pulse(self, pulse_ms: int) -> None:
        self.last_edge_us = time.time_ns() // 1000
        self.pulses += 1
        print(f"  [MOCK GPIO pin {self.pin}] TTL pulse #{self.pulses} "
              f"({pulse_ms} ms)")

    async def pulse_async(self, pulse_ms: int) -> None:
        """Mock has nothing to hold high, so it just yields for the same duration - the async
        path must behave the same way with or without real GPIO."""
        self.last_edge_us = time.time_ns() // 1000
        await asyncio.sleep(pulse_ms / 1000.0)
        self.pulses += 1
        print(f"  [MOCK GPIO pin {self.pin}] TTL pulse #{self.pulses} "
              f"({pulse_ms} ms)")

    def close(self) -> None:
        pass


def make_gpio(pin, mock=False):
    """Return a GPIO backend, never None.
    mock=True  -> MockGPIO (hardware never touched)
    mock=False -> real LgpioGPIO, falling back to MockGPIO if lgpio is
                  unavailable (e.g. on the Mac)."""
    if mock:
        print("You're working with a MOCK GPIO. If this was not the intention, check your wiring.")
        return MockGPIO(pin)
    try:
        return LgpioGPIO(pin)
    except Exception as e:
        print(f"  (no real GPIO: {type(e).__name__}: {e} -> mock)")
        return MockGPIO(pin)
    
class TTLTrigger: # own class so we can keep track of last fire time and pulse count
    def __init__(self, pin=16, pulse_ms=20, refractory_s=8.0, mock=False):
        self.pulse_ms = pulse_ms
        self.refractory_s = refractory_s
        self.gpio = make_gpio(pin, mock)
        self.last_fire_t = -float('inf') # shouldn't this be a list if i want to keep them? 
        self.pulse_count = 0
        self.fire_history = [] # list of (timestamp, pulse_count) for testing
        self.is_mock = isinstance(self.gpio, MockGPIO)

    def fire_if_ready(self, now_t_us):
        now_t = now_t_us / 1e6
        if now_t - self.last_fire_t >= self.refractory_s:
            self.gpio.pulse(self.pulse_ms)
            self.last_fire_t = now_t
            self.pulse_count += 1
            self.fire_history.append((now_t_us, self.pulse_count))
            #self.fire_history.append((s.sensor_t_us, s.host_t_us, val))
            return True
        return False

    async def fire_if_ready_async(self, now_t_us):
        """Non-blocking twin of fire_if_ready, for callers already in the event loop.

        The bookkeeping is committed BEFORE the await: the pulse now spans 20 ms of event-loop
        time during which another notification can call this again, and a check-then-await
        ordering would let both pass the refractory test and double-fire.
        """
        now_t = now_t_us / 1e6
        if now_t - self.last_fire_t < self.refractory_s:
            return False
        self.last_fire_t = now_t
        self.pulse_count += 1
        self.fire_history.append((now_t_us, self.pulse_count))
        await self.gpio.pulse_async(self.pulse_ms)
        return True

    def reset(self):
        self.last_fire_t = -float('inf')
        self.pulse_count = 0
        self.fire_history = []
    def close(self):
        if self.gpio:
            self.gpio.close()