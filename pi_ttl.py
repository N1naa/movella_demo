import time

class LgpioGPIO:
    def __init__(self, pin):
        import lgpio
        self._lg = lgpio
        self.pin = pin
        self._h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self._h, pin, 0)  # start LOW
        self.pulses = 0

    def pulse(self, ms = 20): # pulses should last 20ms usually
        self._lg.gpio_write(self._h, self.pin, 1)
        time.sleep(ms / 1000.0)
        self._lg.gpio_write(self._h, self.pin, 0)
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

    def pulse(self, pulse_ms: int) -> None:
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

    def reset(self):
        self.last_fire_t = -float('inf')
        self.pulse_count = 0
        self.fire_history = []
    def close(self):
        if self.gpio:
            self.gpio.close()