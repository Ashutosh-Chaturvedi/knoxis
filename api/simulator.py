"""
Knoxis Backend - Step 3: The Telemetry Simulator
================================================
This file generates continuous 1Hz satellite readings (like Aditya-L1 in orbit).
It can run in two modes:
1. QUIET SUN: Normal small fluctuations around ~45 counts/sec.
2. FLARE ERUPTION: A realistic M-class solar flare eruption that rises,
   peaks at ~3,500 counts/sec (M4.8 flare), and slowly declines.
"""

import asyncio
import math
import random
from datetime import datetime, timezone
from api.engine import engine


class TelemetrySimulator:
    def __init__(self):
        self.is_running: bool = False
        self.mode: str = "QUIET"       # "QUIET" or "FLARE"
        self.flare_step: int = 0
        self.listeners: list[asyncio.Queue] = []

    def start_flare(self):
        """Triggers an M-class solar flare sequence!"""
        self.mode = "FLARE"
        self.flare_step = 0

    def set_quiet(self):
        """Returns the sun to quiet baseline state."""
        self.mode = "QUIET"
        self.flare_step = 0

    def generate_next_count(self) -> float:
        """Calculates the next second's photon count based on active solar activity."""
        base_noise = random.gauss(0, 2.0)  # Random statistical fluctuation

        if self.mode == "QUIET":
            # Normal quiet background sun around 45 counts/sec
            return max(10.0, 45.0 + base_noise)

        # FLARE SIMULATION CURVE:
        # Step 0-20: Rise phase (rapid surge)
        # Step 20-60: Peak phase (~3,500 counts = M4.8 flare)
        # Step 60-150: Gradual cooling and decline back to quiet
        self.flare_step += 1
        t = self.flare_step

        if t <= 20:
            # Steep exponential rise
            rise_factor = math.exp(t / 4.5)
            counts = 45.0 + rise_factor * 35.0
        elif t <= 60:
            # Peak fluctuation around 3,200 - 3,600 counts
            counts = 3400.0 + random.gauss(0, 80.0)
        elif t <= 140:
            # Slow exponential decay
            decay = math.exp(-(t - 60) / 25.0)
            counts = 45.0 + 3355.0 * decay + base_noise
        else:
            # Flare has completely subsided
            self.mode = "QUIET"
            self.flare_step = 0
            counts = 45.0 + base_noise

        return round(max(10.0, counts), 1)

    async def run_loop(self, broadcast_callback):
        """Runs an infinite loop sending a new tick every 1.0 second."""
        self.is_running = True
        while self.is_running:
            now = datetime.now(timezone.utc)
            counts = self.generate_next_count()
            
            # Feed reading to Knoxis physics engine
            state = engine.process_tick(counts=counts, timestamp=now, is_valid=True)
            
            # Prepare packet for WebSocket clients
            packet = {
                "timestamp": now.isoformat(),
                "time_str": now.strftime("%H:%M:%S"),
                "counts": counts,
                "baseline": round(engine.baseline, 1),
                "ratio": round(counts / engine.baseline, 2),
                "goes_class": state["current_reading"]["goes_class"],
                "flux_w_m2": state["current_reading"]["flux_w_m2"],
                "alert_level": state["alert_level"],
                "active_flare": state["active_flare"],
                "sustained_seconds": state["sustained_seconds"],
                "simulation_mode": self.mode,
            }
            
            # Broadcast to all connected browsers
            await broadcast_callback(packet)
            
            # Wait exactly 1 second before the next tick
            await asyncio.sleep(1.0)


# Global simulator instance
simulator = TelemetrySimulator()
