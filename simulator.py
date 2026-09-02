"""
simulator.py
Generates realistic simulated queue data for multiple counters using a
Poisson arrival process + Gaussian service times. This is the PRIMARY
data source for the demo (reliable, controllable) — a camera/video feed
can later be plugged in as an alternate source producing the same
row format: timestamp, counter_id, people_in_queue, arrivals, served,
avg_service_time.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class QueueSimulator:
    """
    Call `.tick()` once per simulated minute to advance all counters
    and get their updated state as a small DataFrame (one row per counter).
    """

    def __init__(self, counter_ids, base_lambda=None, base_service_time=None, seed=None):
        self.counter_ids = counter_ids
        self.rng = np.random.default_rng(seed)

        # average arrivals/minute per counter (differs per counter -> realistic imbalance)
        # kept modest so the queue is roughly stable at baseline and a surge is
        # clearly visible against it (arrival rate close to service capacity)
        self.base_lambda = base_lambda or {c: self.rng.uniform(0.3, 0.6) for c in counter_ids}
        # average service time (minutes/person) per counter
        self.base_service_time = base_service_time or {c: self.rng.uniform(1.5, 3.0) for c in counter_ids}

        self.queue_length = {c: int(self.rng.integers(0, 5)) for c in counter_ids}
        self.surge_until = {c: None for c in counter_ids}  # (end_time, multiplier)
        self.current_time = datetime.now()

        self.history = []  # list of dict rows

    def trigger_surge(self, counter_id, duration_minutes=15, multiplier=5):
        """Manually spike arrival rate at a counter — use this live during the demo
        to show the system predicting and reacting to a sudden crowd."""
        self.surge_until[counter_id] = (self.current_time + timedelta(minutes=duration_minutes), multiplier)

    def _effective_lambda(self, counter_id):
        lam = self.base_lambda[counter_id]
        surge = self.surge_until.get(counter_id)
        if surge and self.current_time < surge[0]:
            return lam * surge[1]
        return lam

    def tick(self):
        """Advance simulation by one minute for all counters. Returns the new rows."""
        self.current_time += timedelta(minutes=1)
        rows = []
        for c in self.counter_ids:
            lam = self._effective_lambda(c)
            arrivals = int(self.rng.poisson(lam))
            service_time = max(0.5, float(self.rng.normal(self.base_service_time[c], 0.4)))

            service_rate = 1 / service_time  # people this counter can clear per minute
            served = min(self.queue_length[c], int(self.rng.poisson(service_rate)))

            self.queue_length[c] = max(0, self.queue_length[c] + arrivals - served)

            row = {
                "timestamp": self.current_time,
                "counter_id": c,
                "people_in_queue": self.queue_length[c],
                "arrivals": arrivals,
                "served": served,
                "avg_service_time": round(service_time, 2),
            }
            rows.append(row)
            self.history.append(row)
        return pd.DataFrame(rows)

    def history_df(self):
        return pd.DataFrame(self.history)


if __name__ == "__main__":
    # quick smoke test
    sim = QueueSimulator(["Counter 1", "Counter 2", "Counter 3"], seed=1)
    for _ in range(15):
        sim.tick()
    print(sim.history_df().tail(9))