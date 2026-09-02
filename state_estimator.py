"""
state_estimator.py
Converts raw arrival/service history into operational per-counter metrics:
queue length, arrival rate, service time, and estimated wait time
(via Little's Law: L = lambda * W  ->  W = L / lambda).
"""

import pandas as pd

WINDOW_MINUTES = 10  # rolling window used to estimate current rates


def compute_counter_states(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    history_df columns: timestamp, counter_id, people_in_queue, arrivals, served, avg_service_time
    Returns one row per counter with current queue length + estimated wait time.
    """
    if history_df.empty:
        return pd.DataFrame(columns=[
            "counter_id", "people_in_queue", "arrival_rate_per_min",
            "avg_service_time_min", "estimated_wait_min",
        ])

    results = []
    for counter_id, grp in history_df.groupby("counter_id"):
        grp = grp.sort_values("timestamp")
        recent = grp.tail(WINDOW_MINUTES)

        current_queue = int(grp["people_in_queue"].iloc[-1])
        arrival_rate = float(recent["arrivals"].mean())          # people/min
        avg_service_time = float(recent["avg_service_time"].mean())  # min/person

        # Little's Law; guard divide-by-zero when arrivals are ~0 (near-idle counter)
        if arrival_rate > 0.01:
            wait_time_min = current_queue / arrival_rate
        else:
            wait_time_min = current_queue * avg_service_time

        results.append({
            "counter_id": counter_id,
            "people_in_queue": current_queue,
            "arrival_rate_per_min": round(arrival_rate, 2),
            "avg_service_time_min": round(avg_service_time, 2),
            "estimated_wait_min": round(wait_time_min, 1),
        })

    return pd.DataFrame(results).sort_values("counter_id").reset_index(drop=True)


if __name__ == "__main__":
    from simulator import QueueSimulator

    sim = QueueSimulator(["Counter 1", "Counter 2", "Counter 3"], seed=1)
    for _ in range(15):
        sim.tick()
    print(compute_counter_states(sim.history_df()))