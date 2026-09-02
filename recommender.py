"""
recommender.py
Rule-based recommendation engine. Deliberately NOT machine-learned, so it stays
fully explainable in a live demo: "we recommended this because wait time crossed
X minutes and Counter Y was idle" is a sentence you can say out loud to a judge.
"""

WAIT_THRESHOLD_MIN = 15      # counter counted as "overloaded" past this wait time
IDLE_QUEUE_THRESHOLD = 3     # counter counted as "idle / underused" at or below this queue length
REDIRECT_MARGIN_MIN = 8      # min wait-time gap between counters to suggest redirecting


def generate_recommendations(state_df, forecast_alerts: dict = None):
    """
    state_df: output of state_estimator.compute_counter_states()
    forecast_alerts: {counter_id: minutes_to_threshold_or_None} from predictor.threshold_alert()
    Returns a list of (severity, message) tuples, severity in
    {"high", "medium", "warning", "ok"}.
    """
    recs = []
    if state_df is None or state_df.empty:
        return recs

    overloaded = state_df[state_df["estimated_wait_min"] > WAIT_THRESHOLD_MIN]
    idle = state_df[state_df["people_in_queue"] <= IDLE_QUEUE_THRESHOLD]

    for _, row in overloaded.iterrows():
        c = row["counter_id"]
        wait = row["estimated_wait_min"]
        
        if wait > 15:
            extra_staff_needed = int(wait // 10) # 1 staff for every 10 mins over threshold
            recs.append(("high", f"{c} critical ({wait} min). Deploy {extra_staff_needed} extra staff immediately."))
            
        if not idle.empty:
            target = idle.iloc[0]["counter_id"]
            recs.append(("high",
                f"{c} is overloaded ({row['estimated_wait_min']} min wait). "
                f"Redirect customers to {target} or open a new counter."))
        else:
            recs.append(("high",
                f"{c} is overloaded ({row['estimated_wait_min']} min wait) and "
                f"no idle counter is available — open an additional counter."))

    if len(state_df) >= 2:
        busiest = state_df.loc[state_df["estimated_wait_min"].idxmax()]
        quietest = state_df.loc[state_df["estimated_wait_min"].idxmin()]
        gap = busiest["estimated_wait_min"] - quietest["estimated_wait_min"]
        if gap > REDIRECT_MARGIN_MIN and busiest["counter_id"] != quietest["counter_id"]:
            recs.append(("medium",
                f"{busiest['counter_id']} wait time is {gap:.0f} min higher than "
                f"{quietest['counter_id']} — consider redirecting some customers."))

    if forecast_alerts:
        for c, minutes in forecast_alerts.items():
            if minutes is not None:
                recs.append(("warning",
                    f"{c}: queue is likely to exceed 30 people within the next {minutes} minutes."))

    if not recs:
        recs.append(("ok", "All counters operating normally. No action needed."))

    return recs


if __name__ == "__main__":
    from simulator import QueueSimulator
    from state_estimator import compute_counter_states
    from predictor import forecast_all_counters, threshold_alert

    sim = QueueSimulator(["Counter 1", "Counter 2", "Counter 3"], seed=1)
    sim.trigger_surge("Counter 2", duration_minutes=20, multiplier=6)
    for _ in range(15):
        sim.tick()

    hist = sim.history_df()
    state_df = compute_counter_states(hist)
    forecasts = forecast_all_counters(hist, steps=20)
    alerts = {c: threshold_alert(f, 30) for c, f in forecasts.items()}

    for severity, msg in generate_recommendations(state_df, alerts):
        print(f"[{severity.upper()}] {msg}")