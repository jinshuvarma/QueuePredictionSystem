"""
predictor.py
Forecasts near-future queue length per counter. Uses Holt's (damped) Exponential
Smoothing via statsmodels — a statistical model that is FIT (not "trained" in the
deep-learning sense) fresh on each call, so there's no separate training phase and
no risk of a stale/overfit model breaking the live demo.

Falls back to simple linear extrapolation when a counter doesn't yet have enough
history (e.g. right after startup).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

MIN_POINTS_FOR_MODEL = 8  # below this, use a simple trend line instead


def forecast_counter(series: pd.Series, steps: int = 20) -> np.ndarray:
    """Forecast the next `steps` minutes of queue length for one counter.
    Returns an array of length `steps`, clipped at 0 (queue can't be negative)."""
    # Defensive cast: statsmodels requires numeric (float) dtype. If the
    # incoming series ever arrives as dtype 'object' (e.g. built from an
    # empty-then-concatenated DataFrame upstream), this prevents a crash.
    series = pd.to_numeric(series, errors="coerce").fillna(0).astype(float).reset_index(drop=True)

    if len(series) < MIN_POINTS_FOR_MODEL:
        slope = series.iloc[-1] - series.iloc[-2] if len(series) >= 2 else 0
        last_val = series.iloc[-1] if len(series) else 0
        forecast = [last_val + slope * i for i in range(1, steps + 1)]
    else:
        model = ExponentialSmoothing(series, trend="add", damped_trend=True)
        fit = model.fit(optimized=True)
        forecast = fit.forecast(steps)

    return np.clip(np.array(forecast, dtype=float), 0, None)


def forecast_all_counters(history_df: pd.DataFrame, steps: int = 20) -> dict:
    """Runs forecast_counter() for every counter_id present in history_df."""
    forecasts = {}
    for counter_id, grp in history_df.groupby("counter_id"):
        grp = grp.sort_values("timestamp")
        forecasts[counter_id] = forecast_counter(grp["people_in_queue"], steps=steps)
    return forecasts


def threshold_alert(forecast_values, threshold: int = 30):
    """
    Returns minutes-until-threshold-crossing (int), or None if the forecast
    never crosses the threshold within its horizon.
    Powers the bonus statement:
    "Queue is likely to exceed 30 people within the next N minutes."
    """
    for i, v in enumerate(forecast_values, start=1):
        if v > threshold:
            return i
    return None


if __name__ == "__main__":
    from simulator import QueueSimulator

    sim = QueueSimulator(["Counter 1", "Counter 2", "Counter 3"], seed=1)
    sim.trigger_surge("Counter 2", duration_minutes=20, multiplier=6)
    for _ in range(15):
        sim.tick()

    hist = sim.history_df()
    forecasts = forecast_all_counters(hist, steps=20)
    for c, f in forecasts.items():
        print(c, "next 5 min forecast:", np.round(f[:5], 1),
              "| crosses 30 in:", threshold_alert(f, 30), "min")