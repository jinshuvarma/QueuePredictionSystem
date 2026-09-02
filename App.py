import json
import time
import pandas as pd
import cv2
import threading
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

from simulator import QueueSimulator
from state_estimator import compute_counter_states
from predictor import forecast_all_counters, threshold_alert
from recommender import generate_recommendations
from audio_manager import AudioAnnouncer

# --- Page Config & CSS ---
st.set_page_config(page_title="VisionAI Queue Manager", page_icon="👁️", layout="wide")
st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] { opacity: 1 !important; transition: none !important; }
        [data-testid="stStatusWidget"] { display: none !important; }
        div[data-testid="metric-container"] {
            background-color: #1E1E2E; border: 1px solid #333344;
            padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
    </style>
    """,
    unsafe_allow_html=True
)

COUNTERS = ["Counter 1", "Counter 2", "Counter 3"]
FORECAST_STEPS = 20
CROWD_THRESHOLD = 30
AUDIO_COOLDOWN_SECONDS = 10 

# --- Audio Initialization ---
@st.cache_resource
def get_announcer():
    return AudioAnnouncer()

announcer = get_announcer()

if "last_audio_time" not in st.session_state:
    st.session_state.last_audio_time = datetime.min

# --- Sidebar Controls ---
st.sidebar.title("⚙️ System Controls")
data_mode = st.sidebar.radio("Operation Mode", ["Live Camera", "Simulation"])

# --- MODE SWITCH HANDLER (Clears old data when switching tabs) ---
if "current_mode" not in st.session_state:
    st.session_state.current_mode = data_mode

if st.session_state.current_mode != data_mode:
    # Wipe the database so charts and tables reset completely
    st.session_state.history_df = pd.DataFrame(columns=[
        "timestamp", "counter_id", "people_in_queue", "arrivals", "served", "avg_service_time"
    ])
    st.session_state.current_mode = data_mode
    st.rerun()

# --- Initialize Empty History if needed ---
if "history_df" not in st.session_state or st.session_state.history_df.empty:
    st.session_state.history_df = pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ns]"),
        "counter_id": pd.Series(dtype="object"),
        "people_in_queue": pd.Series(dtype="int64"),
        "arrivals": pd.Series(dtype="int64"),
        "served": pd.Series(dtype="int64"),
        "avg_service_time": pd.Series(dtype="float64"),
    })

auto_refresh = st.sidebar.checkbox("Auto-Update Dashboard", value=True)

refresh_secs = st.sidebar.slider("UI Refresh Rate (secs)", 0.1, 2.0, 0.2) 

# --- Data Pipeline Execution ---
if data_mode == "Simulation":
    if "sim" not in st.session_state:
        st.session_state.sim = QueueSimulator(COUNTERS, seed=42)
        for _ in range(10): st.session_state.sim.tick()
    sim = st.session_state.sim

    st.sidebar.markdown("---")
    st.sidebar.subheader("Tools")
    surge_counter = st.sidebar.selectbox("Target Counter", COUNTERS)
    if st.sidebar.button("Trigger Crowd Surge"):
        sim.trigger_surge(surge_counter, duration_minutes=15, multiplier=6)
        st.sidebar.success(f"Surge activated at {surge_counter}")

    sim.tick()
    history = sim.history_df()
    
    csv_data = history.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Shift Audit Report",
        data=csv_data,
        file_name="queue_performance_report.csv",
        mime="text/csv",
    )

else:
    from camera_source import CameraQueueSource
    CAMERA_CONFIG = {
        "Counter 1": {
            "video_path": 0, 
            "zone": [(242, 472), (244, 4), (479, 3), (514, 475), (242, 473)] 
        }
    }

    @st.cache_resource
    def start_camera_engine(config):
        shared_memory = {"frames": {}, "latest_rows": [], "status": "Initializing..."}
        sources = {}
        for cid, cfg in config.items():
            try:
                sources[cid] = CameraQueueSource(cid, cfg["video_path"], cfg["zone"], frame_skip=2)
            except Exception as e:
                shared_memory["status"] = f"Error with {cid}: {e}"
                return shared_memory
        shared_memory["status"] = "Running"

        def inference_loop():
            try:
                while True:
                    rows = []
                    for cid, src in sources.items():
                        
                        row = src.tick(duration_seconds=0.2) 
                        if "frame" in row and row["frame"] is not None:
                            shared_memory["frames"][cid] = row.pop("frame")
                        rows.append(row)
                    shared_memory["latest_rows"] = rows
            except Exception as e:
                print(f"\n❌ BACKGROUND THREAD CRASHED: {e}\n")
                shared_memory["status"] = f"Crash: {e}"

        threading.Thread(target=inference_loop, daemon=True).start()
        return shared_memory

    shared_memory = start_camera_engine(CAMERA_CONFIG)
    if shared_memory["status"] != "Running":
        st.error(shared_memory["status"])
        st.stop()

    if shared_memory["latest_rows"]:
        st.session_state.history_df = pd.concat(
            [st.session_state.history_df, pd.DataFrame(shared_memory["latest_rows"])], ignore_index=True
        )
    if len(st.session_state.history_df) > 500:
        st.session_state.history_df = st.session_state.history_df.iloc[-500:]
    history = st.session_state.history_df

    csv_data = history.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Shift Audit Report",
        data=csv_data,
        file_name="queue_performance_report.csv",
        mime="text/csv",
    )
# --- Analytics Processing ---
state_df = compute_counter_states(history)
forecasts = forecast_all_counters(history, steps=FORECAST_STEPS)
forecast_alerts = {c: threshold_alert(f, CROWD_THRESHOLD) for c, f in forecasts.items()}
recommendations = generate_recommendations(state_df, forecast_alerts)

st.title("VisionAI Queue Manager")
st.markdown("Real-time computer vision queue tracking, forecasting, and automated load balancing.")

# Top Level Metrics
col1, col2, col3, col4 = st.columns(4)
total_waiting = state_df["people_in_queue"].sum() if not state_df.empty else 0
max_wait = state_df["estimated_wait_min"].max() if not state_df.empty else 0
busiest_counter = state_df.loc[state_df["estimated_wait_min"].idxmax()]["counter_id"] if not state_df.empty else "N/A"

col1.metric("Total People Waiting", total_waiting)
col2.metric("Max Wait Time", f"{max_wait:.1f} min")
col3.metric("Busiest Node", busiest_counter)
col4.metric("System Status", "CRITICAL" if total_waiting > CROWD_THRESHOLD else "OPTIMAL")

st.markdown("---")

c_left, c_right = st.columns([1.2, 1])

with c_left:
    st.subheader("Camera")
    if data_mode == "Live Camera":
        if "camera_placeholders" not in st.session_state:
            st.session_state.camera_placeholders = {cid: st.empty() for cid in CAMERA_CONFIG.keys()}
        
        for cid in CAMERA_CONFIG.keys():
            frame_bgr = shared_memory["frames"].get(cid)
            if frame_bgr is not None and isinstance(frame_bgr, np.ndarray) and frame_bgr.size > 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                st.session_state.camera_placeholders[cid].image(frame_rgb, caption=f"Processing: {cid}", use_container_width=True)
            else:
                st.session_state.camera_placeholders[cid].info(f"Connecting to optical sensor for {cid}...")
    else:
        st.info("Visual intelligence disabled in Simulation Mode. Switch to Live Camera in sidebar.")

with c_right:
    st.subheader("Live State Estimation")
    display_df = state_df.rename(columns={
        "counter_id": "Node", "people_in_queue": "Queue Len",
        "arrival_rate_per_min": "Arrivals/m", "avg_service_time_min": "Service(m)",
        "estimated_wait_min": "Est. Wait(m)",
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Actions needs to be perform")
    current_time = datetime.now()
    should_speak = (current_time - st.session_state.last_audio_time).total_seconds() > AUDIO_COOLDOWN_SECONDS

    for severity, msg in recommendations:
        if severity == "high":
            st.error(f"ACTION REQUIRED: {msg}")
            if should_speak:
                clean_msg = msg.split(" — ")[0].replace("min", "minutes")
                announcer.announce(f"Attention please. {clean_msg}")
                st.session_state.last_audio_time = current_time
                should_speak = False 
        elif severity == "medium":
            st.warning(f"⚠️ {msg}")
        elif severity == "warning":
            st.info(f"⏳ {msg}")
        else:
            st.success(f"✅ {msg}")

st.markdown("---")

st.subheader("📈 Queue Trajectory & AI Forecast")
fig = go.Figure()
colors = {"Counter 1": "#00F0FF", "Counter 2": "#FF0055", "Counter 3": "#00FF66"}

for c in COUNTERS:
    grp = history[history["counter_id"] == c].sort_values("timestamp")
    fig.add_trace(go.Scatter(
        x=grp["timestamp"], y=grp["people_in_queue"], mode="lines", name=f"{c} (Actual)",
        line=dict(color=colors.get(c, "#FFF"), width=2),
    ))
    if c in forecasts:
        last_ts = grp["timestamp"].iloc[-1] if not grp.empty else pd.Timestamp.now()
        future_ts = [last_ts + pd.Timedelta(minutes=i) for i in range(1, FORECAST_STEPS + 1)]
        fig.add_trace(go.Scatter(
            x=future_ts, y=forecasts[c], mode="lines", name=f"{c} (Forecast)",
            line=dict(color=colors.get(c, "#FFF"), dash="dot", width=2),
        ))

fig.add_hline(y=CROWD_THRESHOLD, line_dash="dash", line_color="red",
              annotation_text=f"Critical Threshold ({CROWD_THRESHOLD})", annotation_position="top right")
fig.update_layout(
    height=400, xaxis_title="Timeline", yaxis_title="People in Queue",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"), legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig, use_container_width=True)

signage_data = {"status": "NORMAL", "message": "✅ PLEASE WAIT IN LINE", "color": "#0a2911"}

# Check if any counter has a highly unusual service time (stuck customer)
stuck_counters = state_df[state_df["avg_service_time_min"] > 5.0]

if not stuck_counters.empty:
    signage_data["message"] = "⚡ EXPRESS LANE OPEN AT COUNTER 2 (Max 2 Items)"
    signage_data["color"] = "#b58900" # Warning Yellow
else:
    for severity, msg in recommendations:
        if severity == "high":
            signage_data["status"] = "FULL"
            signage_data["message"] = f"🚨 {msg.upper()}"
            signage_data["color"] = "#4a0404"
            break

for severity, msg in recommendations:
    if severity == "high":
        signage_data["status"] = "FULL"
        # Create a bold, aggressive message for the display
        signage_data["message"] = f"🚨 {msg.upper()}"
        signage_data["color"] = "#4a0404"  # Deep Red
        break # Display the highest priority alert

# Write to a local file that the Signage page will read
try:
    with open("shared_state.json", "w") as f:
        json.dump(signage_data, f)
except Exception:
    pass

if auto_refresh:
    time.sleep(refresh_secs)
    st.rerun()