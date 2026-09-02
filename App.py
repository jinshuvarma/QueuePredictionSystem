import json
import time
import pandas as pd
import cv2
import threading
import queue
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

from ultralytics import YOLO

from simulator import QueueSimulator
from state_estimator import compute_counter_states
from predictor import forecast_all_counters, threshold_alert
from recommender import generate_recommendations
from audio_manager import AudioAnnouncer
from camera_source import CentroidTracker, point_in_polygon

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
    try:
        return AudioAnnouncer()
    except Exception as e:
        print(f"Audio announcer disabled: {e}")
        return None

announcer = get_announcer()

if "last_audio_time" not in st.session_state:
    st.session_state.last_audio_time = datetime.min

# --- Sidebar Controls ---
st.sidebar.title("⚙️ System Controls")
data_mode = st.sidebar.radio(
    "Operation Mode",
    ["Simulation", "Live Camera"],
    index=0
)

if "current_mode" not in st.session_state:
    st.session_state.current_mode = data_mode

if st.session_state.current_mode != data_mode:
    st.session_state.history_df = pd.DataFrame(columns=[
        "timestamp", "counter_id", "people_in_queue", "arrivals", "served", "avg_service_time"
    ])
    st.session_state.current_mode = data_mode
    st.rerun()

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
    CAMERA_CONFIG = {
        "Counter 1": {
            "zone": [(242, 472), (244, 4), (479, 3), (514, 475), (242, 473)]
        }
    }

    class BrowserCameraProcessor:
        def __init__(self, zone):
            self.zone = zone
            self.model = YOLO("yolov8n.pt")
            self.tracker = CentroidTracker()
            self.entry_ts = {}
            self.seen_ids = set()
            self.service_times = []
            self.latest_row = {
                "timestamp": datetime.now(),
                "counter_id": "Counter 1",
                "people_in_queue": 0,
                "arrivals": 0,
                "served": 0,
                "avg_service_time": 2.5,
            }
            self.latest_frame = None
            self.lock = threading.Lock()
            self.frame_count = 0

        def _scaled_zone(self, width, height):
            sx = width / 514.0
            sy = height / 475.0
            return [(int(x * sx), int(y * sy)) for x, y in self.zone]

        def process(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            if self.frame_count % 2 != 0:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            h, w = img.shape[:2]
            zone = self._scaled_zone(w, h)

            results = self.model(img, classes=[0], conf=0.4, verbose=False)[0]
            centroids = []
            boxes = []

            for box in results.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                centroids.append(((x1 + x2) / 2, (y1 + y2) / 2))
                boxes.append((int(x1), int(y1), int(x2), int(y2)))

            objects = self.tracker.update(centroids)
            current_ids = set()
            arrivals = 0
            served = 0
            now = time.time()

            for oid, centroid in objects.items():
                if point_in_polygon(tuple(centroid), zone):
                    current_ids.add(oid)
                    if oid not in self.entry_ts:
                        self.entry_ts[oid] = now

            new_ids = current_ids - self.seen_ids
            left_ids = self.seen_ids - current_ids
            arrivals = len(new_ids)

            for oid in left_ids:
                if oid in self.entry_ts:
                    dwell_min = (now - self.entry_ts.pop(oid)) / 60.0
                    if dwell_min >= 0.5:
                        served += 1
                        self.service_times.append(dwell_min)

            self.seen_ids = current_ids

            avg_service_time = (
                sum(self.service_times[-20:]) / len(self.service_times[-20:])
                if self.service_times else 2.5
            )

            # Draw queue ROI.
            overlay = img.copy()
            roi_points = np.array(zone, dtype=np.int32)
            cv2.fillPoly(overlay, [roi_points], (0, 255, 255))
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
            cv2.polylines(img, [roi_points], True, (0, 255, 255), 2)

            # Draw YOLO boxes + tracker IDs.
            for box, centroid in zip(boxes, centroids):
                x1, y1, x2, y2 = box
                in_zone = point_in_polygon(centroid, zone)
                box_color = (0, 255, 0) if in_zone else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

            for oid, centroid in objects.items():
                cv2.putText(
                    img,
                    f"ID: {oid}",
                    (int(centroid[0]) - 15, int(centroid[1]) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            row = {
                "timestamp": datetime.now(),
                "counter_id": "Counter 1",
                "people_in_queue": len(current_ids),
                "arrivals": arrivals,
                "served": served,
                "avg_service_time": round(avg_service_time, 2),
            }

            with self.lock:
                self.latest_row = row
                self.latest_frame = img.copy()

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    @st.cache_resource
    def get_browser_processor():
        zone = CAMERA_CONFIG["Counter 1"]["zone"]
        return BrowserCameraProcessor(zone)

    processor = get_browser_processor()

    st.info("📷 Click START below and allow camera permission. YOLO will process your browser camera in real time.")

    rtc_ctx = webrtc_streamer(
        key="visionai-live-camera",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=processor.process,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )

    with processor.lock:
        live_row = dict(processor.latest_row)

    # Add the browser-camera data to the same history schema used by the simulator.
    live_df = pd.DataFrame([live_row])
    st.session_state.history_df = pd.concat(
        [st.session_state.history_df, live_df], ignore_index=True
    )

    if len(st.session_state.history_df) > 500:
        st.session_state.history_df = st.session_state.history_df.iloc[-500:]

    history = st.session_state.history_df

    csv_data = history.to_csv(index=False).encode("utf-8")
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
        st.caption("Browser camera is connected through WebRTC. Press START above and allow camera permission.")
        with processor.lock:
            preview = None if processor.latest_frame is None else processor.latest_frame.copy()
        if preview is not None:
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            st.image(preview_rgb, caption="YOLO Live Detection", use_container_width=True)
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
            if should_speak and announcer is not None:
                clean_msg = msg.split(" — ")[0].replace("min", "minutes")
                try:
                    announcer.announce(f"Attention please. {clean_msg}")
                except Exception as e:
                    print(f"Audio announcement skipped: {e}")
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
        signage_data["message"] = f"🚨 {msg.upper()}"
        signage_data["color"] = "#4a0404"
        break
try:
    with open("shared_state.json", "w") as f:
        json.dump(signage_data, f)
except Exception:
    pass

if auto_refresh:
    time.sleep(refresh_secs)
    st.rerun()
