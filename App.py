import json
import time
import pandas as pd
import threading
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

st.set_page_config(
    page_title="VisionAI Queue Manager",
    page_icon="👁️",
    layout="wide",
)

st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            opacity: 1 !important;
            transition: none !important;
        }

        [data-testid="stStatusWidget"] {
            display: none !important;
        }

        div[data-testid="metric-container"] {
            background-color: #1E1E2E;
            border: 1px solid #333344;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.25);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 10px;
        }

        .camera-card {
            border: 1px solid #333344;
            border-radius: 12px;
            padding: 10px;
            background: #11111B;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


COUNTERS = ["Counter 1", "Counter 2", "Counter 3"]
FORECAST_STEPS = 20
CROWD_THRESHOLD = 30
AUDIO_COOLDOWN_SECONDS = 10

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

st.sidebar.title("⚙️ System Controls")

data_mode = st.sidebar.radio(
    "Operation Mode",
    ["Simulation", "Live Camera"],
    index=0,
)

if "current_mode" not in st.session_state:
    st.session_state.current_mode = data_mode

if st.session_state.current_mode != data_mode:
    st.session_state.history_df = pd.DataFrame(
        columns=[
            "timestamp",
            "counter_id",
            "people_in_queue",
            "arrivals",
            "served",
            "avg_service_time",
        ]
    )
    st.session_state.current_mode = data_mode
    st.rerun()


if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame(
        columns=[
            "timestamp",
            "counter_id",
            "people_in_queue",
            "arrivals",
            "served",
            "avg_service_time",
        ]
    )

auto_refresh = st.sidebar.checkbox(
    "Auto-Update Dashboard",
    value=True,
)

refresh_secs = st.sidebar.slider(
    "UI Refresh Rate (secs)",
    0.5,
    2.0,
    1.0,
)


# ============================================================
# DATA PIPELINE
# ============================================================
processor = None
rtc_ctx = None

if data_mode == "Simulation":

    if "sim" not in st.session_state:
        st.session_state.sim = QueueSimulator(COUNTERS, seed=42)
        for _ in range(10):
            st.session_state.sim.tick()

    sim = st.session_state.sim

    st.sidebar.markdown("---")
    st.sidebar.subheader("Tools")

    surge_counter = st.sidebar.selectbox(
        "Target Counter",
        COUNTERS,
    )

    if st.sidebar.button("Trigger Crowd Surge"):
        sim.trigger_surge(
            surge_counter,
            duration_minutes=15,
            multiplier=6,
        )
        st.sidebar.success(
            f"Surge activated at {surge_counter}"
        )

    sim.tick()
    history = sim.history_df()

    csv_data = history.to_csv(index=False).encode("utf-8")

    st.sidebar.download_button(
        label="📥 Download Shift Audit Report",
        data=csv_data,
        file_name="queue_performance_report.csv",
        mime="text/csv",
    )

else:

    # Original camera ROI coordinate system.
    CAMERA_ZONE = [
        (242, 472),
        (244, 4),
        (479, 3),
        (514, 475),
        (242, 473),
    ]

    # Local lightweight tracker/helpers.
    # Keeping these here avoids importing camera_source.py (which imports
    # regular OpenCV at module startup on some cloud environments).

    def point_in_polygon(point, polygon):
        x, y = point
        inside = False
        j = len(polygon) - 1

        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            intersects = (
                ((yi > y) != (yj > y))
                and (
                    x
                    < (xj - xi) * (y - yi)
                    / ((yj - yi) or 1e-9)
                    + xi
                )
            )

            if intersects:
                inside = not inside

            j = i

        return inside


    class CentroidTracker:
        def __init__(self, max_disappeared=20, max_distance=80):
            self.next_object_id = 0
            self.objects = {}
            self.disappeared = {}
            self.max_disappeared = max_disappeared
            self.max_distance = max_distance

        def register(self, centroid):
            self.objects[self.next_object_id] = centroid
            self.disappeared[self.next_object_id] = 0
            self.next_object_id += 1

        def deregister(self, object_id):
            self.objects.pop(object_id, None)
            self.disappeared.pop(object_id, None)

        def update(self, input_centroids):
            if len(input_centroids) == 0:
                for object_id in list(self.disappeared):
                    self.disappeared[object_id] += 1

                    if (
                        self.disappeared[object_id]
                        > self.max_disappeared
                    ):
                        self.deregister(object_id)

                return dict(self.objects)

            input_centroids = np.asarray(
                input_centroids,
                dtype=np.float32,
            )

            if len(self.objects) == 0:
                for centroid in input_centroids:
                    self.register(centroid)
                return dict(self.objects)

            object_ids = list(self.objects.keys())
            object_centroids = np.asarray(
                [self.objects[i] for i in object_ids],
                dtype=np.float32,
            )

            distances = np.linalg.norm(
                object_centroids[:, None, :]
                - input_centroids[None, :, :],
                axis=2,
            )

            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if distances[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = tuple(
                    input_centroids[col]
                )
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(len(object_ids))) - used_rows
            unused_cols = set(range(len(input_centroids))) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                if (
                    self.disappeared[object_id]
                    > self.max_disappeared
                ):
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

            return dict(self.objects)


    class BrowserCameraProcessor:
        """
        Browser-camera processor.

        Important:
        - WebRTC supplies frames from the user's browser.
        - YOLO runs every few frames for performance.
        - ROI + last detection boxes are drawn on EVERY frame.
          This prevents the ROI from blinking/flickering.
        """

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

            self.lock = threading.Lock()

            self.frame_count = 0
            self.inference_interval = 4

            # Detection state is persistent between YOLO inference frames.
            self.last_boxes = []
            self.last_centroids = []
            self.last_objects = {}

        def _scaled_zone(self, width, height):
            sx = width / 514.0
            sy = height / 475.0

            return [
                (int(x * sx), int(y * sy))
                for x, y in self.zone
            ]

        def _draw_overlay(
            self,
            img,
            zone,
            boxes,
            centroids,
            objects,
        ):
            # Lazy import keeps App.py startup cloud-safe.
            import cv2

            # ---- Permanent ROI: draw on EVERY frame ----
            overlay = img.copy()
            roi_points = np.array(
                zone,
                dtype=np.int32,
            )

            cv2.fillPoly(
                overlay,
                [roi_points],
                (0, 255, 255),
            )

            cv2.addWeighted(
                overlay,
                0.12,
                img,
                0.88,
                0,
                img,
            )

            cv2.polylines(
                img,
                [roi_points],
                True,
                (0, 255, 255),
                3,
            )

            # ---- Detection boxes: use latest YOLO result ----
            for box, centroid in zip(
                boxes,
                centroids,
            ):
                x1, y1, x2, y2 = box

                in_zone = point_in_polygon(
                    tuple(centroid),
                    zone,
                )

                box_color = (
                    (0, 255, 0)
                    if in_zone
                    else (0, 0, 255)
                )

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2,
                )

            # ---- Tracker IDs ----
            for oid, centroid in objects.items():
                cv2.putText(
                    img,
                    f"ID: {oid}",
                    (
                        int(centroid[0]) - 15,
                        int(centroid[1]) - 15,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            # Small fixed label.
            cv2.putText(
                img,
                "QUEUE ROI",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
            )

            return img

        def process(self, frame):
            import cv2

            img = frame.to_ndarray(
                format="bgr24"
            )

            self.frame_count += 1

            h, w = img.shape[:2]
            zone = self._scaled_zone(
                w,
                h,
            )

            # ==================================================
            # YOLO inference only every Nth frame.
            # Detection state remains visible between inferences.
            # ==================================================
            if (
                self.frame_count
                % self.inference_interval
                == 0
            ):
                results = self.model(
                    img,
                    classes=[0],
                    conf=0.35,
                    imgsz=640,
                    verbose=False,
                )[0]

                centroids = []
                boxes = []

                for box in results.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box

                    centroids.append(
                        (
                            (x1 + x2) / 2,
                            (y1 + y2) / 2,
                        )
                    )

                    boxes.append(
                        (
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                        )
                    )

                objects = self.tracker.update(
                    centroids
                )

                current_ids = set()
                arrivals = 0
                served = 0
                now = time.time()

                # ----------------------------------------------
                # Queue-zone membership
                # ----------------------------------------------
                for oid, centroid in objects.items():
                    if point_in_polygon(
                        tuple(centroid),
                        zone,
                    ):
                        current_ids.add(oid)

                        if oid not in self.entry_ts:
                            self.entry_ts[oid] = now

                new_ids = (
                    current_ids
                    - self.seen_ids
                )

                left_ids = (
                    self.seen_ids
                    - current_ids
                )

                arrivals = len(new_ids)

                for oid in left_ids:
                    if oid in self.entry_ts:
                        dwell_min = (
                            now
                            - self.entry_ts.pop(oid)
                        ) / 60.0

                        if dwell_min >= 0.5:
                            served += 1
                            self.service_times.append(
                                dwell_min
                            )

                self.seen_ids = current_ids

                recent_service = (
                    self.service_times[-20:]
                )

                avg_service_time = (
                    sum(recent_service)
                    / len(recent_service)
                    if recent_service
                    else 2.5
                )

                self.last_boxes = boxes
                self.last_centroids = centroids
                self.last_objects = objects

                row = {
                    "timestamp": datetime.now(),
                    "counter_id": "Counter 1",
                    "people_in_queue": len(current_ids),
                    "arrivals": arrivals,
                    "served": served,
                    "avg_service_time": round(
                        avg_service_time,
                        2,
                    ),
                }

                with self.lock:
                    self.latest_row = row

            # ==================================================
            # ALWAYS draw ROI + latest detection state.
            # This is the anti-blinking fix.
            # ==================================================
            annotated = self._draw_overlay(
                img,
                zone,
                self.last_boxes,
                self.last_centroids,
                self.last_objects,
            )

            return av.VideoFrame.from_ndarray(
                annotated,
                format="bgr24",
            )


    @st.cache_resource
    def get_browser_processor():
        return BrowserCameraProcessor(
            CAMERA_ZONE
        )


    processor = get_browser_processor()


# ============================================================
# HEADER
# ============================================================
st.title("VisionAI Queue Manager")

st.markdown(
    "Real-time computer vision queue tracking, forecasting, "
    "and automated load balancing."
)


# ============================================================
# TOP METRICS
# ============================================================
total_waiting = (
    state_df["people_in_queue"].sum()
    if not state_df.empty
    else 0
)

max_wait = (
    state_df["estimated_wait_min"].max()
    if not state_df.empty
    else 0
)

busiest_counter = (
    state_df.loc[
        state_df[
            "estimated_wait_min"
        ].idxmax()
    ]["counter_id"]
    if not state_df.empty
    else "N/A"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total People Waiting", total_waiting)
col2.metric("Max Wait Time", f"{max_wait:.1f} min")
col3.metric("Busiest Node", busiest_counter)
col4.metric(
    "System Status",
    "CRITICAL" if total_waiting > CROWD_THRESHOLD else "OPTIMAL",
)

st.markdown("---")


# ============================================================
# LIVE CAMERA + STATE AREA
# ============================================================
c_left, c_right = st.columns(
    [1.2, 1],
    gap="large",
)


# ------------------------------------------------------------
# LEFT: CAMERA
# ------------------------------------------------------------
with c_left:

    st.subheader("📷 Camera")

    if data_mode == "Live Camera":

        st.caption(
            "Browser camera • YOLO person detection • Queue ROI tracking"
        )

        rtc_ctx = webrtc_streamer(
            key="visionai-live-camera-v2",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=processor.process,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                    "frameRate": {"ideal": 20},
                },
                "audio": False,
            },
            async_processing=True,
            rtc_configuration={
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302"
                        ]
                    }
                ]
            },
        )

        if rtc_ctx.state.playing:
            st.success(
                "🟢 Camera connected — YOLO detection is running"
            )
        else:
            st.info(
                "Click **START** above and allow camera permission."
            )

    else:
        st.info(
            "Visual intelligence disabled in Simulation Mode. "
            "Switch to Live Camera in the sidebar."
        )


# ------------------------------------------------------------
# RIGHT: STATE / ACTIONS
# ------------------------------------------------------------
with c_right:

    st.subheader("Live State Estimation")

    # Get the latest live sample.
    if data_mode == "Live Camera":
        with processor.lock:
            live_row = dict(
                processor.latest_row
            )

        live_df = pd.DataFrame(
            [live_row]
        )

        # Append only a new timestamp.
        if st.session_state.history_df.empty:
            st.session_state.history_df = live_df
        else:
            last_ts = (
                st.session_state.history_df[
                    "timestamp"
                ].iloc[-1]
            )

            if live_row["timestamp"] != last_ts:
                st.session_state.history_df = pd.concat(
                    [
                        st.session_state.history_df,
                        live_df,
                    ],
                    ignore_index=True,
                )

        if len(
            st.session_state.history_df
        ) > 500:
            st.session_state.history_df = (
                st.session_state.history_df.iloc[-500:]
            )

        history = st.session_state.history_df

        csv_data = history.to_csv(
            index=False
        ).encode("utf-8")

        st.sidebar.download_button(
            label="📥 Download Shift Audit Report",
            data=csv_data,
            file_name="queue_performance_report.csv",
            mime="text/csv",
        )

    else:
        history = history


    # ========================================================
    # ANALYTICS
    # ========================================================
    state_df = compute_counter_states(
        history
    )

    forecasts = forecast_all_counters(
        history,
        steps=FORECAST_STEPS,
    )

    forecast_alerts = {
        c: threshold_alert(
            f,
            CROWD_THRESHOLD,
        )
        for c, f in forecasts.items()
    }

    recommendations = generate_recommendations(
        state_df,
        forecast_alerts,
    )


    display_df = state_df.rename(
        columns={
            "counter_id": "Node",
            "people_in_queue": "Queue Len",
            "arrival_rate_per_min": "Arrivals/m",
            "avg_service_time_min": "Service(m)",
            "estimated_wait_min": "Est. Wait(m)",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # ACTIONS
    # ========================================================
    st.subheader("Actions needs to be perform")

    current_time = datetime.now()

    should_speak = (
        (
            current_time
            - st.session_state.last_audio_time
        ).total_seconds()
        > AUDIO_COOLDOWN_SECONDS
    )

    for severity, msg in recommendations:

        if severity == "high":

            st.error(
                f"ACTION REQUIRED: {msg}"
            )

            if (
                should_speak
                and announcer is not None
            ):
                clean_msg = (
                    msg.split(" — ")[0]
                    .replace(
                        "min",
                        "minutes",
                    )
                )

                try:
                    announcer.announce(
                        f"Attention please. {clean_msg}"
                    )
                except Exception as e:
                    print(
                        f"Audio announcement skipped: {e}"
                    )

                st.session_state.last_audio_time = (
                    current_time
                )

                should_speak = False

        elif severity == "medium":
            st.warning(
                f"⚠️ {msg}"
            )

        elif severity == "warning":
            st.info(
                f"⏳ {msg}"
            )

        else:
            st.success(
                f"✅ {msg}"
            )


# FORECAST
# ============================================================
st.markdown("---")

st.subheader(
    "📈 Queue Trajectory & AI Forecast"
)

fig = go.Figure()

colors = {
    "Counter 1": "#00F0FF",
    "Counter 2": "#FF0055",
    "Counter 3": "#00FF66",
}

for c in COUNTERS:

    grp = (
        history[
            history["counter_id"] == c
        ]
        .sort_values("timestamp")
    )

    fig.add_trace(
        go.Scatter(
            x=grp["timestamp"],
            y=grp["people_in_queue"],
            mode="lines",
            name=f"{c} (Actual)",
            line=dict(
                color=colors.get(
                    c,
                    "#FFF",
                ),
                width=2,
            ),
        )
    )

    if c in forecasts:

        last_ts = (
            grp["timestamp"].iloc[-1]
            if not grp.empty
            else pd.Timestamp.now()
        )

        future_ts = [
            last_ts
            + pd.Timedelta(
                minutes=i
            )
            for i in range(
                1,
                FORECAST_STEPS + 1,
            )
        ]

        fig.add_trace(
            go.Scatter(
                x=future_ts,
                y=forecasts[c],
                mode="lines",
                name=f"{c} (Forecast)",
                line=dict(
                    color=colors.get(
                        c,
                        "#FFF",
                    ),
                    dash="dot",
                    width=2,
                ),
            )
        )


fig.add_hline(
    y=CROWD_THRESHOLD,
    line_dash="dash",
    line_color="red",
    annotation_text=(
        f"Critical Threshold "
        f"({CROWD_THRESHOLD})"
    ),
    annotation_position="top right",
)

fig.update_layout(
    height=400,
    xaxis_title="Timeline",
    yaxis_title="People in Queue",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
    ),
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# DIGITAL SIGNAGE STATE
# ============================================================
signage_data = {
    "status": "NORMAL",
    "message": "✅ PLEASE WAIT IN LINE",
    "color": "#0a2911",
}

stuck_counters = state_df[
    state_df["avg_service_time_min"] > 5.0
]

if not stuck_counters.empty:

    signage_data["message"] = (
        "⚡ EXPRESS LANE OPEN AT "
        "COUNTER 2 (Max 2 Items)"
    )

    signage_data["color"] = "#b58900"

else:

    for severity, msg in recommendations:

        if severity == "high":

            signage_data["status"] = "FULL"
            signage_data["message"] = (
                f"🚨 {msg.upper()}"
            )
            signage_data["color"] = "#4a0404"
            break


try:
    with open(
        "shared_state.json",
        "w",
    ) as f:
        json.dump(
            signage_data,
            f,
        )
except Exception:
    pass


if auto_refresh:

    if data_mode == "Simulation":
        time.sleep(refresh_secs)
        st.rerun()

    elif data_mode == "Live Camera":
        pass
