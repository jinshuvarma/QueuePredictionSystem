
import json
import time
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

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
    </style>
    """,
    unsafe_allow_html=True,
)


COUNTERS = ["Counter 1", "Counter 2", "Counter 3"]
FORECAST_STEPS = 20
CROWD_THRESHOLD = 30
AUDIO_COOLDOWN_SECONDS = 20
LIVE_ALERT_SUSTAIN_SECONDS = 5


                                                              
            
                                                              
@st.cache_resource
def get_announcer():
    try:
        return AudioAnnouncer()
    except Exception as e:
        print(f"Audio announcer disabled: {e}")
        return None


announcer = get_announcer()

def browser_speak(message):
    safe_message = json.dumps(message)
    components.html(
        f"""
        <script>
        (() => {{
            const text = {safe_message};
            const speak = () => {{
                try {{
                    if (!window.speechSynthesis) return false;
                    window.speechSynthesis.cancel();
                    const utterance =
                        new SpeechSynthesisUtterance(text);
                    utterance.lang = "en-IN";
                    utterance.rate = 0.92;
                    utterance.pitch = 1.0;
                    utterance.volume = 1.0;
                    window.speechSynthesis.speak(utterance);
                    return true;
                }} catch (e) {{
                    console.log(e);
                    return false;
                }}
            }};
            speak();
        }})();
        </script>
        """,
        height=0,
    )

if "last_audio_time" not in st.session_state:
    st.session_state.last_audio_time = datetime.min

if "browser_last_spoken" not in st.session_state:
    st.session_state.browser_last_spoken = ""

if "vision_started_announced" not in st.session_state:
    st.session_state.vision_started_announced = False

if "vision_stopped_announced" not in st.session_state:
    st.session_state.vision_stopped_announced = False


                                                              
         
                                                              
st.sidebar.title("System Controls")

data_mode = st.sidebar.radio(
    "Operation Mode",
    ["Live Camera", "Simulation"],
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

                                                                      
    st.session_state.browser_last_spoken = ""
    st.session_state.vision_started_announced = False
    st.session_state.vision_stopped_announced = False
    st.session_state.live_alert_since = None
    st.session_state.live_alert_active = False
    st.session_state.last_audio_time = datetime.min

    st.session_state.current_mode = data_mode
    st.rerun()

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
    def __init__(
        self,
        max_disappeared=4,
        max_distance=110,
    ):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = tuple(
            centroid
        )
        self.disappeared[
            self.next_object_id
        ] = 0

        self.next_object_id += 1

    def deregister(self, object_id):
        self.objects.pop(
            object_id,
            None,
        )

        self.disappeared.pop(
            object_id,
            None,
        )

    def update(self, input_centroids):
        if len(input_centroids) == 0:

            for object_id in list(
                self.disappeared
            ):
                self.disappeared[
                    object_id
                ] += 1

                if (
                    self.disappeared[
                        object_id
                    ]
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

        object_ids = list(
            self.objects.keys()
        )

        object_centroids = np.asarray(
            [
                self.objects[i]
                for i in object_ids
            ],
            dtype=np.float32,
        )

        distances = np.linalg.norm(
            object_centroids[:, None, :]
            - input_centroids[None, :, :],
            axis=2,
        )

        rows = distances.min(
            axis=1
        ).argsort()

        cols = distances.argmin(
            axis=1
        )[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(
            rows,
            cols,
        ):
            if (
                row in used_rows
                or col in used_cols
            ):
                continue

            if (
                distances[row, col]
                > self.max_distance
            ):
                continue

            object_id = object_ids[row]

            self.objects[
                object_id
            ] = tuple(
                input_centroids[col]
            )

            self.disappeared[
                object_id
            ] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = (
            set(range(len(object_ids)))
            - used_rows
        )

        unused_cols = (
            set(range(len(input_centroids)))
            - used_cols
        )

        for row in unused_rows:

            object_id = object_ids[row]

            self.disappeared[
                object_id
            ] += 1

            if (
                self.disappeared[
                    object_id
                ]
                > self.max_disappeared
            ):
                self.deregister(object_id)

        for col in unused_cols:
            self.register(
                input_centroids[col]
            )

        return dict(self.objects)


                                                              
                       
                                                              
CAMERA_ZONE = [
    (121, 472),
    (121, 4),
    (393, 4),
    (393, 472),
    (121, 472),
]


class BrowserCameraProcessor:
    """Browser camera -> YOLO person detection -> queue ROI -> live state."""

    def __init__(self, zone):
        self.zone = zone
        self.model = YOLO("yolov8n.pt")
        self.tracker = CentroidTracker(max_disappeared=4, max_distance=110)

        self.entry_ts = {}
        self.seen_ids = set()
        self.service_times = []
        self.candidate_hits = {}
        self.confirmed_ids = set()
        self.confirmed_boxes = {}

        self.latest_row = {
            "timestamp": datetime.now(),
            "counter_id": "Counter 1",
            "people_in_queue": 0,
            "arrivals": 0,
            "served": 0,
            "avg_service_time": 2.5,
        }

        self.lock = threading.Lock()
        self.latest_input = None
        self.latest_output = None
        self.last_boxes = []
        self.last_centroids = []
        self.last_objects = {}
        self.last_ids = []
        self.frame_counter = 0
        self.inference_every_n_frames = 1
        self.worker_running = True
        self.worker_busy = False

        self.worker = threading.Thread(
            target=self._inference_worker,
            daemon=True,
        )
        self.worker.start()

    def _scaled_zone(self, width, height):
        sx = width / 514.0
        sy = height / 475.0
        return [(int(x * sx), int(y * sy)) for x, y in self.zone]

    def _draw_overlay(self, img, zone, boxes, centroids, objects, track_ids=None):
        import cv2

        roi_points = np.array(zone, dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [roi_points], (0, 255, 255))
        cv2.addWeighted(overlay, 0.10, img, 0.90, 0, img)
        cv2.polylines(img, [roi_points], True, (0, 255, 255), 3)

        if track_ids is None:
            track_ids = [None] * len(boxes)

        for box, footpoint, track_id in zip(boxes, centroids, track_ids):
            # Boxes shown here are ONLY confirmed humans standing in queue ROI.
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = (
                f"HUMAN #{track_id}"
                if track_id is not None
                else "HUMAN IN QUEUE"
            )
            cv2.putText(
                img, label, (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2,
            )
            cv2.circle(img, (int(footpoint[0]), int(footpoint[1])), 5, (0, 255, 0), -1)

        cv2.putText(
            img, f"QUEUE HUMANS: {len(centroids)}", (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
        )
        cv2.putText(
            img, "QUEUE ROI", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2,
        )
        return img

    def _inference_worker(self):
        import cv2

        while self.worker_running:
            frame = None
            with self.lock:
                if self.latest_input is not None:
                    frame = self.latest_input
                    self.latest_input = None
                    self.worker_busy = True

            if frame is None:
                time.sleep(0.005)
                continue

            try:
                frame_h, frame_w = frame.shape[:2]
                zone = self._scaled_zone(frame_w, frame_h)

                # Detect ONLY COCO class 0 (person).
                result = self.model(
                    frame,
                    classes=[0],
                    conf=0.40,
                    imgsz=480,
                    max_det=20,
                    verbose=False,
                )[0]

                queue_boxes = []
                queue_footpoints = []
                frame_area = float(frame_h * frame_w)

                if result.boxes is not None and len(result.boxes) > 0:
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    cls_ids = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                else:
                    xyxy, cls_ids, confidences = [], [], []

                for box, cls_id, confidence in zip(xyxy, cls_ids, confidences):
                    # Hard guard: never treat non-person classes as humans.
                    if int(cls_id) != 0:
                        continue
                    if float(confidence) < 0.40:
                        continue

                    x1, y1, x2, y2 = map(float, box)
                    box_w = x2 - x1
                    box_h = y2 - y1
                    if box_w <= 0 or box_h <= 0:
                        continue

                    area_ratio = (box_w * box_h) / max(frame_area, 1.0)
                    aspect_ratio = box_w / max(box_h, 1.0)

                    # Loose geometry filters: remove obvious non-human artifacts
                    # without rejecting small/distant people.
                    if box_h < 22:
                        continue
                    if area_ratio < 0.0008 or area_ratio > 0.75:
                        continue
                    if aspect_ratio < 0.12 or aspect_ratio > 1.65:
                        continue

                    # IMPORTANT: queue membership is based on the person's
                    # standing/foot position, NOT the bbox center.
                    foot_x = (x1 + x2) / 2.0
                    foot_y = y2
                    if not point_in_polygon((foot_x, foot_y), zone):
                        continue

                    queue_footpoints.append((foot_x, foot_y))
                    queue_boxes.append((
                        int(max(0, x1)),
                        int(max(0, y1)),
                        int(min(frame_w - 1, x2)),
                        int(min(frame_h - 1, y2)),
                    ))

                # Tracker receives ONLY queue candidates. Therefore an object
                # outside the ROI can never become a queue member.
                objects = self.tracker.update(queue_footpoints)

                # IMPORTANT: tracker memory is allowed to keep an ID stable
                # for a few missed frames, but only IDs that have a detection
                # in THIS frame may count as currently waiting.
                visible_ids = set()
                unused_objects = set(objects.keys())
                for footpoint in queue_footpoints:
                    best_id = None
                    best_distance = float("inf")
                    for object_id in unused_objects:
                        distance = (
                            (objects[object_id][0] - footpoint[0]) ** 2
                            + (objects[object_id][1] - footpoint[1]) ** 2
                        ) ** 0.5
                        if distance < best_distance:
                            best_distance = distance
                            best_id = object_id
                    if best_id is not None and best_distance <= 110:
                        visible_ids.add(best_id)
                        unused_objects.discard(best_id)

                # Temporal confirmation prevents one-frame false positives.
                for object_id in visible_ids:
                    self.candidate_hits[object_id] = self.candidate_hits.get(object_id, 0) + 1
                    if self.candidate_hits[object_id] >= 3:
                        self.confirmed_ids.add(object_id)

                for object_id in list(self.candidate_hits.keys()):
                    if object_id not in visible_ids:
                        self.candidate_hits[object_id] -= 1
                        if self.candidate_hits[object_id] <= 0:
                            self.candidate_hits.pop(object_id, None)
                            self.confirmed_ids.discard(object_id)
                            self.entry_ts.pop(object_id, None)

                confirmed_ids_now = visible_ids & self.confirmed_ids
                current_ids = set(confirmed_ids_now)
                now = time.time()

                for object_id in current_ids:
                    if object_id not in self.entry_ts:
                        self.entry_ts[object_id] = now

                new_ids = current_ids - self.seen_ids
                left_ids = self.seen_ids - current_ids
                arrivals = len(new_ids)
                served = 0

                for object_id in left_ids:
                    started = self.entry_ts.pop(object_id, None)
                    if started is not None:
                        dwell_min = (now - started) / 60.0
                        if dwell_min >= 0.25:
                            served += 1
                            self.service_times.append(dwell_min)

                self.seen_ids = current_ids
                recent_service = self.service_times[-20:]
                avg_service_time = (
                    sum(recent_service) / len(recent_service)
                    if recent_service else 2.5
                )

                # Match confirmed tracker IDs back to their current YOLO boxes.
                confirmed_box_map = {}
                for object_id in confirmed_ids_now:
                    object_centroid = objects[object_id]
                    best_box = None
                    best_distance = float("inf")
                    for box, footpoint in zip(queue_boxes, queue_footpoints):
                        distance = (
                            (object_centroid[0] - footpoint[0]) ** 2
                            + (object_centroid[1] - footpoint[1]) ** 2
                        ) ** 0.5
                        if distance < best_distance:
                            best_distance = distance
                            best_box = box
                    if best_box is not None and best_distance <= 90:
                        confirmed_box_map[object_id] = best_box

                draw_boxes = []
                draw_points = []
                draw_ids = []
                for object_id in sorted(confirmed_ids_now):
                    box = confirmed_box_map.get(object_id)
                    if box is not None:
                        draw_boxes.append(box)
                        draw_points.append(objects[object_id])
                        draw_ids.append(object_id)

                # The row is the authoritative REAL-TIME queue state.
                row = {
                    "timestamp": datetime.now(),
                    "counter_id": "Counter 1",
                    "people_in_queue": len(current_ids),
                    "arrivals": arrivals,
                    "served": served,
                    "avg_service_time": round(avg_service_time, 2),
                }

                annotated = self._draw_overlay(
                    frame.copy(), zone, draw_boxes, draw_points, objects, draw_ids
                )

                with self.lock:
                    self.last_boxes = draw_boxes
                    self.last_centro    )

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
            import cv2
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
    if st.session_state.history_df.empty:
        st.session_state.history_df = live_df
    else:
        last_ts = st.session_state.history_df["timestamp"].iloc[-1]
        if live_row["timestamp"] != last_ts:
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
            preview_rgb = preview[:, :, ::-1]
            st.image(Image.fromarray(preview_rgb), caption="YOLO Live Detection", use_container_width=True)
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
