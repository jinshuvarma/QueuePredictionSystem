
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

if "last_audio_time" not in st.session_state:
    st.session_state.last_audio_time = datetime.min

if "browser_last_spoken" not in st.session_state:
    st.session_state.browser_last_spoken = ""

st.sidebar.title("System Controls")

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
    st.session_state.browser_last_spoken = ""
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
        max_disappeared=12,
        max_distance=140,
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


# ============================================================
# LIVE CAMERA PROCESSOR
# ============================================================
CAMERA_ZONE = [
    (121, 472),
    (121, 4),
    (393, 4),
    (393, 472),
    (121, 472),
]


class BrowserCameraProcessor:

    def __init__(self, zone):

        self.zone = zone

        # Model is created once per cached processor.
        self.model = YOLO(
            "yolov8n.pt"
        )

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

        # Latest input/output frames.
        self.latest_input = None
        self.latest_output = None

        self.candidate_hits = {}
        self.confirmed_ids = set()
        self.confirmed_boxes = {}

        self.frame_counter = 0

        # Do not let inference queue build up.
        self.inference_every_n_frames = 1

        self.worker_running = True
        self.worker_busy = False

        self.worker = threading.Thread(
            target=self._inference_worker,
            daemon=True,
        )

        self.worker.start()

    def _scaled_zone(
        self,
        width,
        height,
    ):
        sx = width / 514.0
        sy = height / 475.0

        return [
            (
                int(x * sx),
                int(y * sy),
            )
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
        import cv2

        roi_points = np.array(
            zone,
            dtype=np.int32,
        )

        # ROI is ALWAYS drawn. This removes blinking.
        overlay = img.copy()

        cv2.fillPoly(
            overlay,
            [roi_points],
            (0, 255, 255),
        )

        cv2.addWeighted(
            overlay,
            0.10,
            img,
            0.90,
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

        for box, centroid in zip(
            boxes,
            centroids,
        ):

            if not point_in_polygon(
                tuple(centroid),
                zone,
            ):
                continue

            x1, y1, x2, y2 = box

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3,
            )

            cv2.putText(
                img,
                "HUMAN",
                (
                    x1,
                    max(24, y1 - 8),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            img,
            f"Humans in ROI: {len(centroids)}",
            (
                20,
                65,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

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

                h, w = frame.shape[:2]

                zone = self._scaled_zone(
                    w,
                    h,
                )

                results = self.model(
                    frame,
                    classes=[0],
                    conf=0.35,
                    imgsz=480,
                    max_det=20,
                    verbose=False,
                )[0]

                boxes = []
                centroids = []

                if results.boxes is not None:
                    xyxy = results.boxes.xyxy.cpu().numpy()
                    cls_ids = results.boxes.cls.cpu().numpy()
                    confidences = results.boxes.conf.cpu().numpy()
                else:
                    xyxy = []
                    cls_ids = []
                    confidences = []

                frame_h, frame_w = frame.shape[:2]
                frame_area = float(
                    frame_h * frame_w
                )

                for box, cls_id, confidence in zip(
                    xyxy,
                    cls_ids,
                    confidences,
                ):

                    if int(cls_id) != 0:
                        continue

                    confidence = float(confidence)

                    if confidence < 0.35:
                        continue

                    x1, y1, x2, y2 = map(
                        float,
                        box,
                    )

                    box_w = x2 - x1
                    box_h = y2 - y1

                    if box_w <= 0 or box_h <= 0:
                        continue

                    area_ratio = (
                        box_w * box_h
                    ) / max(
                        frame_area,
                        1.0,
                    )

                    aspect_ratio = (
                        box_w
                        / max(
                            box_h,
                            1.0,
                        )
                    )


                    if box_h < 24:
                        continue

                    if area_ratio < 0.001:
                        continue

                    if area_ratio > 0.90:
                        continue

                    if (
                        aspect_ratio < 0.10
                        or aspect_ratio > 2.0
                    ):
                        continue

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    if not (
                        0 <= cx < frame_w
                        and 0 <= cy < frame_h
                    ):
                        continue

                    centroids.append(
                        (cx, cy)
                    )

                    boxes.append(
                        (
                            int(max(0, x1)),
                            int(max(0, y1)),
                            int(min(frame_w - 1, x2)),
                            int(min(frame_h - 1, y2)),
                        )
                    )

                objects = self.tracker.update(
                    centroids
                )

                visible_ids = set(
                    objects.keys()
                )

                for object_id in visible_ids:

                    self.candidate_hits[
                        object_id
                    ] = (
                        self.candidate_hits.get(
                            object_id,
                            0,
                        )
                        + 1
                    )

                    if (
                        self.candidate_hits[
                            object_id
                        ]
                        >= 2
                    ):
                        self.confirmed_ids.add(
                            object_id
                        )

                for object_id in list(
                    self.candidate_hits.keys()
                ):

                    if object_id not in visible_ids:

                        self.candidate_hits[
                            object_id
                        ] -= 1

                        if (
                            self.candidate_hits[
                                object_id
                            ]
                            <= 0
                        ):
                            self.candidate_hits.pop(
                                object_id,
                                None,
                            )

                            self.confirmed_ids.discard(
                                object_id
                            )

                confirmed_objects = {
                    object_id: objects[object_id]
                    for object_id in (
                        visible_ids
                        & self.confirmed_ids
                    )
                }

                current_ids = set()
                now = time.time()

                for object_id, centroid in (
                    confirmed_objects.items()
                ):

                    if point_in_polygon(
                        tuple(centroid),
                        zone,
                    ):

                        current_ids.add(
                            object_id
                        )

                        if (
                            object_id
                            not in self.entry_ts
                        ):

                            self.entry_ts[
                                object_id
                            ] = now

                new_ids = (
                    current_ids
                    - self.seen_ids
                )

                left_ids = (
                    self.seen_ids
                    - current_ids
                )

                arrivals = len(
                    new_ids
                )

                served = 0

                for object_id in left_ids:

                    if object_id in self.entry_ts:

                        dwell_min = (
                            now
                            - self.entry_ts.pop(
                                object_id
                            )
                        ) / 60.0

                        if dwell_min >= 0.25:

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

                confirmed_box_map = {}

                for object_id, object_centroid in (
                    confirmed_objects.items()
                ):
                    best_box = None
                    best_distance = float("inf")

                    for box, detected_centroid in zip(
                        boxes,
                        centroids,
                    ):
                        distance = (
                            (
                                object_centroid[0]
                                - detected_centroid[0]
                            ) ** 2
                            + (
                                object_centroid[1]
                                - detected_centroid[1]
                            ) ** 2
                        ) ** 0.5

                        if distance < best_distance:
                            best_distance = distance
                            best_box = box

                    if (
                        best_box is not None
                        and best_distance < 140
                    ):
                        confirmed_box_map[object_id] = best_box

                self.confirmed_boxes = confirmed_box_map

                boxes = []
                centroids = []

                for object_id, object_centroid in (
                    confirmed_objects.items()
                ):
                    if not point_in_polygon(
                        tuple(object_centroid),
                        zone,
                    ):
                        continue

                    if object_id not in confirmed_box_map:
                        continue

                    boxes.append(
                        confirmed_box_map[object_id]
                    )
                    centroids.append(
                        object_centroid
                    )

                annotated = self._draw_overlay(
                    frame.copy(),
                    zone,
                    boxes,
                    centroids,
                    objects,
                )

                row = {
                    "timestamp": datetime.now(),
                    "counter_id": "Counter 1",
                    "people_in_queue": len(
                        current_ids
                    ),
                    "arrivals": arrivals,
                    "served": served,
                    "avg_service_time": round(
                        avg_service_time,
                        2,
                    ),
                }

                with self.lock:

                    self.last_boxes = boxes
                    self.last_centroids = centroids
                    self.last_objects = objects

                    self.latest_row = row

                    self.latest_output = (
                        annotated
                    )

            except Exception as e:

                print(
                    f"YOLO worker error: {e}"
                )

            finally:

                with self.lock:
                    self.worker_busy = False

    def process(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        self.frame_counter += 1


        if (
            self.frame_counter
            % self.inference_every_n_frames
            == 0
        ):

            with self.lock:

                self.latest_input = img.copy()

        with self.lock:

            latest_output = (
                None
                if self.latest_output is None
                else self.latest_output.copy()
            )

            boxes = getattr(
                self,
                "last_boxes",
                [],
            )

            centroids = getattr(
                self,
                "last_centroids",
                [],
            )

            objects = getattr(
                self,
                "last_objects",
                {},
            )

        if latest_output is not None:
            h, w = img.shape[:2]
            zone = self._scaled_zone(
                w,
                h,
            )

            annotated = self._draw_overlay(
                img,
                zone,
                boxes,
                centroids,
                objects,
            )

            return av.VideoFrame.from_ndarray(
                annotated,
                format="bgr24",
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24",
        )


@st.cache_resource
def get_browser_processor():
    return BrowserCameraProcessor(
        CAMERA_ZONE
    )

st.title(
    "VisionAI Queue Manager"
)

st.markdown(
    "Real-time computer vision queue tracking, "
    "forecasting, and automated load balancing."
)

def calculate_analytics(history):
    state_df = compute_counter_states(
        history
    )

    forecasts = forecast_all_counters(
        history,
        steps=FORECAST_STEPS,
    )

    forecast_alerts = {
        counter: threshold_alert(
            forecast,
            CROWD_THRESHOLD,
        )
        for counter, forecast
        in forecasts.items()
    }

    recommendations = (
        generate_recommendations(
            state_df,
            forecast_alerts,
        )
    )

    return (
        state_df,
        forecasts,
        recommendations,
    )


def render_metrics(state_df):
    col1, col2, col3, col4 = (
        st.columns(4)
    )

    total_waiting = (
        state_df[
            "people_in_queue"
        ].sum()
        if not state_df.empty
        else 0
    )

    max_wait = (
        state_df[
            "estimated_wait_min"
        ].max()
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

    col1.metric(
        "Total People Waiting",
        total_waiting,
    )

    col2.metric(
        "Max Wait Time",
        f"{max_wait:.1f} min",
    )

    col3.metric(
        "Busiest Node",
        busiest_counter,
    )

    col4.metric(
        "System Status",
        (
            "CRITICAL"
            if total_waiting
            > CROWD_THRESHOLD
            else "OPTIMAL"
        ),
    )


def render_state_and_actions(
    state_df,
    recommendations,
    prefix,
    voice_enabled=False,
    actual_queue_count=None,
    require_real_queue_threshold=False,
):
    st.subheader("Live State Estimation")

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

    st.subheader("Actions needs to be perform")

    high_alert = None

    for severity, msg in recommendations:

        if severity == "high":
            st.error(
                f"ACTION REQUIRED: {msg}"
            )
            if high_alert is None:
                high_alert = msg

        elif severity == "medium":
            st.warning(
                f"{msg}"
            )

        elif severity == "warning":
            st.info(
                f"{msg}"
            )

        else:
            st.success(
                f"{msg}"
            )

    if voice_enabled and high_alert:

        now = time.time()

        if require_real_queue_threshold:
            alert_condition = (
                actual_queue_count is not None
                and actual_queue_count >= CROWD_THRESHOLD
            )
        else:
            alert_condition = True

        if alert_condition:

            if st.session_state.live_alert_since is None:
                st.session_state.live_alert_since = now

            sustained_for = (
                now
                - st.session_state.live_alert_since
            )

            if sustained_for >= LIVE_ALERT_SUSTAIN_SECONDS:

                clean_msg = (
                    high_alert
                    .split(" — ")[0]
                    .replace(
                        "min",
                        "minutes",
                    )
                )

                last_spoken_time = (
                    st.session_state.last_audio_time.timestamp()
                    if st.session_state.last_audio_time != datetime.min
                    else 0
                )

                cooldown_ok = (
                    now - last_spoken_time
                    >= AUDIO_COOLDOWN_SECONDS
                )

                # Same alert is not repeated until cooldown expires.
                if (
                    (
                        clean_msg
                        != st.session_state.browser_last_spoken
                    )
                    or cooldown_ok
                ):

                    components.html(
                        f"""
                        <script>
                        (() => {{
                            const text = {json.dumps(
                                "Attention please. "
                                + clean_msg
                            )};

                            try {{
                                window.speechSynthesis.cancel();
                                const u =
                                    new SpeechSynthesisUtterance(text);
                                u.rate = 0.95;
                                u.pitch = 1.0;
                                u.volume = 1.0;
                                window.speechSynthesis.speak(u);
                            }} catch (e) {{
                                console.log(e);
                            }}
                        }})();
                        </script>
                        """,
                        height=1,
                    )

                    st.session_state.browser_last_spoken = clean_msg
                    st.session_state.last_audio_time = datetime.now()

        else:
            st.session_state.live_alert_since = None

    else:
        st.session_state.live_alert_since = None
        st.session_state.live_alert_active = False
        st.session_state.browser_last_spoken = ""

    with st.expander(
        "Voice Alert",
        expanded=False,
    ):
        if st.button(
            "Test Voice",
            key=f"{prefix}_voice_test",
        ):
            components.html(
                """
                <script>
                window.speechSynthesis.cancel();

                const u =
                    new SpeechSynthesisUtterance(
                        "VisionAI voice alert system is working."
                    );

                u.rate = 0.95;
                u.volume = 1.0;
                window.speechSynthesis.speak(u);
                </script>
                """,
                height=1,
            )



def build_forecast_chart(
    history,
    forecasts,
):

    fig = go.Figure()

    colors = {
        "Counter 1": "#00F0FF",
        "Counter 2": "#FF0055",
        "Counter 3": "#00FF66",
    }

    for counter in COUNTERS:

        group = (
            history[
                history[
                    "counter_id"
                ]
                == counter
            ]
            .sort_values(
                "timestamp"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=group[
                    "timestamp"
                ],
                y=group[
                    "people_in_queue"
                ],
                mode="lines+markers",
                name=f"{counter} (Actual)",
                marker=dict(
                    size=5,
                ),
                line=dict(
                    color=colors.get(
                        counter,
                        "#FFF",
                    ),
                    width=2,
                ),
            )
        )

        if counter in forecasts:

            last_ts = (
                group[
                    "timestamp"
                ].iloc[-1]
                if not group.empty
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
                    y=forecasts[
                        counter
                    ],
                    mode="lines",
                    name=f"{counter} (Forecast)",
                    line=dict(
                        color=colors.get(
                            counter,
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

    if history.empty or len(history) < 3:
        fig.add_annotation(
            text="Collecting live data for AI forecast…",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.92,
            showarrow=False,
            font=dict(
                size=12,
            ),
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

    return fig

if data_mode == "Live Camera":

    processor = get_browser_processor()

    st.caption(
        "Click START and allow camera permission."
    )

    c_left, c_right = st.columns(
        [1.2, 1],
        gap="large",
    )

    with c_left:

        st.subheader(
            "Camera"
        )

        rtc_ctx = webrtc_streamer(
            key="visionai-live-camera-v4",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=processor.process,
            media_stream_constraints={
                "video": {
                    "width": {
                        "ideal": 640,
                        "max": 640,
                    },
                    "height": {
                        "ideal": 360,
                        "max": 360,
                    },
                    "frameRate": {
                        "ideal": 15,
                        "max": 15,
                    },
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
                "Camera connected — YOLO detection is running"
            )

        else:

            st.caption(
                "Camera is waiting. Click START and allow access."
            )

    with c_right:

        # Initial placeholder. The actual numbers are refreshed below.
        live_state_placeholder = st.empty()
        live_actions_placeholder = st.empty()

    st.markdown("---")

    live_metrics_placeholder = st.empty()
    live_forecast_placeholder = st.empty()

    @st.fragment(
        run_every=(
            refresh_secs
            if auto_refresh
            else None
        )
    )
    def live_dashboard():
        camera_is_on = bool(
            rtc_ctx.state.playing
        )

        if not camera_is_on:
            with processor.lock:
                live_row = {
                    "timestamp": datetime.now(),
                    "counter_id": "Counter 1",
                    "people_in_queue": 0,
                    "arrivals": 0,
                    "served": 0,
                    "avg_service_time": 2.5,
                }

                processor.latest_row = dict(live_row)
                processor.latest_output = None
                processor.last_boxes = []
                processor.last_centroids = []
                processor.last_objects = {}
                processor.confirmed_ids.clear()
                processor.candidate_hits.clear()
                processor.seen_ids.clear()
                processor.entry_ts.clear()

            st.session_state.history_df = pd.DataFrame(
                [live_row]
            )

        else:
            with processor.lock:
                live_row = dict(
                    processor.latest_row
                )

            live_df = pd.DataFrame(
                [live_row]
            ),
            if (
                st.session_state.history_df.empty
                or (
                    len(st.session_state.history_df) == 1
                    and float(
                        st.session_state.history_df.iloc[0][
                            "people_in_queue"
                        ]
                    ) == 0
                    and float(
                        st.session_state.history_df.iloc[0][
                            "arrivals"
                        ]
                    ) == 0
                )
            ):
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
                st.session_state.history_df.iloc[
                    -500:
                ]
            )

        history = (
            st.session_state.history_df
        )

        if not camera_is_on:
            state_df = pd.DataFrame(
                [{
                    "counter_id": "Counter 1",
                    "people_in_queue": 0,
                    "arrival_rate_per_min": 0.0,
                    "avg_service_time_min": 2.5,
                    "estimated_wait_min": 0.0,
                }]
            )
            forecasts = {
                "Counter 1": []
            }
            recommendations = [
                (
                    "normal",
                    "Camera is off. Start the camera to begin live queue detection.",
                )
            ]
        else:
            state_df, forecasts, recommendations = (
                calculate_analytics(
                    history
                )
            )

        with live_metrics_placeholder.container():
            render_metrics(state_df)

        with live_state_placeholder.container():
            actual_queue_count = int(
                state_df["people_in_queue"].sum()
                if not state_df.empty
                else 0
            )

            render_state_and_actions(
                state_df,
                recommendations,
                prefix="live",
                voice_enabled=True,
                actual_queue_count=actual_queue_count,
                require_real_queue_threshold=True,
            )

        with live_forecast_placeholder.container():

            st.subheader(
                "Queue Trajectory & AI Forecast"
            )

            fig = build_forecast_chart(
                history,
                forecasts,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                },
            )

        signage_data = {
            "status": "NORMAL",
            "message": "PLEASE WAIT IN LINE",
            "color": "#0a2911",
        }

        stuck_counters = state_df[
            state_df[
                "avg_service_time_min"
            ]
            > 5.0
        ]

        if not stuck_counters.empty:

            signage_data["message"] = (
                "EXPRESS LANE OPEN AT "
                "COUNTER 2 (Max 2 Items)"
            )

            signage_data["color"] = (
                "#b58900"
            )

        else:

            for severity, msg in recommendations:

                if severity == "high":

                    signage_data["status"] = "FULL"
                    signage_data["message"] = (
                        f"{msg.upper()}"
                    )
                    signage_data["color"] = (
                        "#4a0404"
                    )
                    break

        try:

            with open(
                "shared_state.json",
                "w",
            ) as file:

                json.dump(
                    signage_data,
                    file,
                )

        except Exception:
            pass

        csv_data = history.to_csv(
            index=False
        ).encode("utf-8")

        st.sidebar.download_button(
            label="Download Shift Audit Report",
            data=csv_data,
            file_name="queue_performance_report.csv",
            mime="text/csv",
            key="live_audit_download",
        )

    live_dashboard()

else:

    if "sim" not in st.session_state:

        st.session_state.sim = (
            QueueSimulator(
                COUNTERS,
                seed=42,
            )
        )

        for _ in range(10):
            st.session_state.sim.tick()

    sim = st.session_state.sim

    st.sidebar.markdown("---")
    st.sidebar.subheader(
        "Tools"
    )

    surge_counter = st.sidebar.selectbox(
        "Target Counter",
        COUNTERS,
    )

    if st.sidebar.button(
        "Trigger Crowd Surge"
    ):

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

    state_df, forecasts, recommendations = (
        calculate_analytics(
            history
        )
    )

    render_metrics(state_df)

    st.markdown("---")

    simulation_queue_count = int(
        state_df["people_in_queue"].sum()
        if not state_df.empty
        else 0
    )

    render_state_and_actions(
        state_df,
        recommendations,
        prefix="simulation",
        voice_enabled=True,
        actual_queue_count=simulation_queue_count,
        require_real_queue_threshold=False,
    )

    st.markdown("---")

    st.subheader(
        "Queue Trajectory & AI Forecast"
    )

    fig = build_forecast_chart(
        history,
        forecasts,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    signage_data = {
        "status": "NORMAL",
        "message": "PLEASE WAIT IN LINE",
        "color": "#0a2911",
    }

    stuck_counters = state_df[
        state_df[
            "avg_service_time_min"
        ]
        > 5.0
    ]

    if not stuck_counters.empty:

        signage_data["message"] = (
            "EXPRESS LANE OPEN AT "
            "COUNTER 2 (Max 2 Items)"
        )

        signage_data["color"] = (
            "#b58900"
        )

    else:

        for severity, msg in recommendations:

            if severity == "high":

                signage_data["status"] = "FULL"
                signage_data["message"] = (
                    f"{msg.upper()}"
                )
                signage_data["color"] = (
                    "#4a0404"
                )
                break

    try:

        with open(
            "shared_state.json",
            "w",
        ) as file:

            json.dump(
                signage_data,
                file,
            )

    except Exception:
        pass

    csv_data = history.to_csv(
        index=False
    ).encode("utf-8")

    st.sidebar.download_button(
        label="Download Shift Audit Report",
        data=csv_data,
        file_name="queue_performance_report.csv",
        mime="text/csv",
        key="simulation_audit_download",
    )

    if auto_refresh:
        time.sleep(refresh_secs)
        st.rerun()
