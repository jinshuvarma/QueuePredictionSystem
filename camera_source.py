"""
camera_source.py
Real camera / video based queue-data source. Produces the EXACT SAME row
schema as simulator.py's tick() output, so app.py can switch from
QueueSimulator to CameraQueueSource without touching state_estimator.py,
predictor.py, or recommender.py at all.

Schema per counter per tick:
    timestamp, counter_id, people_in_queue, arrivals, served, avg_service_time

Pipeline:
    YOLOv8n (person detection, pretrained on COCO -- no training needed)
      -> CentroidTracker (keeps identity of the same person across frames)
      -> queue-zone polygon check (is this person actually IN the line?)
      -> entry/exit counting over a time window -> one summary row

Install:
    pip install ultralytics opencv-python
"""

import time
from datetime import datetime
from collections import OrderedDict

import numpy as np
import cv2
from ultralytics import YOLO


class CentroidTracker:
    """
    Minimal multi-object tracker: matches new detections to existing
    tracked people by nearest-centroid distance frame-to-frame.
    This is what lets us count UNIQUE people instead of re-counting the
    same person in every frame they appear in.
    """

    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_id = 0
        self.objects = OrderedDict()       # id -> (x, y) centroid
        self.disappeared = OrderedDict()   # id -> frames since last seen
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, centroids):
        if len(centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self.objects

        if len(self.objects) == 0:
            for c in centroids:
                self._register(c)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))
        input_centroids = np.array(centroids)

        D = np.linalg.norm(object_centroids[:, None] - input_centroids[None, :], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            oid = object_ids[row]
            self.objects[oid] = input_centroids[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(D.shape[0])) - used_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self._deregister(oid)

        for col in set(range(D.shape[1])) - used_cols:
            self._register(input_centroids[col])

        return self.objects

    def _register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]


def point_in_polygon(point, polygon):
    """polygon: list of (x, y) pixel points drawn around the queue area."""
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0


class CameraQueueSource:
    """
    Real-time / video-file based data source for ONE counter.
    Run one instance per counter (each watching its own camera or ROI
    inside a shared frame).
    """

    def __init__(self, counter_id, video_path, queue_zone_polygon=None,
                 model_path="yolov8n.pt", conf=0.4, frame_skip=2, loop_video=True):
        """
        video_path         : 0 for webcam, or a path/RTSP URL to a video file
        queue_zone_polygon : [(x1,y1), (x2,y2), ...] pixel points marking the
                              queue area in the frame (see roi_picker.py to draw
                              this precisely). Pass None to use the ENTIRE frame
                              as the queue zone -- the fastest way to get data
                              flowing on the first try, before you've bothered
                              drawing a precise zone.
        frame_skip          : process every Nth frame (CPU speed vs accuracy)
        loop_video          : if the source is a finite video file, restart it
                               from frame 0 when it ends (keeps the demo running
                               instead of going silently blank)
        """
        self.counter_id = counter_id
        self.video_path = video_path
        self.loop_video = loop_video
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"[{counter_id}] Could not open video source: {video_path!r}. "
                f"If this is a file path, check it exists relative to where you "
                f"ran the script. If this is 0/1/2 (webcam index), check no other "
                f"app is using the camera and the index is correct."
            )

        if queue_zone_polygon is None:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            queue_zone_polygon = [(0, 0), (w, 0), (w, h), (0, h)]

        self.zone = queue_zone_polygon
        # yolov8n.pt auto-downloads from Ultralytics' GitHub releases on first
        # use if it isn't already cached locally -- no manual download needed,
        # just requires an internet connection the first time.
        self.model = YOLO(model_path)   # pretrained on COCO, class 0 = "person"
        self.conf = conf
        self.frame_skip = frame_skip
        self.tracker = CentroidTracker()

        self.seen_ids_last_tick = set()
        self.service_times = []   # rolling list of observed dwell times (minutes)
        self._entry_ts = {}       # id -> time first seen inside the zone

    def _detect_people(self, frame):
        results = self.model(frame, classes=[0], conf=self.conf, verbose=False)[0]
        centroids = []
        boxes = []
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            centroids.append(((x1 + x2) / 2, (y1 + y2) / 2))
            # Bounding box ke corners save kar rahe hain
            boxes.append((int(x1), int(y1), int(x2), int(y2))) 
        return centroids, boxes

    def tick(self, duration_seconds=60):
        start = time.time()
        frame_idx = 0
        arrivals, served = 0, 0
        in_zone_ids = set()
        
        objects = {}
        current_in_zone = set()

        while time.time() - start < duration_seconds:
            ret, frame = self.cap.read()
            if not ret:
                if self.loop_video and isinstance(self.video_path, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                else:
                    break 
            frame_idx += 1
            if frame_idx % self.frame_skip != 0:
                continue

            # NAYA CHANGE: Yahan centroids ke sath boxes bhi unpack honge
            centroids, boxes = self._detect_people(frame)
            self.last_boxes = boxes
            self.last_centroids = centroids
            
            objects = self.tracker.update(centroids)

            current_in_zone = set()
            for oid, centroid in objects.items():
                if point_in_polygon(tuple(centroid), self.zone):
                    current_in_zone.add(oid)
                    if oid not in self._entry_ts:
                        self._entry_ts[oid] = time.time()

            new_ids = current_in_zone - self.seen_ids_last_tick
            left_ids = self.seen_ids_last_tick - current_in_zone

            arrivals += len(new_ids)
            walk_outs = 0
            
            for oid in left_ids:
                if oid in self._entry_ts:
                    dwell_min = (time.time() - self._entry_ts.pop(oid)) / 60.0
                    # If they left in under 30 seconds, they abandoned the queue
                    if dwell_min < 0.5:
                        walk_outs += 1
                    else:
                        served += 1
                        self.service_times.append(dwell_min)

            in_zone_ids = current_in_zone
            self.seen_ids_last_tick = current_in_zone

        # --- NAYA DRAWING LOGIC (Boxes instead of Dots) ---
        if 'frame' in locals() and frame is not None and frame.size > 0:
            self.last_annotated_frame = frame.copy()
            
            overlay = self.last_annotated_frame.copy()
            roi_points = np.array(self.zone, dtype=np.int32)
            cv2.fillPoly(overlay, [roi_points], (0, 255, 255)) # Yellow fill
            # Blend the overlay with the original frame (20% opacity)
            cv2.addWeighted(overlay, 0.2, self.last_annotated_frame, 0.8, 0, self.last_annotated_frame)
            # Draw the solid border
            cv2.polylines(self.last_annotated_frame, [roi_points], True, (0, 255, 255), 2)
            
            # 2. Draw Bounding Boxes
            if hasattr(self, 'last_boxes'):
                for box, centroid in zip(self.last_boxes, self.last_centroids):
                    x1, y1, x2, y2 = box
                    # Green if inside ROI, Red if outside
                    in_zone = point_in_polygon(centroid, self.zone)
                    color = (0, 255, 0) if in_zone else (0, 0, 255)
                    cv2.rectangle(self.last_annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # 3. Draw IDs
            for oid, c in objects.items():
                cv2.putText(self.last_annotated_frame, f"ID: {oid}", (int(c[0]) - 15, int(c[1]) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        avg_service_time = (
            sum(self.service_times[-20:]) / len(self.service_times[-20:])
            if self.service_times else 2.5 
        )

        return {
            "timestamp": datetime.now(),
            "counter_id": self.counter_id,
            "people_in_queue": len(in_zone_ids),
            "arrivals": arrivals,
            "served": served,
            "avg_service_time": round(avg_service_time, 2),
            "frame": getattr(self, "last_annotated_frame", None)
        }

    def release(self):
        self.cap.release()