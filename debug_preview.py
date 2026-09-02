"""
debug_preview.py
Standalone visual test -- run this FIRST, before touching Streamlit, to
confirm the camera/video, the YOLO model, and your queue-zone polygon are
all working correctly. Opens a window showing:
    - green polygon = your queue zone
    - boxes around detected people
    - a running people-in-zone count

This is the fastest way to debug "nothing seems to be happening" issues,
since app.py's camera_source.py runs headless (no window) by design.

Usage:
    python debug_preview.py videos/counter1.mp4
    python debug_preview.py 0        # webcam

Press 'q' to quit.
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO

from camera_source import CentroidTracker, point_in_polygon


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    source = int(source) if source.isdigit() else source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Could not open source: {source!r}")
        print("   - If it's a file path: check it exists relative to this folder.")
        print("   - If it's a number (webcam index): try 0, 1, or 2.")
        return

    print("✅ Source opened. Loading YOLOv8n model "
          "(auto-downloads on first run, needs internet once)...")
    model = YOLO("yolov8n.pt")
    print("✅ Model loaded.")

    # EDIT this polygon to match your frame -- or better, generate it with
    # roi_picker.py and paste the printed coordinates here.
    zone = [(50, 50), (400, 50), (400, 400), (50, 400)]

    tracker = CentroidTracker()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended (or camera disconnected).")
            break
        frame_count += 1

        results = model(frame, classes=[0], conf=0.4, verbose=False)[0]
        centroids = []
        boxes = results.boxes.xyxy.cpu().numpy()
        for x1, y1, x2, y2 in boxes:
            centroids.append(((x1 + x2) / 2, (y1 + y2) / 2))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        objects = tracker.update(centroids)
        in_zone = 0
        for oid, c in objects.items():
            color = (0, 0, 255)
            if point_in_polygon(tuple(c), zone):
                in_zone += 1
                color = (0, 255, 0)
            cv2.circle(frame, (int(c[0]), int(c[1])), 5, color, -1)
            cv2.putText(frame, f"ID {oid}", (int(c[0]) + 6, int(c[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.polylines(frame, [np.array(zone, dtype=np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, f"People detected: {len(centroids)} | In zone: {in_zone}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Debug preview - press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()