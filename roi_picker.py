"""
roi_picker.py
One-time setup tool: opens the first frame of your video/camera and lets you
click points to mark the "queue zone" polygon. Prints the coordinates you
need to paste into camera_source.py's `queue_zone_polygon` argument.

Usage:
    python roi_picker.py path/to/video.mp4
    python roi_picker.py 0        # webcam

Controls:
    Left-click  -> add a point
    'r'         -> reset points
    'q' / Enter -> confirm and print the polygon
"""

import sys
import cv2

points = []


def click_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    source = int(source) if source.isdigit() else source

    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    if not ret:
        print("Could not read a frame from the source. Check the path/camera index.")
        return

    cv2.namedWindow("Mark queue zone - left click points, 'q' to finish")
    cv2.setMouseCallback("Mark queue zone - left click points, 'q' to finish", click_handler)

    import numpy as np

    while True:
        display = frame.copy()
        for p in points:
            cv2.circle(display, p, 4, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.polylines(display, [np.array(points, dtype=np.int32)], False, (0, 255, 0), 2)

        cv2.imshow("Mark queue zone - left click points, 'q' to finish", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            points.clear()
        elif key in (ord("q"), 13):  # 'q' or Enter
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nQueue zone polygon (paste this into camera_source.py):\n")
    print(points)


if __name__ == "__main__":
    main()