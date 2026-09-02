import cv2
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from camera_source import CameraQueueSource # Your existing file

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"]) # Essential for local web dev

camera = CameraQueueSource(counter_id="Counter 1", video_path=0, frame_skip=3)
latest_stats = {}

async def process_camera():
    global latest_stats
    while True:
        # Runs your existing ML logic
        latest_stats = camera.tick(duration_seconds=1)
        await asyncio.sleep(0.01)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_camera())

def generate_frames():
    while True:
        if hasattr(camera, 'last_annotated_frame') and camera.last_annotated_frame is not None:
            ret, buffer = cv2.imencode('.jpg', camera.last_annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    # Streams the YOLO processed frames continuously
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stats")
def get_stats():
    # Removes the frame object before sending JSON data
    stats_copy = latest_stats.copy()
    stats_copy.pop("frame", None) 
    return JSONResponse(content=stats_copy)