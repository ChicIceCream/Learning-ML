from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import contextlib
import sys
import os
import time

# Function to suppress YOLOv8 logs
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

model = YOLO('yolov8m.pt')  # Using the YOLO medium model

videopath = 'YOLOv8 MOdel for Heatmap/video_files/road_video_trimmed.mp4'
cap = cv2.VideoCapture(videopath)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

track_history = defaultdict(lambda: [])
last_positions = {}

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

heatmap = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.float32)
frame_count = 0

# Start the timer
start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    
    # Get the timestamp of the current frame in milliseconds
    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    timestamp_s = timestamp_ms / 1000.0
    timestamp_str = f"Time: {int(timestamp_s // 60)}:{int(timestamp_s % 60):02d}"

    with suppress_stdout():
        results = model.track(frame, persist=True, classes=2)

    boxes = results[0].boxes.xywh.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    for box, track_id in zip(boxes, track_ids):
        x_center, y_center, width, height = box
        current_position = (float(x_center), float(y_center))

        top_left_x = int(x_center - width / 2)
        top_left_y = int(y_center - height / 2)
        bottom_right_x = int(x_center + width / 2)
        bottom_right_y = int(y_center + height / 2)

        top_left_x = max(0, top_left_x)
        top_left_y = max(0, top_left_y)
        bottom_right_x = min(heatmap.shape[1], bottom_right_x)
        bottom_right_y = min(heatmap.shape[0], bottom_right_y)

        track = track_history[track_id]
        track.append(current_position)
        if len(track) > 1200:
            track.pop(0)

        last_position = last_positions.get(track_id)
        if last_position and calculate_distance(last_position, current_position) > 5:
            heatmap[top_left_y:bottom_right_y, top_left_x:bottom_right_x] += 1

        last_positions[track_id] = current_position

    heatmap_blurred = cv2.GaussianBlur(heatmap, (15, 15), 0)
    heatmap_norm = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    alpha = 0.7
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)

    # Put the timestamp on the frame
    cv2.putText(overlay, timestamp_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    out.write(overlay)

cap.release()
out.release()
cv2.destroyAllWindows()

# End the timer and display the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Processing time: {elapsed_time:.2f} seconds")