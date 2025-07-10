import time
import cv2
from swappr.utils import parse_timestamp_to_frame
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


video_path = "/home/sebastiangarcia/projects/swappr/data/poc/UFC317/BrazilPriEncode_swappr_317.ts"
start_timestamp = "00:36:36" # 00:36:37
end_timestamp = "00:37:40"
yolo_model_path = "/home/sebastiangarcia/projects/swappr/models/poc/v2_budlight_logo_detection/weights/best.pt"

det_model_budlight = YOLO(yolo_model_path)
video_stream = cv2.VideoCapture(video_path)
video_fps: float = video_stream.get(cv2.CAP_PROP_FPS)

start_frame = parse_timestamp_to_frame(start_timestamp, video_fps)
end_frame = parse_timestamp_to_frame(end_timestamp, video_fps)

video_stream.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
current_frame_number = start_frame


# Set up matplotlib for displaying frames
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_title('Video Frame Display')
ax.axis('off')

while video_stream.isOpened():
    start_time_frame: float = time.time()

    ret, frame = video_stream.read()
    if not ret or frame is None:
        print("End of video stream or error reading frame.")
        break

    # Check if we've reached the end timestamp
    if end_frame is not None and current_frame_number >= end_frame:
        print(f"Reached end timestamp at frame {current_frame_number}")
        break
    
    results = det_model_budlight(frame)

    boxes = results[0].boxes.xyxy
    budlight_bbox = None
    if boxes.shape[0] > 0:
        budlight_bbox = boxes.cpu().numpy().astype("int").squeeze()
        print("Detected budlight logo")
    else:
        print("No budlight logo detected")

    # Convert BGR to RGB for matplotlib display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the frame
    ax.clear()
    ax.imshow(frame_rgb)
    ax.set_title(f'Frame {current_frame_number} - Time: {current_frame_number/video_fps:.2f}s')
    ax.axis('off')
    
    plt.draw()
    plt.pause(0.001)  # Small pause to allow matplotlib to update
    
    current_frame_number += 1
    
    # Optional: Add a small delay to control playback speed
    # time.sleep(1/video_fps)  # Uncomment to play at original speed

video_stream.release()
plt.ioff()  # Turn off interactive mode
plt.show()
