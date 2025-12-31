import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# 1. Load the model you just saved
model_path = '/content/drive/MyDrive/accident_model_final.pt'
video_path = 'train_video.mp4' # <--- Make sure you uploaded this!

print(f"Loading up the AI brain from: {model_path}")
model = YOLO(model_path)

# 2. Setup storage for our data
data = [] 
track_history = defaultdict(lambda: [])
previous_speeds = {}

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: I can't find 'train_video.mp4'. Did you upload it?")
else:
    print("Processing video... This creates the 'spreadsheet' for the next step.")
    
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # Skip frames to make it run faster (process every 3rd frame)
        frame_count += 1
        if frame_count % 3 != 0: continue

        # Run the AI Tracking
        results = model.track(frame, persist=True, verbose=False)

        if results[0].boxes.id is not None:
            # Get the boxes, IDs, and what Class the AI thinks it is
            boxes = results[0].boxes.xywh.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()

            for box, track_id, cls in zip(boxes, ids, clss):
                x, y, w, h = box
                
                # --- THE PHYSICS PART ---
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 20: track.pop(0)

                speed = 0
                acceleration = 0

                if len(track) > 2:
                    # Math: Distance moved between frames
                    dist = np.linalg.norm(np.array(track[-1]) - np.array(track[-2]))
                    speed = dist * 30 # Rough speed (pixels per second)
                    
                    # Math: Change in speed (Acceleration)
                    if track_id in previous_speeds:
                        acceleration = speed - previous_speeds[track_id]
                    previous_speeds[track_id] = speed

                # --- SAVE THE DATA ---
                # We assume Class 1 is 'Accident' (based on your dataset)
                # We are saving this row to teach the Random Forest later
                data.append({
                    "id": track_id,
                    "speed": speed,
                    "acceleration": acceleration,
                    "label": 1 if cls == 1 else 0 # 1=Crash, 0=Safe
                })

    cap.release()

    # 3. Save to CSV
    df = pd.DataFrame(data)
    df.to_csv('accident_data.csv', index=False)
    print("Done! I created 'accident_data.csv' with all the physics numbers.")
    print(df.head()) # Show the first few rows
