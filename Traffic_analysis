import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np

# ---------------------------
# Load YOLOv8 model
# ---------------------------
model = YOLO("yolov8n.pt")  # use yolov8n for speed, replace with yolov8s/8m if needed

# ---------------------------
# Input video
# ---------------------------
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

# ---------------------------
# Lane setup (customize as per video)
# Each lane is a vertical section of the frame
# ---------------------------
lanes = {
    1: (0, 200),    # lane 1: x=0 to x=200
    2: (200, 400),  # lane 2: x=200 to x=400
    3: (400, 600)   # lane 3: x=400 to x=600
}
lane_counts = {lane: 0 for lane in lanes}

# ---------------------------
# Output setup
# ---------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("OutputVideo.mp4", fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

vehicle_data = []  # store results for CSV

# ---------------------------
# Process video
# ---------------------------
frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    results = model(frame, verbose=False)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # bounding boxes
        classes = r.boxes.cls.cpu().numpy()  # class IDs
        confs = r.boxes.conf.cpu().numpy()   # confidence

        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)  # center x
            cy = int((y1 + y2) / 2)  # center y

            # Draw detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Assign to lane
            for lane, (x_start, x_end) in lanes.items():
                if x_start <= cx < x_end:
                    lane_counts[lane] += 1
                    vehicle_data.append([frame_num, lane, int(cls), float(conf)])
                    break

    # Draw lane dividers
    for lane, (x_start, x_end) in lanes.items():
        cv2.line(frame, (x_end, 0), (x_end, frame.shape[0]), (255, 0, 0), 2)
        cv2.putText(frame, f"Lane {lane}: {lane_counts[lane]}",
                    (x_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Traffic Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows
