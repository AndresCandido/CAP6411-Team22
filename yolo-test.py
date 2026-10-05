import cv2
from ultralytics import YOLOE

import time

# Load YOLOE model
model = YOLOE("yoloe-11s-seg.pt")

# Tell YOLOE which objects to look for
model.set_classes(["chair", "person", "book"])  # Example classes, adjust as needed

# Open camera
cap = cv2.VideoCapture(1) # 0 for default camera, 1 for OBS camera

previous_time = time.perf_counter()
fps = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLOE detection
    results = model(frame)

    # Draw detections onto the frame
    annotated_frame = results[0].plot()

        # Calculate FPS
    current_time = time.perf_counter()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # Display FPS below mode
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )

    # Display with OpenCV
    cv2.imshow("YOLOE Object Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()