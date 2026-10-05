import cv2
import time
from ultralytics import YOLO

# Load YOLO26 Depth model
# n = Nano (fastest)
# s = Small
# m = Medium
# l = Large
# x = Extra Large (most accurate, slowest)
model = YOLO("yolo26n-depth.pt") # yolo26 variants: yolo26n-depth.pt, yolo26s-depth.pt, yolo26m-depth.pt, yolo26l-depth.pt, yolo26x-depth.pt

# Open camera
# 0 = default camera
# 1 = OBS virtual camera
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

previous_time = time.perf_counter()
fps = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO26 depth estimation
    results = model(frame, verbose=False)

    # Get depth map
    # Values are estimated distance in meters
    depth = results[0].depth.data.cpu().numpy()

    # Normalize depth map to 0-255 for visualization
    depth_display = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype("uint8")

    # Apply color map
    depth_color = cv2.applyColorMap(
        depth_display,
        cv2.COLORMAP_INFERNO
    )

    # Calculate FPS
    current_time = time.perf_counter()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # Display FPS
    cv2.putText(
        depth_color,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Display original camera
    cv2.imshow("Camera", frame)

    # Display depth map
    cv2.imshow("YOLO26 Depth", depth_color)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()