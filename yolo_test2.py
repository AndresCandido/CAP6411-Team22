import cv2
import time
from ultralytics import YOLOE

# --------------------------------------------------
# Load YOLOE
# --------------------------------------------------

model = YOLOE("yoloe-11s-seg.pt")

model.set_classes(["chair", "person", "book"])  # Example classes, adjust as needed


# --------------------------------------------------
# Open camera
# --------------------------------------------------

cap = cv2.VideoCapture(1)  # 0 for default camera, 1 for OBS camera


# --------------------------------------------------
# Settings
# --------------------------------------------------

DETECTION_INTERVAL = 15

frame_count = 0

trackers = []
tracked_classes = []

previous_time = time.perf_counter()
fps = 0

# --------------------------------------------------
# Main loop
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # ==================================================
    # YOLOE FRAME
    # ==================================================

    if frame_count % DETECTION_INTERVAL == 0:

        # Run YOLOE
        results = model(frame, verbose=False)

        # Clear old trackers
        trackers = []
        tracked_classes = []

        boxes = results[0].boxes

        if boxes is not None:

            for box in boxes:

                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)

                # Get class
                class_id = int(box.cls[0])

                class_name = results[0].names[class_id]


                # ------------------------------------------
                # Create OpenCV tracker
                # ------------------------------------------

                # opencv has multiple trackers, you can choose others like TrackerCSRT_create, TrackerKCF_create, TrackerMIL_create, etc.
                tracker = cv2.TrackerMIL_create() # MIL seems fastest and accurate enough

                tracker.init(
                    frame,
                    (x, y, w, h)
                )

                trackers.append(tracker)
                tracked_classes.append(class_name)


        # YOLOE already knows how to draw its detections
        annotated_frame = results[0].plot()


    # ==================================================
    # TRACKING FRAME
    # ==================================================

    else:

        annotated_frame = frame.copy()

        for tracker, class_name in zip(
            trackers,
            tracked_classes
        ):

            success, bbox = tracker.update(frame)

            if success:

                x, y, w, h = [int(v) for v in bbox]

                # Draw tracked bounding box
                cv2.rectangle(
                    annotated_frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # Draw class name
                cv2.putText(
                    annotated_frame,
                    class_name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


    # ==================================================
    # Display information
    # ==================================================

    if frame_count % DETECTION_INTERVAL == 0:
        mode = "YOLOE"
    else:
        mode = "TRACKING"

    cv2.putText(
        annotated_frame,
        f"Mode: {mode}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )

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


    cv2.imshow(
        "YOLOE + OpenCV Tracking",
        annotated_frame
    )


    frame_count += 1


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()