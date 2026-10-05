import cv2
import time
import numpy as np
from ultralytics import YOLOE, YOLO


# --------------------------------------------------
# Load models
# --------------------------------------------------

# YOLOE segmentation model
detection_model = YOLOE("yoloe-11s-seg.pt")
detection_model.set_classes(["chair", "person", "book"])

# YOLO26 Nano Depth
depth_model = YOLO("yolo26n-depth.pt")

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
tracked_depths = []

previous_time = time.perf_counter()
fps = 0

# --------------------------------------------------
# Calculate object depth using segmentation mask
# --------------------------------------------------

def get_mask_depth(depth_map, mask):

    # Resize segmentation mask to match depth map
    mask_resized = cv2.resize(
        mask,
        (depth_map.shape[1], depth_map.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Convert to boolean mask
    object_mask = mask_resized > 0.5

    # Extract depth pixels belonging only to object
    object_depths = depth_map[object_mask]

    # Remove invalid values
    object_depths = object_depths[
        np.isfinite(object_depths) &
        (object_depths > 0)
    ]

    if object_depths.size == 0:
        return None

    # Median is resistant to occasional incorrect pixels
    return float(np.median(object_depths))

# --------------------------------------------------
# Main loop
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # ==================================================
    # YOLOE + YOLO26 DEPTH FRAME
    # ==================================================

    if frame_count % DETECTION_INTERVAL == 0:

        # ----------------------------------------------
        # Run YOLOE
        # ----------------------------------------------

        results = detection_model(
            frame,
            verbose=False
        )

        result = results[0]

        boxes = result.boxes
        masks = result.masks


        # ----------------------------------------------
        # Run YOLO26 Nano Depth
        # ----------------------------------------------

        depth_results = depth_model(
            frame,
            verbose=False
        )

        depth_map = (
            depth_results[0]
            .depth.data
            .cpu()
            .numpy()
        )


        # ----------------------------------------------
        # Clear old trackers
        # ----------------------------------------------

        trackers = []
        tracked_classes = []
        tracked_depths = []

        annotated_frame = frame.copy()


        # ----------------------------------------------
        # Process YOLOE detections
        # ----------------------------------------------

        if boxes is not None and masks is not None:

            for i, box in enumerate(boxes):

                # ------------------------------------------
                # Get bounding box
                # ------------------------------------------

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)


                # ------------------------------------------
                # Get class
                # ------------------------------------------

                class_id = int(box.cls[0])

                class_name = result.names[class_id]


                # ------------------------------------------
                # Get YOLOE segmentation mask
                # ------------------------------------------

                mask = (
                    masks.data[i]
                    .cpu()
                    .numpy()
                )


                # ------------------------------------------
                # Calculate median object depth
                #
                # Only pixels that YOLOE says belong
                # to this object are used.
                # ------------------------------------------

                distance = get_mask_depth(
                    depth_map,
                    mask
                )


                # ------------------------------------------
                # Create OpenCV tracker
                # ------------------------------------------

                tracker = cv2.TrackerMIL_create()

                tracker.init(
                    frame,
                    (x, y, w, h)
                )


                # ------------------------------------------
                # Save tracker information
                # ------------------------------------------

                trackers.append(tracker)

                tracked_classes.append(
                    class_name
                )

                tracked_depths.append(
                    distance
                )


                # ------------------------------------------
                # Draw bounding box
                # ------------------------------------------

                cv2.rectangle(
                    annotated_frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    2
                )


                # ------------------------------------------
                # Create class + depth label
                # ------------------------------------------

                if distance is not None:

                    label = (
                        f"{class_name}: "
                        f"{distance:.2f} m"
                    )

                else:

                    label = class_name


                # ------------------------------------------
                # Draw class + depth
                # ------------------------------------------

                cv2.putText(
                    annotated_frame,
                    label,
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )


    # ==================================================
    # TRACKING FRAME
    # ==================================================

    else:

        annotated_frame = frame.copy()


        for tracker, class_name, distance in zip(
            trackers,
            tracked_classes,
            tracked_depths
        ):

            success, bbox = tracker.update(frame)

            if success:

                x, y, w, h = [
                    int(v) for v in bbox
                ]


                # ------------------------------------------
                # Draw tracked bounding box
                # ------------------------------------------

                cv2.rectangle(
                    annotated_frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )


                # ------------------------------------------
                # Display the last measured depth
                # ------------------------------------------

                if distance is not None:

                    label = (
                        f"{class_name}: "
                        f"{distance:.2f} m"
                    )

                else:

                    label = class_name


                # ------------------------------------------
                # Draw class + depth
                # ------------------------------------------

                cv2.putText(
                    annotated_frame,
                    label,
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


    # ==================================================
    # Display mode
    # ==================================================

    if frame_count % DETECTION_INTERVAL == 0:

        mode = "YOLOE + DEPTH"

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


    # ==================================================
    # Calculate FPS
    # ==================================================

    current_time = time.perf_counter()

    fps = 1 / (
        current_time - previous_time
    )

    previous_time = current_time


    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )


    # ==================================================
    # Display
    # ==================================================

    cv2.imshow(
        "YOLOE + OpenCV Tracking + YOLO26 Depth",
        annotated_frame
    )


    frame_count += 1


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()