import cv2
import time
import numpy as np
from ultralytics import YOLOE, YOLO


# --------------------------------------------------
# Load models
# --------------------------------------------------

# YOLOE segmentation model
detection_model = YOLOE("yoloe-11s-seg.pt")
detection_model.set_classes(["chair", "person", "book", "picture", "cellphone"])

# YOLO26 Depth
# n = Nano (fastest)
# s = Small
# m = Medium
# l = Large
# x = Extra Large (most accurate, slowest)
depth_model = YOLO("yolo26n-depth.pt") # yolo26 variants: yolo26n-depth.pt, yolo26s-depth.pt, yolo26m-depth.pt, yolo26l-depth.pt, yolo26x-depth.pt

# --------------------------------------------------
# Open camera
# --------------------------------------------------

cap = cv2.VideoCapture(1)  # 0 for default camera, 1 for OBS camera

# --------------------------------------------------
# Settings
# --------------------------------------------------

DETECTION_INTERVAL = 15 # Run YOLOE segmentation every n frames, run YOLO26 depth one frame after that

trackers = []           # OpenCV object trackers (1 per detected object)
tracked_classes = []    # Classes of each tracked objects
tracked_depths = []     # Last measured depth of each object
tracked_masks = []      # Segmentation masks (from YOLOE) of each object for depth calculation (YOLO26 depth) 1 frame after

frame_count = 0                     # Frame counter for fps calculation 
previous_time = time.perf_counter() # Time of previous frame for fps calculation
fps = 0                             # Current FPS

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
    # YOLOE SEGMENTATION FRAME
    # ==================================================

    if frame_count % DETECTION_INTERVAL == 0:

        # ----------------------------------------------
        # Run YOLOE
        # ----------------------------------------------

        results = detection_model(
            frame,
            imgsz=640, # IMPORTANT: Set image size 640, 576, 512, 448, etc. Lower values are faster but less accurate. 512 works well and is a bit faster than 640
            verbose=False
        )

        result = results[0]

        boxes = result.boxes
        masks = result.masks

        # ----------------------------------------------
        # Clear old trackers
        # ----------------------------------------------

        trackers = []
        tracked_classes = []
        tracked_depths = []
        tracked_masks = []

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
                # Save mask for depth estimation
                # on the next frame
                # ------------------------------------------

                distance = None

                tracked_masks.append(mask)

                # ------------------------------------------
                # Create OpenCV tracker
                # ------------------------------------------

                # opencv has multiple trackers, you can choose others like TrackerCSRT_create, TrackerKCF_create, TrackerMIL_create, etc.
                tracker = cv2.legacy.TrackerMedianFlow_create() # MedianFlow is fast and accurate enough, seems like best tracker so far

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
    # YOLO26 DEPTH FRAME
    # ==================================================

    elif frame_count % DETECTION_INTERVAL == 1:

        annotated_frame = frame.copy()

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
        # Calculate depth for tracked objects
        # ----------------------------------------------

        for i, (tracker, class_name, mask) in enumerate(
            zip(trackers, tracked_classes, tracked_masks)
        ):

            success, bbox = tracker.update(frame)

            if success:

                x, y, w, h = [
                    int(v) for v in bbox
                ]

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

                tracked_depths[i] = distance

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

        mode = "YOLOE SEGMENTATION"

    elif frame_count % DETECTION_INTERVAL == 1:

        mode = "YOLO26 DEPTH"

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