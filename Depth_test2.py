import cv2
import torch
from depth_anything_v2.dpt import DepthAnythingV2

# Load Depth Anything V2 model
model = DepthAnythingV2(
    encoder="vits",
    features=64,
    out_channels=[48, 96, 192, 384]
)

model.load_state_dict(
    torch.load("depth_anything_v2_vits.pth", map_location="cpu")
)

model.eval()

# Open camera
# 0 = Default camera, 1 = OBS camera
cap = cv2.VideoCapture(r"C:\Users\andre\Videos\2026-09-30 13-39-25.mp4") #Add camera index or video path here to test on camera or video input

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

while True:

    # Read video frame
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # Generate depth map
    depth = model.infer_image(frame)

    # Normalize depth map to 0-255 for display
    depth_display = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype("uint8")

    # Optional: apply a color map
    depth_color = cv2.applyColorMap(
        depth_display,
        cv2.COLORMAP_INFERNO
    )

    # Display original camera and depth
    cv2.imshow("Camera", frame)
    cv2.imshow("Depth", depth_color)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()