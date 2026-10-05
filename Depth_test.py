import cv2
import torch
from depth_anything_v2.dpt import DepthAnythingV2

model = DepthAnythingV2(
    encoder="vits",
    features=64,
    out_channels=[48, 96, 192, 384]
)

model.load_state_dict(
    torch.load("depth_anything_v2_vits.pth", map_location="cpu")
)

model.eval()

image = cv2.imread(r"C:\Users\andre\OneDrive\Desktop\CV_Projects\Face_Identification\Users_Images\Andres-Candido\24.jpg") # Add image path here to test on image

depth = model.infer_image(image)

cv2.imshow("Depth", depth)
cv2.waitKey(0)