# CAP6411-Team22
A Real Time Foundation Model Perception Stack on a Laptop


These are the scripts I've done so far. 

For the yolo tests I tested YOLOE for open-vocabulary object detection, and YOLO26 DEPTH for depth perception. 
The models used in these files are automatically downloaded using the ultralytics library into your working directory, which makes them easy to use. 

For testing Depth-Anything-V2 use the "Depth_test.py" and "Depth_test2.py" files inside in Depth-Anything-V2 folder. "Depth_test.py" is for images, and "Depth_test2.py" for video/livestream input.
For these files to work they must be placed inside the Depth-Anything-V2 model folder you can download from here: [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small)

## ------ For implementing the latest script, yolo_test5.py, do the following: ------

### Step 1: Install the Required Libraries

Download the libraries used in the script by running:

```bash
pip install opencv-contrib-python numpy ultralytics
```

### Step 2: Run the Script

Run `yolo_test5.py`.

The Ultralytics library should automatically download the two YOLO models used in the script into your working directory:

- `yoloe-11s-seg.pt` — YOLOE segmentation model
- `yolo26n-depth.pt` — YOLO26 depth estimation model

**Note:** YOLOE also requires a file called `mobileclip_blt` to work. This file should be created automatically when you run the script for the first time. It takes approximately 0.5 GB of storage.

### Step 3: Check the Camera Settings

When testing the script, I provided a video stream from my webcam using OBS Virtual Camera.

You might need to change **line 27** in the script if you want to use your laptop's camera directly.

Try changing line 27 to:

```python
cap = cv2.VideoCapture(0)  # 0 for default camera, 1 for OBS camera
```

If a window displaying the camera feed does not appear when you run the script, this line might be the issue.
