# CAP6411-Team22
A Real Time Foundation Model Perception Stack on a Laptop


These are the scripts I've done so far. 

For the yolo tests I tested YOLOE for open-vocabulary object detection, and YOLO26 DEPTH for depth perception. 
The models used in these files are automatically downloaded using the ultralytics library into your working directory, which makes them easy to use. 

For testing Depth-Anything-V2 use the "Depth_test.py" and "Depth_test2.py" files inside in Depth-Anything-V2 folder. "Depth_test.py" is for images, and "Depth_test2.py" for video/livestream input.
For these files to work they must be placed inside the Depth-Anything-V2 model folder you can download from here: [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small)

---------------------- For implementing the latest script, yolo_test5.py, do the following: ------------------------------------------

1. Download (pip install) the libraries used in the script.
    pip install opencv-contrib-python numpy ultralytics
   
3. Run the script. The ultralytics library should the try to download the 2 yolo models (yoloe-11s-seg.pt and yolo26n-depth.pt files) we use in the script automatically onto your working directory.
    Note: Yoloe needs to use a file called mobileclip_blt for it to work. This file should be created automatically as well when you try to run the script for the first time. It takes around half a gb of space.
   
5. Check the camera settings:
    When testing the script I was giving the script a video stream from my webcam using OBS's virtual camera. You might have to change line 27 in the script if you want to use your laptop's camera directly.
   Try:
    line 27 -> cap = cv2.VideoCapture(0)  # 0 for default camera, 1 for OBS camera
   If when you run the script a window with the camera feed doesn't appear, this line might be the issue.
