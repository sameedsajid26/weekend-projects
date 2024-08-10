## Javelin Throw Analysis with AI
This project demonstrates how to analyze javelin throw videos using OpenCV and MediaPipe to extract valuable insights and provide real-time feedback to athletes and coaches. By leveraging the power of artificial intelligence, we can enhance sports analytics and support the development of aspiring athletes.

### Features
Pose estimation using MediaPipe to track key body landmarks
Angle calculation between body segments (e.g., elbow, shoulder, wrist)

### Requirements
Python 3.x
OpenCV
NumPy
MediaPipe
You can install the required packages using pip:

$ pip install opencv-python numpy mediapipe

### Usage
Place your javelin throw video in the same directory as the Python script.
Run the script:
$ python javelin_analysis.py 

The script will process the video, displaying the analyzed frames with overlaid angles and metrics.
Press 'q' to exit the video display.
The processed video will be saved as output_video.mp4 in the same directory.