# AR Tag Detection and Tracking
This is a coursework project to implement to corner detection, orientation and tag ID of an AR Tag, and superimposing an image and a 3D cube on the AR tag in a given video. We used Lena image to superimpose it on the tag and orient it according to the orientation of the tag in the video so that it looks like the image is on the video.

<p align="center">
  <img width="500" height="300" src="https://github.com/namangupta98/detection_and_tracking/blob/master/reference_images/Tag%20Detection%20and%20Superimposition.gif">
  <img width="300" height="300" src="https://github.com/namangupta98/detection_and_tracking/blob/master/reference_images/Lena.png">
  <br><b>Figure 1 - Lena Image Superimposed</b><br>
</p>

<p align="center">
  <img src="https://github.com/namangupta98/detection_and_tracking/blob/master/reference_images/3D%20Cube.png">
  <br><b>Figure 2 - 3D Cube Superimposed</b><br>
</p>

## Overview

In this project, an AR Tag is used to project a 3D cube and Lena image on the tag without using in-built functions such as cv2.findHomography() and cv2.warpPerspective().

## Dependencies

- Python3
- OpenCV-Python
- Numpy

## Run

Please open the file in PyCharm. Fixing issues with terminal.

- Download the repo from GitHub: https://github.com/namangupta98/detection_and_tracking
- Open terminal in the same folder and type the following for Lena superimposition
````
python3 Q1_new.py
````
- Type the following for 3-D Cube
```
python3 question_2.py
```

## Results

https://drive.google.com/drive/folders/1bBNtIiOcpX6tTNRHlQWdCpyqdHL5I4ft
