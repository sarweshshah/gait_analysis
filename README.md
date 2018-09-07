# Gait Analysis using opencv-python 3.4 on a webcam/video

Gait analysis is analysing an individual by their walking pattern and turns out to be a reliable indentification source.
As it turns out, it is as reliable and unique as one's fingerprint and retina scan.

Following code explore the possibility to do the same by invovling following steps:
- Capturing image sequence
- Background modeling and Image subtraction
- Extracting binary silouette image
- Performing image correlation
- Applying discrete Fourier transformation
- Normalise results
- Perform primary component analysis
- Taking distance correction into account
- Indentification and verification

#### Pre-requisites:
- python-3.7
- opencv-python-3.4
- opencv-contrib-python-3.4 (optional)
- imutils-0.5.1 (optional)

Pycharm IDE provides an easy interface to setup the environment for the same.
It automatically downloads the dependencies for the packages.

To check if you have successfully installed opencv, run the following command in the terminal:
```
>>> import cv2
>>> print cv2.__version__
```
If the results are printed out without any errors, congratulations !!! You have installed OpenCV-Python successfully.

<br>

#### Step 1 | Capturing video frames:
The first step of the program captures image frames from available video file or webcam (by default) and presents it to the main thread for processing.
The image is turned to grayscale and displayed on the screen.

#### Step 2 | Performing background substraction:
OpenCV provides various [background subtraction algorithms](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html#background-subtraction) for video analysis. 

The most common usage of these algorithms is to extract moving objects from a static background.
Hence, the `createBackgroundSubtractorMOG2()` method is used for performing the second step of gait analysis.
Here is the result:

![](gait_analysis/results/background subtraction.gif)

**NOTE:** The code performs the video processing and polling of video frames, both on the main thread. Since its an I/O bound process, the framerate of the output video becomes slow. (The CPU has to wait for the thread to get a new frame before it can apply background subtraction on it and vice-versa). Hence, we use threads.
[more...](https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/)
