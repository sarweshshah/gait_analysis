# Uses Google Mediapipe to track and detect pose landmarks
# https://google.github.io/mediapipe/solutions/pose.html

# Online References
# https://codepen.io/mediapipe/pen/jOMbvxw
# https://youtu.be/06TE_U21FK4

import cv2
import mediapipe as mp

img = cv2.imread("data/1.jpg")  # Add image in the data folder to test this code

preFrameTime = 0

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

with mpPose.Pose(static_image_mode=False, smooth_landmarks=True, model_complexity=1,
                 min_tracking_confidence=0.5, min_detection_confidence=0.5) as pose:
    # Convert image to RGB format and process poses
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(imgRGB)

    mpDraw.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                          landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=1),
                          connection_drawing_spec=mpDraw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))

    # Render image in a window and fetch next frame
    cv2.imshow("Input", img)

cv2.waitKey(0)
img.release()
cv2.destroyAllWindows()
