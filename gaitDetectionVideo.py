# Uses Google Mediapipe to track and detect pose landmarks
# https://google.github.io/mediapipe/solutions/pose.html

# Online References
# https://codepen.io/mediapipe/pen/jOMbvxw
# https://youtu.be/06TE_U21FK4

import time

import cv2
import mediapipe as mp

cap = cv2.VideoCapture('data/hydrocephalus.mp4')
# cap = cv2.VideoCapture('data/sneak/v_sneak_g1.mp4')

preFrameTime = 0

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

with mpPose.Pose() as pose:  # Pass parameters in Pose() to change sensitivity etc.
    while True:
        # Read next frame
        success, img = cap.read()

        if success:
            img_height, img_width, _ = img.shape

            # Convert image to RGB format and process poses
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Perform pose detection
            results = pose.process(imgRGB)

            try:
                w_landmarks = results.pose_world_landmarks.landmark

                left_hip = w_landmarks[mpPose.PoseLandmark.LEFT_HIP.value]
                right_hip = w_landmarks[mpPose.PoseLandmark.RIGHT_HIP.value]
                left_knee = w_landmarks[mpPose.PoseLandmark.LEFT_KNEE.value]
                right_knee = w_landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value]
                left_ankle = w_landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value]
                right_ankle = w_landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value]

                pose_string = [
                    # "Left Hip: " + "x:" + str('%.3f' % float(left_hip.x * 10)) + ", " + "y:" + str('%.3f' % float(left_hip.y * 10)) + ", " + "z:" + str('%.3f' % float(left_hip.z * 10)),
                    # "Right Hip: " + "x:" + str('%.3f' % float(right_hip.x * 10)) + ", " + "y:" + str('%.3f' % float(right_hip.y * 10)) + ", " + "z:" + str('%.3f' % float(right_hip.z * 10)),
                    "Left Knee: " + "x:" + str('%.3f' % float(left_knee.x * 10)) + ", " + "y:" + str(
                        '%.3f' % float(left_knee.y * 10)) + ", " + "z:" + str('%.3f' % float(left_knee.z * 10)),
                    "Right Knee: " + "x:" + str('%.3f' % float(right_knee.x * 10)) + ", " + "y:" + str(
                        '%.3f' % float(right_knee.y * 10)) + ", " + "z:" + str('%.3f' % float(right_knee.z * 10)),
                    "Left Ankle: " + "x:" + str('%.3f' % float(left_ankle.x * 10)) + ", " + "y:" + str(
                        '%.3f' % float(left_ankle.y * 10)) + ", " + "z:" + str('%.3f' % float(left_ankle.z * 10)),
                    "Right Ankle: " + "x:" + str('%.3f' % float(right_ankle.x * 10)) + ", " + "y:" + str(
                        '%.3f' % float(right_ankle.y * 10)) + ", " + "z:" + str('%.3f' % float(right_ankle.z * 10)),
                    "Joint positions wrt centre of hips (in cms)"
                ]

                # Display landmark points as text on the screen w.r.t hip position
                y0, dy = img_height - 40, 20
                for index in range(len(pose_string)):
                    y = y0 - (index * dy)
                    if index == len(pose_string) - 1:
                        cv2.putText(img, pose_string[index], (35, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
                    else:
                        cv2.putText(img, pose_string[index], (35, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            except:
                pass

            mpDraw.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=1),
                                  connection_drawing_spec=mpDraw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))

            # Calculate the fps for the video
            currentFrameTime = time.time()
            fps = 1 / (currentFrameTime - preFrameTime)
            preFrameTime = currentFrameTime
            # print(f"FPS of video is: {fps}")

            # Add fps text on the screen
            cv2.putText(img, "FPS: " + str(int(fps)), (35, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            # Render image in a window and fetch next frame
            cv2.imshow("Input", img)

        # Pressing the ESC key will quit the window
        if (cv2.waitKey(10) & 0xFF) == 27:
            break

cap.release()
cv2.destroyAllWindows()
