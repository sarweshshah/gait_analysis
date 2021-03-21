# import the necessary packages
import time

import cv2
import imutils
import numpy as np
from imutils.video import FileVideoStream

fvs = FileVideoStream('data/sarwesh1.mp4', queue_size=1024).start()  # with bag
time.sleep(1.0)

openposeProtoFile = "dnn_models/pose/coco/pose_deploy_linevec.prototxt"
openposeWeightsFile = "dnn_models/pose/coco/pose_iter_440000.caffemodel"
nPoints = 18

# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
              [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
              [2, 17], [5, 16]]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
          [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
          [47, 48], [49, 50], [53, 54], [51, 52], [55, 56],
          [37, 38], [45, 46]]

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
          [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
          [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]


def getKeypoints(prob_map, thres=0.1):
    map_smooth = cv2.GaussianBlur(prob_map, (3, 3), 0, 0)

    map_mask = np.uint8(map_smooth > thres)
    keypoints_array = []

    # find the blobs
    _, contours, _ = cv2.findContours(map_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for each blob find the maxima
    for cnt in contours:
        blob_mask = np.zeros(map_mask.shape)
        blob_mask = cv2.fillConvexPoly(blob_mask, cnt, 1)
        masked_prob_map = map_smooth * blob_mask
        _, max_val, _, max_loc = cv2.minMaxLoc(masked_prob_map)
        keypoints_array.append(max_loc + (prob_map[max_loc[1], max_loc[0]],))

    return keypoints_array


keypoint_stack = []

while fvs.more():
    frame = fvs.read()
    frame = imutils.resize(frame, width=1080)

    frameClone = frame.copy()

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (frameWidth, frameHeight), (0, 0, 0), swapRB=False, crop=False)

    net = cv2.dnn.readNetFromCaffe(openposeProtoFile, openposeWeightsFile)

    net.setInput(inpBlob)
    output = net.forward()

    detected_keypoints = []
    keypoints_list = np.zeros((0, 3))
    keypoint_id = 0
    threshold = 0.3

    for part in range(nPoints):
        probMap = output[0, part, :, :]
        probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
        keypoints = getKeypoints(probMap, threshold)

        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)

    keypoint_stack.append(detected_keypoints)

    for frame_index in range(len(keypoint_stack)):
        keypoints_per_frame = keypoint_stack[frame_index]
        for i in range(nPoints):
            for j in range(len(keypoints_per_frame[i])):
                cv2.circle(frame, keypoints_per_frame[i][j][0:2], 2, colors[i], -1, cv2.LINE_AA)

    frame = cv2.addWeighted(frameClone, 0.5, frame, 0.5, 0.0)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(50) & 0xff
    if k == 27:
        break

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()
