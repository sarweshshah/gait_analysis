# import the necessary packages
import time

import cv2
import imutils
import numpy as np
from imutils.video import FileVideoStream

theImage = cv2.imread("data/distracted-walking.jpg")
time.sleep(1.0)

openposeProtoFile = "dnn_models/pose/coco/pose_deploy_linevec.prototxt"
openposeWeightsFile = "dnn_models/pose/coco/pose_iter_440000.caffemodel"
nPoints = 18

objectdetectionProtoFile = "dnn_models/object_detection/MobileNetSSD_deploy.prototxt"
objectdetectionWeightsFile = "dnn_models/object_detection/MobileNetSSD_deploy.caffemodel"

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

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


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


# Find valid connections between the different joints of a all persons present
def getValidPairs(generated_output):
    validpairs = []
    invalidpairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = generated_output[0, mapIdx[k][0], :, :]
        pafB = generated_output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if nA != 0 and nB != 0:
            valid_pair = np.zeros((0, 3))
            for i in range(nA):
                max_j = -1
                max_score = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores) / len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                        if avg_paf_score > max_score:
                            max_j = j
                            max_score = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], max_score]], axis=0)

            # Append the detected connections to the global list
            validpairs.append(valid_pair)
        else:  # If no keypoints are detected
            invalidpairs.append(k)
            validpairs.append([])
    return validpairs, invalidpairs


# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(validpairs, invalidpairs):
    # the last number in each row is the overall score
    personwise_keypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalidpairs:
            partAs = validpairs[k][:, 0]
            partBs = validpairs[k][:, 1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(validpairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwise_keypoints)):
                    if personwise_keypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwise_keypoints[person_idx][indexB] = partBs[i]
                    personwise_keypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + validpairs[k][i][
                        2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[validpairs[k][i, :2].astype(int), 2]) + validpairs[k][i][2]
                    personwise_keypoints = np.vstack([personwise_keypoints, row])
    return personwise_keypoints


frameClone = theImage.copy()

frameWidth = theImage.shape[1]
frameHeight = theImage.shape[0]

# Fix the input Height and get the width according to the Aspect Ratio
inHeight = 368
inWidth = int((inHeight / frameHeight) * frameWidth)
inpBlob = cv2.dnn.blobFromImage(theImage, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

net = cv2.dnn.readNetFromCaffe(openposeProtoFile, openposeWeightsFile)
objnet = cv2.dnn.readNetFromCaffe(objectdetectionProtoFile, objectdetectionWeightsFile)

net.setInput(inpBlob)
output = net.forward()

# pass the blob through the network and obtain the detections and predictions
objnet.setInput(inpBlob)
detections = objnet.forward()

# loop over the detections
for i in np.arange(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with
    # the prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > 0.6:
        # extract the index of the class label from the
        # `detections`, then compute the (x, y)-coordinates of
        # the bounding box for the object
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the prediction on the frame
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        cv2.rectangle(theImage, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frameClone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

detected_keypoints = []
keypoints_list = np.zeros((0, 3))
keypoint_id = 0
threshold = 0.1

for part in range(nPoints):
    probMap = output[0, part, :, :]
    probMap = cv2.resize(probMap, (theImage.shape[1], theImage.shape[0]))
    keypoints = getKeypoints(probMap, threshold)

    keypoints_with_id = []
    for i in range(len(keypoints)):
        keypoints_with_id.append(keypoints[i] + (keypoint_id,))
        keypoints_list = np.vstack([keypoints_list, keypoints[i]])
        keypoint_id += 1

    detected_keypoints.append(keypoints_with_id)

valid_pairs, invalid_pairs = getValidPairs(output)
personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

for i in range(17):
    for n in range(len(personwiseKeypoints)):
        index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
        if -1 in index:
            continue
        B = np.int32(keypoints_list[index.astype(int), 0])
        A = np.int32(keypoints_list[index.astype(int), 1])
        cv2.line(theImage, (B[0], A[0]), (B[1], A[1]), colors[i], 2, cv2.LINE_AA)

frame = cv2.addWeighted(frameClone, 0.5, theImage, 0.5, 0.0)

cv2.imshow("Detected Pose", frame)
cv2.waitKey(0)

