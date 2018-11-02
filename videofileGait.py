# import the necessary packages
import time
import cv2
import imutils
import numpy as np
from imutils.video import FileVideoStream

# fvs = FileVideoStream('data/sarwesh.mov').start()  #without bag
fvs = FileVideoStream('data/sanghveer.mov', queueSize=512).start()  # with bag
time.sleep(1.0)

kernelSize = 7
backgroundHistory = 15

# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=1500)
fgbg = cv2.createBackgroundSubtractorMOG2(history=backgroundHistory, detectShadows=True)

kernel = np.ones((kernelSize, kernelSize), np.uint8)

while fvs.more():
    frame = fvs.read()

    frame = imutils.resize(frame, width=960)

    # Applying background subtraction on the capture frame
    frame = fgbg.apply(frame)

    # img = frame
    # size = np.size(img)
    # skel = np.zeros(img.shape, np.uint8)

    # ret, img = cv2.threshold(img, 127, 255, 0)
    # element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    # done = False
    #
    # while not done:
    #     eroded = cv2.erode(img, element)
    #     temp = cv2.dilate(eroded, element)
    #     temp = cv2.subtract(img, temp)
    #     skel = cv2.bitwise_or(skel, temp)
    #     img = eroded.copy()
    #
    #     zeros = size - cv2.countNonZero(img)
    #     if zeros == size:
    #         done = True

    # Morphological transformation on the substracted image to remove noise.
    # frame = cv2.dilate(frame, kernel, iterations=1)

    # Otsu's thresholding after Gaussian filtering
    # frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # ret3, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Canny edge detection to detect the outermost silhouette of person.
    # fgmask = cv2.Canny(frame, 100, 200)

    # cv2.imshow("skel", skel)
    cv2.imshow("Output", frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # if cv2.waitKey(33) == ord('w'):  # UP:
    #     backgroundHistory += 1
    #     print("Updated history = ", backgroundHistory)
    # elif cv2.waitKey(33) == ord('s'):  # DOWN:
    #     if backgroundHistory > 1: backgroundHistory -= 1
    #     print("Updated history = ", backgroundHistory)
    # elif cv2.waitKey(33) == ord('d'):  # RIGHT:
    #     kernelSize += 2
    #     print("Updated kernel size = ", kernelSize)
    # elif cv2.waitKey(33) == ord('a'):  # LEFT:
    #     if kernelSize > 1: kernelSize -= 2
    #     print("Updated kernel size = ", kernelSize)

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()
