# import the necessary packages
# import numpy as np
import cv2
import imutils
from imutils.video import WebcamVideoStream

stream = WebcamVideoStream(0).start()
fgbg = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)

# kernel = np.ones((5, 5), np.uint8)

while 1:
    frame = stream.read()
    frame = imutils.resize(frame, width=960)

    # Applying background subtraction on the capture frame
    frame = fgbg.apply(frame)

    # Morphological transformation on the substracted image to remove noise.
    # frame = cv2.dilate(frame, kernel, iterations=1)

    # Otsu's thresholding after Gaussian filtering
    # frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # ret3, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Canny edge detection to detect the outermost silhouette of person.
    # fgmask = cv2.Canny(frame, 100, 200)

    cv2.imshow("output", frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# do a bit of cleanup
stream.stop()
cv2.destroyAllWindows()
