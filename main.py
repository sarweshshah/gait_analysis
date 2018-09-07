# import the necessary packages
import cv2
import imutils

from webcamStream import WebcamVideoStream

stream = WebcamVideoStream(0).start()
fgbg = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=False)

while 1:
    frame = stream.read()
    frame = imutils.resize(frame, width=1000)

    fgmask = fgbg.apply(frame)
    # fgmask = cv2.Canny(fgmask, 500, 600)

    # fgmask = cv2.Canny(frame, 100, 200)

    cv2.imshow("output", fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# do a bit of cleanup
stream.stop()
cv2.destroyAllWindows()
