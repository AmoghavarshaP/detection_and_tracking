# Code for feature extraction corner detection
# Importing the libraries
import cv2
import numpy as np
# import imgproc
# import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-i","--image",required = True, help = "image path")
# args = vars(ap.parse_args())

# image_train = cv.imread('index1.png',1) 				# reads the training image 
# image_test = cv.imread('index1.png',1)				# reads the testing video

cap = cv2.VideoCapture('Video_dataset/multipleTags.mp4')

while cap.isOpened():
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Converts the gray image to float32 format because corner Harris accepts image in float32 format.

    # # find Harris corners
    # gray = np.float32(gray)
    # dst = cv.cornerHarris(gray, 5, 3, 0.04)
    # ret, dst = cv.threshold(dst, 0.1*dst.max(), 255, 0)
    # dst = np.uint8(dst)
    # frame[dst > 0.01*dst.max()] = [0, 0, 255]		# Threshold for an optimal value, it may vary depending on the image.
    #
    # # find corner coordinates
    # ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # find Shi-Tomasi corners
    # corners = cv.goodFeaturesToTrack(gray, 100, 0.3, 2)
    # corners = np.int0(corners)
    # print(corners)

    # draw corners in video frames
    # for c in corners:
        # i, j = c.ravel()
        # print('i:', i, 'j:', j)
        # cv.circle(frame, (i, j), 3, 255, -1)
        # print(corners[i])

    cv2.imshow('frame', thresh)					# Displays the image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()
