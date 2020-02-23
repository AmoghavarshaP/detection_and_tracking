# Code for feature extraction corner detection
# Importing the libraries
import cv2 as cv
import numpy as np
# import imgproc
# import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-i","--image",required = True, help = "image path")
# args = vars(ap.parse_args())

# image_train = cv.imread('index1.png',1) 				# reads the training image 
# image_test = cv.imread('index1.png',1)				# reads the testing video

cap = cv.VideoCapture('Tag0.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow('frame',gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
# cv2.destroyAllWindows()   
gray = np.float32(gray)							    # Converts the gray image to float32 format because corner Harris accepts image in float32 format.


# Loads Corner Harris function for edge detection

# The parameters are:
        # img - Input image, it should be grayscale and float32 type.
        # blockSize - It is the size of neighbourhood considered for corner detection
        # ksize - Aperture parameter of Sobel derivative used.
        # k - Harris detector free parameter in the equation.
dst = cv.cornerHarris(gray,2,3,0.04)
ret, dst = cv.threshold(dst,0.1*dst.max(),255,0)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
for i in range(1, len(corners)):
    print(corners[i])

frame[dst>0.01*dst.max()]=[0,0,255]		# Threshold for an optimal value, it may vary depending on the image.
cv.imshow('dst',frame)					# Displays the image
cap.release()
cv.waitKey(0) 									# creates a wait
# cv.imwrite("tag.png",image)  					# Saves the image to the folder with the name tag.png