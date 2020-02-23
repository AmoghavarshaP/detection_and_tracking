# Code for feature extraction corner detection
# Importing the libraries
import cv2 as cv
import numpy as np
# import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-i","--image",required = True, help = "image path")
# args = vars(ap.parse_args())

image_train = cv.imread('index.png', 1) 				# reads the training image
image_test = cv.VideoCapture('tag0.mp4')			# reads the testing video


gray = cv.cvtColor(image_train, cv.COLOR_BGR2GRAY)	# changes the color space of an 'image' to grayscale     
gray = np.float32(gray)							    # Converts the gray image to float32 format because corner Harris accepts image in float32 format.


# Loads Corner Harris function for edge detection

# The parameters are:
        # img - Input image, it should be grayscale and float32 type.
        # blockSize - It is the size of neighbourhood considered for corner detection
        # ksize - Aperture parameter of Sobel derivative used.
        # k - Harris detector free parameter in the equation.
dst = cv.cornerHarris(gray, 2, 3, 0.04)

cv.imshow('dst_image', dst)
image_train[dst > 0.01*dst.max()] = [0, 0, 255]		# Threshold for an optimal value, it may vary depending on the image.
cv.imshow('dst', image_train)					# Displays the image
cv.waitKey(0) 									# creates a wait