import cv2 as cv
import numpy as np
import homogrphy_function

MIN_MATCHES = 15
image_1 = cv.imread('index.png', 0)        	# read the image
# image_test = cv.VideoCapture('tag0.mp4')
image_3 = cv.imread('index3.png', 0)


orb = cv.ORB_create()                    	# initiate the ORB detector
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
kp_model, des_model = orb.detectAndCompute(image_3, None) 				# finding the keypoints
kp_frame, des_frame = orb.detectAndCompute(image_1, None)				# compute the descriptors
matches = bf.match(des_model, des_frame)
matches = sorted(matches, key=lambda x: x.distance)
# image2 = cv.drawKeypoints(image_1, kp, image_1, color=(0,255,0), flags=0)

# if len(matches) > MIN_MATCHES:
    # draw first 15 matches.
image_1 = cv.drawMatches(image_3, kp_model, image_1, kp_frame,
                          matches[:MIN_MATCHES], 0, flags=2)
    # show result
cv.imshow('frame', image_1)

src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)



# Draw a rectangle that marks the found model in the frame
h, w = image_3.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
# project corners into frame
dst = cv.perspectiveTransform(pts, M)  
# connect them with lines
img2 = cv.polylines(image_1, [np.int32(dst)], True, 255, 3, cv.LINE_AA) 
cv.imshow('frame', img2)
cv.waitKey(0)



# else:
#     print "Not enough matches have been found - %d/%d" % (len(matches),
#                                                           MIN_MATCHES)

# cv.imshow('keypoints',image2)
# cv.waitKey(0)
# homogrphy_function.homography()
