import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy
import imutils
import math
import time
import numpy.linalg as la
import numpy.linalg as LA
from numpy.linalg import inv
from numpy.linalg import norm



cap = cv2.VideoCapture("Tag0.mp4")
src = cv2.imread("Lena.png")
while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	smooth = cv2.GaussianBlur(gray, (5,5), 1)
	(True, thresh) = cv2.threshold(smooth, 200, 255, cv2.THRESH_BINARY)
	contour, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
 	rel_h = []
 	for h in hierarchy[0]:
 		if h[3] != -1 and h[2] == -1 and h[0] == -1:
 			rel_h.append(h)
 			# print rel_h
 	rel_c = []
 	for c in rel_h[0]:
 		if cv2.contourArea(contour[c]) > 500:
 			rel_c.append(contour[c])
 			# print rel_c
 	for cntrs in rel_c:
 		epsilon = 0.01*cv2.arcLength(cntrs, 1)
 		approx = cv2.approxPolyDP(cntrs, epsilon, 1)
 		# print approx


	p2 = []
	for a in approx:
		p2.append([a[0][1], a[0][0]])
	


	p1 = np.array([[0, 0], [200, 0], [0, 200], [200, 200]], dtype="float32")
	A_Matrix = []

	for points in range(len(p2)):
		x_1, y_1 = p1[points]
		x_2, y_2 = p2[points]
		A_Matrix.append([[x_1, y_1, 1, 0, 0, 0, -x_2 * x_1, -x_2 * y_1, -x_2]])
		A_Matrix.append([[0, 0, 0, x_1, y_1, 1, -y_2 * x_1, -y_2 * y_1, -y_2]])
	A_Matrix = np.array(A_Matrix)
	A = np.reshape(A_Matrix, (8, 9))
	[_, _, V] = np.linalg.svd(A)
	H = V[:, -1]
	H1 = np.reshape(H, (3, 3))
   
###############################################################################################
# code for cube plotting starts here

	pts_dst = np.array(p2,dtype="float32")
	pts_src = np.array([[0,0],[200, 0],[200, 200],[0,200]],dtype="float32")
	# M, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC,5.0)
	temp = cv2.warpPerspective(frame, H1,(frame.shape[1],frame.shape[0]));
	cv2.fillConvexPoly(frame, pts_dst.astype(int), 0, 16);
	frame = frame + temp;
	# cv2.imshow("screen1",frame)

# Defining the camera intinsic matrix
	K = np.array([[1406.08415449821,0,0],
       			  [ 2.20679787308599, 1417.99930662800,0],
       			  [ 1014.13643417416, 566.347754321696,1]])
	K = K.T
	K_inv = np.linalg.inv(K)
	transform = np.dot(K_inv, H1 )
	column_1 = np.asarray(transform[:,0]).T
	column_2 = np.asarray(transform[:,1]).T
	column_3 = np.asarray(transform[:,2]).T
   	l = math.sqrt(la.norm(column_1, 2) * la.norm(column_2, 2))
   	rotation_1 = column_1 / l
   	rotation_2 = column_2 / l
   	translation = column_3 / l
   	c = rotation_1 + rotation_2
   	p = np.cross(rotation_1, rotation_2)
   	d = np.cross(c, p)
   	
   	rotation_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
   	rotation_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
   	rotation_3 = np.cross(rotation_1, rotation_2)
   	projection = np.stack((rotation_1, rotation_2, rotation_3, translation)).T
   	proj_mat = np.dot(K, projection)
   	print proj_mat

   	axis = np.float32([[0,0,0,1],[0,512,0,1],[512,512,0,1],[512,0,0,1],[0,0,-512,1],[0,512,-512,1],[512,512,-512,1],[512,0,-512,1]])
   	Proj = np.matmul(axis,proj_mat.T)
   

   	n1 = np.divide(Proj[0],Proj[0][2])
   	n2 = np.divide(Proj[1],Proj[1][2])
   	n3 = np.divide(Proj[2],Proj[2][2])
   	n4 = np.divide(Proj[3],Proj[3][2])
   	n5 = np.divide(Proj[4],Proj[4][2])
   	n6 = np.divide(Proj[5],Proj[5][2])
   	n7 = np.divide(Proj[6],Proj[6][2])
   	n8 = np.divide(Proj[7],Proj[7][2])
   	points = np.vstack((n1,n2,n3,n4,n5,n6,n7,n8))
   	
   	final_2d = np.delete(points,2, axis=1)
   	# draw(image,final_2d)

   	imgpts = np.int32(final_2d).reshape(-1,2)
   	img = cv2.drawContours(frame, [imgpts[:4]],-1,(255,0,255),-3)
   	for i,j in zip(range(4),range(4,8)):
   		img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(0,0,0),3)
        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]],-1,(255,0,0),3)
    	cv2.imshow("try",img)
 
    	# M = cv2.getPerspectiveTransform(pts_src,pts_dst)
    	warped_img = cv2.warpPerspective(frame, H1, (200, 200))
    	# cv2.imshow("Image_Warped", warped_img)
	if cv2.waitKey(1) & 0xff == ord("q"):
		break
cap.release()
