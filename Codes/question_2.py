import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy
import imutils
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm

# to take the input from the user to use the video
choice = int(input("Select option 0. Tag0    1. Tag1     2. Tag2     3. Multiple Tags: "))

if choice == 0:
    vid = cv2.VideoCapture('Video_dataset/Tag0.mp4')
elif choice == 1:
    vid = cv2.VideoCapture('Video_dataset/Tag1.mp4')
elif choice == 2:
    vid = cv2.VideoCapture('Video_dataset/Tag2.mp4')
elif choice == 3:
    vid = cv2.VideoCapture('Video_dataset/multipleTags.mp4')


dim = 200
p1 = np.array([
    [0, 0],
    [dim - 1, 0],
    [dim - 1, dim - 1],
    [0, dim - 1]], dtype="float32")

def tag_id(image):
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    crop = thresh[50:150, 50:150]

    cell_1, cell_2, cell_3, cell_4 = crop[37, 37], crop[37, 63], crop[63, 63], crop[63, 37]

    cell_1 = 1 if cell_1 == 255 else 0
    cell_2 = 1 if cell_2 == 255 else 0
    cell_3 = 1 if cell_3 == 255 else 0
    cell_4 = 1 if cell_4 == 255 else 0

    if crop[87, 87] == 255:
        return [cell_4, cell_3, cell_2, cell_1], 'Bottom_Right'
        # print('Tag ID: ', cell_4, cell_3, cell_2, cell_1, ' at Bottom Right')
    elif crop[13, 13] == 255:
        return [cell_2, cell_1, cell_4, cell_3], 'Top_Left'
        # print('Tag ID: ', cell_2, cell_1, cell_4, cell_3, 'Top Left')
    elif crop[87, 13] == 255:
        return [cell_3, cell_2, cell_1, cell_4], 'Bottom_Left'
        # print('Tag ID: ', cell_3, cell_2, cell_1, cell_4, 'Bottom Left')
    elif crop[13, 87] == 255:
        return [cell_1, cell_4, cell_3, cell_2], 'Top_Right'
        # print('Tag ID: ', cell_1, cell_4, cell_3, cell_2, 'Top_Right')


def draw_cube(img, imgpts):  # To draw the cube
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (255, 0, 255), 3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 255, 255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (255, 0, 255), 3)
    return img

def ordering(points):
    rect = np.zeros((4, 2), dtype="float32")

    # The top left point has the smallest sum
    # The bottom right has the largest sum

    summer = points.sum(axis=1)  # adding the x and y coordinates of rectangle by specifying "axis=1"
    rect[0] = points[np.argmin(summer)]
    rect[2] = points[np.argmax(summer)]

    difference = np.diff(points, axis=1)  # Here the difference of the x & y coordinates
    rect[1] = points[np.argmin(difference)]  # top right will have minimum difference
    rect[3] = points[np.argmax(difference)]  # bottom right will have the largest difference

    return rect


def homographyFunction(p, p1):
    A = []
    p2 = ordering(p)
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    l = V[-1, :] / V[-1, -1]
    h = np.reshape(l, (3, 3))
    return h

# Function to calculate Projection matrix
def calculator(h):
    K = np.array(
        [[1406.08415449821, 0, 0], [2.20679787308599, 1417.99930662800, 0], [1014.13643417416, 566.347754321696, 1]]).T
    h = inv(h)
    b_new = np.dot(inv(K), h)
    b1 = b_new[:, 0].reshape(3, 1)
    b2 = b_new[:, 1].reshape(3, 1)
    r3 = np.cross(b_new[:, 0], b_new[:, 1])
    b3 = b_new[:, 2].reshape(3, 1)
    L = 2 / (norm((inv(K)).dot(b1)) + norm((inv(K)).dot(b2)))
    r1 = L * b1
    r2 = L * b2
    r3 = (r3 * L * L).reshape(3, 1)
    t = L * b3
    r = np.concatenate((r1, r2, r3), axis=1)

    return r, t, K


def contour_generator(frame):
    test_img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    test_blur = cv2.GaussianBlur(test_img1, (3, 3), 0)
    edge = cv2.Canny(test_blur, 75, 200)
    edge1 = copy.copy(edge)
    countour_list = list()
    r, ctrs, h = cv2.findContours(edge1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index = list()
    for hier in h[0]:
        if hier[3] != -1:
            index.append(hier[3])

    for c in index:
        peri = cv2.arcLength(ctrs[c], True)
        approx = cv2.approxPolyDP(ctrs[c], 0.02 * peri, True)

        if len(approx) > 4:
            peri1 = cv2.arcLength(ctrs[c - 1], True)
            corners = cv2.approxPolyDP(ctrs[c - 1], 0.02 * peri1, True)
            countour_list.append(corners)

    new_contour_list = list()
    for contour in countour_list:
        if len(contour) == 4:
            new_contour_list.append(contour)
    final_contour_list = list()
    for element in new_contour_list:
        if cv2.contourArea(element) < 2500:
            final_contour_list.append(element)

    return final_contour_list


def reorient(location, maxDim):
    if location == "BR":
        p1 = np.array([
            [0, 0],
            [maxDim - 1, 0],
            [maxDim - 1, maxDim - 1],
            [0, maxDim - 1]], dtype="float32")
        return p1
    elif location == "TR":
        p1 = np.array([
            [maxDim - 1, 0],
            [maxDim - 1, maxDim - 1],
            [0, maxDim - 1],
            [0, 0]], dtype="float32")
        return p1
    elif location == "TL":
        p1 = np.array([
            [maxDim - 1, maxDim - 1],
            [0, maxDim - 1],
            [0, 0],
            [maxDim - 1, 0]], dtype="float32")
        return p1

    elif location == "BL":
        p1 = np.array([
            [0, maxDim - 1],
            [0, 0],
            [maxDim - 1, 0],
            [maxDim - 1, maxDim - 1]], dtype="float32")
        return p1


def image_process(frame, p1):
    final_contour_list = contour_generator(frame)
    cube_list = list()
    axis = np.float32(
        [[0, 0, 0], [0, 200, 0], [200, 200, 0], [200, 0, 0], [0, 0, -200], [0, 200, -200], [200, 200, -200],
         [200, 0, -200]])
    mask = np.full(frame.shape, 0, dtype='uint8')
    for i in range(len(final_contour_list)):
        cv2.drawContours(frame, [final_contour_list[i]], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", frame)
        c_rez = final_contour_list[i][:, 0]
        H_matrix = homographyFunction(p1, ordering(c_rez))
        tag = cv2.warpPerspective(frame, H_matrix, (200, 200))

        cv2.imshow("Outline", frame)
        cv2.imshow("Tag after homogenous", tag)

        tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
        decoded, location = tag_id(tag1)
        #empty = np.full(frame.shape, 0, dtype='uint8')
        if not location == None:
            p2 = reorient(location, 200)
            if not decoded == None:
                r, t, K = calculator(H_matrix)
                points, jac = cv2.projectPoints(axis, r, t, K, np.zeros((1, 4)))
                img = draw_cube(mask, points)
                cube_list.append(img.copy())
    if cube_list != []:  # empty cube list
        for cube in cube_list:
            temp = cv2.add(mask, cube.copy())
            mask = temp

        final_image = cv2.add(frame, mask)
        cv2.imshow("cubes", final_image)
        #cv2.waitKey(0)

    if cv2.waitKey(1) & 0xff == 27:
        cv2.destroyAllWindows()


while vid.isOpened():
    success, frame = vid.read()
    if success == False:
        break
    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    image_process(img, p1)

cap.release()