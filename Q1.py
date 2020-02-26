import cv2
import numpy as np
import copy

# image = cv2.imread('reference_images/ref_marker.png', 0)

p1 = np.array([[0, 0], [199, 0], [199, 199], [0, 199]], dtype="float32")

# def warpPerspective(img, homo, dim):
#     cv2.transpose(img)
#     warped_image = np.zeros((dim[0], dim[1], 3))
#     for index1 in range(0, img.shape[0]):
#         for index2 in range(0, img.shape[1]):
#             new_vec = np.dot(homo, [index1, index2, 1])
#             new_row, new_col, _ = (new_vec / new_vec[2] + 0.4).astype(int)
#             if 5 < new_row < (dim[0] - 5):
#                 if 5 < new_col < (dim[1] - 5):
#                     warped_image[new_row, new_col] = img[index1, index2]
#                     warped_image[new_row - 1, new_col - 1] = img[index1, index2]
#                     warped_image[new_row - 2, new_col - 2] = img[index1, index2]
#                     warped_image[new_row - 3, new_col - 3] = img[index1, index2]
#                     warped_image[new_row + 1, new_col + 1] = img[index1, index2]
#                     warped_image[new_row + 2, new_col + 2] = img[index1, index2]
#                     warped_image[new_row + 3, new_col + 3] = img[index1, index2]
#
#     # convert matrix to image
#     warped_image = np.array(warped_image, dtype=np.uint8)
#     cv2.transpose(warped_image)
#     return np.array(warped_image, dtype=np.uint8)


def tag_id(imager):
    ret, thresh = cv2.threshold(imager, 127, 255, cv2.THRESH_BINARY)
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


def detect_corners(cap):
    req_corners = []
    # _, frame = cap.read()
    gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_image = cv2.Canny(gaussian_blur, 75, 200)
    cnny_img = copy.copy(canny_image)
    # _, thresh = cv2.threshold(gry, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(cnny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)

    # thresh = cv2.drawContours(thresh, contours, -1, (0, 0, 255), 5)
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

    # finding child contour
    contour_num = []
    for h in hierarchy[0]:
        if h[3] != -1:
            contour_num.append(h[3])

    temp_corners = []

    # approx = []
    corners = []

    for cntr_nums in contour_num:
        epsilon = 0.01 * cv2.arcLength(contours[cntr_nums], True)
        approx = cv2.approxPolyDP(contours[cntr_nums], epsilon, True)

        # sorting out the case of more than 4 corners that is not rectangle
        if len(approx) < 5:
            epsilon = 0.1 * cv2.arcLength(contours[cntr_nums - 1], True)
            approx = cv2.approxPolyDP(contours[cntr_nums - 1], epsilon, True)
            corners.append(approx)

    for corner in corners:
        if len(corner) == 4:
            temp_corners.append(corner)

    for curner in temp_corners:
        cntr_area = cv2.contourArea(curner)
        if 700 < cntr_area < 6837:
            req_corners.append(curner)
    return req_corners
    # cv2.drawContours(frame, req_corners, -1, (255, 0, 0), 3)
    #
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     break
    # cap.release()


def homographyFunction(p2, dest):
    A_Matrix = []
    p_1 = ordering(dest)
    for points in range(len(p2)):
        x_1, y_1 = p_1[points]
        x_2, y_2 = p2[points]
        A_Matrix.append([[x_1, y_1, 1, 0, 0, 0, -x_2 * x_1, -x_2 * y_1, -x_2]])
        A_Matrix.append([[0, 0, 0, x_1, y_1, 1, -y_2 * x_1, -y_2 * y_1, -y_2]])

    A_Matrix = np.array(A_Matrix)
    # print(A_Matrix)
    A = np.reshape(A_Matrix, (8, 9))
    [_, _, V] = np.linalg.svd(A)
    H = V[:, -1]
    H1 = np.reshape(H, (3, 3))
    return H1


# def homographyFunction(p):
#     A = []
#     p2 = ordering(p)
#
#     for i in range(0, len(p1)):
#         x, y = p1[i][0], p1[i][1]
#         u, v = p2[i][0], p2[i][1]
#         A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
#         A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
#     A = np.array(A)
#     U, S, Vh = np.linalg.svd(A)
#     l = Vh[-1, :] / Vh[-1, -1]
#     h = np.reshape(l, (3, 3))
#     # print(l)
#     # print(h)
#     return h


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


# def LocationofTag(location):
#     #location=["BR","TR","TL","BL"]
#     if location=='Bottom_Right':
#         p1=np.array([[0,0],[199,0],[199,199],[0,199]],dtype=float32)
#         return p1
#     elif location=="Top_Right":
#         p1=np.array([[199,0],[199,199],[0,199],[0,0]],dtype=float32)
#         return p1
#     elif location=="Top_Left":
#         p1=np.array([[199,199],[0,199],[0,0],[199,0]],dtype=float32)
#         return p1
#     elif location=="Bottom_Left":
#         p1=np.array([[0,199],[0,0],[199,0],[199,199]],dtype=float32)
#         return p1


def run(frame, dst):
    # cv2.imshow('frame', frame)
    # frames = copy.copy(frame)
    Corners = detect_corners(frame)

    for i in range(len(Corners)):
        cv2.drawContours(frame, [Corners[i]], -1, (255, 0, 0), 3)
        # cv2.imshow("Contours", frame)
        corner_rows = np.reshape(Corners[i], (4, 2))
        # corner_rows = Corners[i][:, 0]
        # print(corner_rows)
        homo, _ = cv2.findHomography(ordering(corner_rows), ordering(dst))
        #homo = homographyFunction(ordering(corner_rows), dst)
        # cv2.invert(homo)
        warped_img = cv2.warpPerspective(frame, homo, (200, 200))
        # warped_img = warpPerspective(frame, homo, (200, 200))
        gray_w_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Image_Warped", gray_w_img)
        ID, orientation = tag_id(gray_w_img)  # Updated till here
        print(ID, orientation)
    cv2.imshow("Contours", frame)
        # empty = np.full(frame.shape, 0, dtype='uint8')
        # if orientation:
        #     p2 = LocationofTag(orientation, 200)
        #     if decoded:
        #         print("ID detected: " + str(decoded))
        #     H_Lena = homograph(order(c_rez), p2)
        #     lena_overlap = cv2.warpPerspective(lena_resize, H_Lena, (frame.shape[1], frame.shape[0]))
        #     if not np.array_equal(lena_overlap, empty):
        #         lena_list.append(lena_overlap.copy())
        #         print(lena_overlap.shape)

    # # mask = np.full(frame.shape, 0, dtype='uint8')
    # if lena_list != []:
    #     for lena in lena_list:
    #         temp = cv2.add(mask, lena.copy())
    #         mask = temp
    #
    #     lena_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #     r, lena_bin = cv2.threshold(lena_gray, 10, 255, cv2.THRESH_BINARY)
    #
    #     mask_inv = cv2.bitwise_not(lena_bin)
    #
    #     mask_3d = frame.copy()
    #     mask_3d[:, :, 0] = mask_inv
    #     mask_3d[:, :, 1] = mask_inv
    #     mask_3d[:, :, 2] = mask_inv
    #     img_masked = cv2.bitwise_and(frame, mask_3d)
    #     final_image = cv2.add(img_masked, mask)
    #     cv2.imshow("Lena", final_image)
    #     # cv2.waitKey(0)
    #


# print("Select")
vid = cv2.VideoCapture('Video_dataset/Tag0.mp4')

while vid.isOpened():
    _, frm = vid.read()
    # if flag == False:
    #     break
    resize_image = cv2.resize(frm, (0, 0), fx=0.5, fy=0.5)
    run(resize_image, p1)
    if cv2.waitKey(1) & 0xff == ord('q'):
        cv2.destroyAllWindows()
vid.release()

# if cap:
#     while True:
#         # Corners = detect_corners(cap)
#         Homo = homographyFunction(Corners)
#     # cap.release()
# else:
#     pass
