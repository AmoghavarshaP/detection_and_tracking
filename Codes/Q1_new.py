import cv2
import numpy as np
import copy

p1 = np.array([[0, 0], [199, 0], [199, 199], [0, 199]], dtype="float32")
lena_img = cv2.imread('reference_images/Lena.png', 1)
lena_img = cv2.resize(lena_img, (199, 199))
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


def tag_reorientation(orient):
    if orient == 'Top_Left':
        new_orient = np.array([[199, 199], [0, 199], [0, 0], [199, 0]])
    elif orient == 'Top_Right':
        new_orient = np.array([[199, 0], [199, 199], [0, 199], [0, 0]])
    elif orient == 'Bottom_Left':
        new_orient = np.array([[0, 199], [0, 0], [199, 0], [199, 199]])
    elif orient == 'Bottom_Right':
        new_orient = np.array([[0, 0], [199, 0], [199, 199], [0, 199]])
    return new_orient


def homographyFunction(p2,p1):
    A=[]
    # p2=ordering(p0)
    for points in range(0, len(p1)):
        x1,y1=p1[points][0],p1[points][1]
        x2,y2=p2[points][0],p2[points][1]
        A.append([x1,y1,1,0,0,0,-x2*x1,-x2*y1,-x2])
        A.append([0,0,0,x1,y1,1,-y2*x1,-y2*y1,-y2])
    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    # A = np.reshape(A, (8, 9))
    # H = V[:, -1]
    # H1 = np.reshape(H, (3, 3))
    H = V[8, :] / V[8, 8]
    H1 = np.reshape(H, (3, 3))
    return H1


def warpTag(fram, homo):
    inv_homo = np.linalg.inv(homo)
    fram_gray = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(fram_gray, 127, 255, cv2.THRESH_BINARY)
    warp_tag = np.zeros((200, 200, 3))

    for i in range(200):
        for j in range(200):
            x, y, z = np.matmul(inv_homo, [i, j, 1])
            if (540 > int(y / z) > 0) and (960 > int(x / z) > 0):
                warp_tag[i][j][0] = fram_gray[int(y / z)][int(x / z)]
                warp_tag[i][j][1] = fram_gray[int(y / z)][int(x / z)]
                warp_tag[i][j][2] = fram_gray[int(y / z)][int(x / z)]

    warp_tag = warp_tag.astype('uint8')
    return warp_tag


def warpLena(lena, homo, size):
    m, n, o = size
    lena_warpp = np.zeros((m, n, 3))
    gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    ret, grayImage = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    q, w = grayImage.shape
    for ii in range(m):
        for jj in range(n):
            x, y, z = np.matmul(homo, [ii, jj, 1])
            if y / z != float('inf') and (x / z != float('inf')):
                if (int(q) > int(y / z) > 0) and (int(w) > int(x / z) > 0):
                    lena_warpp[jj][ii][0] = lena[int(y / z)][int(x / z)][0]
                    lena_warpp[jj][ii][1] = lena[int(y / z)][int(x / z)][1]
                    lena_warpp[jj][ii][2] = lena[int(y / z)][int(x / z)][2]

    lena_warpp = lena_warpp.astype('uint8')
    return lena_warpp


def run(frame, dst):
    Corners = detect_corners(frame)

    for i in range(len(Corners)):
        cv2.drawContours(frame, [Corners[i]], -1, (255, 0, 0), 1)
        # cv2.imshow("Contours", frame)
        corner_rows = Corners[i][:, 0]
        # corner_rows = np.reshape(Corners[i], (4, 2))
        # print(corner_rows)
        # homo, _ = cv2.findHomography(ordering(corner_rows), ordering(dst))
        homo = homographyFunction(ordering(dst), ordering(corner_rows))
        # homo = homographyFunction(ordering(corner_rows), ordering(dst))
        # cv2.invert(homo)
        # warped_img = cv2.warpPerspective(frame, homo, (200, 200))
        warped_img = warpTag(frame, homo)
        gray_w_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Image_Warped", gray_w_img)
        ID, orientation = tag_id(gray_w_img)
        print(ID, orientation)
        position = tag_reorientation(orientation)

        new_homo = homographyFunction(position, ordering(corner_rows))
        # new_homo = np.linalg.inv(new_homo)

        fram = copy.deepcopy(frame)

        lena_warp = warpLena(lena_img, new_homo, frame.shape)
        # lena_org = copy.deepcopy(lena_warp)
        lena_gray = cv2.cvtColor(lena_warp, cv2.COLOR_BGR2GRAY)
        _, lena_thresh = cv2.threshold(lena_gray, 0, 250, cv2.THRESH_BINARY_INV)
        fram_bit = cv2.bitwise_and(fram, fram, mask=lena_thresh)
        # lena_warp = cv2.warpPerspective(lena_img, new_homo, (frame.shape[1], frame.shape[0]))
        new_lena = cv2.add(fram_bit, lena_warp)
        # cv2.putText(new_lena, [ID + orientation], (corner_rows[0][0] - 50, corner_rows[0][0] - 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        out.write(new_lena)
        cv2.imshow('lena_warped', new_lena)     # Updated till here

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


choice = int(input("Select option 0. Tag0    1. Tag1     2. Tag2     3. Multiple Tags: "))

if choice == 0:
    vid = cv2.VideoCapture('Video_dataset/Tag0.mp4')
elif choice == 1:
    vid = cv2.VideoCapture('Video_dataset/Tag1.mp4')
elif choice == 2:
    vid = cv2.VideoCapture('Video_dataset/Tag2.mp4')
elif choice == 3:
    vid = cv2.VideoCapture('Video_dataset/multipleTags.mp4')

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (1920, 1080))

while True:
    _, frm = vid.read()
    resize_image = cv2.resize(frm, (0, 0), fx=0.5, fy=0.5)
    run(resize_image, p1)
    if cv2.waitKey(1) & 0xff == ord('q'):
        cv2.destroyAllWindows()
vid.release()
out.release()
