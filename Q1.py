import cv2
import numpy as np

image = cv2.imread('reference_images/ref_marker.png', 0)
cap = cv2.VideoCapture('Video_dataset/Tag0.mp4')


def tag_id(image):
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    crop = thresh[50:150, 50:150]

    cell_1 = crop[37, 37]
    cell_2 = crop[37, 63]
    cell_3 = crop[63, 63]
    cell_4 = crop[63, 37]

    cell_1 = 1 if cell_1 == 255 else 0
    cell_2 = 1 if cell_2 == 255 else 0
    cell_3 = 1 if cell_3 == 255 else 0
    cell_4 = 1 if cell_4 == 255 else 0

    if crop[87, 87] == 255:
        return cell_4, cell_3, cell_2, cell_1, 'Bottom_Right'
        # print('Tag ID: ', cell_4, cell_3, cell_2, cell_1, ' at Bottom Right')
    elif crop[13, 13] == 255:
        return cell_2, cell_1, cell_4, cell_3, 'Top_Left'
        # print('Tag ID: ', cell_2, cell_1, cell_4, cell_3, 'Top Left')
    elif crop[87, 13] == 255:
        return cell_3, cell_2, cell_1, cell_4, 'Bottom_Left'
        # print('Tag ID: ', cell_3, cell_2, cell_1, cell_4, 'Bottom Left')
    elif crop[13, 87] == 255:
        return cell_1, cell_4, cell_3, cell_2, 'Top_Right'
        # print('Tag ID: ', cell_1, cell_4, cell_3, cell_2, 'Top_Right')


def detect_corners(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.3, 2)
        corners = np.int0(corners)
        # print(corners)

        # draw corners in video frames
        for c in corners:
            i, j = c.ravel()
            # print('i:', i, 'j:', j)
            cv2.circle(frame, (i, j), 3, 255, -1)
            # print(corners[i])

        cv2.imshow('frame', frame)  # Displays the image
        ID = tag_id(gray)       # if error occurs check gray
        print('Tag ID:', ID)
        # cv.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    cap.release()
    return None


if cap:
    detect_corners(cap)
else:
    pass
