import cv2
import numpy as np

image = cv2.imread('reference_images/ref_marker.png', 0)

ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
crop = thresh[50:150, 50:150]

# cv2.imshow('cropped image', crop)

block_1 = crop[37, 37]
block_2 = crop[37, 63]
block_4 = crop[63, 37]
block_3 = crop[63, 63]

block_1 = 1 if block_1 == 255 else 0
block_2 = 1 if block_2 == 255 else 0
block_3 = 1 if block_3 == 255 else 0
block_4 = 1 if block_4 == 255 else 0

if crop[87, 87] == 255:
    print('Tag ID: ', block_4, block_3, block_2, block_1, ' at Bottom Right')
elif crop[13, 13] == 255:
    print('Tag ID: ', block_2, block_1, block_4, block_3, 'Top Left')
elif crop[13, 13] == 255:
    print('Tag ID: ', block_3, block_2, block_1, block_4, 'Bottom Left')
elif crop[13, 13] == 255:
    print('Tag ID: ', block_1, block_4, block_3, block_2, 'Bottom Right')
# print(block_1, block_2, block_3, block_4)
# cv2.waitKey(0)
