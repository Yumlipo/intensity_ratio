# import cv2
# from screeninfo import get_monitors
#
# # img_ori = cv2.imread("pics/1.bmp")
# #
# # cv2.namedWindow('Kymo', cv2.WINDOW_FULLSCREEN)
# # win = cv2.getWindowImageRect("Kymo")
# #
# # cv2.imshow("Kymo", img_ori)
# #
# # while cv2.getWindowProperty("Kymo", cv2.WND_PROP_VISIBLE) > 0:
# #     # display the image and wait for a keypress
# #     key = cv2.waitKey(1) & 0xFF
# #
# #     if key == ord("q"):
# #         cv2.destroyAllWindows()
# #         break
#
# def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
#     dim = None
#     (h, w) = image.shape[:2]
#
#     if width is None and height is None:
#         return image
#     if width is None:
#         r = height / float(h)
#         dim = (int(w * r), height)
#     else:
#         r = width / float(w)
#         dim = (width, int(h * r))
#
#     return cv2.resize(image, dim, interpolation=inter)
#
#
#
# image = cv2.imread('pics/1.jpg')
# resize = ResizeWithAspectRatio(image, width=1280) # Resize by width OR
# # resize = ResizeWithAspectRatio(image, height=1280) # Resize by height
#
# cv2.imshow('resize', resize)
# cv2.waitKey()

#
#
# cv2.imshow("Display window", img)
# k = cv2.waitKey(0)
# if k == ord("s"):
#     cv2.imwrite("starry_night.png", img)

# import the necessary packages
import argparse
# import cv2
# # initialize the list of reference points and boolean indicating
# # whether cropping is being performed or not
# refPt = []
# cropping = False
# def click_and_crop(event, x, y, flags, param):
# 	# grab references to the global variables
# 	global refPt, cropping
# 	# if the left mouse button was clicked, record the starting
# 	# (x, y) coordinates and indicate that cropping is being
# 	# performed
# 	if event == cv2.EVENT_LBUTTONDOWN:
# 		refPt = [(x, y)]
# 		cropping = True
# 	# check to see if the left mouse button was released
# 	elif event == cv2.EVENT_LBUTTONUP:
# 		# record the ending (x, y) coordinates and indicate that
# 		# the cropping operation is finished
# 		refPt.append((x, y))
# 		cropping = False
# 		# draw a rectangle around the region of interest
# 		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
# 		cv2.imshow("image", image)
#
#
# image = cv2.imread('pics/1.jpg')
# clone = image.copy()
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", click_and_crop)
# # keep looping until the 'q' key is pressed
# while True:
# 	# display the image and wait for a keypress
# 	cv2.imshow("image", image)
# 	key = cv2.waitKey(1) & 0xFF
# 	# if the 'r' key is pressed, reset the cropping region
# 	if key == ord("r"):
# 		image = clone.copy()
# 	# if the 'c' key is pressed, break from the loop
# 	elif key == ord("c"):
# 		break
# # if there are two reference points, then crop the region of interest
# # from teh image and display it
# if len(refPt) == 2:
# 	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
# 	cv2.imshow("ROI", roi)
# 	cv2.waitKey(0)
# # close all open windows
# cv2.destroyAllWindows()

import cv2
import numpy as np
img1 = cv2.imread('pics/1.jpg')
img2 = cv2.imread('pics/1.jpg')
vis = np.concatenate((img1, img2), axis=1)
cv2.imwrite('out.png', vis)

cv2.namedWindow("image")
cv2.imshow("image", vis)
cv2.waitKey(0)

