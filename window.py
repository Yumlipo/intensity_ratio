import cv2
from screeninfo import get_monitors
import numpy as np


import ctypes  # An included library with Python install.
def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

message = "    To highlight a point, press it with a double click.\n    Once you select a point, click next to the background to compare the intensity for it.\n    You can select several points. Do not forget for all of them to select a pair with background immediately.\n    After you have selected all the points, press the Enter for further processing."
Mbox('Program usage rules', message, 1)

SELECT = 0
screen = get_monitors()
# print(screen[0].height)


for monitor in get_monitors():
    work_area = [monitor.width, monitor.height - 100]

    print(str(work_area[0]) + 'x' + str(work_area[1]))



crds = np.array([])
def mouse_action(event, x, y, flags, param):
    global crds, img, selected_img, union_img, img_ori, SELECT, I

    if event == cv2.EVENT_LBUTTONDBLCLK:
        SELECT = 1
        crds = np.append(crds, [x, y])#save click coordinates
        print("crds", crds)
        cv2.rectangle(img, (x-3, y-3), (x+3, y+3), (255, 255, 0), 1)#draw rectangle on full img

        selected_img = img_ori[y-4:y+5, x-4:x+5].copy()#copy selected area with extension so that the frame does not overlap data
        cv2.rectangle(selected_img, (0, 0), (8, 8), (255, 255, 255), 1)
        I = np.append(I, calculate_I(selected_img[1:8, 1:8]))
        # print("III", I)

        if crds.shape[0] > 2:#we need to add this selected area to previuos
            union_img = np.concatenate((union_img, selected_img), axis=1)
            # cv2.imwrite('out.png', union_img)
        else:
            union_img = selected_img
        cv2.imshow("Selected Images", union_img)

        text = str(int((crds.shape[0] + 2) / 4))#add number of selected area
        coordinates = (x-5, y-5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3
        color = (255, 0, 255)
        thickness = 1
        img = cv2.putText(img, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Z project", img)

def calculate_I(img):
    Intens = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            Intens += 0.299 * img[i, j, 2] + 0.587 * img[i, j, 1] + 0.114 * img[i, j, 0]
    return round(Intens)



#------------change file name
img_ori = cv2.imread("pics/1.jpg")

cv2.namedWindow('Z project', cv2.WINDOW_FULLSCREEN)
cv2.moveWindow('Z project', int(0.1 * work_area[0]), int(0.1 * work_area[1]))
win = cv2.getWindowImageRect("Z project")



img_ori_h, img_ori_w = img_ori.shape[0:2] # original image width and height
img = img_ori.copy()
selected_img = []
union_img = []
cv2.imshow("Z project", img)
I = np.array([])


cv2.setMouseCallback('Z project', mouse_action)
