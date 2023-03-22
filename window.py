import cv2
from screeninfo import get_monitors
import numpy as np

SELECT = 0
screen = get_monitors()
# print(screen[0].height)


for monitor in get_monitors():
    work_area = [monitor.width, monitor.height - 100]

    print(str(work_area[0]) + 'x' + str(work_area[1]))



crds = np.array([])
def mouse_action(event, x, y, flags, param):
    global crds, img, selected_img, union_img, img_ori, SELECT

    if event == cv2.EVENT_LBUTTONDBLCLK:
        SELECT = 1
        crds = np.append(crds, [x, y])#save click coordinates
        print("crds", crds)
        cv2.rectangle(img, (x-3, y-3), (x+3, y+3), (255, 255, 0), 1)#draw rectangle on full img

        selected_img = img_ori[y-4:y+5, x-4:x+5].copy()#copy selected area with extension so that the frame does not overlap data
        cv2.rectangle(selected_img, (0, 0), (8, 8), (255, 255, 255), 1)

        if crds.shape[0] > 2:#we need to add this selected area to previuos
            union_img = np.concatenate((union_img, selected_img), axis=1)
            cv2.imwrite('out.png', union_img)
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

    cv2.imshow("I Ratio", img)
    cv2.waitKey(1)




#------------change file name
img_ori = cv2.imread("pics/1.jpg")

cv2.namedWindow('I Ratio', cv2.WINDOW_FULLSCREEN)
cv2.moveWindow('I Ratio', int(0.1 * work_area[0]), int(0.1 * work_area[1]))
win = cv2.getWindowImageRect("I Ratio")



img_ori_h, img_ori_w = img_ori.shape[0:2] # original image width and height
img = img_ori.copy()
selected_img = []
union_img = []
cv2.imshow("I Ratio", img)

cv2.setMouseCallback('I Ratio', mouse_action)