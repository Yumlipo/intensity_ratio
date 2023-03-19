import io

import cv2
from matplotlib import pyplot as plt
import numpy as np
from screeninfo import get_monitors
from numpy import genfromtxt
# import json
import statistics
import os
from scipy.optimize import curve_fit
from win32api import GetMonitorInfo, MonitorFromPoint

import window


screen = get_monitors()
# print(screen[0].height)

from screeninfo import get_monitors

for monitor in get_monitors():
    width = monitor.width
    height = monitor.height

    print(str(width) + 'x' + str(height))

monitor_info = GetMonitorInfo(MonitorFromPoint((0,0)))
work_area = monitor_info.get("Work")
print("The work area size is {}x{}.".format(work_area[2], work_area[3]))

STATE_NO_SELECTION = 0
STATE_SELECTING = 1
STATE_SELECTED = 2

crds = []
selection_state = 0

#Mouse event
def mouse(event, x, y, flags, param):
    global crds, selection_state
    global flag, horizontal, vertical, flag_hor, flag_ver, dx, dy, sx, sy, dst, x1, y1, x2, y2, x3, y3, f1, f2
    global old_w, board_x, z_y, z_x, zoom_h, zoom_w, zoom_max, zoom_min, zoom, zoom_pos, scroll_har, scroll_var, img_w, img_h, img, dst1, win_w, win_h, show_w, show_h
    if selection_state != STATE_SELECTING:
        dst = img
        if event == cv2.EVENT_LBUTTONDOWN: # Left click
            if flag == 0:
                if horizontal and 0 < x < win_w and win_h-scroll_w < y < win_h:
                    flag_hor = 1 # The mouse is on the horizontal scroll bar
                elif vertical and win_w-scroll_w <x <win_w and 0 <y <win_h:
                    flag_ver = 1 # The mouse is on the vertical scroll bar
                if flag_hor or flag_ver:
                    flag = 1 # Make the scroll bar vertical
                    x1, y1, x2, y2, x3, y3 = x, y, dx, dy, sx, sy # Make the mouse move distance relative to the initial scroll bar click position, not relative to the previous position
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON): # Hold down the left button and drag
            if flag == 1:
                if flag_hor:
                    w = x-x1 # Moving width
                    dx = x2 + w * f1 # original image x
                    if dx < 0: # position correction
                        dx = 0
                    elif dx > img_w-show_w:
                        dx = img_w-show_w
                    sx = x3 + w # scroll bar x
                    if sx < 0: # position correction
                        sx = 0
                    elif sx > win_w-scroll_har:
                        sx = win_w-scroll_har
                if flag_ver:
                    h = y-y1 # moving height
                    dy = y2 + h * f2 # original image y
                    if dy < 0: # position correction
                        dy = 0
                    elif dy > img_h-show_h:
                        dy = img_h-show_h
                    sy = y3 + h # scroll bar y
                    if sy <0: # position correction
                        sy = 0
                    elif sy> win_h-scroll_var:
                        sy = win_h-scroll_var
                dx, dy = int(dx), int(dy)
                img1 = img[dy:dy + show_h, dx:dx + show_w] # Take a screenshot for display
                # print(dy, dy + show_h, dx, dx + show_w)
                dst = img1.copy()
        elif event == cv2.EVENT_LBUTTONUP: # Left key release
            flag, flag_hor, flag_ver = 0, 0, 0
            x1, y1, x2, y2, x3, y3 = 0, 0, 0, 0, 0, 0
        elif event == cv2.EVENT_MOUSEWHEEL: # scroll wheel
            if flags> 0: # scroll wheel up
                zoom_pos += 1
                # print("zoom_w", (zoom_w-old_w*2), "win_w", win_w)
                if (zoom_w-old_w*2) >= win_w/4 and z_x != 1:
                    zoom_min = zoom_pos + 1
                    z_x = 1
                if zoom_pos > zoom_min: # zoom factor adjustment
                    zoom_pos = zoom_min
            else: # scroll wheel down
                zoom_pos -= 1
                # print("zoom_h", zoom_h, "win_h", win_h)
                if zoom_h <= win_h and z_y != 1:
                    zoom_max = -zoom_pos-1
                    z_y = 1
                if zoom_pos < -zoom_max: # zoom factor adjustment
                    zoom_pos = -zoom_max

            # print("zoom_pos", zoom_pos)
            zoom = (1 + wheel_step) ** zoom_pos
            zoom = round(zoom, 2) # Take 2 significant digits
            img_w, img_h = int(img_ori_w * zoom), int(img_ori_h * zoom) # zooming is relative to the original image, not iteration
            img = cv2.resize(img_ori, (img_w, img_h), interpolation=cv2.INTER_AREA)
            img, new_w = draw_borders(img, win_w, img_w)
            board_x = old_w - new_w
            old_w = new_w
            zoom_w, zoom_h = img.shape[1], img.shape[0]
            # print("img_h", img_h, "img_zoom_h", zoom_h, "show_h", show_h)
            # print("img_w", img_w, "img_zoom_w", zoom_w, "show_w", show_w)
            horizontal, vertical = 0, 0
            if img_h <= win_h and img_w <= win_w:
                dst1 = img
                cv2.resizeWindow("Kymo", win_w, win_h)
                scroll_har, scroll_var = 0, 0
                f1, f2 = 0, 0
            else:
                if img_w > win_w and img_h > win_h:
                    horizontal, vertical = 1, 1
                    scroll_har, scroll_var = win_w * show_w/img_w, win_h * show_h/img_h
                    f1, f2 = (img_w-show_w)/(win_w-scroll_har), (img_h-show_h)/(win_h-scroll_var)
                elif img_w > win_w and img_h <= win_h:
                    #show_h = img_h
                    # win_h = show_h + scroll_w
                    horizontal = 1
                    scroll_har, scroll_var = win_w * show_w/img_w, 0
                    f1, f2 = (img_w-show_w)/(win_w-scroll_har), 0
                elif img_w <= win_w and img_h> win_h:
                    #show_w = img_w
                    # win_w = show_w + scroll_w
                    vertical = 1
                    scroll_har, scroll_var = 0, win_h * show_h/img_h
                    f1, f2 = 0, (img_h-show_h)/(win_h-scroll_var)
                dx, dy = dx * zoom, dy * zoom # After zooming, display the coordinates of the relative zoomed image
                sx, sy = dx/img_w * (win_w-scroll_har), dy/img_h * (win_h-scroll_var)
                dx, dy = int(dx), int(dy)
                dst = img[dy:dy + show_h, dx:dx + show_w]
        dst1 = draw_scroll_bar(dst, horizontal, vertical, sx, sy)

    if flag == 0:
        if x < old_w:
            x = old_w
        elif x >= old_w + img_w:
            x = old_w + img_w + 1
        if event == cv2.EVENT_LBUTTONDOWN and selection_state != STATE_SELECTING:
            crds = [[int((x+dx - old_w) / zoom), int((y + dy) / zoom)],
                    [int((x+dx - old_w) / zoom), int((y + dy) / zoom)]]
            # print(crds)
            selection_state = STATE_SELECTING
        elif selection_state == STATE_SELECTING:
            crds[1] = [int((x+dx - old_w) / zoom), int((y + dy) / zoom)]
            if event == cv2.EVENT_LBUTTONUP:
                if crds[1] == crds[0]:
                    selection_state = STATE_NO_SELECTION
                else:
                    selection_state = STATE_SELECTED
    if selection_state != STATE_NO_SELECTION:
        dst2 = dst1.copy()
        new_crds = [[int(crds[0][0]*zoom - dx + old_w), int(crds[0][1] * zoom - dy)],
                    [int(crds[1][0]*zoom - dx + old_w), int(crds[1][1] * zoom - dy)]]
        # print("new", new_crds)
        draw_rect(dst2, new_crds)
        cv2.imshow("Kymo", dst2)
    else:
        cv2.imshow("Kymo", dst1)
    cv2.waitKey(1)



def draw_scroll_bar(dst, horizontal, vertical, sx, sy):
    if horizontal and vertical:
        sx, sy = int(sx), int(sy)
        # Draw a picture on dst1 instead of dst to avoid continuous refreshing of mouse events so that the displayed picture is constantly filled
        dst1 = cv2.copyMakeBorder(dst, 0, scroll_w, 0, scroll_w, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.rectangle(dst1, (sx, show_h), (int(sx + scroll_har), win_h), (181, 181, 181),
                      -1)  # draw horizontal scroll bar
        cv2.rectangle(dst1, (show_w, sy), (win_w, int(sy + scroll_var)), (181, 181, 181),
                      -1)  # draw vertical scroll bar
    elif horizontal == 0 and vertical:
        sx, sy = int(sx), int(sy)
        dst1 = cv2.copyMakeBorder(dst, 0, 0, 0, scroll_w, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.rectangle(dst1, (show_w, sy), (win_w, int(sy + scroll_var)), (181, 181, 181),
                      -1)  # draw vertical scroll bar
    elif horizontal and vertical == 0:
        sx, sy = int(sx), int(sy)
        dst1 = cv2.copyMakeBorder(dst, 0, scroll_w, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.rectangle(dst1, (sx, show_h), (int(sx + scroll_har), win_h), (181, 181, 181),
                      -1)  # draw horizontal scroll bar
    else:
        dst1 = dst.copy()
    return dst1

def draw_rect(img, coordinates, color=(0, 0, 255), thickness=1):
    cv2.rectangle(img, coordinates[0], coordinates[1], color, thickness)

def draw_borders(img, win_w, img_w):
    # create borders
    top = 0  # shape[0] = rows
    bottom = top
    left = int((win_w - img_w) / 2)  # shape[1] = cols
    right = left
    # print("right", right)
    borderType = cv2.BORDER_CONSTANT
    return cv2.copyMakeBorder(img, top, bottom, left, right, borderType, None, [0, 0, 0]), right
    # cv2.imshow("Kymo", img)



#contrast correction tiff
# img_ori = cv2.imread("1.bmp", cv2.IMREAD_GRAYSCALE)

#------------ИЗМЕНИТЬ НАЗВАНИЕ ФАЙЛА
img_ori = cv2.imread("pics/1.bmp")
# img_ori = (img_ori - img_ori.min()) * 255 // (img_ori.max() - img_ori.min())
# img_ori = img_ori.astype(np.uint8)
# print(img_ori.max(), img_ori.min())

#create a window
cv2.namedWindow('Kymo', cv2.WINDOW_FULLSCREEN)
win = cv2.getWindowImageRect("Kymo")

img_ori_h, img_ori_w = img_ori.shape[0:2] # original image width and height
img = img_ori.copy()

scroll_w = 30 # scroll bar width
show_w, show_h = work_area[2] - scroll_w*2, work_area[3] - scroll_w*2

img, old_w = draw_borders(img_ori, show_w, img.shape[1])
img_b = img.copy()

img_h, img_w = img.shape[0:2] # original image width and height

horizontal, vertical = 0, 0 # Whether the original image exceeds the displayed image
dx, dy = 0, 0 # Display the coordinates of the picture relative to the original picture
sx, sy = 0, 0 # The coordinates of the scroll block relative to the scroll bar
flag, flag_hor, flag_ver = 0, 0, 0 # Mouse operation type, whether the mouse is on the horizontal scroll bar, and whether the mouse is on the vertical scroll bar
x1, y1, x2, y2, x3, y3 = 0, 0, 0, 0, 0, 0 # intermediate variables
board_x = 0

win_w, win_h = show_w + scroll_w, show_h + scroll_w # window width and height
# win_w, win_h = work_area[2], work_area[3]
scroll_har, scroll_var = win_w * show_w/img_w, win_h * show_h/img_h # scroll bar horizontal and vertical length
wheel_step, zoom, zoom_pos = 0.15, 1, 0  # zoom factor, zoom value
zoom_max, zoom_min = 100, 100
z_y, z_x = 0, 0
# print("zoom_min", zoom_min)
zoom_w, zoom_h = img_w, img_h # zoom image width and height

# print(img_ori_w, img_w, show_w, win_w, scroll_har)
# print(img_ori_h, img_h, show_h, win_h, scroll_var)

if win_w == scroll_har:
    f1, f2 = 0, (img_ori_h-show_h)/(win_h-scroll_var) # The proportion of the movable part of the original image to the movable part of the scroll bar
elif win_h == scroll_har:
    f1, f2 = (img_ori_w-show_w)/(win_w-scroll_har), 0 # The proportion of the movable part of the original image to the movable part of the scroll bar
else:
    f1, f2 = (img_ori_w-show_w)/(win_w-scroll_har), (img_ori_h-show_h)/(win_h-scroll_var) # The proportion of the movable part of the original image to the movable part of the scroll bar

print (f1, f2)

# cv2.namedWindow('Kymo', cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("Kymo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# print("img_h", img_h, "show_h", show_h)
# print("img_w", img_w, "show_w", show_w)

if img_h <= show_h and img_w <= show_w:
    cv2.imshow("Kymo", img)
    # print("1")
else:
    if img_w > show_w:
        horizontal = 1
    if img_h > show_h:
        vertical = 1
    i = img[dy:dy + show_h, dx:dx + show_w]
    dst = i.copy()
    # dst = draw_scroll_bar(dst, horizontal, vertical, sx, sy)
    cv2.imshow("Kymo", dst)

print(win)
# cv2.resizeWindow("Kymo", win_w, win_h)
cv2.setMouseCallback('Kymo', mouse)

# file_data = open('data.csv', 'w')
# file_MSD = open('data_MSD.csv', 'w')

data_t = np.array([])
data_D = np.array([])
data_MSD = []
rename_flag = 0

while cv2.getWindowProperty("Kymo", cv2.WND_PROP_VISIBLE) > 0:
    # display the image and wait for a keypress
    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # If 'enter' is pressed, apply stats
        if selection_state == STATE_SELECTED:
            if cv2.getWindowProperty("Selected Region", cv2.WND_PROP_VISIBLE):
                cv2.destroyWindow("Selected Region")
            if crds[0][1] > crds[1][1]:
                crds[0][1], crds[1][1] = crds[1][1], crds[0][1]
            if crds[0][0] > crds[1][0]:
                crds[0][0], crds[1][0] = crds[1][0], crds[0][0]
            image_roi = img_ori[:, crds[0][0]:crds[1][0]]

        #в этом массиве хранится картинка
        numpydata = np.asarray(img_ori[0:img_ori.shape[0], x1:x2])
        #массив для хранения интенсивностей из каждой строки
        # fit_I0 = np.array([0])
        # x_data = np.linspace(0, numpydata.shape[1], numpydata.shape[1], dtype=int).flatten()
        # for i, row in enumerate(numpydata):
        #     #переводим данные о цвете пиксела в градации серого, те в яркость
        #     I = 0.299 * row[:, 2] + 0.587 * row[:, 1] + 0.114 * row[:, 0]
        #     I = I.flatten()
        #
        #     #берем среднюю интенсивность в строке и добавляем в большой массив
        #     fit_I0 = np.append(fit_I0, np.mean(I))
        #
        #     # печать 10-ой строки, чтобы проверять, всё ли на месте
        #     if (i == 10):
        #         # fit_I0 = np.delete(fit_I0, 0)
        #         plt.plot(x_data, I, 'o', label='data')
        #         plt.show()
        # #уменьшаем количество данные, записывая каждое 10 значение, на всякий случай
        # I0 = fit_I0[::10]
        # yI0 = np.linspace(1, numpydata.shape[0]+1, int(numpydata.shape[0]/10), dtype=int)
        #
        # #массив времени
        # tt = np.linspace(0, numpydata.shape[0]+1, numpydata.shape[0]+1, dtype=int)
        # plt.plot(tt, fit_I0, label="интенсивность в каждой строке")
        #
        #
        # #линейная аппроксимация интенсивности
        # # z = np.polyfit(yI0, I0, 1)
        # # p=np.poly1d(z)
        # # plt.plot(yI0, I0, '.', label="интенсивность в каждой 10 точке")
        # # plt.plot(tt, p(tt), '-', label="линейное приближение")
        #
        #
        # plt.xlabel("t, кадры")
        # plt.ylabel("интенсивность полоски")
        # # plt.xlim(0, 500)
        #
        # # #горизонтальные линии, чтобы определить ступеньки
        # # #ВВЕСТИ ВРЕМЕННЫЕ ОТРЕЗКИ, ГДЕ НАБЛЮДАЮТСЯ СПАДЫ ИНТЕНСИВНОСТИ
        # I1_mean = np.mean(fit_I0[:700])
        # I2_mean = np.mean(fit_I0[700:])
        # # I3_mean = np.mean(fit_I0[1400:2000])
        # # # I4_mean = np.mean(fit_I0[1200:])
        # #
        # # plt.hlines(I1_mean, xmin=0, xmax=700, color="red")
        # # plt.hlines(I2_mean, xmin=700, xmax=2000, color="red")
        # # # plt.hlines(I3_mean, xmin=1400, xmax=2000, color="red")
        # # text = "dI12 = " + str(round((I1_mean-I2_mean)/I1_mean*100, 2)) + "%"# + " dI23 = " + str(round((I2_mean - I3_mean) / I2_mean * 100, 2)) #+ "%" + " dI34 = " + str(round((I3_mean - I4_mean) / I3_mean * 100, 2)) + "%"
        # # plt.text(850, 122, text)
        # # # text2 = "tau1 = " + str(round(850 * 0.02, 2)) + "s" + " tau2 = " + str(round(550 * 0.02, 2)) + "s" + " tau3 = " + str(round(600 * 0.02, 2)) + "s"
        # # # plt.text(850, 110, text2)
        # #
        # plt.legend()
        # plt.show()

        # numpydata = np.asarray(img[10:11, 0:img.shape[1]])
        #Gauss fitting
        fit_I0 = np.array([0])
        fit_x0 = np.array([0])
        numpydata = np.asarray(image_roi)
        for i, row in enumerate(numpydata):
            # Для каждой строки пикселей фиттим данных гауссом и получаем константы
            # print(img_d.shape[1])
            I = 0.299 * row[:, 2] + 0.587 * row[:, 1] + 0.114 * row[:, 0]
            I = I.flatten()
            xx_data = np.linspace(0, row.shape[0], row.shape[0], dtype=int).flatten()
            xx_fit = np.linspace(0, row.shape[0], row.shape[0] * 10).flatten()

            fit_I0 = np.append(fit_I0, np.mean(I))

            # Define the Gaussian function
            # def gauss(x, I0, x0, sigma, y0):
            #     return I0 * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + y0
            #
            # try:
            #     parameters, covariance = curve_fit(gauss, xx_data, I, p0=(100, 5, 5, 20),
            #                                        bounds=([0, 0, -100, 0], [1000, 100, 100, 100]))
            #     fit_I0 = np.append(fit_I0, parameters[0])
            #     fit_x0 = np.append(fit_x0, parameters[1])
            #     fit_sigma = parameters[2]
            #     fit_y0 = parameters[3]
            # except RuntimeError:
            #     fit_I0 = np.append(fit_I0, np.median(fit_I0))
            #     fit_x0 = np.append(fit_x0, np.median(fit_x0))
            #     fit_sigma = 1
            #     fit_y0 = 0
            #
            # # печать 10-ой строки
            # if (i == 10):
            #     fit_I0 = np.delete(fit_I0, 0)
            #     fit_x0 = np.delete(fit_x0, 0)
            #     fit_y = gauss(xx_fit, fit_I0[-1], fit_x0[-1], fit_sigma, fit_y0)
            #     plt.plot(xx_data, I, 'o', label='data')
            #     plt.plot(xx_fit, fit_y, '-', label='fit')
            #     plt.legend()
            #     plt.show()
        # fit_I0 = np.delete(fit_I0, 0)
        # fit_x0 = np.delete(fit_x0, 0)

        # По ближайшим точкам
        fit_I0[np.flatnonzero(np.abs(fit_I0[:-1] - fit_I0[1:]) >= 100) + 1] = np.nan

        # обрезаем все, что слишком сильно выбивается
        err = 0.2
        amax = np.median(fit_I0) * (1. + err)
        amin = np.median(fit_I0) * (1. - err)
        fit_I0[(fit_I0 > amax) | (fit_I0 < amin)] = np.nan

        fit_I0[(fit_I0 > 400)] = np.nan

        # интерполяция пропусков
        mask = np.isnan(fit_I0)
        fit_I0[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), fit_I0[~mask])

        x_data = np.linspace(0, fit_I0.shape[0], fit_I0.shape[0], dtype=int).flatten()


        # уменьшаем количество данные, записывая каждое 10 значение, на всякий случай
        I0 = fit_I0[::10]
        # yI0 = np.linspace(0, numpydata.shape[0], int(numpydata.shape[0]/10), dtype=int)
        yI0 = np.arange(0, fit_I0.shape[0], 10)
        print(I0.shape, yI0.shape)

        #массив времени
        tt = np.linspace(0, fit_I0.shape[0], fit_I0.shape[0], dtype=int)
        plt.plot(tt[1:], fit_I0[1:], label="интенсивность в каждой строке")

        #линейная аппроксимация интенсивности
        z = np.polyfit(yI0, I0, 1)
        p=np.poly1d(z)
        plt.plot(yI0, I0, '.', label="интенсивность в каждой 10 точке")
        # plt.plot(tt, p(tt), '-', label="линейное приближение")

        # #горизонтальные линии, чтобы определить ступеньки
        # #ВВЕСТИ ВРЕМЕННЫЕ ОТРЕЗКИ, ГДЕ НАБЛЮДАЮТСЯ СПАДЫ ИНТЕНСИВНОСТИ
        I1_mean = np.mean(fit_I0[:280])
        I2_mean = np.mean(fit_I0[280:4000])
        # I3_mean = np.mean(fit_I0[1750:])
        # # I4_mean = np.mean(fit_I0[1200:])
        #
        # plt.hlines(I1_mean, xmin=0, xmax=280, color="red")
        # plt.hlines(I2_mean, xmin=280, xmax=4000, color="red")
        # plt.hlines(I3_mean, xmin=1750, xmax=4000, color="red")
        # text = "dI12 = " + str(round((I1_mean - I2_mean) / I1_mean * 100,
        #                              2)) + "%"#  + " dI23 = " + str(round((I2_mean - I3_mean) / I2_mean * 100, 2)) #+ "%" + " dI34 = " + str(round((I3_mean - I4_mean) / I3_mean * 100, 2)) + "%"
        # plt.text(0.1*np.max(tt), 0.9*np.max(fit_I0), text)
        # # text2 = "tau1 = " + str(round(850 * 0.02, 2)) + "s" + " tau2 = " + str(round(550 * 0.02, 2)) + "s" + " tau3 = " + str(round(600 * 0.02, 2)) + "s"
        # # plt.text(850, 110, text2)
        #

        plt.xlabel("t, кадры")
        plt.ylabel("интенсивность полоски")
        plt.legend()
        plt.show()


    if key == ord("q"):
        cv2.destroyAllWindows()
        break

