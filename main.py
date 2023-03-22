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
import window
import ctypes  # An included library with Python install.
def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

message = "    To highlight a point, press it with a double click.\n    Once you select a point, click next to the background to compare the intensity for it.\n    You can select several points. Do not forget for all of them to select a pair with background immediately."
Mbox('Program usage rules', message, 1)


data_t = np.array([])
data_D = np.array([])
data_MSD = []
rename_flag = 0

while cv2.getWindowProperty("I Ratio", cv2.WND_PROP_VISIBLE) > 0:
    # display the image and wait for a keypress
    key = cv2.waitKey(1) & 0xFF

    if len(window.crds) != 0:
        print(window.selected_img)

    # if key == 13:  # If 'enter' is pressed, apply stats
    #     if window.selection_state == window.STATE_SELECTED:
    #         if cv2.getWindowProperty("Selected Region", cv2.WND_PROP_VISIBLE):
    #             cv2.destroyWindow("Selected Region")
    #         if window.crds[0][1] > window.crds[1][1]:
    #             window.crds[0][1], window.crds[1][1] = window.crds[1][1], window.crds[0][1]
    #         if window.crds[0][0] > window.crds[1][0]:
    #             window.crds[0][0], window.crds[1][0] = window.crds[1][0], window.crds[0][0]
    #         image_roi = window.img_ori[:, window.crds[0][0]:window.crds[1][0]]
    #
    #     #в этом массиве хранится картинка
    #     numpydata = np.asarray(window.img_ori[0:window.img_ori.shape[0], window.x1:window.x2])
    #     #массив для хранения интенсивностей из каждой строки
    #     # fit_I0 = np.array([0])
    #     # x_data = np.linspace(0, numpydata.shape[1], numpydata.shape[1], dtype=int).flatten()
    #     # for i, row in enumerate(numpydata):
    #     #     #переводим данные о цвете пиксела в градации серого, те в яркость
    #     #     I = 0.299 * row[:, 2] + 0.587 * row[:, 1] + 0.114 * row[:, 0]
    #     #     I = I.flatten()
    #     #
    #     #     #берем среднюю интенсивность в строке и добавляем в большой массив
    #     #     fit_I0 = np.append(fit_I0, np.mean(I))
    #     #
    #     #     # печать 10-ой строки, чтобы проверять, всё ли на месте
    #     #     if (i == 10):
    #     #         # fit_I0 = np.delete(fit_I0, 0)
    #     #         plt.plot(x_data, I, 'o', label='data')
    #     #         plt.show()
    #     # #уменьшаем количество данные, записывая каждое 10 значение, на всякий случай
    #     # I0 = fit_I0[::10]
    #     # yI0 = np.linspace(1, numpydata.shape[0]+1, int(numpydata.shape[0]/10), dtype=int)
    #     #
    #     # #массив времени
    #     # tt = np.linspace(0, numpydata.shape[0]+1, numpydata.shape[0]+1, dtype=int)
    #     # plt.plot(tt, fit_I0, label="интенсивность в каждой строке")
    #     #
    #     #
    #     # #линейная аппроксимация интенсивности
    #     # # z = np.polyfit(yI0, I0, 1)
    #     # # p=np.poly1d(z)
    #     # # plt.plot(yI0, I0, '.', label="интенсивность в каждой 10 точке")
    #     # # plt.plot(tt, p(tt), '-', label="линейное приближение")
    #     #
    #     #
    #     # plt.xlabel("t, кадры")
    #     # plt.ylabel("интенсивность полоски")
    #     # # plt.xlim(0, 500)
    #     #
    #     # # #горизонтальные линии, чтобы определить ступеньки
    #     # # #ВВЕСТИ ВРЕМЕННЫЕ ОТРЕЗКИ, ГДЕ НАБЛЮДАЮТСЯ СПАДЫ ИНТЕНСИВНОСТИ
    #     # I1_mean = np.mean(fit_I0[:700])
    #     # I2_mean = np.mean(fit_I0[700:])
    #     # # I3_mean = np.mean(fit_I0[1400:2000])
    #     # # # I4_mean = np.mean(fit_I0[1200:])
    #     # #
    #     # # plt.hlines(I1_mean, xmin=0, xmax=700, color="red")
    #     # # plt.hlines(I2_mean, xmin=700, xmax=2000, color="red")
    #     # # # plt.hlines(I3_mean, xmin=1400, xmax=2000, color="red")
    #     # # text = "dI12 = " + str(round((I1_mean-I2_mean)/I1_mean*100, 2)) + "%"# + " dI23 = " + str(round((I2_mean - I3_mean) / I2_mean * 100, 2)) #+ "%" + " dI34 = " + str(round((I3_mean - I4_mean) / I3_mean * 100, 2)) + "%"
    #     # # plt.text(850, 122, text)
    #     # # # text2 = "tau1 = " + str(round(850 * 0.02, 2)) + "s" + " tau2 = " + str(round(550 * 0.02, 2)) + "s" + " tau3 = " + str(round(600 * 0.02, 2)) + "s"
    #     # # # plt.text(850, 110, text2)
    #     # #
    #     # plt.legend()
    #     # plt.show()
    #
    #     # numpydata = np.asarray(img[10:11, 0:img.shape[1]])
    #     #Gauss fitting
    #     fit_I0 = np.array([0])
    #     fit_x0 = np.array([0])
    #     numpydata = np.asarray(image_roi)
    #     for i, row in enumerate(numpydata):
    #         # Для каждой строки пикселей фиттим данных гауссом и получаем константы
    #         # print(img_d.shape[1])
    #         I = 0.299 * row[:, 2] + 0.587 * row[:, 1] + 0.114 * row[:, 0]
    #         I = I.flatten()
    #         xx_data = np.linspace(0, row.shape[0], row.shape[0], dtype=int).flatten()
    #         xx_fit = np.linspace(0, row.shape[0], row.shape[0] * 10).flatten()
    #
    #         fit_I0 = np.append(fit_I0, np.mean(I))
    #
    #         # Define the Gaussian function
    #         # def gauss(x, I0, x0, sigma, y0):
    #         #     return I0 * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + y0
    #         #
    #         # try:
    #         #     parameters, covariance = curve_fit(gauss, xx_data, I, p0=(100, 5, 5, 20),
    #         #                                        bounds=([0, 0, -100, 0], [1000, 100, 100, 100]))
    #         #     fit_I0 = np.append(fit_I0, parameters[0])
    #         #     fit_x0 = np.append(fit_x0, parameters[1])
    #         #     fit_sigma = parameters[2]
    #         #     fit_y0 = parameters[3]
    #         # except RuntimeError:
    #         #     fit_I0 = np.append(fit_I0, np.median(fit_I0))
    #         #     fit_x0 = np.append(fit_x0, np.median(fit_x0))
    #         #     fit_sigma = 1
    #         #     fit_y0 = 0
    #         #
    #         # # печать 10-ой строки
    #         # if (i == 10):
    #         #     fit_I0 = np.delete(fit_I0, 0)
    #         #     fit_x0 = np.delete(fit_x0, 0)
    #         #     fit_y = gauss(xx_fit, fit_I0[-1], fit_x0[-1], fit_sigma, fit_y0)
    #         #     plt.plot(xx_data, I, 'o', label='data')
    #         #     plt.plot(xx_fit, fit_y, '-', label='fit')
    #         #     plt.legend()
    #         #     plt.show()
    #     # fit_I0 = np.delete(fit_I0, 0)
    #     # fit_x0 = np.delete(fit_x0, 0)
    #
    #     # По ближайшим точкам
    #     fit_I0[np.flatnonzero(np.abs(fit_I0[:-1] - fit_I0[1:]) >= 100) + 1] = np.nan
    #
    #     # обрезаем все, что слишком сильно выбивается
    #     err = 0.2
    #     amax = np.median(fit_I0) * (1. + err)
    #     amin = np.median(fit_I0) * (1. - err)
    #     fit_I0[(fit_I0 > amax) | (fit_I0 < amin)] = np.nan
    #
    #     fit_I0[(fit_I0 > 400)] = np.nan
    #
    #     # интерполяция пропусков
    #     mask = np.isnan(fit_I0)
    #     fit_I0[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), fit_I0[~mask])
    #
    #     x_data = np.linspace(0, fit_I0.shape[0], fit_I0.shape[0], dtype=int).flatten()
    #
    #
    #     # уменьшаем количество данные, записывая каждое 10 значение, на всякий случай
    #     I0 = fit_I0[::10]
    #     # yI0 = np.linspace(0, numpydata.shape[0], int(numpydata.shape[0]/10), dtype=int)
    #     yI0 = np.arange(0, fit_I0.shape[0], 10)
    #     print(I0.shape, yI0.shape)
    #
    #     #массив времени
    #     tt = np.linspace(0, fit_I0.shape[0], fit_I0.shape[0], dtype=int)
    #     plt.plot(tt[1:], fit_I0[1:], label="интенсивность в каждой строке")
    #
    #     #линейная аппроксимация интенсивности
    #     z = np.polyfit(yI0, I0, 1)
    #     p=np.poly1d(z)
    #     plt.plot(yI0, I0, '.', label="интенсивность в каждой 10 точке")
    #     # plt.plot(tt, p(tt), '-', label="линейное приближение")
    #
    #     # #горизонтальные линии, чтобы определить ступеньки
    #     # #ВВЕСТИ ВРЕМЕННЫЕ ОТРЕЗКИ, ГДЕ НАБЛЮДАЮТСЯ СПАДЫ ИНТЕНСИВНОСТИ
    #     I1_mean = np.mean(fit_I0[:280])
    #     I2_mean = np.mean(fit_I0[280:4000])
    #     # I3_mean = np.mean(fit_I0[1750:])
    #     # # I4_mean = np.mean(fit_I0[1200:])
    #     #
    #     # plt.hlines(I1_mean, xmin=0, xmax=280, color="red")
    #     # plt.hlines(I2_mean, xmin=280, xmax=4000, color="red")
    #     # plt.hlines(I3_mean, xmin=1750, xmax=4000, color="red")
    #     # text = "dI12 = " + str(round((I1_mean - I2_mean) / I1_mean * 100,
    #     #                              2)) + "%"#  + " dI23 = " + str(round((I2_mean - I3_mean) / I2_mean * 100, 2)) #+ "%" + " dI34 = " + str(round((I3_mean - I4_mean) / I3_mean * 100, 2)) + "%"
    #     # plt.text(0.1*np.max(tt), 0.9*np.max(fit_I0), text)
    #     # # text2 = "tau1 = " + str(round(850 * 0.02, 2)) + "s" + " tau2 = " + str(round(550 * 0.02, 2)) + "s" + " tau3 = " + str(round(600 * 0.02, 2)) + "s"
    #     # # plt.text(850, 110, text2)
    #     #
    #
    #     plt.xlabel("t, кадры")
    #     plt.ylabel("интенсивность полоски")
    #     plt.legend()
    #     plt.show()


    if key == ord("q"):
        cv2.destroyAllWindows()
        break

