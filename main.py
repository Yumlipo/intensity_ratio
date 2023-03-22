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



data_t = np.array([])
data_D = np.array([])
data_MSD = []
rename_flag = 0
Imain = 0

IminusBG = np.array([])

while cv2.getWindowProperty("I Ratio", cv2.WND_PROP_VISIBLE) > 0:
    # display the image and wait for a keypress
    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # If 'enter' is pressed calculate I(point) - I(BG)
        print("I in main", window.I)
        if (window.I.shape[0] % 2) > 0:
            window.Mbox("Warning", "You need to select a pair of rectangles: the desired point and its background! Now you have an odd number of squares.", 1)
            break
        else:
            print("I[::2]", window.I[::2])
            print("I[1::2]", window.I[1::2])
            IminusBG = np.append(IminusBG, window.I[::2] - window.I[1::2])
            print("I-BG", IminusBG)


    if key == ord("q"):
        cv2.destroyAllWindows()
        break

