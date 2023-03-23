import cv2
import numpy as np
from nd2reader import ND2Reader
import matplotlib.pyplot as plt

import window

stack = ND2Reader('pics/example.nd2')
# plt.imshow(stack[5])
print(stack.metadata)
# plt.show()


IminusBG = np.array([])

while cv2.getWindowProperty("Z project", cv2.WND_PROP_VISIBLE) > 0:
    # display the image and wait for a keypress
    # print("waiting")
    key = cv2.waitKey(0) & 0xFF
    # print(key)

    if key == 13:  # If 'enter' is pressed calculate I(point) - I(BG)
        IminusBG = []
        print("I in main", window.I)
        if (window.crds.shape[0] % 4) > 0:
            window.Mbox("Warning", "You need to select a pair of rectangles: the desired point and its background! Now you have an odd number of squares.", 1)
            break
        else:
            for img in stack:
                I_temp = np.array([])
                for x, y in zip(window.crds[::2], window.crds[1::2]):
                    copy_img_from_stack = img[int(y-3):int(y+4), int(x-3):int(x+4)].copy()
                    I_temp = np.append(I_temp, np.sum(copy_img_from_stack))
                # I_point = I_temp[::2]
                # I_BG = I_temp[1::2]
                IminusBG += [I_temp[::2] - I_temp[1::2]]
            IminusBG_arr = np.stack(IminusBG, axis=1)
            print("I-BG, ", IminusBG_arr, " shape ", IminusBG_arr.shape)

            # num_of_points = IminusBG_arr.shape[1]
            t = np.linspace(0, IminusBG_arr.shape[1], IminusBG_arr.shape[1])
            # for i in range(IminusBG_arr.shape[0]):
            #     plt.plot(t, IminusBG_arr[i], label=str(i))
            # plt.xlabel('frames, x ms')
            # plt.ylabel('I(point) - I(BG)')
            # plt.title("Signal to noise from time")
            # plt.legend()
            # plt.show()
            for i in range(IminusBG_arr.shape[0]):
                fig, ax = plt.subplots()
                ax.plot(t, IminusBG_arr[i], label=str(i+1))
                ax.set_xlabel('frames, x ms')
                ax.set_ylabel('I(point) - I(BG)')
                ax.set_title("Signal to noise from time for №" + str(i+1) + " point")
                ax.legend()
            plt.show()



    if key == ord("q"):
        cv2.destroyAllWindows()
        break

