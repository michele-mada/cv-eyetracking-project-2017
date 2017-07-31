import cv2
import matplotlib.pyplot as plt
import numpy as np


class TrackingBoard:

    dot_radius = 4
    wintitle = "trackingboard"

    def __init__(self):
        mng = plt.get_current_fig_manager()
        self.screen_size = (
            mng.window.winfo_screenheight(),
            mng.window.winfo_screenwidth(),
        )
        #self.screen_size = (768,1366)
        print(self.screen_size)
        plt.close()
        cv2.namedWindow(self.wintitle)
        cv2.setWindowProperty(self.wintitle, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def update_right(self, pos):
        intpos = (int(pos[0]), self.screen_size[0] - int(pos[1]))
        white = np.ones(self.screen_size, dtype=np.ubyte)
        pic_to_display = cv2.cvtColor(white, cv2.COLOR_GRAY2RGB)
        cv2.circle(pic_to_display, intpos, self.dot_radius, (0, 255, 0), -1)
        cv2.imshow(self.wintitle, pic_to_display)
