import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


class TrackingBoard:

    dot_radius = 4
    wintitle = "trackingboard"

    def __init__(self, override_screensize=None):
        self.leftpos = (0, 0)
        self.rightpos = (0, 0)
        self.centroid = (0, 0)
        if override_screensize:
            self.screen_size = override_screensize
        else:
            mng = plt.get_current_fig_manager()
            self.screen_size = (
                mng.window.winfo_screenwidth(),
                mng.window.winfo_screenheight(),
            )
        print(self.screen_size)
        plt.close()
        cv2.namedWindow(self.wintitle)
        cv2.setWindowProperty(self.wintitle, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def update(self, right, left, centroid):
        self.rightpos = right
        self.leftpos = left
        self.centroid = centroid
        self.draw_dots()

    def draw_dots(self):
        black = np.ones((self.screen_size[1], self.screen_size[0]), dtype=np.ubyte)
        intpos_right = (int(self.rightpos[0]), self.screen_size[1] - int(self.rightpos[1]))
        intpos_left = (int(self.leftpos[0]), self.screen_size[1] - int(self.leftpos[1]))
        intpos_centroid = (int(self.centroid[0]), self.screen_size[1] - int(self.centroid[1]))
        pic_to_display = cv2.cvtColor(black, cv2.COLOR_GRAY2RGB)
        cv2.circle(pic_to_display, intpos_right, self.dot_radius, (0, 255, 0), -1)
        cv2.circle(pic_to_display, intpos_left, self.dot_radius, (255, 0, 0), -1)
        cv2.circle(pic_to_display, intpos_centroid, self.dot_radius*2, (0, 0, 255), -1)
        cv2.imshow(self.wintitle, pic_to_display)
