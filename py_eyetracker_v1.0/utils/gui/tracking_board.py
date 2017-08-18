import cv2
import matplotlib.pyplot as plt
import numpy as np


class TrackingBoard:

    dot_radius = 4
    wintitle = "trackingboard"

    def __init__(self):
        self.leftpos = (0, 0)
        self.rightpos = (0, 0)
        mng = plt.get_current_fig_manager()
        self.screen_size = (
            mng.window.winfo_screenheight(),
            mng.window.winfo_screenwidth(),
        )
        self.screen_size = (768,1366)
        print(self.screen_size)
        plt.close()
        cv2.namedWindow(self.wintitle)
        cv2.setWindowProperty(self.wintitle, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def update_right(self, pos):
        self.rightpos = pos
        self.draw_dots()

    def update_left(self, pos):
        self.leftpos = pos
        self.draw_dots()

    def draw_dots(self):
        black = np.ones(self.screen_size, dtype=np.ubyte)
        intpos_right = (int(self.rightpos[0]), self.screen_size[0] - int(self.rightpos[1]))
        intpos_left = (int(self.leftpos[0]), self.screen_size[0] - int(self.leftpos[1]))
        pic_to_display = cv2.cvtColor(black, cv2.COLOR_GRAY2RGB)
        cv2.circle(pic_to_display, intpos_right, self.dot_radius, (0, 255, 0), -1)
        cv2.circle(pic_to_display, intpos_left, self.dot_radius, (255, 0, 0), -1)
        cv2.imshow(self.wintitle, pic_to_display)
