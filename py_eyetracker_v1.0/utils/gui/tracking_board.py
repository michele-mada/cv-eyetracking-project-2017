import cv2
import numpy as np


class TrackingBoard:

    dot_radius = 4
    wintitle = "trackingboard"

    def __init__(self, screensize):
        self.leftpos = (0, 0)
        self.rightpos = (0, 0)
        self.centroid = (0, 0)
        self.screen_size = screensize
        print(self.screen_size)
        cv2.namedWindow(self.wintitle)
        cv2.setWindowProperty(self.wintitle, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def update(self, right, left, centroid, head_pose):
        self.rightpos = right
        self.leftpos = left
        self.centroid = centroid
        self.head_pose = head_pose
        self.draw_dots()

    def draw_dots(self):
        black = np.ones((self.screen_size[1], self.screen_size[0]), dtype=np.ubyte)
        intpos_right = (int(self.rightpos[0]), self.screen_size[1] - int(self.rightpos[1]))
        intpos_left = (int(self.leftpos[0]), self.screen_size[1] - int(self.leftpos[1]))
        intpos_centroid = (int(self.centroid[0]), self.screen_size[1] - int(self.centroid[1]))
        intpos_head_pose = (int(self.head_pose[0]*self.screen_size[0]),int(self.head_pose[1]*self.screen_size[1]))
        pic_to_display = cv2.cvtColor(black, cv2.COLOR_GRAY2RGB)
        cv2.circle(pic_to_display, intpos_right, self.dot_radius, (0, 255, 0), -1)
        cv2.circle(pic_to_display, intpos_left, self.dot_radius, (255, 0, 0), -1)
        cv2.circle(pic_to_display, intpos_centroid, self.dot_radius*2, (0, 0, 255), -1)
        cv2.circle(pic_to_display, intpos_head_pose, self.dot_radius, (255,255, 255), -1)
        cv2.imshow(self.wintitle, pic_to_display)
