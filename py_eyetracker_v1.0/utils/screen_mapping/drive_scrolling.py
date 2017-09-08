import numpy as np

from classes import Tracker
from app_control.interface import AppControlInterface




def drive_scrolling(tracker: Tracker, iface: AppControlInterface, screen_size):
    (centroid_x, centroid_y) = tracker.centroid
    (screen_w, screen_h) = screen_size
    centroid_array = np.array(tracker.centroid)
    if centroid_array[0] > screen_size[0] * 2 / 3:
        if centroid_array[1] < screen_size[1] / 3:
            iface.scroll_down(1.0)
        elif centroid_array[1] > screen_size[1] * 2 / 3:
            iface.scroll_up(1.0)
