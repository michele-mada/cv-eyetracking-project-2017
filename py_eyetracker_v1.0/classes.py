from collections import namedtuple
import numpy as np
import pickle

from utils.screen_mapping.map_function import map_function
from utils.gazetracker.calibrator import cal_param_storage_path


Rect = namedtuple("Rect", ["x", "y", "width", "height"])
Point = namedtuple("Point", ["x", "y"])



class Tracker:

    def __init__(self):
        self.face = Face()
        self.cal_param_right = (np.ones((5,)), np.ones((5,)))
        self.cal_param_left = (np.ones((5,)), np.ones((5,)))
        # TODO: historical queue to do some smoothing?

    def update(self, face):
        self.face = face

    def load_saved_cal_params(self):
        with open(cal_param_storage_path, "rb") as fp:
            (self.cal_param_right, self.cal_param_left) = pickle.load(fp)

    def get_onscreen_gaze_mapping(self):
        right_eye_screen_pos = map_function(self.face.right_eye.eye_vector, *self.cal_param_right)
        left_eye_screen_pos = map_function(self.face.left_eye.eye_vector, *self.cal_param_left)
        # TODO: combine the values?
        # TODO: apply head rotation offset
        return right_eye_screen_pos


class Face:

    def __init__(self):
        self.dlib68_points = []
        self.right_eye = None
        self.left_eye = None
        self.orientation = np.zeros((3,))
        self.translation = np.zeros((3,))


class Eye:

    def __init__(self, area, is_right):
        self.area = Rect(*area)
        self.is_right = is_right
        self.pupil_relative = Point(0,0)
        self.inner_corner_relative = Point(0, 0)
        self.outer_corner_relative = Point(0, 0)

    @property
    def pupil(self):
        return Point(self.area.x + self.pupil_relative.x, self.area.y + self.pupil_relative.y)

    @property
    def inner_corner(self):
        return Point(self.area.x + self.inner_corner_relative.x, self.area.y + self.inner_corner_relative.y)

    @inner_corner.setter
    def inner_corner(self, value):
        self.inner_corner_relative = Point(value[0] - self.area.x, value[1] - self.area.y)

    @property
    def outer_corner(self):
        return Point(self.area.x + self.outer_corner_relative.x, self.area.y + self.outer_corner_relative.y)

    @outer_corner.setter
    def outer_corner(self, value):
        self.outer_corner_relative = Point(value[0] - self.area.x, value[1] - self.area.y)

    @property
    def absolute_area_center(self):
        return Point(self.relative_area_center.x + self.area.x, self.relative_area_center.y + self.area.y)

    @property
    def relative_area_center(self):
        return Point(self.area.width / 2, self.area.height / 2)

    @property
    def eye_vector(self):
        return Point(self.inner_corner_relative[0] - self.pupil_relative[0],
                     self.inner_corner_relative[1] - self.pupil_relative[1])

    def set_leftmost_corner(self, point):
        if self.is_right:
            self.outer_corner_relative = point
        else:
            self.inner_corner_relative = point

    def set_rightmost_corner(self, point):
        if self.is_right:
            self.inner_corner_relative = point
        else:
            self.outer_corner_relative = point

    def __str__(self):
        return "Eye(%s, area=%s, pupil=%s, inner_corner=%s, outer_corner=%s" % (
            "Right" if self.is_right else "Left",
            str(self.area),
            str(self.pupil),
            str(self.inner_corner),
            str(self.outer_corner),
        )