import dill as pickle
from collections import namedtuple, deque
from functools import reduce

import numpy as np

Rect = namedtuple("Rect", ["x", "y", "width", "height"])
Point = namedtuple("Point", ["x", "y"])
Observation = namedtuple("Observation", ["screen_point", "left_eyevectors", "right_eyevectors"])



class Tracker:

    def __init__(self, model_class, smooth_frames=5, smooth_weight_fun=lambda x: 1.0):
        self.face = Face()
        self.smooth_weight_fun = smooth_weight_fun
        self.cal_model_right = model_class()
        self.cal_model_left = model_class()
        self.history = deque(maxlen=smooth_frames)

    def update(self, face):
        assert(isinstance(face, Face))
        self.history.append(face)
        self.face = face

    def get_smooth_face(self):
        if len(self.history) > 0:
            facesum = reduce(lambda a,b: a + b,
                             map(lambda i: self.smooth_weight_fun(i[0])*i[1],
                                 enumerate(self.history)),
                             Face.zero())
            avgface = facesum / sum(map(lambda n: self.smooth_weight_fun(n),
                                        range(len(self.history))))
            avgface.force_int()
            return avgface
        else:
            return self.face

    def load_saved_cal_params(self):
        from utils.screen_mapping.calibrator import cal_param_storage_path
        with open(cal_param_storage_path + ".bag", "rb") as fp:
            observations = pickle.load(fp)
            self.cal_model_right.before_training(observations)
            self.cal_model_left.before_training(observations)
            self.cal_model_right.train_from_data(observations, is_left=False)
            self.cal_model_left.train_from_data(observations, is_left=True)

    def get_onscreen_gaze_mapping(self, smooth=False):
        face = self.face
        if smooth:
            face = self.get_smooth_face()
        right_eye_screen_pos = self.cal_model_right.map_point(face.right_eye.eye_vector)
        left_eye_screen_pos = self.cal_model_left.map_point(face.left_eye.eye_vector)
        # TODO: combine the values?
        # TODO: apply head rotation offset
        return right_eye_screen_pos, left_eye_screen_pos


class Face:

    def __init__(self):
        self.dlib68_points = np.zeros((68,2))
        self.right_eye = None
        self.left_eye = None
        self.orientation = np.zeros((3,1))
        self.translation = np.zeros((3,1))

    @staticmethod
    def zero():
        new_face = Face()
        new_face.right_eye = Eye.zero(True)
        new_face.left_eye = Eye.zero(False)
        return new_face

    def force_int(self):
        self.right_eye.force_int()
        self.left_eye.force_int()

    def __str__(self):
        return "Face(points: %s\nright_eye: %s\nleft_eye: %s\norientation: %s\ntranslation: %s\n)" % \
               (str(self.dlib68_points), str(self.right_eye), str(self.left_eye), str(self.orientation), str(self.translation))

    def __add__(self, other):
        if not isinstance(other, Face):
            raise Exception("summing a face with something else")
        new_face = Face()
        new_face.dlib68_points = self.dlib68_points + other.dlib68_points
        new_face.right_eye = self.right_eye + other.right_eye
        new_face.left_eye = self.left_eye + other.left_eye
        new_face.orientation = self.orientation + other.orientation
        new_face.translation = self.translation + other.translation
        return new_face

    def __truediv__(self, other):
        if not isinstance(other, (int, float, complex)):
            raise Exception("dividing a face by <not a number>")
        new_face = Face()
        new_face.dlib68_points = self.dlib68_points / other
        new_face.right_eye = self.right_eye / other
        new_face.left_eye = self.left_eye / other
        new_face.orientation = self.orientation / other
        new_face.translation = self.translation / other
        return new_face

    def __mul__(self, other):
        if not isinstance(other, (int, float, complex)):
            raise Exception("multiplying a face by <not a number>")
        new_face = Face()
        new_face.dlib68_points = self.dlib68_points * other
        new_face.right_eye = self.right_eye * other
        new_face.left_eye = self.left_eye * other
        new_face.orientation = self.orientation * other
        new_face.translation = self.translation * other
        return new_face

    __rmul__ = __mul__

class Eye:

    def __init__(self, area, is_right):
        self.area = Rect(*area)
        self.is_right = is_right
        self.pupil_relative = Point(0,0)
        self.inner_corner_relative = Point(0, 0)
        self.outer_corner_relative = Point(0, 0)

    @staticmethod
    def zero(is_right):
        new_eye = Eye((0,0,0,0), is_right)
        return new_eye

    def force_int(self):
        self.area = Rect(x=int(self.area.x), y=int(self.area.y), width=int(self.area.width), height=int(self.area.height))
        self.pupil_relative = Point(x=int(self.pupil_relative.x), y=int(self.pupil_relative.y))
        self.inner_corner_relative = Point(x=int(self.inner_corner_relative.x), y=int(self.inner_corner_relative.y))
        self.outer_corner_relative = Point(x=int(self.outer_corner_relative.x), y=int(self.outer_corner_relative.y))

    @property
    def pupil(self):
        return Point(self.area.x + self.pupil_relative.x, self.area.y + self.pupil_relative.y)

    @pupil.setter
    def pupil(self, value):
        self.pupil_relative = Point(value[0] - self.area.x, value[1] - self.area.y)

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

    def __add__(self, other):
        if not isinstance(other, Eye):
            raise Exception("summing an eye with something else")
        new_area = Rect(*((np.array(self.area) + np.array(other.area)).tolist()))
        new_eye = Eye(new_area, self.is_right and other.is_right)
        new_eye.pupil = Point(*((np.array(self.pupil) + np.array(other.pupil)).tolist()))
        new_eye.inner_corner = Point(*((np.array(self.inner_corner) + np.array(other.inner_corner)).tolist()))
        new_eye.outer_corner = Point(*((np.array(self.outer_corner) + np.array(other.outer_corner)).tolist()))
        return new_eye

    def __truediv__(self, other):
        if not isinstance(other, (int, float, complex)):
            raise Exception("dividing an eye by <not a number>")
        new_area = Rect(*((np.array(self.area) / other).tolist()))
        new_eye = Eye(new_area, self.is_right)
        new_eye.pupil = Point(*((np.array(self.pupil) / other).tolist()))
        new_eye.inner_corner = Point(*((np.array(self.inner_corner) / other).tolist()))
        new_eye.outer_corner = Point(*((np.array(self.outer_corner) / other).tolist()))
        return new_eye

    def __mul__(self, other):
        if not isinstance(other, (int, float, complex)):
            raise Exception("multiplying an eye by <not a number>")
        new_area = Rect(*((np.array(self.area) * other).tolist()))
        new_eye = Eye(new_area, self.is_right)
        new_eye.pupil = Point(*((np.array(self.pupil) * other).tolist()))
        new_eye.inner_corner = Point(*((np.array(self.inner_corner) * other).tolist()))
        new_eye.outer_corner = Point(*((np.array(self.outer_corner) * other).tolist()))
        return new_eye

    __rmul__ = __mul__
