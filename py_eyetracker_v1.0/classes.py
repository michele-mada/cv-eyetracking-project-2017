import dill as pickle
from collections import namedtuple, deque
from functools import reduce

import numpy as np


# Named tuples for "dumb" data objects

Rect = namedtuple("Rect", ["x", "y", "width", "height"])
Point = namedtuple("Point", ["x", "y"])
Observation = namedtuple("Observation", ["screen_point", "left_eyevectors", "right_eyevectors"])



class Tracker:

    """
        Object representing the face-to-screen mapping data, in the current frame and possibly in the last
        N past frames for the purpose of stabilizing through a weighted sum function.

        _face: current-frame face object
        smooth_weight_fun: function generating the weights for the face-averaging process
        cal_model_right: model which computes the right eyevector-to-screen mapping
        cal_model_left: model which computes the left eyevector-to-screen mapping
        smooth_frames: number of frames to average across
        history: fixed-size queue with the most recend faces captured
        centroid_history: fixed-size queue with the most recent left eye and right eye screen positions computed

    """

    def __init__(self, model_class, smooth_frames=5, centroid_history_frames=5, smooth_weight_fun=lambda x: 1.0):
        """
        :param model_class: class implementing MapperInterface
        :param smooth_frames: smooth the tracking data averaging across N frames. 1 to disable smoothing
        :param centroid_history_frames: calculate the centroid using the eye data from the last N frames
        :param smooth_weight_fun: weight function to use in computing the smoothing
        """
        self._face = Face()
        self.smooth_weight_fun = smooth_weight_fun
        self.cal_model_right = model_class()
        self.cal_model_left = model_class()
        self.smooth_frames = smooth_frames
        self.history = deque(maxlen=smooth_frames)
        self.centroid_history = deque(maxlen=centroid_history_frames*2)

    def update(self, face):
        assert(isinstance(face, Face))
        self.history.append(face)
        self._face = face

    @property
    def smoothing_enabled(self):
        return self.smooth_frames > 1

    @property
    def face(self):
        """
        Returns automatically either the current face or the stabilized current face according to the settings
        :return: current face
        """
        if self.smoothing_enabled:
            return self._get_smooth_face()
        else:
            return self._face

    def _get_smooth_face(self):
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
            return self._face

    def load_saved_cal_params(self):
        from utils.screen_mapping.calibrator import cal_param_storage_path
        with open(cal_param_storage_path + ".bag", "rb") as fp:
            observations = pickle.load(fp)
            self.cal_model_right.before_training(observations)
            self.cal_model_left.before_training(observations)
            self.cal_model_right.train_from_data(observations, is_left=False)
            self.cal_model_left.train_from_data(observations, is_left=True)

    @property
    def centroid(self):
        if len(self.centroid_history) == 0:
            return (0, 0)
        x = sum(map(lambda pair: pair[0], self.centroid_history))
        y = sum(map(lambda pair: pair[1], self.centroid_history))
        return (x / len(self.centroid_history), y / len(self.centroid_history))

    def get_onscreen_gaze_mapping(self):
        face = self.face
        right_eye_screen_pos = self.cal_model_right.map_point(face.right_eye.eye_vector)
        left_eye_screen_pos = self.cal_model_left.map_point(face.left_eye.eye_vector)
        self.centroid_history.append(right_eye_screen_pos)
        self.centroid_history.append(left_eye_screen_pos)
        return right_eye_screen_pos, left_eye_screen_pos, self.centroid


class Face:

    """
        Object representing a face in a single time frame, with:
        dlib68_points: (68,2) array with coordinates of specific landmark
        right_eye: right Eye object
        left_eye: left Eye object
        head_pose: (2,1) array with the projection of the nose onto the screen
        orientation: 3D rotation of the head
        translation: 3D translation of the head

        Implements sum between faces, multiplication and division between a face and a number
    """

    def __init__(self):
        self.dlib68_points = np.zeros((68,2))
        self.right_eye = Eye.zero(True)
        self.left_eye = Eye.zero(True)
        self.head_pose = (0,0)
        self.orientation = np.zeros((3,1))
        self.translation = np.zeros((3,1))

    @staticmethod
    def zero():
        return Face()

    def force_int(self):
        """
        Coerce the eye coordinates to integer
        """
        self.right_eye.force_int()
        self.left_eye.force_int()

    # Implement some Python builtin operations

    def __str__(self):  # cast to string
        return "Face(points: %s\nright_eye: %s\nleft_eye: %s\nhead_pose: %s\norientation: %s\ntranslation: %s\n)" % \
               (str(self.dlib68_points), str(self.right_eye), str(self.left_eye), str(self.head_pose), str(self.orientation), str(self.translation))

    def __add__(self, other):  # addition between Eye objects
        if not isinstance(other, Face):
            raise Exception("summing a face with something else")
        new_face = Face()
        new_face.dlib68_points = self.dlib68_points + other.dlib68_points
        new_face.right_eye = self.right_eye + other.right_eye
        new_face.left_eye = self.left_eye + other.left_eye
        new_face.orientation = self.orientation + other.orientation
        new_face.translation = self.translation + other.translation
        new_face.head_pose = (self.head_pose[0] + other.head_pose[0],
                              self.head_pose[1] + other.head_pose[1])
        return new_face

    def __truediv__(self, other):  # division by a number
        if not isinstance(other, (int, float, complex)):
            raise Exception("dividing a face by <not a number>")
        new_face = Face()
        new_face.dlib68_points = self.dlib68_points / other
        new_face.right_eye = self.right_eye / other
        new_face.left_eye = self.left_eye / other
        new_face.orientation = self.orientation / other
        new_face.translation = self.translation / other
        new_face.head_pose = (self.head_pose[0] / other,
                              self.head_pose[1] / other)
        return new_face

    def __mul__(self, other):  # multiplication by a number
        if not isinstance(other, (int, float, complex)):
            raise Exception("multiplying a face by <not a number>")
        new_face = Face()
        new_face.dlib68_points = self.dlib68_points * other
        new_face.right_eye = self.right_eye * other
        new_face.left_eye = self.left_eye * other
        new_face.orientation = self.orientation * other
        new_face.translation = self.translation * other
        new_face.head_pose = (self.head_pose[0] * other,
                              self.head_pose[1] * other)
        return new_face

    __rmul__ = __mul__

class Eye:
    """
        Object representing an eye in a single time frame, with:
        area: Rect named tuple (x,y,w,h) describing the eye position relative to the whole picture
        is_right: boolean describing if the eye is the right one
        pupil_relative: Pointnamed tuple (x,y) describing the location of the pupil relative to the area
        inner_corner_relative: Pointnamed tuple (x,y) describing the location of the inner eye-corner relative
            to the area
        outer_corner_relative: Pointnamed tuple (x,y) describing the location of the outer eye-corner relative
            to the area

        Implements sum between eyes, multiplication and division between an eye and a number.
        Perchè fare occhio per occhio sarebbe da barbari.
    """

    def __init__(self, area, is_right):
        self.area = Rect(*area)
        self.is_right = is_right
        self.pupil_relative = Point(0,0)
        self.inner_corner_relative = Point(0, 0)
        self.outer_corner_relative = Point(0, 0)

    @staticmethod
    def zero(is_right):
        return Eye((0,0,0,0), is_right)

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

    # Implement some Python builtin operations

    def __str__(self):  # cast to string
        return "Eye(%s, area=%s, pupil=%s, inner_corner=%s, outer_corner=%s" % (
            "Right" if self.is_right else "Left",
            str(self.area),
            str(self.pupil),
            str(self.inner_corner),
            str(self.outer_corner),
        )

    def __add__(self, other):  # addition between Eye objects
        if not isinstance(other, Eye):
            raise Exception("summing an eye with something else")
        new_area = Rect(*((np.array(self.area) + np.array(other.area)).tolist()))
        new_eye = Eye(new_area, self.is_right and other.is_right)
        new_eye.pupil = Point(*((np.array(self.pupil) + np.array(other.pupil)).tolist()))
        new_eye.inner_corner = Point(*((np.array(self.inner_corner) + np.array(other.inner_corner)).tolist()))
        new_eye.outer_corner = Point(*((np.array(self.outer_corner) + np.array(other.outer_corner)).tolist()))
        return new_eye

    def __truediv__(self, other):  # division by a number
        if not isinstance(other, (int, float, complex)):
            raise Exception("dividing an eye by <not a number>")
        new_area = Rect(*((np.array(self.area) / other).tolist()))
        new_eye = Eye(new_area, self.is_right)
        new_eye.pupil = Point(*((np.array(self.pupil) / other).tolist()))
        new_eye.inner_corner = Point(*((np.array(self.inner_corner) / other).tolist()))
        new_eye.outer_corner = Point(*((np.array(self.outer_corner) / other).tolist()))
        return new_eye

    def __mul__(self, other):  # multiplication by a number
        if not isinstance(other, (int, float, complex)):
            raise Exception("multiplying an eye by <not a number>")
        new_area = Rect(*((np.array(self.area) * other).tolist()))
        new_eye = Eye(new_area, self.is_right)
        new_eye.pupil = Point(*((np.array(self.pupil) * other).tolist()))
        new_eye.inner_corner = Point(*((np.array(self.inner_corner) * other).tolist()))
        new_eye.outer_corner = Point(*((np.array(self.outer_corner) * other).tolist()))
        return new_eye

    __rmul__ = __mul__
