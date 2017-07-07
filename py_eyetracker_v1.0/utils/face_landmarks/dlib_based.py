from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

from classes import Rect, Eye
from utils.face_landmarks import point_map


detector = None
predictor = None


#Start and end indexes for left (l) and right (r) eyes in 68-points
#I did it manually cause i can't access the proper imutils attributes
rstart = 36
rend = 42
lstart = 42
lend = 48


def detect_faces(image_cv2format, image_cv2format_equalized):
    global detector, predictor
    detect_attempt = "equalized"
    rects = detector(image_cv2format_equalized.copy(), 1)
    if len(rects) == 0:
        detect_attempt = "raw"
        gray = cv2.equalizeHist(image_cv2format.copy())
        rects = detector(gray, 1)
        if len(rects) == 0:
            return [], "gave up", image_cv2format
        else:
            return rects, detect_attempt, image_cv2format
    else:
        return rects, detect_attempt, image_cv2format_equalized


def bounding_rect(points, border=5):
    x_min = min(points[:, 0])
    x_max = max(points[:, 0])
    y_min = min(points[:, 1])
    y_max = max(points[:, 1])
    return Rect(x=x_min-border, y=y_min-border, width=x_max-x_min+2*border, height=y_max-y_min+2*border)


def eye_area_detection_step(image_cv2format, image_cv2format_equalized, model="data/shape_predictor_68_face_landmarks.dat"):
    global detector, predictor
    if detector is None:
        detector = dlib.get_frontal_face_detector()
    if predictor is None:
        predictor = dlib.shape_predictor(model)

    faces, detection_method, image_chosen = detect_faces(image_cv2format, image_cv2format_equalized)
    if len(faces) == 0:
        return False, detection_method, [], []

    face = faces[0]  #TODO: select the most appropriate face
    # detect facial landmarks for the face region
    shape = predictor(image_chosen.copy(), face)
    # convert the facial landmark coordinates to NumPy array
    points = face_utils.shape_to_np(shape)

    left_eye_rect = bounding_rect(points[lstart:lend])
    right_eye_rect = bounding_rect(points[rstart:rend])

    if left_eye_rect.width * left_eye_rect.height <= 0:
        return False, "left eye fail", [], points
    if right_eye_rect.width * right_eye_rect.height <= 0:
        return False, "right eye fail", [], points

    return True, detection_method, [left_eye_rect, right_eye_rect], points


def pick_eye_corners(eye_obj, points):
    assert(isinstance(eye_obj, Eye))
    if eye_obj.is_right:
        eye_obj.inner_corner = points[point_map.right_eye_inner_corner]
        eye_obj.outer_corner = points[point_map.right_eye_outer_corner]
    else:
        eye_obj.inner_corner = points[point_map.left_eye_inner_corner]
        eye_obj.outer_corner = points[point_map.left_eye_outer_corner]
