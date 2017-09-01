import numpy as np
import cv2

from utils.face_landmarks import point_map
from utils.camera.parameters import *

""""#Antropometric constant values of the human head.
#Found on wikipedia and on:
# "Head-and-Face Anthropometric Survey of U.S. Respirator Users"
#
#X-Y-Z with X pointing forward and Y on the left.
#The X-Y-Z coordinates used are like the standard
model_points = np.array([
#   coordinates                 dlib point  description
    [-100.0, -77.5, -5.0],      # 0         right side
    [-110.0, -77.5, -85.0],     # 4         gonion right
    [0.0, 0.0, -122.7],         # 8         chin
    [-110.0, 77.5, -85.0],      # 12        gonion left
    [-100.0, 77.5, -5.0],       # 16        left side
    [-20.0, -56.1, 10.0],       # 17        frontal breadth right
    [-20.0, 56.1, 10.0],        # 26        frontal breadth left
    [0.0, 0.0, 0.0],            # 27        sellion
    [21.1, 0.0, -48.0],         # 30        nose tip
    [5.0, 0.0, -52.0],          # 33        sub nose
    [-20.0, -65.5,-5.0],        # 36        right eye outer corner
    [-10.0, -40.5,-5.0],        # 39        right eye inner corner
    [-10.0, 40.5,-5.0],         # 42        left eye inner corner
    [-20.0, 65.5,-5.0],         # 45        left eye outer corner
    [10.0, 0.0, -75.0],         # 62        stomion
])

model_dlib_indices = [
    point_map.right_side,
    point_map.gonion_right,
    point_map.chin,
    point_map.gonion_left,
    point_map.left_side,
    point_map.frontal_breadth_right,
    point_map.frontal_breadth_left,
    point_map.sellion,
    point_map.nose_tip,
    point_map.sub_nose,
    point_map.right_eye_outer_corner,
    point_map.right_eye_inner_corner,
    point_map.left_eye_inner_corner,
    point_map.left_eye_outer_corner,
    point_map.stomion
]"""

model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner

])

model_dlib_indices = [
    point_map.nose_tip,
    point_map.chin,
    point_map.left_eye_outer_corner,
    point_map.right_eye_outer_corner,
    point_map.left_mouth_corner,
    point_map.right_mouth_corner,
]

assert(len(model_points) == len(model_dlib_indices))


def six_points(dlib_68points, image_shape):

    image_points = np.array(
        list(map(lambda pi: dlib_68points[pi], model_dlib_indices)),
        dtype="double")

    # camera parameters
    camera_matrix = camera_matrix_from_picture_shape(image_shape)
    dist_coeffs = get_dist_coeffs()

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]),
                                                     rotation_vector,
                                                     translation_vector,
                                                     camera_matrix, dist_coeffs)
    head_pose = ((-(nose_end_point2D[0][0][0]- image_shape[1]/2) + image_shape[1]/2)/image_shape[1], nose_end_point2D[0][0][1]/image_shape[0])

    return rotation_vector, translation_vector, head_pose
