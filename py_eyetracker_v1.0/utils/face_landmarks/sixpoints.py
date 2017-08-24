import numpy as np
import cv2

from utils.face_landmarks import point_map
from utils.camera.parameters import *


model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corne
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner

])



def six_points(dlib_68points, image_shape):

    image_points = np.array([
        dlib_68points[point_map.nose_tip],  # Nose tip
        dlib_68points[point_map.chin],  # Chin
        dlib_68points[point_map.left_eye_outer_corner],  # Left eye left corner
        dlib_68points[point_map.right_eye_outer_corner],  # Right eye right corne
        dlib_68points[point_map.left_mouth_corner],  # Left Mouth corner
        dlib_68points[point_map.right_mouth_corner]  # Right mouth corner
    ], dtype="double")

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
