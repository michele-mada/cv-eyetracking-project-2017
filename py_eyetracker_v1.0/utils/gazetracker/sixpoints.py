import numpy as np
import cv2

from utils.face_landmarks import point_map


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
    focal_length = image_shape[1]
    center = (image_shape[1] / 2, image_shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.CV_ITERATIVE)
    return rotation_vector, translation_vector
