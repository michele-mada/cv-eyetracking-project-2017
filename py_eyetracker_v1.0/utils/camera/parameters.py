import numpy as np


def camera_matrix_from_picture_shape(image_shape):
    focal_length = image_shape[1]
    center = (image_shape[1] / 2, image_shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    return camera_matrix

def get_dist_coeffs():
    return np.zeros((4, 1))  # Assuming no lens distortion