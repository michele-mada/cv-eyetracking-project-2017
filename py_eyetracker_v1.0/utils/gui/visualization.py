import numpy as np
import cv2
from skimage import img_as_ubyte

from utils.camera.parameters import *
from utils.face_landmarks import point_map
from classes import Face


blue = (255, 0, 0)
red = (0, 0, 255)
green = (0, 255, 0)

dot_radius = 2


def draw_rect(image, rect, color, thickness):
    cv2.rectangle(image,
                  (rect.x, rect.y),
                  (rect.x + rect.width, rect.y + rect.height),
                  color,
                  thickness)

def draw_dot(image, point, color):
    cv2.circle(image, point, dot_radius, color, -1)


def draw_gaze_vector(image, face: Face, length=500.0):
    camera_matrix = camera_matrix_from_picture_shape(image.shape)
    dist_coeffs = get_dist_coeffs()
    #si potrebbe togliere e usare face.head_pose anche se è specchiato
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, length)]),
                                                     face.orientation,
                                                     face.translation,
                                                     camera_matrix, dist_coeffs)

    p1 = (int(face.dlib68_points[point_map.nose_tip][0]), int(face.dlib68_points[point_map.nose_tip][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    cv2.line(image, p1, p2, blue, 2)


def draw_routine(picture, face: Face, not_eyes, title, draw_unicorn=True):
    """
    Draw the main window
    :param picture: 
    :param face
    :param not_eyes: list of Rect
    :param title: window title
    :return: 
    """

    pic_to_display = cv2.cvtColor(img_as_ubyte(picture), cv2.COLOR_GRAY2RGB)

    if face is not None:
        right_eye = face.right_eye
        left_eye = face.left_eye

        if draw_unicorn:
            draw_gaze_vector(pic_to_display, face)

        # draw eye regions (found using the haar cascade)
        draw_rect(pic_to_display, right_eye.area, red, 1)
        draw_rect(pic_to_display, left_eye.area, red, 1)

        for rect in not_eyes:
            draw_rect(pic_to_display, rect, blue, 1)

        # draw pupil spots (red)
        draw_dot(pic_to_display, right_eye.pupil, red)
        draw_dot(pic_to_display, left_eye.pupil, red)
        # draw inner corners (green)
        draw_dot(pic_to_display, right_eye.inner_corner, green)
        draw_dot(pic_to_display, left_eye.inner_corner, green)
        # draw outer corners (blue)
        draw_dot(pic_to_display, right_eye.outer_corner, blue)
        draw_dot(pic_to_display, left_eye.outer_corner, blue)

    cv2.imshow(title, pic_to_display)
