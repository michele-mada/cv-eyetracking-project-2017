from matplotlib import patches as patches
import cv2
from skimage import img_as_ubyte


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


def draw_routine(picture, right_eye, left_eye, not_eyes, title):
    """
    Draw the main window
    :param picture: 
    :param right_eye: 
    :param left_eye: 
    :param not_eyes: list of Rect
    :param title: window title
    :return: 
    """
    pic_to_display = cv2.cvtColor(img_as_ubyte(picture), cv2.COLOR_GRAY2RGB)
    if right_eye is not None and left_eye is not None:
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