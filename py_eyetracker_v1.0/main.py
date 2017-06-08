import argparse

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float

from classes import Eye
from utils.camera import WebcamVideoStream
from utils.eyecenter.timm.timm_and_barth import find_eye_center_and_corners_cl as find_eye_center_and_corners_tb
from utils.eyecenter.timm.timm_and_barth import context as opencl_context
from utils.eyecenter.py_eyecenter import find_eye_center_and_corners_hough

camera_port = 0

algos = {
    "hough": find_eye_center_and_corners_hough,
    "timm": find_eye_center_and_corners_tb
}


def find_eyes(image, cascPath):
    """
    Uses an OpenCV haar cascade to locate the rectangles of the image in which
    each eye resides
    :param imagePath: file path of the image
    :return: list of lists representing the eye recangles [x, y, width, height]
    """
    cascade = cv2.CascadeClassifier(cascPath)

    # Detect faces in the image
    eyes = cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    return eyes


def cropped_rect(image, rect):
    """
    Crops an image represented by a 2d numpy array
    :param image: numpy 2d array
    :param rect: list or tuple of four elements: x,y,width,height
    :return: numpy 2d array containing the cropped region
    """
    (x1, y1, width, height) = rect
    return np.copy(image[y1:y1+height,x1:x1+width])


def split_eyes(eyes):
    """
    Select right eye and left eyes.
    The criteria is simple: the right eye is always the one with smaller x
    :param eyes: list of rectangles representing the eyes
    :return: tuple (right_eye, left_eye) of Eye class
    """
    right_eye = eyes[0]
    left_eye = eyes[1]

    if right_eye[0] > left_eye[0]:
        right_eye, left_eye = left_eye, right_eye

    return Eye(right_eye, True), Eye(left_eye, False)


def process_frame(image_cv2format, algo, debug=False):
    find_eye_center_and_corners = algos[algo]
    picture_float = img_as_float(image_cv2format)
    #picture = exposure.equalize_hist(picture_float)
    picture = picture_float
    eyes = find_eyes(image_cv2format, cli.cascade_file)
    if len(eyes) < 2:
        return picture, None, None

    right_eye, left_eye = split_eyes(eyes)

    left_eyepatch = cropped_rect(picture, left_eye.area)
    find_eye_center_and_corners(left_eyepatch, left_eye, debug=debug)

    right_eyepatch = cropped_rect(picture, right_eye.area)
    find_eye_center_and_corners(right_eyepatch, right_eye, debug=debug)

    return picture, right_eye, left_eye


def one_shot(cli):
    image_cv2 = cv2.imread(cli.file)
    #image_hsv = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)
    #(channel_h, channel_s, channel_v) = cv2.split(image_hsv)
    image_cv2_gray = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2GRAY)

    picture, right_eye, left_eye  = process_frame(image_cv2_gray, cli.algo, debug=cli.debug)

    print("Eyes (R,L):\n  %s,\n  %s" % (str(right_eye), str(left_eye)))

    # draw stuff
    fig, ax = plt.subplots(1)

    ax.imshow(picture, cmap="gray")

    if right_eye is not None and left_eye is not None:
        # draw eye regions (found using the haar cascade)
        ax.add_patch(patches.Rectangle(
            (right_eye.area.x, right_eye.area.y),
            right_eye.area.width,
            right_eye.area.height,
            linewidth=1, edgecolor='r', facecolor='none'
        ))
        ax.add_patch(patches.Rectangle(
            (left_eye.area.x, left_eye.area.y),
            left_eye.area.width,
            left_eye.area.height,
            linewidth=1, edgecolor='r', facecolor='none'
        ))
        # draw pupil spots (red)
        ax.plot(right_eye.pupil[0], right_eye.pupil[1], "r+")
        ax.plot(left_eye.pupil[0], left_eye.pupil[1], "r+")
        # draw inner corners (green)
        ax.plot(right_eye.inner_corner[0], right_eye.inner_corner[1], "g+")
        ax.plot(left_eye.inner_corner[0], left_eye.inner_corner[1], "g+")
        # draw outer corners (blue)
        ax.plot(right_eye.outer_corner[0], right_eye.outer_corner[1], "b+")
        ax.plot(left_eye.outer_corner[0], left_eye.outer_corner[1], "b+")

    plt.show()


def live(cli):
    camera = WebcamVideoStream(src=camera_port,
                               debug=cli.debug,
                               contrast=None if cli.contrast == "None" else float(cli.contrast),
                               saturation=None if cli.saturation == "None" else float(cli.saturation)
                               )
    camera.start()
    plt.ion()
    fig, ax = plt.subplots(1)

    while plt.fignum_exists(1):
        image_cv2 = camera.read()
        image_cv2_gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)

        picture, right_eye, left_eye = process_frame(image_cv2_gray, cli.algo, debug=cli.debug)

        plt.cla()
        ax.imshow(picture, cmap="gray")
        if right_eye is not None and left_eye is not None:
            # draw eye regions (found using the haar cascade)
            ax.add_patch(patches.Rectangle(
                (right_eye.area.x, right_eye.area.y),
                right_eye.area.width,
                right_eye.area.height,
                linewidth=1, edgecolor='r', facecolor='none'
            ))
            ax.add_patch(patches.Rectangle(
                (left_eye.area.x, left_eye.area.y),
                left_eye.area.width,
                left_eye.area.height,
                linewidth=1, edgecolor='r', facecolor='none'
            ))
            # draw pupil spots (red)
            ax.plot(right_eye.pupil[0], right_eye.pupil[1], "r+")
            ax.plot(left_eye.pupil[0], left_eye.pupil[1], "r+")
            # draw inner corners (green)
            ax.plot(right_eye.inner_corner[0], right_eye.inner_corner[1], "g+")
            ax.plot(left_eye.inner_corner[0], left_eye.inner_corner[1], "g+")
            # draw outer corners (blue)
            ax.plot(right_eye.outer_corner[0], right_eye.outer_corner[1], "b+")
            ax.plot(left_eye.outer_corner[0], left_eye.outer_corner[1], "b+")
        plt.pause(0.1)

    camera.stop()


def main(cli):
    if cli.algo == "timm":
        opencl_context.load_program(cli.program_timm)
    if cli.file == "-":
        live(cli)
    else:
        one_shot(cli)


def parsecli():
    parser = argparse.ArgumentParser(description="Find eyes and eye-centers from an image")
    parser.add_argument('file', help='filename of the picture; - for webcam', type=str)
    parser.add_argument('-c', '--cascade-file', help='path to the .xml file with the eye-detection haar cascade',
                        type=str, default="haarcascade_righteye_2splits.xml")
    parser.add_argument('--saturation', help='override the webcam default saturation setting (value [0.0, 1.0])',
                        type=str, default="None")
    parser.add_argument('--contrast', help='override the webcam default contrast setting (value [0.0, 1.0])',
                        type=str, default="None")
    parser.add_argument('-a','--algo', help='algorithm to use (hough or timm)(default to timm)',
                        type=str, default="timm")
    parser.add_argument('-p', '--program-timm', help='path to the opencl kernel implementing timm and barth algorithm',
                        type=str, default="timm_barth_kernel.cl")
    parser.add_argument('-d','--debug', help='enable debug mode', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    cli = parsecli()
    main(cli)

