import argparse

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float

from utils.camera import WebcamVideoStream
from utils.eye_area import detect_haar_cascade, split_eyes, eye_regions_from_face
from utils.eyecenter.py_eyecenter import PyHoughEyecenter
from utils.eyecenter.timm.timm_and_barth import TimmAndBarth

camera_port = 0

algos = {
    "hough": PyHoughEyecenter,
    "timm": TimmAndBarth
}


def cropped_rect(image, rect):
    """
    Crops an image represented by a 2d numpy array
    :param image: numpy 2d array
    :param rect: list or tuple of four elements: x,y,width,height
    :return: numpy 2d array containing the cropped region
    """
    (x1, y1, width, height) = rect
    return np.copy(image[y1:y1+height,x1:x1+width])


def process_frame(image_cv2format, algo):
    picture_float = img_as_float(image_cv2format)
    image_cv2format_equalized = cv2.equalizeHist(image_cv2format)
    picture = picture_float
    eyes = detect_haar_cascade(image_cv2format_equalized, cli.eye_cascade_file)
    if len(eyes) < 2:  # if eyes not found, try without the equalization
        eyes = detect_haar_cascade(image_cv2format, cli.eye_cascade_file)
        if len(eyes) < 2:  # if eyes not found again, try just detecting the face
            face = detect_haar_cascade(image_cv2format_equalized, cli.face_cascade_file)
            if len(face) < 1:  # give up
                return img_as_float(image_cv2format_equalized), None, None
            else:
                eyes = eye_regions_from_face(face[0])  # TODO: use the most central face in case many faces are detected

    right_eye, left_eye = split_eyes(eyes)

    right_eyepatch = cropped_rect(picture, right_eye.area)
    algo.detect_eye_features(right_eyepatch, right_eye)

    left_eyepatch = cropped_rect(picture, left_eye.area)
    algo.detect_eye_features(left_eyepatch, left_eye)

    return img_as_float(image_cv2format_equalized), right_eye, left_eye


def draw_routine(ax, picture, right_eye, left_eye):
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


def one_shot(cli, algo):
    image_cv2 = cv2.imread(cli.file)
    image_cv2_gray = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2GRAY)

    fig, ax_img = plt.subplots(1)
    if cli.debug:
        fig2, axes_2 = algo.create_debug_figure()
        algo.setup_debug_parameters(True, axes_2)

    picture, right_eye, left_eye = process_frame(image_cv2_gray, algo)

    print("Eyes (R,L):\n  %s,\n  %s" % (str(right_eye), str(left_eye)))

    draw_routine(ax_img, picture, right_eye, left_eye)
    plt.show()


def live(cli, algo):
    camera = WebcamVideoStream(src=camera_port,
                               debug=cli.debug,
                               contrast=None if cli.contrast == "None" else float(cli.contrast),
                               saturation=None if cli.saturation == "None" else float(cli.saturation)
                               )
    camera.start()
    plt.ion()
    fig, ax_img = plt.subplots(1)
    if cli.debug:
        fig2, axes_2 = algo.create_debug_figure()
        algo.setup_debug_parameters(True, axes_2)

    while plt.fignum_exists(1):
        image_cv2 = camera.read()
        image_cv2_gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)

        ax_img.clear()
        algo.clean_debug_axes()

        picture, right_eye, left_eye = process_frame(image_cv2_gray, algo)

        draw_routine(ax_img, picture, right_eye, left_eye)
        plt.pause(0.1)

    camera.stop()


def main(cli):
    algo = algos[cli.algo]()
    if cli.algo == "timm":
        algo.context.load_program(cli.program_timm)

    if cli.file == "-":
        live(cli, algo)
    else:
        one_shot(cli, algo)


def parsecli():
    parser = argparse.ArgumentParser(description="Find eyes and eye-centers from an image")
    parser.add_argument('file', help='filename of the picture; - for webcam', type=str)
    parser.add_argument('-e', '--eye-cascade-file', help='path to the .xml file with the eye-detection haar cascade',
                        type=str, default="../haarcascades/haarcascade_righteye_2splits.xml")
    parser.add_argument('-f', '--face-cascade-file', help='path to the .xml file with the face-detection haar cascade',
                        type=str, default="../haarcascades/haarcascade_frontalface_default.xml")
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

