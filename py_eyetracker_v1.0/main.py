from skimage import img_as_float, measure
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.filters import threshold_otsu
from skimage.feature import corner_harris, corner_peaks
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import argparse

from utils.camera import WebcamVideoStream
from classes import Eye, Point


camera_port = 0



def find_eye_center_and_corners(eye_image, eye_object, debug=False):
    assert(isinstance(eye_object, Eye))

    # Part 1: finding the center of the eye

    # apply a threshold to the image
    thresh = threshold_otsu(eye_image) * 0.5
    eye_binary = eye_image < thresh

    # split the thresholded image into regions
    blobs_labels, nlabels = measure.label(eye_binary, background=0, return_num=True)

    centers = []
    accums = []
    radii = []
    hough_radii = np.arange(10, 25, 2)

    # foreach region, use the hough transform to find the most prominent circle
    for n in range(1, nlabels):
        img = blobs_labels == n
        hough_res = hough_circle(img, hough_radii)
        _accums, cx, cy, _radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
        accums.extend(_accums)
        centers.extend(zip(cx, cy))
        radii.extend(_radii)

    # the center of the pupil is the center of the best circular region found
    best_circle_index = np.argmax(np.array(accums))
    center_x, center_y = centers[best_circle_index]

    # Part 1: finding the corners of the eye

    # heuristic to find the best corner, given its position relative to the pupil center
    def corner_measure(corner, center, leftness=1):
        cy, cx = corner
        zx, zy = center
        return (cx-zx) * leftness - 0.5 * abs(cy-zy)

    # find the centers (y,x) of the corners, using harris
    corners = corner_peaks(corner_harris(eye_binary))

    # give each corner a score, according to our heuristic
    left_corner_score = map(lambda corner: (corner,
                                            corner_measure(corner, (center_x, center_y), leftness=1)
                                            ),
                            corners)
    # find the best corner
    left_corner = max(left_corner_score, key=lambda cs: cs[1])[0]

    # give each corner a score, according to our heuristic
    right_corner_score = map(lambda corner: (corner,
                                             corner_measure(corner, (center_x, center_y), leftness=-1)
                                             ),
                             corners)
    # find the best corner
    right_corner = max(right_corner_score, key=lambda cs: cs[1])[0]

    if debug:
        fig, ax = plt.subplots(1)

        ax.imshow(eye_binary, cmap="gray")

        ax.imshow(blobs_labels, cmap='spectral')
        ax.plot(center_x, center_y, "w+")

        ax.plot(corners[:,1], corners[:,0], "bo")

        ax.plot(left_corner[1], left_corner[0], "ro")
        ax.plot(right_corner[1], right_corner[0], "go")

    eye_object.pupil_relative = Point(center_x, center_y)
    eye_object.set_leftmost_corner(Point(x=left_corner[1], y=left_corner[0]))
    eye_object.set_rightmost_corner(Point(x=right_corner[1], y=right_corner[0]))



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


def process_frame(image_cv2format, debug=False):
    picture_float = img_as_float(image_cv2format)
    picture = exposure.equalize_hist(picture_float)
    eyes = find_eyes(image_cv2format, cli.cascade_file)
    if len(eyes) < 2:
        return picture, None, None

    right_eye, left_eye = split_eyes(eyes)

    right_eyepatch = cropped_rect(picture, right_eye.area)
    find_eye_center_and_corners(right_eyepatch, right_eye, debug=debug)

    left_eyepatch = cropped_rect(picture, left_eye.area)
    find_eye_center_and_corners(left_eyepatch, left_eye, debug=debug)

    return picture, right_eye, left_eye


def one_shot(cli):
    image_cv2 = cv2.imread(cli.file)
    #image_hsv = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)
    #(channel_h, channel_s, channel_v) = cv2.split(image_hsv)
    image_cv2_gray = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2GRAY)

    picture, right_eye, left_eye  = process_frame(image_cv2_gray, debug=cli.debug)

    print("Eyes (R,L):\n  %s,\n  %s" % (str(right_eye), str(left_eye)))

    # draw stuff
    fig, ax = plt.subplots(1)

    ax.imshow(picture, cmap="gray")

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
    #TODO: make this function work again
    camera = WebcamVideoStream(src=camera_port)
    camera.start()
    plt.ion()
    fig, ax = plt.subplots(1)

    while plt.fignum_exists(1):
        image_cv2 = camera.read()
        image_cv2_gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)

        picture, right_eye, left_eye, right_pupil, left_pupil = process_frame(image_cv2_gray)

        eye1 = right_eye
        eye2 = left_eye
        plt.cla()
        ax.imshow(picture, cmap="gray")
        ax.add_patch(patches.Rectangle(
            (eye1[0], eye1[1]),  # (x,y)
            eye1[2],  # width
            eye1[3],  # height
            linewidth=1, edgecolor='r', facecolor='none'
        ))
        ax.add_patch(patches.Rectangle(
            (eye2[0], eye2[1]),  # (x,y)
            eye2[2],  # width
            eye2[3],  # height
            linewidth=1, edgecolor='r', facecolor='none'
        ))
        ax.plot(right_pupil[0], right_pupil[1], "r+")
        ax.plot(left_pupil[0], left_pupil[1], "r+")
        #plt.show()
        plt.pause(0.1)

    camera.stop()


def main(cli):
    if cli.file == "-":
        live(cli)
    else:
        one_shot(cli)


def parsecli():
    parser = argparse.ArgumentParser(description="Find eyes and eye-centers from an image")
    parser.add_argument('file', help='filename of the picture; - for webcam', type=str)
    parser.add_argument('-c', '--cascade-file', help='path to the .xml file with the eye-detection haar cascade',
                        type=str, default="haarcascade_righteye_2splits.xml")
    parser.add_argument('-d','--debug', help='enable debug mode', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    cli = parsecli()
    main(cli)

