from skimage import color, io, feature
from skimage.feature import peak_local_max
from skimage.transform import hough_circle
from skimage.draw import circle_perimeter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import argparse


def find_eye_center(eye):
    """
    Finds the center of the eye's pupil using circular hough transform
    :param eye: numpy 2d array of image intensities [0, 1]
    :return: center_x, center_y coordinates of the center of the pupil
    """
    edges = feature.canny(eye, sigma=3)

    # Detect two radii
    hough_radii = np.arange(15, 30, 2)
    hough_res = hough_circle(edges, hough_radii)

    centers = []
    accums = []
    radii = []

    for radius, h in zip(hough_radii, hough_res):
        # For each radius, extract two circles
        num_peaks = 2
        peaks = peak_local_max(h, num_peaks=num_peaks)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)

    best_circle_index = np.argsort(accums)[::-1][0]
    center_y, center_x = centers[best_circle_index]
    #cx, cy = circle_perimeter(center_x, center_y, radii[best_circle_index])
    #image = color.gray2rgb(eye)
    #image[cy, cx] = (220, 20, 20)
    return center_x, center_y


def find_eyes(imagePath, cascPath):
    """
    Uses an OpenCV haar cascade to locate the rectangles of the image in which
    each eye resides
    :param imagePath: file path of the image
    :return: list of lists representing the eye recangles [x, y, width, height]
    """
    cascade = cv2.CascadeClassifier(cascPath)
    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    eyes = cascade.detectMultiScale(
        gray,
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
    :return: tuple (right_eye, left_eye) of rectangles
    """
    right_eye = eyes[0]
    left_eye = eyes[1]

    if right_eye[0] > left_eye[0]:
        right_eye, left_eye = left_eye, right_eye

    return right_eye, left_eye


def main(cli):
    eyes = find_eyes(cli.file, cli.cascade_file)
    picture = color.rgb2gray(io.imread(cli.file))
    right_eye, left_eye = split_eyes(eyes)

    right_eyepatch = cropped_rect(picture, right_eye)
    right_pupil_x, right_pupil_y = find_eye_center(right_eyepatch)
    right_pupil = (right_pupil_x + right_eye[0], right_pupil_y + right_eye[1])

    left_eyepatch = cropped_rect(picture, left_eye)
    left_pupil_x, left_pupil_y = find_eye_center(left_eyepatch)
    left_pupil = (left_pupil_x + left_eye[0], left_pupil_y + left_eye[1])

    print("Eyes: %s\n  Right pupil: %s\n  Left pupil: %s" % (", ".join(map(str, eyes)), str(right_pupil), str(left_pupil)))

    # draw stuff
    eye1 = right_eye
    eye2 = left_eye
    fig, ax = plt.subplots(1)
    ax.imshow(picture, cmap="gray")
    ax.add_patch(patches.Rectangle(
        (eye1[0], eye1[1]),   # (x,y)
        eye1[2],          # width
        eye1[3],          # height
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
    plt.show()


def parsecli():
    parser = argparse.ArgumentParser(description="Find eyes and eye-centers from an image")
    parser.add_argument('file', help='picture file name', type=str)
    parser.add_argument('-c', '--cascade-file', help='path to the .xml file with the eye-detection haar cascade',
                        type=str, default="haarcascade_righteye_2splits.xml")
    return parser.parse_args()


if __name__ == "__main__":
    cli = parsecli()
    main(cli)

