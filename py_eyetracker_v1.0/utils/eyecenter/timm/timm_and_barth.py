import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from skimage import exposure
from skimage.feature import peak_local_max
from skimage.filters import scharr_h, scharr_v, gaussian
from skimage.transform import resize

from classes import Point
from utils.eyecenter.timm.cl_runner import CLTimmBarth


def precomputation(eye_image):
    # compute gradients across x and y
    # (scharr gives better results than sobel)
    x_gradient = scharr_v(eye_image)
    y_gradient = scharr_h(eye_image)
    edgemap = np.sqrt(x_gradient ** 2 + y_gradient ** 2) + 0.00001

    # remove gradients under a specific threshold
    # (as suggested on http://thume.ca/projects/2012/11/04/simple-accurate-eye-center-tracking-in-opencv/)
    mean_gradient = np.mean(edgemap)
    std_gradient = np.std(edgemap)
    threshold = 0.3 * std_gradient + mean_gradient
    under_threshold_indices = edgemap < threshold
    x_gradient[under_threshold_indices] = 0
    y_gradient[under_threshold_indices] = 0

    # normalize the gradients
    maxedge = np.amax(edgemap)
    x_gradient /= maxedge
    y_gradient /= maxedge

    # compute the inverse-colored image
    inverse = gaussian(1.0 - eye_image)
    return x_gradient, y_gradient, inverse


# prepare the opencl context
context = CLTimmBarth(precomputation=precomputation)


def find_eye_center_and_corners_cl(eye_image, eye_object, debug=False, locality=0.15):  # locality=0.15

    # scale the image if needed
    width, height = np.shape(eye_image)
    max_w = 120
    scale_factor = 1.0
    if width > max_w:
        max_h = (height / width) * max_w
        scale_factor = height / max_h
        eye_image = resize(eye_image, (max_w, max_h))

    # important: histogram equalization
    eye_image = exposure.equalize_hist(eye_image)
    width, height = np.shape(eye_image)
    # run the actual timm & barth filter
    tb_image = context.compute(eye_image, locality=width * locality)
    # find the local peaks in the function
    maxima = peak_local_max(tb_image, min_distance=10)

    # select the actual center among the candidates
    def center_euristic(point):  # point must be central, and have a strong t&b value
        y,x = point
        return 5 * tb_image[x, y] + \
               -1 * sqrt((x - (width / 2)) ** 2 + (y - (height / 2)) ** 2)
    true_center = max(maxima, key=center_euristic)

    eye_object.pupil_relative = Point(true_center[1] * scale_factor, true_center[0] * scale_factor)

    if debug:
        fig, ax = plt.subplots(1)
        plt.title("right eye" if eye_object.is_right else "left eye")
        ax.imshow(tb_image, cmap="gray")
        for point in maxima:
            ax.plot(point[1], point[0], "b+")
        ax.plot(true_center[1], true_center[0], "r+")

