import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.feature import peak_local_max
from skimage.filters import scharr_h, scharr_v, gaussian
from skimage.transform import resize

from classes import Point
from utils.eyecenter.timm.cl_runner import CLTimmBarth


def precomputation(eye_image):
    x_gradient = scharr_v(eye_image)
    y_gradient = scharr_h(eye_image)
    edgemap = np.sqrt(x_gradient ** 2 + y_gradient ** 2) + 0.00001
    maxedge = np.amax(edgemap)
    x_gradient /= maxedge
    y_gradient /= maxedge
    inverse = gaussian(1.0 - eye_image)
    return x_gradient, y_gradient, inverse


context = CLTimmBarth(precomputation=precomputation)


def find_eye_center_and_corners_cl(eye_image, eye_object, debug=False):
    width, height = np.shape(eye_image)
    max_w = 120
    scale_factor = 1.0
    if width > max_w:
        max_h = (height / width) * max_w
        scale_factor = height / max_h
        eye_image = resize(eye_image, (max_w, max_h))

    eye_image = exposure.equalize_hist(eye_image)
    width, height = np.shape(eye_image)
    tb_image = context.compute(eye_image, locality=width/2)
    maxima = peak_local_max(tb_image, min_distance=10)

    true_center = max(maxima, key=lambda point: tb_image[point[1], point[0]])

    eye_object.pupil_relative = Point(true_center[1] * scale_factor, true_center[0] * scale_factor)

    if debug:
        fig, ax = plt.subplots(1)
        plt.title("right eye" if eye_object.is_right else "left eye")
        ax.imshow(tb_image, cmap="gray")
        for point in maxima:
            ax.plot(point[1], point[0], "b+")
        ax.plot(true_center[1], true_center[0], "r+")

