from skimage import measure
import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import corner_harris, corner_peaks
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from skimage import exposure

from classes import Eye, Point
from utils.eyecorners import find_eye_corners
from utils.eyecenter.interface import EyeFeaturesExtractor
from utils.histogram.lsh_equalization import lsh_equalization


class PyHoughEyecenter(EyeFeaturesExtractor):

    def __init__(self):
        super().__init__()

    def create_debug_figure(self):
        fig, (r1, l1) = plt.subplots(1, 2)
        axes = (r1, l1)
        return fig, axes

    def detect_eye_features(self, eye_image, eye_object):
        assert(isinstance(eye_object, Eye))

        # Part 1: finding the center of the eye

        if self.equalization != lsh_equalization:
            eye_image = exposure.equalize_hist(eye_image)
        # apply a threshold to the image
        try:
            thresh = threshold_otsu(eye_image) * 0.3
            eye_binary = eye_image < thresh
        except Exception:
            print("\nerror shape:")
            print(eye_image.shape)
            print("eye shape:")
            print(eye_object.area)

        centers = []
        accums = []
        radii = []

        base_width = eye_object.area.width/2.0
        hough_radii = np.arange(base_width/4.0, base_width/3.0, 2)

        hough_res = hough_circle(eye_binary, hough_radii)
        _accums, cx, cy, _radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
        accums.extend(_accums)
        centers.extend(zip(cx, cy))
        radii.extend(_radii)

        # the center of the pupil is the center of the best circular region found
        try:
            best_circle_index = np.argmax(np.array(accums))
            center_x, center_y = centers[best_circle_index]
        except ValueError:
            return

        # Part 2: finding the corners of the eye

        if self.debug_mode:
            debug_ax = self.debug_axes[0]
            if not eye_object.is_right:
                debug_ax = self.debug_axes[1]

            debug_ax.set_title("right eye" if eye_object.is_right else "left eye")
            debug_ax.imshow(eye_image, cmap="gray")

            debug_ax.plot(center_x, center_y, "r+")

        eye_object.pupil_relative = Point(center_x, center_y)

