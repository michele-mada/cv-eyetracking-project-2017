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
        fig, ((r1, l1), (r2, l2), (r3, l3)) = plt.subplots(3, 2)
        axes = (r1, r2, r3, l1, l2, l3)
        return fig, axes

    def detect_eye_features(self, eye_image, eye_object):
        assert(isinstance(eye_object, Eye))

        # Part 1: finding the center of the eye

        if self.equalization != lsh_equalization:
            eye_image = exposure.equalize_hist(eye_image)
        # apply a threshold to the image
        try:
            thresh_eyeball = threshold_otsu(eye_image) * 0.5
            thresh_eyecorner = threshold_otsu(eye_image)
            eye_binary_ball = eye_image < thresh_eyeball
            eye_binary_corner = eye_image < thresh_eyecorner
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

        hough_res = hough_circle(eye_binary_ball, hough_radii)
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
        right_corner, left_corner, all_corners = find_eye_corners(eye_binary_corner, Point(center_x, center_y))

        if self.debug_mode:
            (ax_1, ax_2, ax_3) = self.debug_axes[0:3]
            if not eye_object.is_right:
                (ax_1, ax_2, ax_3) = self.debug_axes[3:6]

            ax_1.set_title("right eye" if eye_object.is_right else "left eye")
            ax_1.imshow(eye_image, cmap="gray")
            ax_2.imshow(eye_binary_ball, cmap="gray")
            ax_3.imshow(eye_binary_corner, cmap="gray")

            ax_1.plot(center_x, center_y, "r+")
            ax_2.plot(center_x, center_y, "r+")
            corner_x, corner_y = zip(*all_corners)
            ax_3.plot(corner_x, corner_y, "b.")
            ax_1.plot(right_corner.x, right_corner.y, "ro")
            ax_1.plot(left_corner.x, left_corner.y, "go")
            ax_3.plot(right_corner.x, right_corner.y, "ro")
            ax_3.plot(left_corner.x, left_corner.y, "go")



        eye_object.pupil_relative = Point(center_x, center_y)
        #eye_object.set_rightmost_corner(right_corner)
        #eye_object.set_leftmost_corner(left_corner)

