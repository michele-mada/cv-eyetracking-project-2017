import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from skimage import exposure
from skimage.feature import peak_local_max
from skimage.filters import scharr_h, scharr_v, gaussian
from skimage.transform import resize
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu

from classes import Point
from utils.eyecenter.interface import EyeFeaturesExtractor
from utils.eyecenter.timm.cl_runner import CLTimmBarth
from utils.eyecorners import find_eye_corners
from utils.histogram.lsh_equalization import lsh_equalization


class TimmAndBarth(EyeFeaturesExtractor):


    def __init__(self):
        super().__init__()
        # prepare the opencl context
        self.context = CLTimmBarth(precomputation=self.precomputation)
        self.debug_edgemap = None
        self.locality_factor = 0.2 # 0.15

    def create_debug_figure(self):
        fig, ((r1, l1), (r2, l2), (r3, l3)) = plt.subplots(3, 2)
        axes = (r1, r2, r3, l1, l2, l3)
        return fig, axes

    def precomputation(self, eye_image):
        # compute gradients across x and y
        # (scharr gives better results than sobel)
        x_gradient = scharr_v(eye_image)
        y_gradient = scharr_h(eye_image)
        edgemap = np.sqrt(x_gradient ** 2 + y_gradient ** 2) + 0.00001

        # remove gradients under a specific threshold
        # (as suggested on http://thume.ca/projects/2012/11/04/simple-accurate-eye-center-tracking-in-opencv/)
        mean_gradient = np.mean(edgemap)
        #width, height = np.shape(eye_image)
        std_gradient = np.std(edgemap)
        threshold = 0.3 * std_gradient + mean_gradient
        under_threshold_indices = edgemap < threshold
        x_gradient[under_threshold_indices] = 0
        y_gradient[under_threshold_indices] = 0

        # normalize the gradients
        maxedge = np.amax(edgemap)
        x_gradient /= maxedge
        y_gradient /= maxedge
        self.debug_edgemap = np.sqrt(x_gradient ** 2 + y_gradient ** 2)

        # compute the inverse-colored image
        inverse = gaussian(1.0 - eye_image)
        return x_gradient, y_gradient, inverse

    def detect_eye_features(self, eye_image, eye_object):

        # scale the image if needed
        width, height = np.shape(eye_image)
        max_w = 120
        scale_factor = 1.0
        if width > max_w:
            max_h = int((height / width) * max_w)
            scale_factor = height / max_h
            eye_image = resize(eye_image, (max_w, max_h))

        # important: histogram equalization
        if self.equalization != lsh_equalization:
            eye_image = exposure.equalize_hist(eye_image)
        #eye_image = gaussian(eye_image)
        width, height = np.shape(eye_image)
        # run the actual timm & barth filter
        tb_image = self.context.compute(eye_image, locality=int(width * self.locality_factor))

        #mask = tb_image > np.amax(tb_image) * 0.5
        #mask_clear_border = clear_border(mask)
        #mask_clear_border = mask
        #tb_image = tb_image * mask_clear_border

        thresh = threshold_otsu(eye_image)
        eye_binary = eye_image < thresh

        (ax_1, ax_2, ax_3) = [None] * 3
        if self.debug_mode:
            (ax_1, ax_2, ax_3) = self.debug_axes[0:3]
            if not eye_object.is_right:
                (ax_1, ax_2, ax_3) = self.debug_axes[3:6]
            eye_title = "right eye" if eye_object.is_right else "left eye"
            ax_1.set_title(eye_title + " capture")
            ax_2.set_title("gradient")
            ax_3.set_title("TB value")
            ax_1.imshow(eye_image, cmap="gray")
            ax_2.imshow(self.debug_edgemap, cmap="gray")
            ax_3.imshow(tb_image, cmap="viridis", alpha=.5)

        # find the local peaks in the function
        maxima = peak_local_max(tb_image, min_distance=int(width * self.locality_factor * 0.9))
        # in our imperfect world, we must also check that the peaks reside within the bounds of the image
        def is_in_bounds(point):
            y, x = point
            return x > 0 and x < width and y > 0 and y < height
        maxima = list(filter(is_in_bounds, maxima))
        if len(maxima) == 0:
            return
        # select the actual center among the candidates
        def center_euristic(point):
            y,x = point
            # objective function must have a high value; point must be central; low point are preferred
            return 5 * tb_image[x, y] + \
                   -1 * sqrt((x - (width / 2)) ** 2 + (y - (height / 2)) ** 2) + \
                   y * 0.1
        true_center = max(maxima, key=center_euristic)
        center_x = int(true_center[1] * scale_factor)
        center_y = int(true_center[0] * scale_factor)

        right_corner, left_corner, corners = find_eye_corners(eye_binary, (center_x, center_y))

        eye_object.pupil_relative = Point(center_x, center_y)
        eye_object.set_leftmost_corner(Point(x=left_corner[1], y=left_corner[0]))
        eye_object.set_rightmost_corner(Point(x=right_corner[1], y=right_corner[0]))

        if self.debug_mode:
            ax_1.plot(corners[:, 1], corners[:, 0], "b.")
            ax_1.plot(left_corner[1], left_corner[0], "r.")
            ax_1.plot(right_corner[1], right_corner[0], "g.")

            for point in maxima:
                ax_3.plot(point[1], point[0], "b+")
            ax_3.plot(true_center[1], true_center[0], "r+")
            ax_2.plot(true_center[1], true_center[0], "r+")
            ax_1.plot(true_center[1], true_center[0], "r+")

