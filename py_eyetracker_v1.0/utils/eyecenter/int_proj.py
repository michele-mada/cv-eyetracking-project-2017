import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

from classes import Eye, Point
from utils.eyecenter.interface import EyeFeaturesExtractor
from utils.histogram.lsh_equalization import lsh_equalization


class GeneralIntegralProjection(EyeFeaturesExtractor):

    def __init__(self, alpha=0.4):
        super().__init__()
        self.alpha = alpha

    def create_debug_figure(self):

        r1 = plt.subplot(2, 4, 1)
        r2 = plt.subplot(2, 4, 5, sharex=r1)
        r3 = plt.subplot(2, 4, 2, sharey=r1)
        l1 = plt.subplot(2, 4, 3)
        l2 = plt.subplot(2, 4, 7, sharex=l1)
        l3 = plt.subplot(2, 4, 4, sharey=l1)

        axes = (r1, r2, r3, l1, l2, l3)
        return None, axes

    def detect_eye_features(self, eye_image, eye_object):
        assert(isinstance(eye_object, Eye))

        if self.equalization != lsh_equalization:
            eye_image = exposure.equalize_hist(eye_image)

        intensity = 1 - eye_image
        (height, width) = intensity.shape

        ipf_x = np.sum(intensity, axis=0) / height
        ipf_y = np.sum(intensity, axis=1) / width
        ipf_y_vert = np.array(np.transpose(np.matrix(ipf_y)))

        ipf_x_tiled = np.tile(ipf_x, (height, 1))
        ipf_y_tiled = np.tile(ipf_y_vert, (1, width))

        vpf_x = np.sum((intensity - ipf_x_tiled) ** 2, axis=0) / height
        vpf_y = np.sum((intensity - ipf_y_tiled) ** 2, axis=1) / width

        gpf_x = (1 - self.alpha) * ipf_x + self.alpha * vpf_x
        gpf_y = (1 - self.alpha) * ipf_y + self.alpha * vpf_y

        center_x = np.argmax(gpf_x)
        center_y = np.argmax(gpf_y)

        if self.debug_mode:
            (ax_1, ax_2, ax_3) = self.debug_axes[0:3]
            if not eye_object.is_right:
                (ax_1, ax_2, ax_3) = self.debug_axes[3:6]

            ax_1.set_xlim(width)
            ax_1.set_ylim(height)

            ax_1.imshow(eye_image, cmap="gray")
            ax_1.plot(center_x, center_y, "r+")
            ax_2.plot(list(range(width)), gpf_x)
            ax_3.plot(gpf_y, list(range(height)))

        eye_object.pupil_relative = Point(center_x, center_y)

