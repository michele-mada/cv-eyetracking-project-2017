import cv2
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from skimage import img_as_float

from utils.histogram.lsh import locality_sensitive_histogram_hybrid as locality_sensitive_histogram
from utils.histogram.iif import illumination_invariant_features_cl as illumination_invariant_features


if __name__ == "__main__":
    image_cv2 = cv2.imread("../test_images/webcam.jpg")
    image_cv2_gray = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2GRAY)
    picture_float = img_as_float(image_cv2_gray)

    fig, ax_img = plt.subplots(1)

    histogram = locality_sensitive_histogram(picture_float, debug_ax=ax_img)
    iif = illumination_invariant_features(picture_float, histogram, debug_ax=ax_img)

    ax_img.imshow(iif, cmap="gray")
    plt.show()