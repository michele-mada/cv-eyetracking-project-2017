from skimage import measure
import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import corner_harris, corner_peaks
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from skimage import exposure

from classes import Eye, Point


def find_eye_center_and_corners_hough(eye_image, eye_object, debug=False, debug_ax=None):
    assert(isinstance(eye_object, Eye))

    # Part 1: finding the center of the eye

    eye_image = exposure.equalize_hist(eye_image)
    # apply a threshold to the image
    thresh = threshold_otsu(eye_image) * 0.3
    eye_binary = eye_image < thresh

    # split the thresholded image into regions
    blobs_labels, nlabels = measure.label(eye_binary, background=0, return_num=True)

    centers = []
    accums = []
    radii = []

    base_width = eye_object.area.width/2.0
    hough_radii = np.arange(base_width/4.0, base_width/3.0, 2)

    """# foreach region, use the hough transform to find the most prominent circle
    for n in range(1, nlabels):
        img = blobs_labels == n
        hough_res = hough_circle(img, hough_radii)
        _accums, cx, cy, _radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
        accums.extend(_accums)
        centers.extend(zip(cx, cy))
        radii.extend(_radii)"""

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

    # Part 1: finding the corners of the eye

    # heuristic to find the best corner, given its position relative to the pupil center
    def corner_measure(corner, center, leftness=1):
        cy, cx = corner
        zx, zy = center
        return (cx-zx) * leftness - 1.5 * abs(cy-zy)

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
        if debug_ax is None:
            fig, debug_ax = plt.subplots(1)

        plt.title("right eye" if eye_object.is_right else "left eye")
        debug_ax.imshow(eye_binary, cmap="gray")

        debug_ax.imshow(blobs_labels, cmap='spectral')
        debug_ax.plot(center_x, center_y, "w+")

        debug_ax.plot(corners[:,1], corners[:,0], "bo")

        debug_ax.plot(left_corner[1], left_corner[0], "ro")
        debug_ax.plot(right_corner[1], right_corner[0], "go")

    eye_object.pupil_relative = Point(center_x, center_y)
    eye_object.set_leftmost_corner(Point(x=left_corner[1], y=left_corner[0]))
    eye_object.set_rightmost_corner(Point(x=right_corner[1], y=right_corner[0]))

