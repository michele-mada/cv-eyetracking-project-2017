from skimage.feature import corner_harris, corner_peaks
from skimage.filters import gaussian


# heuristic to find the best corner, given its position relative to the pupil center
def corner_measure(corner, center, leftness=1):
    cy, cx = corner
    zx, zy = center
    return ( cx -zx) * leftness - 1.5 * abs( cy -zy)


def find_eye_corners(eye_image, eye_center):
    all_corners = corner_peaks(corner_harris(eye_image))

    # give each corner a score, according to our heuristic
    left_corner_score = map(lambda corner: (corner,
                                            corner_measure(corner, eye_center, leftness=1)
                                            ),
                            all_corners)
    # find the best corner
    try:
        left_corner = max(left_corner_score, key=lambda cs: cs[1])[0]
    except Exception:
        left_corner = (0, 0)

    # give each corner a score, according to our heuristic
    right_corner_score = map(lambda corner: (corner,
                                             corner_measure(corner, eye_center, leftness=-1)
                                             ),
                             all_corners)
    # find the best corner
    try:
        right_corner = max(right_corner_score, key=lambda cs: cs[1])[0]
    except Exception:
        right_corner = (0, 0)

    return right_corner, left_corner, all_corners  #TODO: fix x and y coords being reversed
