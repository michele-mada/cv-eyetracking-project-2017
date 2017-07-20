import numpy as np
from skimage import img_as_float, img_as_ubyte

from classes import Rect
from utils.eye_area import detect_haar_cascade, eye_regions_from_face, split_eyes
from utils.face_landmarks.dlib_based import eye_area_detection_step, pick_eye_corners


def cropped_rect(image, rect):
    """
    Crops an image represented by a 2d numpy array
    :param image: numpy 2d array
    :param rect: list or tuple of four elements: x,y,width,height
    :return: numpy 2d array containing the cropped region
    """
    (x1, y1, width, height) = rect
    if x1 < 0:
        width += x1
        x1 = 0
    if y1 < 0:
        height += y1
        y1 = 0
    return np.copy(image[y1:y1+height,x1:x1+width])


def image_preprocessing_step(image_cv2format, algo):
    picture_float = img_as_float(image_cv2format)
    picture_float_equalized = algo.equalization(picture_float)
    image_cv2format_equalized = img_as_ubyte(picture_float_equalized)
    picture = picture_float_equalized
    return picture, image_cv2format, image_cv2format_equalized


def geometric_eye_area_selection_step(eyes):
    right_eye, left_eye = split_eyes(eyes)

    not_eyes = list(map(tuple, eyes))
    not_eyes.remove(tuple(left_eye.area))
    not_eyes.remove(tuple(right_eye.area))
    return right_eye, left_eye, not_eyes


def eye_features_extraction_step(picture, right_eye, left_eye, algo):
    right_eyepatch = cropped_rect(picture, right_eye.area)
    algo.detect_eye_features(right_eyepatch, right_eye)

    left_eyepatch = cropped_rect(picture, left_eye.area)
    algo.detect_eye_features(left_eyepatch, left_eye)


def process_frame(image_cv2format, algo, cascade_files):
    """
    Preprocess and image and extract useful features
    :param image_cv2format: cv2 image (numpy 2d array of ubyte)
    :param algo: algorithm to use to preprocess / extract features
    :return: tuple containing:
        postprocessed image (in the skimage format, aka numpy 2d array of float)
        right eye object (see classes.py)
        left eye object (see classes.py)
        debug string describing the detection method
        list of Rect which could be eyes but were refused by the geometric estimator
    """

    # pre-processing
    picture, image_cv2format, image_cv2format_equalized = image_preprocessing_step(image_cv2format, algo)

    # eye area detection
    success, detect_method, eyes, points68 = eye_area_detection_step(image_cv2format, image_cv2format_equalized)
    #success, detect_method, eyes = eye_area_detection_step_haar(image_cv2format, image_cv2format_equalized, cascade_files)
    if not success:
        return picture, None, None, detect_method, []

    # geometrically select right and left eye
    right_eye, left_eye, not_eyes = geometric_eye_area_selection_step(eyes)
    pick_eye_corners(right_eye, points68)
    pick_eye_corners(left_eye, points68)

    # extract the eye features (updates the objects right_eye and left_eye)
    eye_features_extraction_step(picture, right_eye, left_eye, algo)

    return picture, right_eye, left_eye, detect_method, not_eyes