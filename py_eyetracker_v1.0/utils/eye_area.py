import cv2

from classes import Eye, Rect


def detect_haar_cascade(image, cascPath):
    """
    Uses an OpenCV haar cascade to locate the rectangles of the image in which
    each eye resides
    :param imagePath: file path of the image
    :return: list of lists representing the eye recangles [x, y, width, height]
    """
    cascade = cv2.CascadeClassifier(cascPath)

    # Detect faces in the image
    eyes = cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    return eyes


def split_eyes(eyes):
    """
    Select right eye and left eyes.
    The criteria is simple: the right eye is always the one with smaller x
    :param eyes: list of rectangles representing the eyes
    :return: tuple (right_eye, left_eye) of Eye class
    """
    pairs = [(a, b) for a in eyes for b in eyes]

    def evaluate_pair(pair):
        right_eye, left_eye = pair
        score = 0
        if right_eye[0] < left_eye[0]:
            score += 100
        if left_eye[0] > right_eye[0] + right_eye[2] * 1.5:
            score += 50
        if abs(left_eye[1] - right_eye[1]) < left_eye[3]:
            score += 40
        if abs(left_eye[1] - right_eye[1]) < right_eye[3]:
            score += 40
        return score

    true_pair = max(pairs, key=evaluate_pair)
    right_eye, left_eye = true_pair

    return Eye(right_eye, True), Eye(left_eye, False)


def eye_regions_from_face(face):
    """
    Given a rect representing the face, compute two smaller rectangles
    covering the eye regions. The eye regions are determined geometrically.
    :param face: tuple describing the face rect (x,y,width,height)
    :return: 
    """
    facearea = Rect(*face)
    eye_percent_top = 0.25
    eye_percent_side = 0.13
    eye_percent_height = 0.25
    eye_percent_width = 0.30

    eye_region_width = facearea.width * eye_percent_width
    eye_region_height = facearea.height * eye_percent_height
    eye_percent_top = facearea.height * eye_percent_top

    right_eye_rect = Rect(x=int(facearea.x + facearea.width - eye_region_width - facearea.width * eye_percent_side),
                          y=int(facearea.y + eye_percent_top),
                          width=int(eye_region_width), height=int(eye_region_height))
    left_eye_rect = Rect(x=int(facearea.x + facearea.width * eye_percent_side),
                         y=int(facearea.y + eye_percent_top),
                         width=int(eye_region_width), height=int(eye_region_height))
    return [right_eye_rect, left_eye_rect]