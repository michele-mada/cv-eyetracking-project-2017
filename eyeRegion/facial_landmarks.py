from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

#Start and end indexes for left (l) and right (r) eyes in 68-points
#I did it manually cause i can't access the proper imutils attributes
rstart = 36
rend = 42
lstart = 42
lend = 48

#lists of left and rright eyes detected (for future use)
leyes = list()
reyes = list()

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat", help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector(HOG-based)
# and then create the facial landmark preditor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
# lowering the image size increase performances but can cause
# problems during the facial detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in grayscale image
rects = detector(gray, 1)
if len(rects) == 0:
    gray = cv2.equalizeHist(gray)
    rects = detector(gray, 1)

# search for the landmarks of all the detected faces
for (i, rect) in enumerate(rects):
    # detect facial landmarks for the face region
    shape = predictor(gray, rect)
    # convert the facial landmark coordinates to NumPy array
    shape = face_utils.shape_to_np(shape)

    leye = shape[lstart:lend]
    reye = shape[rstart:rend]

    #left eye rectangle
    x_min = min(leye[:,0])
    x_max = max(leye[:,0])
    y_min = min(leye[:,1])
    y_max = max(leye[:,1])
    diff_x = int((x_max-x_min)/2)
    diff_y = int((y_max - y_min)/2)
    l_a = [x_min, y_max]
    l_b = [x_max, y_max]
    l_c = [x_max, y_min]
    l_d = [x_min, y_min]
    # draw left eye rectangle corners
    cv2.circle(image, (l_a[0], l_a[1]), 1, (255, 0, 255), -1)
    cv2.circle(image, (l_b[0], l_b[1]), 1, (255, 0, 255), -1)
    cv2.circle(image, (l_c[0], l_c[1]), 1, (255, 0, 255), -1)
    cv2.circle(image, (l_d[0], l_d[1]), 1, (255, 0, 255), -1)
    leye_img=image[(y_min-diff_y):(y_max+diff_y),(x_min-diff_x):(x_max+diff_x)]
    cv2.imshow("left eye", leye_img)

    # right eye rectangle
    x_min = min(reye[:, 0])
    x_max = max(reye[:, 0])
    y_min = min(reye[:, 1])
    y_max = max(reye[:, 1])
    diff_x = int((x_max-x_min)/2)
    diff_y = int((y_max - y_min)/2)
    r_a = [x_min, y_max]
    r_b = [x_max, y_max]
    r_c = [x_max, y_min]
    r_d = [x_min, y_min]
    # draw right eye rectangle corners
    cv2.circle(image, (r_a[0], r_a[1]), 1, (255, 0, 255), -1)
    cv2.circle(image, (r_b[0], r_b[1]), 1, (255, 0, 255), -1)
    cv2.circle(image, (r_c[0], r_c[1]), 1, (255, 0, 255), -1)
    cv2.circle(image, (r_d[0], r_d[1]), 1, (255, 0, 255), -1)
    reye_img = image[(y_min - diff_y):(y_max + diff_y), (x_min - diff_x):(x_max + diff_x)]
    cv2.imshow("right_eye", reye_img)
    # add eyes point to the proper list
    leyes.append(leye)
    reyes.append(reye)

    # convert dlib's rectangle to a OpenCV-style bounding box
    #  (x,y,width,height)
    (x,y,w,h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x,y),(x+w,y+w),(0,255,0),2)

    # loop over the coordinates for the facial landmark
    # and draw them on the image
    # for(x,y) in shape:
    #     cv2.circle(image, (x,y),1,(0,0,255),-1)

    # draw eyes boundary
    for(x,y) in leye:
        cv2.circle(image, (x,y),1,(0,0,255),-1)
    for (x, y) in reye:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # draw pupil

cv2.imshow("Output", image)
cv2.waitKey(0)
