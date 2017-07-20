import argparse
import sys
from math import sqrt

import cv2
import matplotlib

from utils.process_frame import process_frame
from utils.visualization import draw_routine

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import exposure

from utils.camera import WebcamVideoStream
from utils.eyecenter.py_eyecenter import PyHoughEyecenter
from utils.eyecenter.timm.timm_and_barth import TimmAndBarth
from utils.histogram.lsh_equalization import lsh_equalization

from utils.bioID import BioIDFaceDatabase



algos = {
    "hough": PyHoughEyecenter,
    "timm": TimmAndBarth
}
equaliz = {
    "h": exposure.equalize_hist,
    "ah": lambda img: exposure.equalize_adapthist(img, clip_limit=0.03),
    "lsh": lsh_equalization,
}


def get_cascade_files(cli):
    return {
        "eye": cli.eye_cascade_file,
        "face": cli.face_cascade_file
    }


def one_shot(cli, algo):
    image_cv2 = cv2.imread(cli.file)
    image_cv2_gray = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2GRAY)

    if cli.debug:
        fig2, axes_2 = algo.create_debug_figure()
        algo.setup_debug_parameters(True, axes_2)

    cascade_files = get_cascade_files(cli)

    picture, right_eye, left_eye, detect_string, not_eyes = process_frame(image_cv2_gray, algo, cascade_files)

    print("Eye detection successful (%s)" % detect_string)
    print("Eyes (R,L):\n  %s,\n  %s" % (str(right_eye), str(left_eye)))

    draw_routine(picture, right_eye, left_eye, not_eyes, "detection")
    plt.show()


def test_run(cli, algo):
    cascade_files = get_cascade_files(cli)

    # define various accuracy metrics
    accuracy_tiers = [
        [0.25, "distance eye center-corner", 0],
        [0.10, "iris diameter", 0],
        [0.05, "pupil diameter", 0],
    ]

    def distance(expected, detected):
        return sqrt((detected.pupil.x - expected.x)**2 +
                    (detected.pupil.y - expected.y)**2)

    def error_estimate(face, detect_righteye, detect_lefteye):
        return max(distance(face.left_eye, detect_lefteye),
                   distance(face.right_eye, detect_righteye)) / face.eye_center_distance

    total_error = 0.0
    missed_detections = 0

    # initialize the BioID face database
    facedb = BioIDFaceDatabase(cli.bioid_folder)
    if len(facedb.faces) == 0:
        print("Error: bioid face database not found at \"%s\"" % cli.bioid_folder)
        sys.exit(-1)

    # run the tests
    print("Testing against %d faces" % len(facedb.faces))
    for n, face in enumerate(facedb.faces):
        picture, right_eye, left_eye, detect_string, not_eyes = process_frame(face.load_cv2(), algo, cascade_files)
        if right_eye is None or left_eye is None:
            missed_detections += 1
        else:
            e = error_estimate(face, right_eye, left_eye)
            total_error += e
            for i in range(len(accuracy_tiers)):
                if e <= accuracy_tiers[i][0]:
                    accuracy_tiers[i][2] += 1
        print("testing: done %.2f%%" % ((float(n)*100) / float(len(facedb.faces)),), end="\r")

    total_detected = len(facedb.faces) - missed_detections
    total_error /= total_detected

    # print the results
    print("Test results:                   ")
    print("Correct detections: %d out of %d (%.2f)" % (total_detected,
                                                       len(facedb.faces),
                                                       float(total_detected) / len(facedb.faces)))
    print("Average error: %f" % total_error)
    print("Accuracy (tiered):")
    for tier in accuracy_tiers:
        print("    e <= %.2f (%s): %.2f" % (tier[0],tier[1], float(tier[2]) / total_detected))


def live(cli, algo):
    camera = WebcamVideoStream(src=cli.camera_port,
                               debug=cli.debug,
                               contrast=None if cli.contrast == "None" else float(cli.contrast),
                               saturation=None if cli.saturation == "None" else float(cli.saturation)
                               )
    camera.start()
    plt.ion()
    if cli.debug:
        fig2, axes_2 = algo.create_debug_figure()
        algo.setup_debug_parameters(True, axes_2)

    cascade_files = get_cascade_files(cli)

    while True:
        image_cv2 = camera.read()
        image_cv2_gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)

        algo.clean_debug_axes()

        picture, right_eye, left_eye, detect_string, not_eyes = process_frame(image_cv2_gray, algo, cascade_files)

        draw_routine(picture, right_eye, left_eye, not_eyes, "detection")

        key = cv2.waitKey(1)
        if key == 27: break

    cv2.destroyAllWindows()  # I like the name of this function
    camera.stop()


def main(cli):
    algo = algos[cli.algo]()
    algo.equalization = equaliz[cli.equalization]

    if cli.algo == "timm":
        algo.context.load_program(program_path="cl_kernels/timm_barth_smallpic_kernel.cl")
        #algo.context.load_program()

    if cli.file == "-":
        live(cli, algo)
    elif cli.file == "test":
        test_run(cli, algo)
    else:
        one_shot(cli, algo)


def parsecli():
    parser = argparse.ArgumentParser(description="Find eyes and eye-centers from an image")
    # main generic parameters
    parser.add_argument('file', help='filename of the picture; - for webcam; \"test\" to run a performance test', type=str)
    parser.add_argument('-d', '--debug', help='enable debug mode', action='store_true')
    # haar cascade detection parameters
    parser.add_argument('--eye-cascade-file', help='path to the .xml file with the eye-detection haar cascade',
                        type=str, default="../haarcascades/haarcascade_righteye_2splits.xml")
    parser.add_argument('--face-cascade-file', help='path to the .xml file with the face-detection haar cascade',
                        type=str, default="../haarcascades/haarcascade_frontalface_default.xml")
    # camera parameters
    parser.add_argument('--saturation', help='override the webcam default saturation setting (value [0.0, 1.0])',
                        type=str, default="None")
    parser.add_argument('--contrast', help='override the webcam default contrast setting (value [0.0, 1.0])',
                        type=str, default="None")
    parser.add_argument('-c','--camera-port', help='numeric index of the camera to use',
                        type=int, default=0)
    # algorithm selection parameters
    parser.add_argument('-a','--algo', help='pupil center algorithm to use',
                        type=str, default="hough", choices=algos.keys())
    parser.add_argument('-e', '--equalization', help='type of histogram equalization to use',
                        type=str, default="ah", choices=equaliz.keys())
    # other
    parser.add_argument('--bioid-folder', metavar='BIOID_FOLDER', help='BioID face database folder, to use in the \"test\" mode',
                        type=str, default="../../BioID-FaceDatabase-V1.2")
    return parser.parse_args()


if __name__ == "__main__":
    cli = parsecli()
    main(cli)

