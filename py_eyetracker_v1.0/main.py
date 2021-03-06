import argparse
import sys
from math import sqrt, exp

import cv2
import matplotlib

from utils.gui.visualization import draw_routine
from utils.process_frame import process_frame

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import exposure

from classes import Tracker
from utils.camera.capture import WebcamVideoStream
from utils.eyecenter.hough import PyHoughEyecenter
from utils.eyecenter.timm.timm_and_barth import TimmAndBarth
from utils.eyecenter.int_proj import GeneralIntegralProjection
from utils.histogram.lsh_equalization import lsh_equalization

from utils.bioID import BioIDFaceDatabase

from utils.gui.tracking_board import TrackingBoard
from utils.screen_mapping import mapper_implementations
from utils.screen_mapping.screen_tools import get_screen_size
from utils.screen_mapping.drive_scrolling import drive_scrolling

from app_control import app_controllers


algos = {
    "hough": PyHoughEyecenter,
    "timm": TimmAndBarth,
    "gip": GeneralIntegralProjection,
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

    if cli.debug:
        fig2, axes_2 = algo.create_debug_figure()
        algo.setup_debug_parameters(True, axes_2)

    cascade_files = get_cascade_files(cli)

    picture, face, detect_string, not_eyes = process_frame(image_cv2, algo, cascade_files)

    print("Eye detection successful (%s)" % detect_string)
    print("Eyes (R,L):\n  %s,\n  %s" % (str(face.right_eye), str(face.left_eye)))

    if cli.debug:
        plt.pause(0.01)

    draw_routine(picture, face, not_eyes, "detection", draw_unicorn=cli.unicorn)
    cv2.waitKey(-1)


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

    def error_estimate(bioid_face, detect_righteye, detect_lefteye):
        return max(distance(bioid_face.left_eye, detect_lefteye),
                   distance(bioid_face.right_eye, detect_righteye)) / bioid_face.eye_center_distance

    total_error = 0.0
    missed_detections = 0

    # initialize the BioID face database
    facedb = BioIDFaceDatabase(cli.bioid_folder)
    if len(facedb.faces) == 0:
        print("Error: bioid face database not found at \"%s\"" % cli.bioid_folder)
        sys.exit(-1)

    # run the tests
    print("Testing against %d faces" % len(facedb.faces))
    for n, bioid_face in enumerate(facedb.faces):
        picture, face, detect_string, not_eyes = process_frame(bioid_face.load_cv2(), algo, cascade_files, already_grayscale=True)
        if face is None:
            missed_detections += 1
        else:
            e = error_estimate(bioid_face, face.right_eye, face.left_eye)
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
    # initialize asynchronus camera polling system
    camera = WebcamVideoStream(src=cli.camera_port,
                               debug=cli.debug,
                               contrast=None if cli.contrast == "None" else float(cli.contrast),
                               saturation=None if cli.saturation == "None" else float(cli.saturation)
                               )
    # get the correct screen size
    override_screensize = None
    if cli.override_screensize != "None":
        override_screensize = tuple(map(int, cli.override_screensize.split("x")))
    screensize = get_screen_size(override_screensize=override_screensize)
    # initialize the scroll-controlled application
    app = app_controllers[cli.app]()
    # start capturing
    camera.start()

    # initialize matplotlib, needed for debug purposes
    if not cli.quiet:
        plt.ion()
    if cli.debug and not cli.quiet:
        fig2, axes_2 = algo.create_debug_figure()
        algo.setup_debug_parameters(True, axes_2)

    # load haar cascade files (note: haar cascades are disabled right now)
    cascade_files = get_cascade_files(cli)
    # initialize the tracker
    tracker = Tracker(mapper_implementations[cli.mapping_function],
                      smooth_frames=cli.smoothing,
                      centroid_history_frames=cli.centroid_history,
                      #smooth_weight_fun=lambda x: exp(-x*0.5)
                      )
    # initialize the tracking blackboard
    if cli.tracking:
        if not cli.quiet:
            trackboard = TrackingBoard(screensize=screensize)
        tracker.load_saved_cal_params()

    while True:
        image_cv2 = camera.read()

        if cli.debug and not cli.quiet:
            algo.clean_debug_axes()

        picture, face, detect_string, not_eyes = process_frame(image_cv2, algo, cascade_files)

        if face is not None and face.right_eye is not None and face.left_eye is not None:
            tracker.update(face)
            drive_scrolling(tracker, app, screensize)
            print("right eyevector:", tracker.face.normalized_right_eye_vector)
            print("left eyevector:", tracker.face.normalized_left_eye_vector)

            if cli.tracking and not cli.quiet:
                coord_right, coord_left, coord_centroid = tracker.get_onscreen_gaze_mapping()
                print("right, left: ", coord_right, coord_left)
                head_pose = tracker.face.head_pose
                trackboard.update(coord_right, coord_left, coord_centroid, head_pose)

        smooth_face = tracker.face
        if smooth_face is not None and smooth_face.right_eye is not None and smooth_face.left_eye is not None:
            if not cli.quiet:
                draw_routine(picture, smooth_face, not_eyes, "detection", draw_unicorn=cli.unicorn)

        key = cv2.waitKey(1)
        if key == 27: break

        if cli.debug:
            plt.pause(0.01)

    if not cli.quiet:
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
    parser = argparse.ArgumentParser(description="Eye tracking experiment")
    # main generic parameters
    parser.add_argument('file', help='filename of the picture; - for webcam; \"test\" to run a performance test', type=str)
    parser.add_argument('-d', '--debug', help='enable debug mode', action='store_true')
    parser.add_argument('-A', '--app', help='drive the scrolling of an application. Default is None.', type=str, default="None", choices=app_controllers.keys())
    parser.add_argument('-q', '--quiet', help='suppress all windows, except the application one (see -A)', action='store_true')

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
    # gaze tracking parameters
    parser.add_argument('-u', '--unicorn', help='draw a debug vector indicating the face orientation', action='store_true')
    parser.add_argument('-t', '--tracking', help='display the eye tracking whiteboard', action='store_true')
    parser.add_argument('-m', '--mapping-function', help='eye-vector to screen mapping function to use',
                        type=str, default="poly_quad", choices=mapper_implementations.keys())
    parser.add_argument('-o', '--override-screensize', metavar='1366x768', help='specify the size of your tracking board',
                        type=str, default="None")
    parser.add_argument('--smoothing', metavar='int',
                        help='smooth the tracking data averaging across the last N frames',
                        type=int, default=1)
    parser.add_argument('--centroid-history', metavar='int',
                        help='compute the gaze centroid across the last N frames',
                        type=int, default=5)
    # other
    parser.add_argument('--bioid-folder', metavar='BIOID_FOLDER', help='BioID face database folder, to use in the \"test\" mode',
                        type=str, default="../../BioID-FaceDatabase-V1.2")
    return parser.parse_args()


if __name__ == "__main__":
    cli = parsecli()
    main(cli)

