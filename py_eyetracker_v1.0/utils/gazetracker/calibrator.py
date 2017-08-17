from threading import Thread, Lock
import time
import cv2
import numpy as np
import pickle
import logging
from skimage import exposure

from utils.logging import LogMaster

from utils.camera.capture import WebcamVideoStream
from utils.process_frame import process_frame
from utils.screen_mapping.optimizer import compute_params

from utils.eyecenter.hough import PyHoughEyecenter
from utils.eyecenter.timm.timm_and_barth import TimmAndBarth
from utils.eyecenter.int_proj import GeneralIntegralProjection
from utils.histogram.lsh_equalization import lsh_equalization

import utils.screen_mapping.map_function


cal_param_storage_path = "calibration.dat"


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


def get_cascade_files():
    return {
        "eye": "../haarcascades/haarcascade_righteye_2splits.xml",
        "face": "../haarcascades/haarcascade_frontalface_default.xml"
    }


def remove_outlier(fromlist, m=1):
    (x_coords, y_coords) = zip(*list(map(lambda i: i.tolist(), fromlist)))
    mean_x = np.mean(x_coords, axis=0)
    std_x = np.std(x_coords, axis=0)
    mean_y = np.mean(y_coords, axis=0)
    std_y = np.std(y_coords, axis=0)
    def is_inlier(point):
        return abs(point[0] - mean_x) < m * std_x and \
               abs(point[1] - mean_y) < m * std_y
    return list(filter(is_inlier, fromlist))


class Calibrator(LogMaster):

    def __init__(self, camera_port=0, algo="hough", equaliz="ah", mapping="quadratic", loglevel=logging.DEBUG):
        self.setLogger(self.__class__.__name__, loglevel)
        self.screen_points_captured = []
        self.right_eye_vectors_captured = []
        self.left_eye_vectors_captured = []
        self.params_right_eye = None
        self.params_left_eye = None
        self.refresh()
        self.setup_algo(algo, equaliz)
        self.mapping = mapping
        self._worker = None
        self._traffic_man = Lock()
        self.logger.info("Calibrator ready; using algo=%s, equalizer=%s, mapping fun=%s" % (algo, equaliz, mapping))
        self.camera = WebcamVideoStream(src=camera_port)
        self.camera.start()

    ## Internal functions

    def setup_algo(self, algoname, equalizername):
        self.algo = algos[algoname]()
        self.algo.equalization = equaliz[equalizername]
        if algoname == "timm":
            self.algo.context.load_program(program_path="cl_kernels/timm_barth_smallpic_kernel.cl")

    def refresh(self):
        self.data_bag_left = []
        self.data_bag_right = []

    def work_thread(self, duration, wait_before, screen_point):
        self._traffic_man.acquire()
        time.sleep(wait_before)
        self.logger.debug("Aquiring point %s" % str(screen_point))
        self.refresh()
        time_started = time.time()
        while time.time() - time_started < duration:
            image_cv2 = self.camera.read()
            image_cv2_gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
            picture, face, detect_string, not_eyes = process_frame(image_cv2_gray, self.algo, get_cascade_files())
            self.data_bag_right.append(np.array(face.right_eye.eye_vector).astype(np.float))
            self.data_bag_left.append(np.array(face.left_eye.eye_vector).astype(np.float))

        right_eye_vector = np.mean(remove_outlier(self.data_bag_right), axis=0)
        left_eye_vector = np.mean(remove_outlier(self.data_bag_left), axis=0)

        self.logger.debug("Acquired point %s, right eye-vector: %s, left eye-vector: %s" %
                          (str(screen_point), str(right_eye_vector), str(left_eye_vector)))

        self.screen_points_captured.append(screen_point)
        self.right_eye_vectors_captured.append(right_eye_vector)
        self.left_eye_vectors_captured.append(left_eye_vector)
        self._traffic_man.release()

    ## Interface

    def stop(self):
        self.camera.stop()

    def capture_point(self, duration, wait_before, screen_point):
        self._worker = Thread(target=self.work_thread,args=(duration, wait_before, screen_point))
        self._worker.start()

    def compute_mapping_parameters(self):
        utils.screen_mapping.map_function.current_profile = self.mapping
        self.params_right_eye = tuple(compute_params(self.screen_points_captured, self.right_eye_vectors_captured))
        self.params_left_eye = tuple(compute_params(self.screen_points_captured, self.left_eye_vectors_captured))
        return self.params_right_eye, self.params_left_eye

    def save_mapping_parameters(self):
        if self.params_right_eye is None or self.params_left_eye is None:
            self.compute_mapping_parameters()
        with open(cal_param_storage_path + "." + self.mapping, "wb") as fp:
            pickle.dump((self.params_right_eye, self.params_left_eye), fp)
        self.logger.debug("Calibration parameters (over)written to file %s" % cal_param_storage_path)


