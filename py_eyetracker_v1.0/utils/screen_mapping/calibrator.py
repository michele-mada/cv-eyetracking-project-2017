from threading import Thread, Lock
import time
import cv2
import numpy as np
import dill as pickle
import logging
from skimage import exposure

from utils.logging import LogMaster

from classes import Observation
from utils.camera.capture import WebcamVideoStream
from utils.process_frame import process_frame

from utils.eyecenter.hough import PyHoughEyecenter
from utils.eyecenter.timm.timm_and_barth import TimmAndBarth
from utils.eyecenter.int_proj import GeneralIntegralProjection
from utils.histogram.lsh_equalization import lsh_equalization


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


class CaptureCalibrator(LogMaster):

    def __init__(self, camera_port=0, algo="hough", equaliz="ah", mapping="quadratic", loglevel=logging.DEBUG):
        self.setLogger(self.__class__.__name__, loglevel)
        self.screen_points_captured = []
        self.observations = []
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
        self.logger.debug("Acquiring point #%d, %s" % (len(self.screen_points_captured), str(screen_point)))
        self.refresh()
        time_started = time.time()
        while time.time() - time_started < duration:
            image_cv2 = self.camera.read()
            image_cv2_gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
            picture, face, detect_string, not_eyes = process_frame(image_cv2_gray, self.algo, get_cascade_files())
            if face is not None and face.right_eye is not None and face.left_eye is not None:
                self.data_bag_right.append(np.array(face.right_eye.eye_vector).astype(np.float))
                self.data_bag_left.append(np.array(face.left_eye.eye_vector).astype(np.float))

        self.observations.append(Observation(screen_point=screen_point,
                                             right_eyevectors=remove_outlier(self.data_bag_right),
                                             left_eyevectors=remove_outlier(self.data_bag_left)))

        self.logger.debug("Acquired point %s" % (str(screen_point),))

        self.screen_points_captured.append(screen_point)
        self._traffic_man.release()

    ## Interface

    def stop(self):
        self.camera.stop()

    def capture_point(self, duration, wait_before, screen_point):
        self._worker = Thread(target=self.work_thread,args=(duration, wait_before, screen_point))
        self._worker.start()

    def save_mapping_parameters(self):
        with open(cal_param_storage_path + ".bag", "wb") as fp:
            pickle.dump(self.observations, fp)
        self.logger.debug("Observation data (over)written to file %s" % cal_param_storage_path)


