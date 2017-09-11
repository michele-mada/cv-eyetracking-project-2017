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
from utils.gui.visualization import draw_routine

from utils.eyecenter.hough import PyHoughEyecenter
from utils.eyecenter.timm.timm_and_barth import TimmAndBarth
from utils.eyecenter.int_proj import GeneralIntegralProjection
from utils.histogram.lsh_equalization import lsh_equalization
from utils.screen_mapping.mappers.fuzzy_mapper import FuzzyMapper

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

    def __init__(self, camera_port=0, algo="hough", equaliz="ah", mapping="quadratic", show_gui=False, loglevel=logging.DEBUG):
        self.setLogger(self.__class__.__name__, loglevel)
        self.show_gui = show_gui
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

        self.detection()  # have the detection window appear

    ## Internal functions

    def setup_algo(self, algoname, equalizername):
        self.algo = algos[algoname]()
        self.algo.equalization = equaliz[equalizername]
        if algoname == "timm":
            self.algo.context.load_program(program_path="cl_kernels/timm_barth_smallpic_kernel.cl")

    def refresh(self):
        self.data_bag_left = []
        self.data_bag_right = []

    def detection(self):
        image_cv2 = self.camera.read()
        picture, face, detect_string, not_eyes = process_frame(image_cv2, self.algo, get_cascade_files())
        if self.show_gui:
            draw_routine(picture, face, not_eyes, "detection", draw_unicorn=False)
            key = cv2.waitKey(1)
        return picture, face, detect_string, not_eyes

    def work_thread(self, duration, wait_before, screen_point):
        self._traffic_man.acquire()
        time.sleep(wait_before)
        self.logger.debug("Acquiring point #%d, %s" % (len(self.screen_points_captured), str(screen_point)))
        self.refresh()
        time_started = time.time()
        while time.time() - time_started < duration:
            picture, face, detect_string, not_eyes = self.detection()
            if face is not None and face.right_eye is not None and face.left_eye is not None:
                self.data_bag_right.append(
                    np.array(face.normalized_right_eye_vector).astype(np.float))
                self.data_bag_left.append(
                    np.array(face.normalized_left_eye_vector).astype(np.float))

        self.observations.append(Observation(screen_point=screen_point,
                                             right_eyevectors=remove_outlier(self.data_bag_right),
                                             left_eyevectors=remove_outlier(self.data_bag_left)))

        self.logger.debug("Acquired point %s (%d data items)" % (str(screen_point), len(remove_outlier(self.data_bag_right))))

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
        self.observations = []

    def evaluate_calibration(self, distance, mapping_method):
        from utils.screen_mapping.calibrator import cal_param_storage_path
        with open(cal_param_storage_path + ".bag", "rb") as fp:
            stored_observations = pickle.load(fp)
            mapper_right = mapping_method()
            mapper_left = mapping_method()
            mapper_right.train_from_data(stored_observations, is_left=False)
            mapper_left.train_from_data(stored_observations, is_left=True)
            mean_errors_left=[]
            mean_errors_right=[]
            screen_pos = []
            for obs in self.observations:
                assert(isinstance(obs, Observation))
                right_vectors = obs.right_eyevectors
                left_vectors = obs.left_eyevectors
                right_eye_screen_pos = mapper_right.map_point(np.mean(right_vectors, axis=0))
                left_eye_screen_pos = mapper_left.map_point(np.mean(left_vectors, axis=0))
                true_screen_point = np.array(obs.screen_point)
                
                estimated_screen_point = np.mean([right_eye_screen_pos, left_eye_screen_pos], axis=0)
                screen_pos.append(estimated_screen_point)
                
                error_left = np.linalg.norm(left_eye_screen_pos-true_screen_point)
                error_right = np.linalg.norm(right_eye_screen_pos-true_screen_point)
                
                mean_errors_left.append(error_left)
                mean_errors_right.append(error_right)
            mean_error_left = np.mean(mean_errors_left)
            mean_error_right = np.mean(mean_errors_right)
           
            mean_error = (mean_error_left+mean_error_right)/2
            mean_angular_error = np.degrees(np.arctan(mean_error/distance))
            print(mapping_method)
            print("mean_angular_error")
            print(mean_angular_error)
            return screen_pos
            
            
        



