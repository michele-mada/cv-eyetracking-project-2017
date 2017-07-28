from threading import Thread
import time
import cv2
import numpy as np
import pickle

from utils.camera.capture import WebcamVideoStream
from utils.process_frame import process_frame
from utils.screen_mapping.optimizer import compute_params


cal_param_storage_path = "calibration.dat"


def get_cascade_files():
    return {
        "eye": "../haarcascades/haarcascade_righteye_2splits.xml",
        "face": "../haarcascades/haarcascade_frontalface_default.xml"
    }


class Calibrator:

    def __init__(self, camera_port=0, algo="hough"):
        self.screen_points_captured = []
        self.right_eye_vectors_captured = []
        self.left_eye_vectors_captured = []
        self.params_right_eye = None
        self.params_left_eye = None
        self.refresh()
        self.algo = algo
        self._worker = None
        self.camera = WebcamVideoStream(src=camera_port)
        self.camera.start()

    ## Internal functions

    def refresh(self):
        self.data_bag_left = []
        self.data_bag_right = []

    def work_thread(self, duration, screen_point):
        time_started = time.time()
        while time.time() - time_started < duration:
            image_cv2 = self.camera.read()
            image_cv2_gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
            picture, face, detect_string, not_eyes = process_frame(image_cv2_gray, self.algo, get_cascade_files())
            self.data_bag_right.append(np.array(face.right_eye.eye_vector))
            self.data_bag_right.append(np.array(face.left_eye.eye_vector))
        right_eye_vector = np.mean(self.data_bag_right, axis=0)
        left_eye_vector = np.mean(self.data_bag_left, axis=0)

        self.screen_points_captured.append(screen_point)
        self.right_eye_vectors_captured.append(right_eye_vector)
        self.left_eye_vectors_captured.append(left_eye_vector)

    ## Interface

    def stop(self):
        self.camera.stop()

    def capture_point(self, duration, screen_point):
        self.refresh()
        self._worker = Thread(target=self.work_thread,args=(duration, screen_point))
        self._worker.start()

    def compute_mapping_parameters(self):
        self.params_right_eye = tuple(compute_params(self.screen_points_captured, self.right_eye_vectors_captured))
        self.params_left_eye = tuple(compute_params(self.screen_points_captured, self.left_eye_vectors_captured))
        return self.params_right_eye, self.params_left_eye

    def save_mapping_parameters(self):
        if self.params_right_eye is None or self.params_left_eye is None:
            self.compute_mapping_parameters()
        with open(cal_param_storage_path, "wb") as fp:
            pickle.dump((self.params_right_eye, self.params_left_eye), fp)


