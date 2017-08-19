import logging

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from utils.logging import LogMaster
from utils.screen_mapping.mapper_interface import MapperInterface
from classes import Observation, Point


class NeuralMapper(LogMaster, MapperInterface):

    def __init__(self, hidden_layers_size=(8, 3), loglevel=logging.DEBUG):
        self.setLogger(self.__class__.__name__, loglevel)
        self.model = MLPRegressor(solver="lbfgs", activation="relu", max_iter=2000, early_stopping=True,
                                  hidden_layer_sizes=hidden_layers_size)
        self.scaler = StandardScaler()

    def before_training(self, observations):
        X_list = []
        self.logger.debug("Preparing data scaler")
        for obs in observations:
            assert(isinstance(obs, Observation))
            for eyevector in obs.right_eyevectors:
                X_list.append(eyevector)
            for eyevector in obs.left_eyevectors:
                X_list.append(eyevector)
        self.scaler.fit(np.array(X_list))

    def train_from_data(self, observations, is_left=False):
        X_list = []
        y_list = []
        for obs in observations:
            assert(isinstance(obs, Observation))
            domain = obs.right_eyevectors
            if is_left:
                domain = obs.left_eyevectors

            self.logger.debug("Training %s neural model with %d data-points; target=%s" %
                              (("left" if is_left else "right"), len(domain), str(obs.screen_point)))

            for eyevector in domain:
                X_list.append(eyevector)
                y_list.append(obs.screen_point)

        X_train = self.scaler.transform(np.array(X_list))
        #X_train = np.array(X_list)
        self.model.fit(X_train, np.array(y_list))

    def map_point(self, eyevector):
        X_predict = self.scaler.transform(np.array(np.array([eyevector])))
        #X_predict = np.array(np.array([eyevector]))
        return Point(*list(self.model.predict(X_predict)[0]))
