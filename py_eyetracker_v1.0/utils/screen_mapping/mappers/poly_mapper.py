import logging

import numpy as np

from utils.logging import LogMaster
from utils.screen_mapping.mapper_interface import MapperInterface
from classes import Observation, Point
from utils.screen_mapping.mappers.optimizer import compute_params


class PolyMapper(LogMaster, MapperInterface):

    def __init__(self, num_params, loglevel=logging.DEBUG):
        self.setLogger(self.__class__.__name__, loglevel)
        self.num_params = num_params
        self.params_x = np.array((num_params,))
        self.params_y = np.array((num_params,))
        self.logger.debug("Polynomial mapper ready (%d parameters)" % (self.num_params,))

    def oneD_map_function(self, eye_vector, params):
        return 0

    def train_from_data(self, observations, is_left=False):
        screen_points = []
        eye_vectors = []
        for obs in observations:
            assert(isinstance(obs, Observation))
            domain = obs.right_eyevectors
            if is_left:
                domain = obs.left_eyevectors

            eye_vector = np.mean(domain, axis=0)
            screen_points.append(obs.screen_point)
            eye_vectors.append(eye_vector)

        self.params_x, self.params_y = tuple(compute_params(screen_points, eye_vectors,
                                                            self.oneD_map_function, self.num_params))

    def map_point(self, eyevector):
        x_screen = self.oneD_map_function(eyevector, self.params_x)
        y_screen = self.oneD_map_function(eyevector, self.params_y)
        return Point(x=x_screen, y=y_screen)


class PolyQuadMapper(PolyMapper):

    def __init__(self, loglevel=logging.DEBUG):
        super().__init__(6, loglevel)

    def oneD_map_function(self, eye_vector, params):
        (ex, ey) = eye_vector
        return params[0] + ex * params[1] + ey * params[2] + \
               ex * ey * params[3] + (ex ** 2) * params[4] + (ey ** 2) * params[5]


class PolyLinMapper(PolyMapper):

    def __init__(self, loglevel=logging.DEBUG):
        super().__init__(3, loglevel)

    def oneD_map_function(self, eye_vector, params):
        (ex, ey) = eye_vector
        return params[0] + ex * params[1] + ey * params[2]
