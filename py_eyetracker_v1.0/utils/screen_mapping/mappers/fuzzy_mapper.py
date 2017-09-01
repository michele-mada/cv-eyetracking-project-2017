import logging

import numpy as np
from math import sqrt, exp
from itertools import combinations

from utils.logging import LogMaster
from utils.screen_mapping.mapper_interface import MapperInterface
from classes import Observation, Point


class FuzzyMapper(LogMaster, MapperInterface):

    def __init__(self, loglevel=logging.DEBUG):
        self.setLogger(self.__class__.__name__, loglevel)

        self.screen_points = []
        self.eye_vectors = []
        self.neighbourhood = 0.0

        self.logger.debug("Fuzzy mapper ready")

    def train_from_data(self, observations, is_left=False):
        for obs in observations:
            assert(isinstance(obs, Observation))
            domain = obs.right_eyevectors
            if is_left:
                domain = obs.left_eyevectors

            eye_vector = np.mean(domain, axis=0)
            self.screen_points.append(obs.screen_point)
            self.eye_vectors.append(eye_vector)

        pairs = list(combinations(self.eye_vectors, 2))
        self.neighbourhood = max(map(lambda pair: np.linalg.norm(pair[0] - pair[1]), pairs))


    def map_point(self, eyevector: Point):
        top_n = 3
        x_component, y_component = 0.0, 0.0
        def closeness(reference, candidate: Point):
            dist = sqrt((reference[0] - candidate[0]) **2 + (reference[1] - candidate[1]) **2)
            nd = dist / self.neighbourhood
            return exp(-nd)

        weights = list(map(lambda cand: closeness(cand, eyevector), self.eye_vectors))
        top_n_weights = sorted(weights, reverse=True)[:top_n]
        normalization = sum(top_n_weights) + 0.0001
        for index, coordinates, weight in zip(range(len(weights)), self.screen_points, weights):
            if weight in top_n_weights:
                x_component += weight * coordinates[0]
                y_component += weight * coordinates[1]
        x_component /= normalization
        y_component /= normalization

        return Point(x=x_component, y=y_component)