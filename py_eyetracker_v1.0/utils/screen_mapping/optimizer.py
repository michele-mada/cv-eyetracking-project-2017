import numpy as np
from scipy.optimize import least_squares

from utils.screen_mapping.map_function import oneD_map_function


def compute_error(screen_coordinate, eye_vector, candidate_params):
    computed_coord = oneD_map_function(eye_vector, candidate_params)
    return screen_coordinate - computed_coord


def compute_params(screen_points, eye_vectors):
    x_screen_points, y_screen_points = map(list, zip(*screen_points))

    def error_function_x(params):
        return np.array(map(lambda pair: compute_error(pair[0], pair[1], params),
                            zip(x_screen_points, eye_vectors)))
    def error_function_y(params):
        return np.array(map(lambda pair: compute_error(pair[0], pair[1], params),
                            zip(y_screen_points, eye_vectors)))

    optimization_res_x = least_squares(error_function_x, np.ones((5,)))
    optimization_res_y = least_squares(error_function_y, np.ones((5,)))

    params_x = optimization_res_x.x
    params_y = optimization_res_y.x
    return params_x, params_y
