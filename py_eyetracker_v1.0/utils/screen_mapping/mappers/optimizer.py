import numpy as np
from scipy.optimize import least_squares


def compute_error(screen_coordinate, eye_vector, candidate_params, mapfun):
    computed_coord = mapfun(eye_vector, candidate_params)
    return screen_coordinate - computed_coord


def compute_params(screen_points, eye_vectors, mapfun, num_params):
    x_screen_points, y_screen_points = map(list, zip(*screen_points))

    def error_function_x(params):
        return np.array(list(map(lambda pair: compute_error(pair[0],
                                                            pair[1],
                                                            params, mapfun),
                                 zip(x_screen_points, eye_vectors))))
    def error_function_y(params):
        return np.array(list(map(lambda pair: compute_error(pair[0],
                                                            pair[1],
                                                            params, mapfun),
                                 zip(y_screen_points, eye_vectors))))

    optimization_res_x = least_squares(error_function_x,
                                       np.ones((num_params,)))
    optimization_res_y = least_squares(error_function_y,
                                       np.ones((num_params,)))

    params_x = optimization_res_x.x
    params_y = optimization_res_y.x
    return params_x, params_y
