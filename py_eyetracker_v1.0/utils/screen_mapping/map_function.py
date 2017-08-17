
from classes import Point





def oneD_map_function_quadratic(eye_vector, fixed_params):
    (ex, ey) = eye_vector
    return fixed_params[0] + ex * fixed_params[1] + ey * fixed_params[2] + \
           ex * ey * fixed_params[3] + (ex ** 2) * fixed_params[4] + (ey ** 2) * fixed_params[5]


def oneD_map_function_linear(eye_vector, fixed_params):
    (ex, ey) = eye_vector
    return fixed_params[0] + ex * fixed_params[1] + ey * fixed_params[2]


profiles = {
    "quadratic": (oneD_map_function_quadratic, 6),
    "linear":  (oneD_map_function_linear, 3),
}

current_profile = "quadratic"

def oneD_map_function(*args):
    return profiles[current_profile][0](*args)

def calibration_params_vector_length():
    return profiles[current_profile][1]


def map_function(eye_vector, fixed_params_x, fixed_params_y):
    x_screen = oneD_map_function(eye_vector, fixed_params_x)
    y_screen = oneD_map_function(eye_vector, fixed_params_y)
    return Point(x=x_screen, y=y_screen)
