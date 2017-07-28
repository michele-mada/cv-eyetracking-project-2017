
from classes import Point


def oneD_map_function(eye_vector, fixed_params):
    (ex, ey) = eye_vector
    return fixed_params[0] + ex * fixed_params[1] + ey * fixed_params[2] + \
           ex * ey * fixed_params[3] + (ex ** 2) * fixed_params[4] + (ey ** 2) * fixed_params[4]


def map_function(eye_vector, fixed_params_x, fixed_params_y):
    x_screen = oneD_map_function(eye_vector, fixed_params_x)
    y_screen = oneD_map_function(eye_vector, fixed_params_y)
    return Point(x=x_screen, y=y_screen)
