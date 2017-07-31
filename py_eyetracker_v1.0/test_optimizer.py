import numpy as np
from utils.screen_mapping.optimizer import compute_params
from scipy.optimize import least_squares

screen_x = 1366
screen_y = 768

radius = 30
padding = 20

centers = [(radius + padding, screen_y - radius - padding),
           (screen_x/2, screen_y - radius - padding),
           (screen_x - radius - padding, screen_y - radius - padding),
           (radius + padding, screen_y/2),
           (screen_x/2, screen_y/2),
           (screen_x - radius - padding, screen_y/2),
           (radius + padding, radius + padding),
           (screen_x/2, radius + padding),
           (screen_x - radius - padding, radius + padding)]

random_vectors = [np.random.rand(2) for _ in range(len(centers))]


params_x, params_y = compute_params(centers, random_vectors)
print(params_x)
print(params_y)
