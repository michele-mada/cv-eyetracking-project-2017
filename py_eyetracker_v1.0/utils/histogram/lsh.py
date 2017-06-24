from math import exp, sqrt
import numpy as np

from utils.histogram.cl_run_Q import CL_Q
from utils.histogram.cl_run_1D import CL_hist_1D


def locality_sensitive_histogram_hybrid(image, sigma=0.15, num_bins=32, debug_ax=None):
    width, height = np.shape(image)
    alpha_x = exp(-sqrt(2.0) / (sigma * width))
    alpha_y = exp(-sqrt(2.0) / (sigma * height))

    fast_Q = CL_Q()
    fast_Q.load_program()
    q_mtx = fast_Q.compute(image, num_bins)

    #debug_ax.imshow(q_mtx[:,:,11], cmap="gray")

    # compute H and normalization factor
    hist_mtx = np.copy(q_mtx)
    f_mtx = np.ones_like(q_mtx)

    # ------------x dimension
    # compute left part
    hist_mtx_l = np.copy(hist_mtx)
    f_mtx_l = np.copy(f_mtx)
    for i in range(1, height):
        hist_mtx_l[:, i,:] = hist_mtx_l[:, i,:] + alpha_x * hist_mtx_l[:, i - 1,:]
        f_mtx_l[:, i,:] = f_mtx_l[:, i,:] + alpha_x * f_mtx_l[:, i - 1,:]
    # compute right part
    hist_mtx_r = np.copy(hist_mtx)
    f_mtx_r = np.copy(f_mtx)
    for i in reversed(range(-1, height-1)):
        hist_mtx_l[:, i, :] = hist_mtx_l[:, i, :] + alpha_x * hist_mtx_l[:, i + 1, :]
        f_mtx_l[:, i, :] = f_mtx_l[:, i, :] + alpha_x * f_mtx_l[:, i + 1, :]
    # combine right and left parts
    hist_mtx = hist_mtx_r + hist_mtx_l - q_mtx
    f_mtx = f_mtx_r + f_mtx_l - 1

    # ------------y dimension
    # compute left part
    hist_mtx_l = np.copy(hist_mtx)
    f_mtx_l = np.copy(f_mtx)
    for i in range(1, width):
        hist_mtx_l[i,:,:] = hist_mtx_l[i,:,:] + alpha_y * hist_mtx_l[i - 1,:,:]
        f_mtx_l[i,:,:] = f_mtx_l[i,:,:] + alpha_y * f_mtx_l[i - 1,:,:]
    # compute right part
    hist_mtx_r = np.copy(hist_mtx)
    f_mtx_r = np.copy(f_mtx)
    for i in reversed(range(-1, width-1)):
        hist_mtx_l[i,:,:] = hist_mtx_l[i,:,:] + alpha_y * hist_mtx_l[i + 1,:,:]
        f_mtx_l[i,:,:] = f_mtx_l[i,:,:] + alpha_y * f_mtx_l[i + 1,:,:]
    # combine right and left parts
    hist_mtx = hist_mtx_r + hist_mtx_l - q_mtx
    f_mtx = f_mtx_r + f_mtx_l - 1

    # normalize H using normailization factor
    hist_mtx = hist_mtx / f_mtx

    return hist_mtx


fast_Q = CL_Q()
fast_hist_1D_x = CL_hist_1D(direction="x")
fast_hist_1D_y = CL_hist_1D(direction="y")
fast_Q.load_program()
fast_hist_1D_x.load_program()
fast_hist_1D_y.load_program()


def locality_sensitive_histogram_cl(image, sigma=0.15, num_bins=32, debug_ax=None):
    width, height = np.shape(image)

    Q = fast_Q.compute(image, num_bins)

    #TODO: fix this second part of the code

    alpha_x = exp(-sqrt(2) / (sigma * width))
    left_hist_x, right_hist_x, left_norm_x, right_norm_x = fast_hist_1D_y.compute(alpha_x, Q)
    hist_x = left_hist_x + right_hist_x - Q
    norm_x = left_norm_x + right_norm_x - 1.0

    alpha_y = exp(-sqrt(2) / (sigma * height))
    left_hist_y, right_hist_y, left_norm_y, right_norm_y = fast_hist_1D_x.compute(alpha_y, hist_x, init_norm=norm_x)
    hist_y = left_hist_y + right_hist_y - Q
    norm_y = left_norm_y + right_norm_y - 1.0

    hist = hist_y / norm_y

    return hist
