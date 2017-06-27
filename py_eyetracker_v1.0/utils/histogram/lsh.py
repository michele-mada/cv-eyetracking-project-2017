from math import exp, sqrt
import numpy as np

from utils.histogram.cl_run_Q import CL_Q
from utils.histogram.cl_run_1D import CL_hist_1D


num_bins = 32


def locality_sensitive_histogram_hybrid(image, sigma=0.15, num_bins=32, debug_ax=None):
    # it has some weird bug (try a solid white image)
    # but it generally works.
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
    for i in reversed(range(0, height-1)):
        hist_mtx_r[:, i, :] = hist_mtx_r[:, i, :] + alpha_x * hist_mtx_r[:, i + 1, :]
        f_mtx_r[:, i, :] = f_mtx_r[:, i, :] + alpha_x * f_mtx_r[:, i + 1, :]
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
    for i in reversed(range(0, width-1)):
        hist_mtx_r[i,:,:] = hist_mtx_r[i,:,:] + alpha_y * hist_mtx_r[i + 1,:,:]
        f_mtx_r[i,:,:] = f_mtx_r[i,:,:] + alpha_y * f_mtx_r[i + 1,:,:]
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


def locality_sensitive_histogram_cl(image, sigma=0.15, debug_ax=None):
    width, height = np.shape(image)

    Q = fast_Q.compute(image, num_bins)

    alpha_x = exp(-sqrt(2) / (sigma * width))
    alpha_y = exp(-sqrt(2) / (sigma * height))
    shapetuple = (width, height, num_bins)
    linearized_Q = Q.reshape((np.size(Q)))
    F_1 = np.ones_like(linearized_Q)

    hist_1, norm_1 = fast_hist_1D_y.compute(linearized_Q, F_1, alpha_y, shapetuple)
    hist_2, norm_2 = fast_hist_1D_x.compute(hist_1, norm_1, alpha_x, shapetuple)

    hist = hist_2 / norm_2

    return hist.reshape(Q.shape)
