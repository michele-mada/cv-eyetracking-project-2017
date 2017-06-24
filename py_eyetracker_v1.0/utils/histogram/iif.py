from math import exp, sqrt
import numpy as np

from utils.histogram.cl_run_iif import CL_IIF, CL_IIF_BINID


def illumination_invariant_features_hybrid(image, histogram, k=0.1, debug_ax=None):
    width, height, nbins = np.shape(histogram)
    fast_binid = CL_IIF_BINID()
    fast_binid.load_program()
    bp_mtx = fast_binid.compute(image, nbins)

    b_mtx = np.ndarray((1,1, nbins))
    b_mtx[0, 0,:] = np.arange(0, nbins)
    b_mtx = np.tile(b_mtx, (width, height, 1))

    # contruct pixel intensity matrix
    i_mtx = np.repeat(image[:, :, np.newaxis], nbins, axis=2)
    i_mtx *= 255
    i_mtx = k * i_mtx
    i_mtx[i_mtx < k] = k

    # compute illumination invariant features
    X = -((b_mtx - bp_mtx) ** 2) / (2 * (i_mtx ** 2))
    e_mtx = np.exp(X)
    Ip_mtx = e_mtx * histogram
    feature_img = np.sum(Ip_mtx, 2)
    return feature_img


fast_iif = CL_IIF()
fast_iif.load_program()


def illumination_invariant_features_cl(image, histogram, k=0.1, debug_ax=None):
    return fast_iif.compute(image, histogram, k)