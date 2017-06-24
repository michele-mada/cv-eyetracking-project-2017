

from utils.histogram.lsh import locality_sensitive_histogram_hybrid as locality_sensitive_histogram
from utils.histogram.iif import illumination_invariant_features_cl as illumination_invariant_features


def lsh_equalization(picture_float):
    histogram = locality_sensitive_histogram(picture_float)
    iif = illumination_invariant_features(picture_float, histogram)
    return iif
