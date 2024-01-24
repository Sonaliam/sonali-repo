import logging
import numpy as np
import scipy.spatial as sp
from sklearn import metrics


def cossim(f1, f2):
    cos_dist = sp.distance.cdist(f1, f2, 'cosine')  # 0 equal, 2 different
    norm_cos_dist = cos_dist / 2.0  # 0 equal, 1 different
    return 1 - norm_cos_dist  # 1 equal, 0 different



