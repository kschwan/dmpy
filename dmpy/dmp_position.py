from __future__ import division, print_function

import numpy as np

from dmp_nd import NDDMP
from canonicalsystem import CanonicalSystem


class PositionDMP(NDDMP):
    def __init__(self, n_bfs=10, alpha=48.0, beta=None, cs_alpha=None):
        super(PositionDMP, self).__init__(3, n_bfs, alpha, beta, cs_alpha)
