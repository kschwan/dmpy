from .dmp_nd import NDDMP


class PositionDMP(NDDMP):
    def __init__(self, n_bfs=10, alpha=48.0, beta=None, cs_alpha=None):
        super().__init__(3, n_bfs, alpha, beta, cs_alpha)
