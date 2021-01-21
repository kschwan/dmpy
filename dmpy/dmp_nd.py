import numpy as np

from .canonicalsystem import CanonicalSystem


class NDDMP:
    def __init__(self, n_dims=1, n_bfs=10, alpha=48.0, beta=None, cs_alpha=None):
        self.n_dims = n_dims
        self.n_bfs = n_bfs
        self.alpha = alpha
        self.beta = beta if beta is not None else alpha / 4
        self.cs = CanonicalSystem(alpha=cs_alpha)

        # Centres of the Gaussian basis functions
        self.c = np.exp(-self.cs.alpha * np.linspace(0, 1, n_bfs))

        # Variance of the Gaussian basis functions
        self.h = 1.0 / np.gradient(self.c)**2

        # Scaling factor
        self.Dp = np.identity(n_dims)

        # Initially weights are zero (no forcing term)
        self.w = np.zeros((n_dims, n_bfs))

        # Initial- and goal positions
        self.p0 = np.zeros(n_dims)
        self.gp = np.zeros(n_dims)

        self.reset()

    def step(self, x, dt, tau):
        def fp(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return self.Dp @ (self.w @ psi / psi.sum() * xj)

        # DMP system acceleration
        self.ddp = (self.alpha * (self.beta * (self.gp - self.p) - tau * self.dp) + fp(x)) / tau**2

        # Integrate acceleration to obtain velocity
        self.dp += self.ddp * dt

        # Integrate velocity to obtain position
        self.p += self.dp * dt

        return self.p, self.dp, self.ddp

    def rollout(self, ts, tau=None):
        self.reset()

        if tau is None:
            tau = ts[-1]

        if np.isscalar(tau):
            tau = np.full_like(ts, tau)

        x = self.cs.rollout(ts, tau)  # Integrate canonical system
        dt = np.gradient(ts) # Differential time vector

        n_steps = len(ts)
        p = np.empty((n_steps, self.n_dims))
        dp = np.empty((n_steps, self.n_dims))
        ddp = np.empty((n_steps, self.n_dims))

        for i in range(n_steps):
            p[i], dp[i], ddp[i] = self.step(x[i], dt[i], tau[i])

        return p, dp, ddp

    def reset(self):
        self.p = self.p0.copy()
        self.dp = np.zeros(self.n_dims)
        self.ddp = np.zeros(self.n_dims)

    def train(self, positions, ts, tau):
        p = positions

        # Sanity-check input
        if len(p) != len(ts):
            raise RuntimeError("len(p) != len(ts)")

        # Initial- and goal positions
        self.p0 = p[0]
        self.gp = p[-1]

        # Scaling factor
        self.Dp = np.diag(self.gp - self.p0)
        Dp_inv = np.linalg.inv(self.Dp)

        # Desired velocities and accelerations
        d_p = np.gradient(p, ts, axis=0)
        dd_p = np.gradient(d_p, ts, axis=0)

        # Integrate canonical system
        x = self.cs.rollout(ts, tau)

        # Set up system of equations to solve for weights
        def features(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return xj * psi / psi.sum()

        def forcing(j):
            return Dp_inv @ (tau**2 * dd_p[j] - self.alpha * (self.beta * (self.gp - p[j]) - tau * d_p[j]))

        A = np.stack([features(xj) for xj in x])
        f = np.stack([forcing(j) for j in range(len(ts))])

        # Least squares solution for Aw = f (for each column of f)
        self.w = np.linalg.lstsq(A, f, rcond=None)[0].T

        # Cache variables for later inspection
        self.train_p = p
        self.train_d_p = d_p
        self.train_dd_p = dd_p
