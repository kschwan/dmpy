import numpy as np
import quaternion  # add-on numpy quaternion type (https://github.com/moble/quaternion)

from .canonicalsystem import CanonicalSystem


class QuaternionDMP:
    """

    [1] A. Ude, B. Nemec, T. Petric, and J. Morimoto, "Orientation in Cartesian
    space dynamic movement primitives", in 2014 IEEE International Conference on
    Robotics and Automation (ICRA), 2014, no. 3, pp 2997-3004.
    """

    def __init__(self, n_bfs=10, alpha=48.0, beta=None, cs_alpha=None):
        """

        Parameters
        ----------
        n_bfs : int
            Number of basis functions.
        alpha : float
            Filter constant.
        beta : float
            Filter constant.
        """

        self.n_bfs = n_bfs
        self.alpha = alpha
        self.beta = beta if beta is not None else self.alpha / 4
        self.cs = CanonicalSystem(alpha=cs_alpha if cs_alpha is not None else self.alpha/2)

        # Centres of the Gaussian basis functions
        self.c = np.exp(-self.cs.alpha * np.linspace(0, 1, self.n_bfs))

        # Variance of the Gaussian basis functions
        self.h = 1.0 / np.gradient(self.c)**2

        # Scaling factor
        self.Do = np.identity(3)

        # Initially weights are zero (no forcing term)
        self.w = np.zeros((3, self.n_bfs))

        # Initial- and goal orientations
        self.q0 = quaternion.one
        self.go = quaternion.one

        self.reset()

    def step(self, x, dt, tau):
        def fo(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return self.Do @ (self.w @ psi / psi.sum() * xj)

        # DMP system acceleration
        self.d_omega = (self.alpha * (self.beta * 2 * np.log(self.go * self.q.conjugate()).vec - tau * self.omega) + fo(x)) / tau**2

        # Integrate rotational acceleration
        self.omega += self.d_omega * dt

        # Integrate rotational velocity (to obtain quaternion)
        self.q = np.exp(dt / 2 * np.quaternion(0, *self.omega)) * self.q

        return self.q, self.omega, self.d_omega

    def rollout(self, ts, tau=None):
        self.reset()

        if tau is None:
            tau = ts[-1]

        if np.isscalar(tau):
            tau = np.full_like(ts, tau)

        x = self.cs.rollout(ts, tau)  # Integrate canonical system
        dt = np.gradient(ts) # Differential time vector

        n_steps = len(ts)
        q = np.empty((n_steps,), dtype=np.quaternion)
        omega = np.empty((n_steps, 3))
        d_omega = np.empty((n_steps, 3))

        for i in range(n_steps):
            q[i], omega[i], d_omega[i] = self.step(x[i], dt[i], tau[i])

        return q, omega, d_omega

    def reset(self):
        self.q = self.q0.copy()
        self.omega = np.zeros(3)
        self.d_omega = np.zeros(3)

    def train(self, quaternions, ts, tau):
        # View input as numpy quaternion type
        if quaternions.dtype == np.quaternion:
            quats = quaternions
        else:
            quats = quaternion.as_quat_array(quaternions)

        # Sanity-check input
        if len(quats) != len(ts):
            raise RuntimeError("len(quats) != len(ts)")

        not_normalized = np.abs(np.norm(quats) - 1.0) > quaternion._eps
        if np.any(not_normalized):
            raise RuntimeError("Input contains unnormalized quaternions at indices:\n{}".format(np.nonzero(not_normalized)))

        # Initial- and goal orientations
        self.q0 = quats[0]
        self.go = quats[-1]

        # Differential time vector
        dt = np.gradient(ts)[:,np.newaxis]

        # Scaling factor
        self.Do = np.diag((2 * np.log(self.go * self.q0.conjugate())).vec)
        Do_inv = np.linalg.inv(self.Do)

        # Compute finite difference velocity between orientations
        omega = 2 * np.log(np.roll(quats, -1) * quats.conjugate())  # In unit time
        omega[-1] = omega[-2]  # Last element is no good
        omega = quaternion.as_float_array(omega)[:,1:] / dt  # Scale by dt

        # Alternatively: Fit cubic splines to data and integrate them
        # omega = quaternion.quaternion_time_series.angular_velocity(quats, ts)

        # Compute desired angular accelerations
        d_omega = np.gradient(omega, ts, axis=0)

        # Integrate canonical system at time points
        x = self.cs.rollout(ts, tau)

        # Set up system of equations to solve for weights
        def features(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return xj * psi / psi.sum()

        def forcing(j):
            return Do_inv @ (tau**2 * d_omega[j] - self.alpha * (self.beta * (2 * np.log(self.go * quats[j].conjugate())).vec - tau * omega[j]))

        A = np.stack([features(xj) for xj in x])
        f = np.stack([forcing(j) for j in range(len(ts))])

        # Least squares solution for Aw = f (for each column of f)
        self.w = np.linalg.lstsq(A, f, rcond=None)[0].T

        # Cache variables for later inspection
        self.train_quats = quats
        self.train_omega = omega
        self.train_d_omega = d_omega
