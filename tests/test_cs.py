from __future__ import division, print_function

import numpy as np
from matplotlib import pyplot as plt

import dmpy

if __name__ == '__main__':
    cs = dmpy.CanonicalSystem(alpha=1.0)

    T = 5.0
    N = 400
    ts = np.linspace(0, T, N)

    vartau1 = 1.1 + np.sin(ts * 12)
    vartau2 = np.random.uniform(low=0.3, high=3.0, size=N)
    vartau3 = np.concatenate((np.linspace(0.3, 3.0, N//2), np.linspace(2.0, 0.3, N//2)))

    xs = [
        cs.rollout(ts, vartau1),
        cs.rollout(ts, vartau2),
        cs.rollout(ts, vartau3),
        cs.rollout(ts, 0.5),
        cs.rollout(ts, 1.0),
        cs.rollout(ts, 2.0),
    ]

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(ts, xs[0], label=r'$x\ (\tau_1)$')
    ax1.plot(ts, xs[1], label=r'$x\ (\tau_2)$')
    ax1.plot(ts, xs[2], label=r'$x\ (\tau_3)$')
    ax1.plot(ts, xs[3], label=r'$x\ (\tau=0.5)$')
    ax1.plot(ts, xs[4], label=r'$x\ (\tau=1.0)$')
    ax1.plot(ts, xs[5], label=r'$x\ (\tau=2.0)$')
    ax1.legend()
    ax1.set_ylabel('x')

    ax2.plot(ts, vartau1, label=r'$\tau_1$')
    ax2.plot(ts, vartau2, label=r'$\tau_2$')
    ax2.plot(ts, vartau3, label=r'$\tau_3$')
    ax2.legend()
    ax2.set_ylabel(r'$\tau$')
    ax2.set_xlabel('t')

    plt.show()
