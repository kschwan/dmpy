import numpy as np
from matplotlib import pyplot as plt

import dmpy


if __name__ == '__main__':
    data = np.load('trajectory.npz')
    pos = data['position']
    vel = data['vel_linear']

    dt = 1/30
    N = len(pos)
    ts = np.linspace(0, N*dt, N)
    ts_train = np.linspace(0, 1, N)
    dmp = dmpy.PositionDMP(100)

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(ts, pos)
    ax1.legend([r'$s_{}$'.format(i) for i in 'xyz'], loc='upper right')
    ax2.plot(ts, vel)
    ax2.legend([r'$v_{}$'.format(i) for i in 'xyz'], loc='upper right')
    ax3.plot(ts, np.gradient(vel, axis=0) / dt)
    ax3.legend([r'$a_{}$'.format(i) for i in 'xyz'], loc='upper right')
    fig.suptitle('Input trajectory')

    dmp.train(pos, ts_train, 1.0)

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(ts_train, dmp.train_p)
    ax1.legend([r'$s_{}$'.format(i) for i in 'xyz'], loc='upper right')
    ax2.plot(ts_train, dmp.train_d_p)
    ax2.legend([r'$v_{}$'.format(i) for i in 'xyz'], loc='upper right')
    ax3.plot(ts_train, dmp.train_dd_p)
    ax3.legend([r'$a_{}$'.format(i) for i in 'xyz'], loc='upper right')
    fig.suptitle('Train trajectory')

    p, dp, ddp = dmp.rollout(ts)
    # p, dp, ddp = dmp.rollout(ts, ts[N//2])

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(ts, p)
    ax1.legend([r'$s_{}$'.format(i) for i in 'xyz'], loc='upper right')
    ax2.plot(ts, dp)
    ax2.legend([r'$v_{}$'.format(i) for i in 'xyz'], loc='upper right')
    ax3.plot(ts, ddp)
    ax3.legend([r'$a_{}$'.format(i) for i in 'xyz'], loc='upper right')
    fig.suptitle('DMP trajectory')
    plt.show()
