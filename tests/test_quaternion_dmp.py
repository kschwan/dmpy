import numpy as np
import quaternion
from matplotlib import pyplot as plt

import dmpy


# Open loop
def test1():
    print("Test 1")

    # Repeat the first experiment in A. Ude et al. "Orientation in Cartesian
    # space dynamic movement primitives"
    tau = 3.5

    dmp = dmpy.QuaternionDMP(25, alpha=48.0, beta=12.0, cs_alpha=2.0)

    dmp.q0 = np.quaternion(0.3717, -0.4993, -0.6162, 0.4825)
    dmp.go = np.quaternion(0.2471, 0.1797, 0.3182, -0.8974)

    ts = np.linspace(0, 2, 1000)
    q, omega, domega = dmp.rollout(ts, tau)

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(ts, quaternion.as_float_array(q))
    ax1.set_xlim([0, 1.4])
    ax1.legend([r'$q_{}$'.format(i) for i in 'wxyz'], loc='upper right')
    ax2.plot(ts, omega)
    ax2.legend([r'$\omega_{}$'.format(i) for i in 'xyz'], loc='upper right')
    ax3.plot(ts, domega)
    ax3.legend([r'$\dot\omega_{}$'.format(i) for i in 'xyz'], loc='upper right')
    plt.suptitle('DMP trajectory')
    plt.show()


# Training
def test2():
    print("Test 2")

    dmp = dmpy.QuaternionDMP(100)
    data = np.load('trajectory.npz')
    quats = quaternion.as_quat_array(data['quaternions'])
    vel = data['vel_angular']

    N = len(quats)
    dt = 1 / 30
    ts = np.linspace(0, dt*N, N)
    ts_train = np.linspace(0, 1, N)

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(ts, quaternion.as_float_array(quats))
    ax1.legend([r'$q_{}$'.format(i) for i in 'wxyz'], loc='upper right')
    ax2.plot(ts, vel)
    ax2.legend([r'$\omega_{}$'.format(i) for i in 'xyz'], loc='upper right')
    ax3.plot(ts, np.gradient(vel, axis=0) / dt)
    ax3.legend([r'$\dot\omega_{}$'.format(i) for i in 'xyz'], loc='upper right')
    fig.suptitle('Input trajectory')

    dmp.train(quats, ts_train, 1.0)

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(ts_train, quaternion.as_float_array(dmp.train_quats))
    ax1.legend([r'$q_{}$'.format(i) for i in 'wxyz'], loc='upper right')
    ax2.plot(ts_train, dmp.train_omega)
    ax2.legend([r'$\omega_{}$'.format(i) for i in 'xyz'], loc='upper right')
    ax3.plot(ts_train, dmp.train_d_omega)
    ax3.legend([r'$\dot\omega_{}$'.format(i) for i in 'xyz'], loc='upper right')
    fig.suptitle('Train trajectory')

    q, omega, domega = dmp.rollout(ts)
    # q, omega, domega = dmp.rollout(ts, ts[N//2])

    # vartau = 40 + 10*np.sin(ts*0.2)
    # plt.figure()
    # plt.plot(ts, vartau)
    # q, omega, domega = dmp.rollout(ts, vartau)

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(ts, quaternion.as_float_array(q))
    ax1.legend([r'$q_{}$'.format(i) for i in 'wxyz'], loc='upper right')
    ax2.plot(ts, omega)
    ax2.legend([r'$\omega_{}$'.format(i) for i in 'xyz'], loc='upper right')
    ax3.plot(ts, domega)
    ax3.legend([r'$\dot\omega_{}$'.format(i) for i in 'xyz'], loc='upper right')
    fig.suptitle('DMP trajectory')

    plt.show()


if __name__ == '__main__':
    test1()
    test2()
