import numpy as np
import matplotlib.pyplot as plt

from goc_mpc.splines import CubicSpline
from goc_mpc.sec_mpc import TimingMPC


def needle_threading_example():
    waypoints = np.array([[0.0, 0.0],
                          [0.0, 1.0]])

    sp = CubicSpline()
    mpc = TimingMPC(waypoints, 1.0, 1.0)
    positions = np.arange(0.0, 0.8, 0.05)

    fig, ax = plt.subplots(1, 1, figsize=(8, 12))

    ax.set_xlim(-0.1, 0.11)
    ax.set_ylim(-1.0, 1.0)

    for pos in positions:
        start_x = np.array([0.1, -1.0 + pos])
        start_v = np.array([0.0, 2.0])

        mpc.solve(start_x, start_v, 1);
        mpc.fill_cubic_spline(sp, start_x, start_v)

        t_vals = np.linspace(sp.begin(), sp.end(), 100)
        positions = sp.eval_multiple(t_vals, 0)
        ax.plot(positions[:, 0], positions[:, 1], label=f'Path for {pos}')

    fig.savefig("./test.png")


if __name__ == "__main__":
    needle_threading_example()
