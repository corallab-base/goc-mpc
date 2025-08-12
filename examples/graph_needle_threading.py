import numpy as np
import matplotlib.pyplot as plt

from goc_mpc.splines import CubicSpline
from goc_mpc.goc_mpc import GraphTimingMPC


def graph_needle_threading_example():
    waypoints = np.array([[0.0, 0.0],
                          [1.0, -0.5],
                          [5.0, 0.5],
                          [6.0, 0.0]])

    # ij means i must come before j
    graph = np.array([[0, 1, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])

    sp = CubicSpline()
    mpc = GraphTimingMPC(waypoints, graph, 1.0, 1.0)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(-0.6, 6.1)
    ax.set_ylim(-1.0, 1.0)

    start_x = np.array([-0.5, 0])
    start_v = np.array([0.0, 0.0])
    mpc.solve(start_x, start_v, 1);

    mpc.fill_cubic_spline(sp, start_x, start_v)

    t_vals = np.linspace(sp.begin(), sp.end(), 100)
    positions = sp.eval_multiple(t_vals, 0)
    ax.plot(positions[:, 0], positions[:, 1], label=f'Path')

    fig.savefig("./graph-test.png")


if __name__ == "__main__":
    graph_needle_threading_example()
