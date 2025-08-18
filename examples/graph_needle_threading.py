import numpy as np
import matplotlib.pyplot as plt

from goc_mpc.splines import CubicSpline
from goc_mpc.goc_mpc import GraphOfConstraints, GraphTimingMPC


def graph_needle_threading_example():

    max_vel = -1.0
    max_acc = -1.0
    max_jerk = -1.0

    num_agents = 1
    dim = 2

    state_lower_bound = np.ones(dim) * -100.0
    state_upper_bound = np.ones(dim) * 100.0

    graph = GraphOfConstraints(num_agents, dim, state_lower_bound, state_upper_bound)
    graph.structure.add_nodes(3)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(1, 2, True)
    graph.add_linear_eq(0, np.eye(2), np.array([0.0, 0.0]))
    graph.add_linear_eq(1, np.eye(2), np.array([0.0, 1.0]))
    graph.add_linear_eq(2, np.eye(2), np.array([0.1, 2.0]))

    waypoints = np.array([[0.0, 0.0],
                          [0.0, 1.0],
                          [0.3, 2.0]])
    assignments = [-1, -1, -1]

    splines = [CubicSpline()]
    mpc = GraphTimingMPC(graph, 1.0, 1.0, max_vel, max_acc, max_jerk)

    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    # ax.set_xlim(-0.1, 0.11)
    # ax.set_ylim(-1.0, 2.0)

    positions = np.arange(0.0, 0.8, 0.05)
    for pos in positions:
        start_x = np.array([0.1, -1.0 + pos])
        start_v = np.array([0.0, 0.5])
    
        remaining_phases = list(range(waypoints.shape[0]))
        success = mpc.solve(start_x, start_v, remaining_phases, waypoints, assignments);
        mpc.fill_cubic_splines(splines, start_x, start_v)
    
        sp = splines[0]
        t_vals = np.linspace(sp.begin(), sp.end(), 100)
        positions = sp.eval_multiple(t_vals, 0)
        ax.plot(positions[:, 0], positions[:, 1], label=f'Path')

    fig.savefig("./graph-test.png")


if __name__ == "__main__":
    graph_needle_threading_example()
