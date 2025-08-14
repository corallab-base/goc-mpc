import numpy as np
import matplotlib.pyplot as plt

from goc_mpc.splines import CubicSpline
from goc_mpc.goc_mpc import GraphWaypointMPC


def graph_assignment_example():
    num_agents = 2
    dim = 2

    # ij means i must come before j
    graph = np.array([[0, 1, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])

    mpc = GraphWaypointMPC(graph, num_agents, dim,
                           np.array([-10.0, -10.0]),
                           np.array([10.0, 10.0]))

    A4 = np.eye(4)
    A2 = np.eye(2)
    b0 = np.asarray([1.0, 2.0, 1.0, 0.0])
    b1 = np.asarray([2.0, 2.0])
    b2 = np.asarray([2.0, 0.0])
    b3 = np.asarray([3.0, 2.0])

    mpc.add_linear_eq(0, A4, b0)
    mpc.add_assignable_linear_eq(1, A2, b1)
    mpc.add_assignable_linear_eq(2, A2, b2)
    mpc.add_assignable_linear_eq(3, A2, b3)

    x = np.array([0.0, 0.0, 0.0, 0.0])

    remaining_vertices = {0, 1, 2, 3}
    wps, assignments = mpc.solve(remaining_vertices, x);

    print(wps)
    print(assignments)



if __name__ == "__main__":
    graph_assignment_example()
