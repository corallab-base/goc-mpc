import numpy as np
import matplotlib.pyplot as plt

from goc_mpc.splines import CubicSpline
from goc_mpc.goc_mpc import GraphOfConstraints, GraphWaypointMPC


def graph_assignment_example():
    num_agents = 2
    dim = 2
    state_lower_bound = np.ones(dim) * -10.0
    state_upper_bound = np.ones(dim) * 10.0

    graph = GraphOfConstraints(num_agents, dim, 
                               state_lower_bound,
                               state_upper_bound)
    graph.structure.add_nodes(4)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(0, 2, True)
    graph.structure.add_edge(1, 3, True)

    A4 = np.eye(4)
    A2 = np.eye(2)
    b0 = np.asarray([1.0, 2.0, 1.0, 0.0])
    b1 = np.asarray([2.0, 2.0])
    b2 = np.asarray([2.0, 0.0])
    b3 = np.asarray([3.0, 2.0])

    graph.add_linear_eq(0, A4, b0)
    graph.add_assignable_linear_eq(1, A2, b1)
    graph.add_assignable_linear_eq(2, A2, b2)
    graph.add_assignable_linear_eq(3, A2, b3)

    mpc = GraphWaypointMPC(graph)

    x = np.array([0.0, 0.0, 0.0, 0.0])
    remaining_vertices = [0, 1, 2, 3]
    wps, assignments = mpc.solve(remaining_vertices, x);

    print(wps)
    print(assignments)



if __name__ == "__main__":
    graph_assignment_example()
