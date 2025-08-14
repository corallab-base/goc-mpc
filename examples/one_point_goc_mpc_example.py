import os
import time
import imageio
import numpy as np
import mujoco as mj
import matplotlib.pyplot as plt

from mujoco import viewer

from goc_mpc.systems import OnePointMassEnv
from goc_mpc.goc_mpc import GraphOfConstraintsMPC
from goc_mpc.graphs import Graph
from goc_mpc.utils.mesh_cat_mirror import MeshCatMirror


def one_point_example():
    env = OnePointMassEnv(mode="servo", n_substeps=5)

    mirror = MeshCatMirror(env.model, env.data, bodies=["p1"], radius=0.05)

    # GoC-MPC
    graph = Graph()
    graph.add_nodes(2)
    graph.add_edge(0, 1)

    num_agents = 1
    dim = 3

    goc_mpc = GraphOfConstraintsMPC(graph, num_agents, dim)
    goc_mpc.add_linear_eq(0, np.eye(3), np.array([0.0, 0.0, 2.0]))
    goc_mpc.add_linear_eq(1, np.eye(3), np.array([0.0, 0.0, 3.0]))

    observed_qs = []

    dt = 1.0 / 30
    obs, _ = env.reset(qpos=np.array([0.0, 0.0, 1.0]))
    mirror.push()

    for k in range(60):
        x, x_dot = obs[:3], obs[3:]
        xi_h, _ = goc_mpc.step(k * dt, x, x_dot)

        qpos = xi_h[0]
        obs, rew, done, trunc, info = env.step(qpos)

        # If you want to slow it down to (roughly) real-time:
        time.sleep(dt)
        mirror.push()


if __name__ == "__main__":
    one_point_example()
