import os
import time
import imageio
import numpy as np
import mujoco as mj
import matplotlib.pyplot as plt

from mujoco import viewer

from goc_mpc.systems import Two2DPointMassEnv
from goc_mpc.goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC
from goc_mpc.utils.mesh_cat_mirror import MeshCatMirror


def visualize_last_cycle(goc_mpc):

    if goc_mpc.graph.dim != 2:
        return

    fig, axes = plt.subplots(3)
    for i, spline in enumerate(goc_mpc.last_cycle_splines):
        begin_time = spline.begin()
        end_time = spline.end()
        times = np.linspace(begin_time, end_time, 200)
        positions = spline.eval_multiple(times, 0)
        axes[0].plot(positions[:, 0], positions[:, 1], label=f"agent {i}")

        velocities = spline.eval_multiple(times, 1)
        accelerations = spline.eval_multiple(times, 2)
        for j in range(velocities.shape[1]):
            axes[1].plot(times, velocities[:, j], label=f"agent {i} v_{j}")
            axes[2].plot(times, accelerations[:, j], label=f"agent {i} a_{j}")

    axes[1].legend()
    axes[2].legend()

    short_path_points = goc_mpc.last_cycle_short_path[0]

    # visualize short path
    for ag_i in range(len(goc_mpc.last_cycle_splines)):
        axes[0].plot(short_path_points[:, ag_i * 2 + 0],
                     short_path_points[:, ag_i * 2 + 1], color="red")

    fig.show()

    return fig

def two_points_example():
    # env and visualization
    env = Two2DPointMassEnv(mode="teleport", n_substeps=5)
    mirror = MeshCatMirror(env.model, env.data, bodies=["p1", "p2"], radius=0.05)

    # problem set-up
    num_agents = 2
    dim = 2

    state_lower_bound = np.ones(dim) * -100.0
    state_upper_bound = np.ones(dim) * 100.0

    graph = GraphOfConstraints(num_agents, dim, state_lower_bound, state_upper_bound)
    graph.structure.add_nodes(4)
    graph.structure.add_edge(0, 2, True)
    graph.structure.add_edge(1, 3, True)

    graph.add_variable()
    graph.add_variable()

    graph.add_assignable_linear_eq(0, 0, np.eye(2), np.array([0.0, 1.0]))
    graph.add_assignable_linear_eq(2, 0, np.eye(2), np.array([0.0, 2.0]))

    graph.add_assignable_linear_eq(1, 1, np.eye(2), np.array([1.0, 1.0]))
    graph.add_assignable_linear_eq(3, 1, np.eye(2), np.array([1.0, 2.0]))

    # GoC-MPC
    goc_mpc = GraphOfConstraintsMPC(graph,
                                    short_path_time_per_step = 0.1)
                                    # max_vel = 0.1,  # maximum velocity for every joint
                                    # max_acc = 0.1,  # maximum acceleration for every joint
                                    # max_jerk = 0.1) # maximum jerk for every joint

    observed_qs = []

    dt = 1.0 / 30

    while True:
        obs, _ = env.reset(qpos=np.array([0.0, 0.0, 1.0, 0.0]))
        goc_mpc.reset()
        mirror.push()
    
        for k in range(1500):
            x, x_dot = obs[:4], obs[4:]
            xi_h, _, _ = goc_mpc.step(k * dt, x, x_dot)
    
            # if k % 100 == 0:
            #     fig = visualize_last_cycle(goc_mpc)
            #     input("Continue?")
            #     plt.close(fig)

            qpos = xi_h[0]
            obs, rew, done, trunc, info = env.step(qpos)
    
            # If you want to slow it down to (roughly) real-time:
            mirror.push()
    

        resp = input("Repeat?")
        if resp == 'q':
            break




if __name__ == "__main__":
    two_points_example()
