import os
import time
import imageio
import numpy as np
import mujoco as mj
import matplotlib.pyplot as plt

from mujoco import viewer

from goc_mpc.plants import build_1pm_2cube_plant
from goc_mpc.systems import OnePointMassEnv
from goc_mpc.goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC
from goc_mpc.utils.mesh_cat_mirror import MeshCatMirror
from goc_mpc.simple_drake_env import SimpleDrakeGym


def visualize_last_cycle(goc_mpc):

    if goc_mpc.graph.dim != 3:
        return

    fig, axes = plt.subplots(3)
    for i, spline in enumerate(goc_mpc.last_cycle_splines):

        print(i, "spline length", spline.num_pieces())

        begin_time = spline.begin()
        end_time = spline.end()
        times = np.linspace(begin_time, end_time, 200)
        positions = spline.eval_multiple(times, 0)
        axes[0].plot(positions[:, 0], positions[:, 2], label=f"agent {i}")

        velocities = spline.eval_multiple(times, 1)
        accelerations = spline.eval_multiple(times, 2)
        for j in [0, 2]:
            axes[1].plot(times, velocities[:, j], label=f"agent {i} v_{j}")
            axes[2].plot(times, accelerations[:, j], label=f"agent {i} a_{j}")

    axes[1].legend()
    axes[2].legend()

    short_path_points = goc_mpc.last_cycle_short_path[0]

    # visualize short path
    for ag_i in range(len(goc_mpc.last_cycle_splines)):
        axes[0].plot(short_path_points[:, ag_i * 3 + 0],
                     short_path_points[:, ag_i * 3 + 2], color="red")

    fig.show()

    return fig


def main():
    # env and visualization
    env = SimpleDrakeGym(["point_mass_0", "point_mass_1"], ["cube_0", "cube_1", "cube_2"])

    # see if this can be improved
    dim = env.plant.num_positions()
    state_lower_bound = np.ones(dim) * -100.0
    state_upper_bound = np.ones(dim) * 100.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["point_mass_0", "point_mass_1"], ["cube_0", "cube_1", "cube_2"],
                               state_lower_bound, state_upper_bound)

    graph.structure.add_nodes(4)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(2, 3, True)
    graph.structure.add_edge(1, 3, True)

    phi0 = graph.add_robot_above_cube_constraint(0, 0, 0, 0.1);
    graph.add_grasp_change(phi0, "grab", 0, 0);

    phi1 = graph.add_robot_above_cube_constraint(1, 0, 1, 0.2);
    graph.add_grasp_change(phi1, "release", 0, 0);

    phi2 = graph.add_robot_above_cube_constraint(2, 1, 2, 0.1);
    graph.add_grasp_change(phi2, "grab", 1, 2);

    phi3 = graph.add_robot_above_cube_constraint(3, 1, 0, 0.2);
    graph.add_grasp_change(phi3, "release", 1, 2);

    # GoC-MPC
    goc_mpc = GraphOfConstraintsMPC(graph, short_path_time_per_step = 0.1)
                                    # max_vel = 0.1,  # maximum velocity for every joint
                                    # max_acc = 0.1,  # maximum acceleration for every joint
                                    # max_jerk = 0.1) # maximum jerk for every joint

    observed_qs = []

    dt = 1.0 / 30

    # do it again?
    env._diagram.ForcedPublish(env._context)
    input("Continue?")

    while True:
        obs, _ = env.reset()
        goc_mpc.reset()

        for k in range(1000):
            x, x_dot = obs[:graph.total_dim], obs[graph.total_dim:]
            xi_h, _, _ = goc_mpc.step(k * dt, x, x_dot)

            if k % 200 == 0:
                fig = visualize_last_cycle(goc_mpc)
                input("Continue?")
                plt.close(fig)

            qpos = xi_h[0]
            obs, rew, done, trunc, info = env.step(qpos, grasp_cmds=goc_mpc.last_grasp_commands)

        resp = input("Repeat?")
        if resp == 'q':
            break




if __name__ == "__main__":
    main()
