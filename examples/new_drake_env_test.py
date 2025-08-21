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


def main():
    # env and visualization
    env = SimpleDrakeGym(["point_mass_1"], ["cube_1", "cube_2"])

    # see if this can be improved
    dim = env.plant.num_positions()
    state_lower_bound = np.ones(dim) * -100.0
    state_upper_bound = np.ones(dim) * 100.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["point_mass_1"], ["cube_1", "cube_2"],
                               state_lower_bound, state_upper_bound)

    graph.structure.add_nodes(2)
    graph.structure.add_edge(0, 1, True)

    graph.add_robot_above_cube_constraint(0, "point_mass_1", "cube_1", 0.1);
    graph.add_grasp_change(0, "grab", "point_mass_1", "cube_1");

    graph.add_robot_above_cube_constraint(1, "point_mass_1", "cube_2", 0.2);
    graph.add_grasp_change(1, "release", "point_mass_1", "cube_1");

    # graph.add_linear_eq(0, np.eye(3), np.array([0.0, 1.0, 1.0]))
    # graph.add_linear_eq(1, np.eye(3), np.array([1.0, 2.0, 1.0]))

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

            # if len(goc_mpc.last_grasp_commands) > 0:
            #     breakpoint()

            # if k % 200 == 0:
            #     fig = visualize_last_cycle(goc_mpc)
            #     input("Continue?")
            #     plt.close(fig)

            qpos = xi_h[0]
            obs, rew, done, trunc, info = env.step(qpos, grasp_cmds=goc_mpc.last_grasp_commands)

        resp = input("Repeat?")
        if resp == 'q':
            break




if __name__ == "__main__":
    main()
