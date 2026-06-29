import os
import time
import imageio
import numpy as np
import mujoco as mj
import matplotlib.pyplot as plt

from pydrake.geometry import Meshcat

from goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC, WaypointSolver, WaypointObjective
from goc_mpc.splines import Block
from goc_mpc.simple_drake_env import SimpleDrakeGym



def pointmass_example_setup():
    state_lower_bound = -10.0
    state_upper_bound = 10.0

    robot_spec = [Block.R(3)]

    graph = GraphOfConstraints([robot_spec], [],
                               state_lower_bound, state_upper_bound,
                               robot_names=["point_mass"],
                               object_names=[])

    joint_agent_dim = graph.num_agents * graph.dim;

    graph.structure.add_nodes(3)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(1, 2, True)

    origin = np.array([0.0, 0.0, 0.0])

    goal_position_1 = origin + np.array([0.0, 0.1, 0.0])
    phi0 = graph.add_robots_linear_eq(0, np.eye(joint_agent_dim), goal_position_1)

    goal_position_2 = origin + np.array([0.0, -0.1, 0.0])
    phi1 = graph.add_robots_linear_eq(1, np.eye(joint_agent_dim), goal_position_2)

    home_position_1 = origin + np.array([0.0, 0.0, 0.1])
    phi2 = graph.add_robots_linear_eq(2, np.eye(joint_agent_dim), home_position_1)

    # GoC-MPC
    spline_spec = robot_spec
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec,
                                    # for waypoint solver:
                                    waypoint_solver = WaypointSolver.kGurobi,
                                    waypoint_objective = WaypointObjective.kSquaredDistance,
                                    waypoint_enforce_rigidity = False,
                                    # for timing solver:
                                    time_delta_cutoff = 0.3,
                                    short_path_time_per_step = 0.1,
                                    phi_tolerance = 0.05,
                                    # max_vel = 0.05,  # maximum velocity for every joint
                                    max_acc = 1.00,  # maximum acceleration for every joint
                                    # max_jerk = 0.05 # maximum jerk for every joint
                                    linear_interpolation = True)

    # goc_mpc.reset()

    return graph, goc_mpc


def main():
    meshcat = Meshcat(port=7000)
    env = SimpleDrakeGym(["point_mass"], [], meshcat=meshcat)

    graph, goc_mpc = pointmass_example_setup()

    input("Continue?")

    while True:
        obs, _ = env.reset()

        breakpoint()

        goc_mpc.reset()

        goc_mpc.unpause()

        t = 0.0
        teleport = False

        step = 1
        for k in range(0, 2000, step):
            x, x_dot = obs

            try:
                xi_h, xi_dot_h, times_h = goc_mpc.step(t, x, x_dot, teleport=teleport)
            except RuntimeError as e:
                # print(e)
                # xi_h, _, _ = goc_mpc.last_cycle_short_path
                # if xi_h.shape[0] > 1:
                #     xi_h = xi_h[1:]
                print("RuntimeError, not overwriting xi_h")
                pass
            except TypeError as e:
                # print(e)
                # xi_h, _, _ = goc_mpc.last_cycle_short_path
                # if xi_h.shape[0] > 1:
                #     xi_h = xi_h[1:]
                print("TypeError, not overwriting xi_h")
                pass


            # get target point on xi_h and associated time
            qpos = xi_h[step]
            time_delta = times_h[step]

            # jump to target point
            obs, rew, done, trunc, info = env.step(qpos, grasp_cmds=goc_mpc.last_grasp_commands)

            # advance time by the associated time delta
            t += time_delta

            if len(goc_mpc.remaining_phases) == 0:
                print("Finished!")
                goc_mpc.pause()
                break

            if teleport:
                time.sleep(1.0)
            else:
                time.sleep(0.1)

        resp = input("Repeat?")
        if resp == 'q':
            break


if __name__ == "__main__":
    main()
