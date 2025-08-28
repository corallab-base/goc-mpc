import os
import time
import imageio
import numpy as np
import mujoco as mj
import matplotlib.pyplot as plt

from mujoco import viewer

from goc_mpc.splines import Block
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


def two_pm_block_stacking():

    # env and visualization
    env = SimpleDrakeGym(["point_mass_0", "point_mass_1"], ["cube_0", "cube_1", "cube_2"])

    state_lower_bound = -100.0
    state_upper_bound =  100.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["point_mass_0", "point_mass_1"], ["cube_0", "cube_1", "cube_2"],
                               state_lower_bound, state_upper_bound)

    graph.structure.add_nodes(4)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(2, 3, True)
    graph.structure.add_edge(1, 3, True)

    phi0 = graph.add_robot_above_cube_constraint(0, 0, 0, 0.1);
    graph.add_grasp_change(phi0, "grab", 0, 0);

    graspPhi0 = graph.add_robot_holding_cube_constraint(0, 1, 0, 0, 0.1);

    phi1 = graph.add_robot_above_cube_constraint(1, 0, 1, 0.2);
    graph.add_grasp_change(phi1, "release", 0, 0);

    phi2 = graph.add_robot_above_cube_constraint(2, 1, 2, 0.1);
    graph.add_grasp_change(phi2, "grab", 1, 2);

    graspPhi1 = graph.add_robot_holding_cube_constraint(2, 3, 1, 2, 0.1);

    phi3 = graph.add_robot_above_cube_constraint(3, 1, 0, 0.2);
    graph.add_grasp_change(phi3, "release", 1, 2);

    # GoC-MPC
    spline_spec = [Block.R(3)]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1)
    return env, graph, goc_mpc

def two_gripper_block_stacking():
    # env and visualization
    env = SimpleDrakeGym(["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"])

    state_lower_bound = -100.0
    state_upper_bound =  100.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"],
                               state_lower_bound, state_upper_bound)

    graph.structure.add_nodes(4)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(2, 3, True)
    graph.structure.add_edge(1, 3, True)

    phi0 = graph.add_robot_above_cube_constraint(0, 0, 0, 0.1);
    graph.add_grasp_change(phi0, "grab", 0, 0);

    graspPhi0 = graph.add_robot_holding_cube_constraint(0, 1, 0, 0, 0.1);

    phi1 = graph.add_robot_above_cube_constraint(1, 0, 1, 0.2);
    graph.add_grasp_change(phi1, "release", 0, 0);

    phi2 = graph.add_robot_above_cube_constraint(2, 1, 2, 0.1);
    graph.add_grasp_change(phi2, "grab", 1, 2);

    graspPhi1 = graph.add_robot_holding_cube_constraint(2, 3, 1, 2, 0.1);

    phi3 = graph.add_robot_above_cube_constraint(3, 1, 0, 0.2);
    graph.add_grasp_change(phi3, "release", 1, 2);

    # GoC-MPC
    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1)
    return env, graph, goc_mpc


def two_ur5e_block_stacking():
    # env and visualization
    env = SimpleDrakeGym(["ur5e_0", "ur5e_1"], ["cube_0", "cube_1", "cube_2"])

    state_lower_bound = -100.0
    state_upper_bound =  100.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["ur5e_0", "ur5e_1"], ["cube_0", "cube_1", "cube_2"],
                               state_lower_bound, state_upper_bound)

    graph.structure.add_nodes(4)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(2, 3, True)
    graph.structure.add_edge(1, 3, True)

    phi0 = graph.add_robot_above_cube_constraint(0, 0, 0, 0.1);
    graph.add_grasp_change(phi0, "grab", 0, 0);

    graspPhi0 = graph.add_robot_holding_cube_constraint(0, 1, 0, 0, 0.1);

    phi1 = graph.add_robot_above_cube_constraint(1, 0, 1, 0.2);
    graph.add_grasp_change(phi1, "release", 0, 0);

    phi2 = graph.add_robot_above_cube_constraint(2, 1, 2, 0.1);
    graph.add_grasp_change(phi2, "grab", 1, 2);

    graspPhi1 = graph.add_robot_holding_cube_constraint(2, 3, 1, 2, 0.1);

    phi3 = graph.add_robot_above_cube_constraint(3, 1, 0, 0.2);
    graph.add_grasp_change(phi3, "release", 1, 2);

    # GoC-MPC
    spline_spec = [Block.T(6)]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1)
    return env, graph, goc_mpc

def main():
    env, graph, goc_mpc = two_gripper_block_stacking()
    
    observed_qs = []

    dt = 1.0 / 30

    # do it again?
    env._diagram.ForcedPublish(env._context)
    # input("Continue?")

    breakpoint()

    while True:
        obs, _ = env.reset()
        goc_mpc.reset()

        for k in range(1000):
            x, x_dot = obs
            xi_h, _, _ = goc_mpc.step(k * dt, x, x_dot)
            
            # print("real cube 0 q:", x[6:9])
            # ag0_next_node = goc_mpc.timing_mpc.get_agent_spline_nodes(0)[0]
            # print("agent 0 next spline node:", ag0_next_node)
            # print("agent 0 next goal:", goc_mpc.waypoint_mpc.view_waypoints()[ag0_next_node, 6:9])

            if k == 250:
                # detach grasp to see if backtracking is possible.
                assert "cube_0" in env._grasps
                env.release_grasp("cube_0")

            # if k % 200 == 0:
            #     fig = visualize_last_cycle(goc_mpc)
            #     breakpoint()
            #     input("Continue?")
            #     plt.close(fig)

            qpos = xi_h[0]
            obs, rew, done, trunc, info = env.step(qpos, grasp_cmds=goc_mpc.last_grasp_commands)

        resp = input("Repeat?")
        if resp == 'q':
            break




if __name__ == "__main__":
    main()
