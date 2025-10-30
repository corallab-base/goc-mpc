import os
import argparse
import time
import pickle
import datetime
import itertools
import imageio
import numpy as np
import mujoco as mj
import matplotlib.pyplot as plt

from collections import namedtuple

from mujoco import viewer

from pydrake.math import RollPitchYaw
from pydrake.geometry import Meshcat
from pydrake.common.eigen_geometry import Quaternion

from goc_mpc.splines import Block
from goc_mpc.systems import OnePointMassEnv
from goc_mpc.goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC
from goc_mpc.utils.mesh_cat_mirror import MeshCatMirror
from goc_mpc.simple_drake_env import SimpleDrakeGym

from visualize_pose_spline import render_spline_to_html


Disturbance = namedtuple('Disturbance', ['delay', 'func', 'agent'])


def visualize_last_cycle(goc_mpc):
    if goc_mpc.graph.dim != 3 and goc_mpc.graph.dim != 7:
        return

    fig, axes = plt.subplots(3)
    for i, spline in enumerate(goc_mpc.last_cycle_splines):

        breakpoint()
        # render_spline_to_html(spline)
        print(i, "spline length", spline.num_pieces())

        # begin_time = spline.begin()
        # end_time = spline.end()
        # times = np.linspace(begin_time, end_time, 200)
        # positions = spline.eval_multiple(times, 0)
        # axes[0].plot(positions[:, 0], positions[:, 2], label=f"agent {i}")

        # velocities = spline.eval_multiple(times, 1)
        # accelerations = spline.eval_multiple(times, 2)
        # for j in [0, 2]:
        #     axes[1].plot(times, velocities[:, j], label=f"agent {i} v_{j}")
        #     axes[2].plot(times, accelerations[:, j], label=f"agent {i} a_{j}")

    # axes[1].legend()
    # axes[2].legend()

    # short_path_points = goc_mpc.last_cycle_short_path[0]

    # visualize short path
    # for ag_i in range(len(goc_mpc.last_cycle_splines)):
    #     axes[0].plot(short_path_points[:, ag_i * 3 + 0],
    #                  short_path_points[:, ag_i * 3 + 2], color="red")

    # fig.show()

    # return fig


def test_two_gripper_block_stacking(n_points=3, quat=np.array([0.0, -0.70710678,  0.70710678, 0.0]), meshcat=None):
    env = SimpleDrakeGym(["free_body_0", "free_body_1"], [f"cube_{i}" for i in range(n_points)], meshcat=meshcat)

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], [f"cube_{i}" for i in range(n_points)],
                               state_lower_bound, state_upper_bound)

    joint_agent_dim = graph.num_agents * graph.dim;

    # RESET
    start_node = graph.structure.add_node()
    home_position_1 = np.array([-0.30, -0.30, 1.0, *quat,
                                -0.30, 0.30, 1.0, *quat])
    graph.add_robots_linear_eq(start_node, np.eye(joint_agent_dim), home_position_1)

    pick_up1, put_down1, pick_up2, put_down2 = graph.structure.add_nodes(4)
    graph.structure.add_edge(start_node, pick_up1, True)
    graph.structure.add_edge(start_node, pick_up2, True)

    graph.structure.add_edge(pick_up1, put_down1, True)
    graph.structure.add_edge(pick_up2, put_down2, True)
    graph.structure.add_edge(put_down1, put_down2, True)

    phi0 = graph.add_robot_to_point_displacement_constraint(pick_up1, 0, 0, np.array([0.0, 0.0, -0.14]));
    graph.add_robot_quat_linear_eq(pick_up1, 0, np.eye(4), quat)
    graph.add_grasp_change(phi0, "grab", 0, 0);

    graspPhi0 = graph.add_robot_holding_cube_constraint(pick_up1, put_down1, 0, 0, 0.25);

    phi1 = graph.add_robot_to_point_displacement_constraint(put_down1, 0, 1, np.array([0.0, 0.0, -0.20]));
    graph.add_robot_quat_linear_eq(put_down1, 0, np.eye(4), quat)
    graph.add_grasp_change(phi1, "release", 0, 0);

    phi2 = graph.add_robot_to_point_displacement_constraint(pick_up2, 1, 2, np.array([0.0, 0.0, -0.14]));
    graph.add_robot_quat_linear_eq(pick_up2, 1, np.eye(4), quat)
    graph.add_grasp_change(phi2, "grab", 1, 2);

    graspPhi1 = graph.add_robot_holding_cube_constraint(pick_up2, put_down2, 1, 2, 0.25);

    phi3 = graph.add_robot_to_point_displacement_constraint(put_down2, 1, 0, np.array([0.0, 0.0, -0.20]));
    graph.add_robot_quat_linear_eq(put_down2, 1, np.eye(4), quat)
    graph.add_grasp_change(phi3, "release", 1, 2);

    left_safe_pos_node = graph.structure.add_node()
    graph.structure.add_edge(put_down1, left_safe_pos_node, True)

    # also, don't go to release the second block until the left safe
    # position has been reached
    # graph.structure.add_edge(left_safe_pos_node, release2_approach, True)

    left_safe_position = np.array([-0.30, -0.30, 1.0])
    graph.add_robot_pos_linear_eq(left_safe_pos_node, 0, np.eye(3), left_safe_position);

    right_safe_pos_node = graph.structure.add_node()
    graph.structure.add_edge(put_down2, right_safe_pos_node, True)

    right_safe_position = np.array([-0.30, 0.30, 1.0])
    graph.add_robot_pos_linear_eq(right_safe_pos_node, 1, np.eye(3), right_safe_position);

    # # GoC-MPC
    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = False,
                                    time_delta_cutoff = 0.0,
                                    phi_tolerance = 0.03)
    return env, graph, goc_mpc

def n_gripper_n_block_stacking(n_grippers=3, n_blocks=5, quat=np.array([0.0, 0.0, 1.0, 0.0])):
    agents = [f"free_body_{j}" for j in range(n_grippers)]
    objects = [f"cube_{i}" for i in range(n_blocks)]
    env = SimpleDrakeGym(agents, objects)

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, agents, objects,
                               state_lower_bound, state_upper_bound)

    joint_agent_dim = graph.num_agents * graph.dim;

    def add_pick_and_place(agent, block_to_pick, destination):
        pick_up, place = graph.structure.add_nodes(2)
        graph.structure.add_edge(pick_up, place, True)

        pick_phi = graph.add_robot_to_point_displacement_constraint(pick_up, agent, block_to_pick, np.array([0.0, 0.0, -0.14]))
        graph.add_robot_quat_linear_eq(pick_up, agent, np.eye(4), quat)
        graph.add_grasp_change(pick_phi, "grab", agent, block_to_pick);

        graph.add_robot_holding_cube_constraint(pick_up, place, agent, block_to_pick, 0.25) # distance threshold

        place_phi = graph.add_robot_to_point_displacement_constraint(place, agent, destination, np.array([0.0, 0.0, -0.20]));
        graph.add_robot_quat_linear_eq(place, agent, np.eye(4), quat)
        graph.add_grasp_change(place_phi, "release", agent, block_to_pick);

        return pick_up, place

    agent_cycle = itertools.cycle(range(n_grippers))
    agent_prevs = {}

    block = 0
    while block < n_blocks - 1:
        agent = next(agent_cycle)
        pick, place = add_pick_and_place(agent, block+1, block)

        if agent in agent_prevs:
            prev, _ = agent_prevs[agent]
            graph.structure.add_edge(prev, pick, True)

        agent_prevs[agent] = place, block+1

        block += 1

    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = False,
                                    time_delta_cutoff = 0.0,
                                    phi_tolerance = 0.03)
    return env, graph, goc_mpc


def n_gripper_n_block_sequence_stacking(n_grippers=3, n_blocks=5, quat=np.array([0.0, 0.0, 1.0, 0.0])):
    agents = [f"free_body_{j}" for j in range(n_grippers)]
    objects = [f"cube_{i}" for i in range(n_blocks)]
    env = SimpleDrakeGym(agents, objects)

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, agents, objects,
                               state_lower_bound, state_upper_bound)

    joint_agent_dim = graph.num_agents * graph.dim;

    # def add_pick_and_place(agent, block_to_pick, destination):
    #     pick_up, place = graph.structure.add_nodes(2)
    #     graph.structure.add_edge(pick_up, place, True)

    #     pick_phi = graph.add_robot_to_point_displacement_constraint(pick_up, agent, block_to_pick, np.array([0.0, 0.0, -0.14]))
    #     graph.add_robot_quat_linear_eq(pick_up, agent, np.eye(4), quat)
    #     graph.add_grasp_change(pick_phi, "grab", agent, block_to_pick);

    #     graph.add_robot_holding_cube_constraint(pick_up, place, agent, block_to_pick, 0.25) # distance threshold

    #     place_phi = graph.add_robot_to_point_displacement_constraint(place, agent, destination, np.array([0.0, 0.0, -0.20]));
    #     graph.add_robot_quat_linear_eq(place, agent, np.eye(4), quat)
    #     graph.add_grasp_change(place_phi, "release", agent, block_to_pick);

    #     return pick_up, place

    agent_to_block = {
        0: 0,
        1: 2
    }

    block_to_destination = {
        0: 1,
        2: 0
    }

    held_blocks = {}
    block_to_agent_and_destination = {}

    pick_up = graph.structure.add_node()
    for agent, block in agent_to_block.items():
        pick_phi = graph.add_robot_to_point_displacement_constraint(pick_up, agent, block, np.array([0.0, 0.0, -0.14]))
        graph.add_robot_quat_linear_eq(pick_up, agent, np.eye(4), quat)
        graph.add_grasp_change(pick_phi, "grab", agent, block);
        held_blocks[agent] = block
        block_to_agent_and_destination[block] = (agent, block_to_destination[block])

    prev = pick_up
    for pick_block, destination in block_to_destination.items():
        place = graph.structure.add_node()
        graph.structure.add_edge(prev, place, True)

        for holding_agent, held_block in held_blocks.items():
            graph.add_robot_holding_cube_constraint(prev, place, holding_agent, held_block, 0.25) # distance threshold

        agent, _ = block_to_agent_and_destination[pick_block]
        place_phi = graph.add_robot_to_point_displacement_constraint(place, agent, destination, np.array([0.0, 0.0, -0.20]));
        graph.add_robot_quat_linear_eq(place, agent, np.eye(4), quat)
        graph.add_grasp_change(place_phi, "release", agent, pick_block);
        del held_blocks[agent]

    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = False,
                                    time_delta_cutoff = 0.0,
                                    phi_tolerance = 0.03)
    return env, graph, goc_mpc


def two_gripper_block_stacking():
    env = SimpleDrakeGym(["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"])

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"],
                               state_lower_bound, state_upper_bound)

    joint_agent_dim = graph.num_agents * graph.dim;

    graph.structure.add_nodes(5)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(2, 3, True)
    graph.structure.add_edge(1, 3, True)
    graph.structure.add_edge(1, 4, True)

    phi0 = graph.add_robot_to_point_displacement_constraint(0, 0, 0, np.array([0.0, 0.0, -0.1]));
    graph.add_robot_quat_linear_eq(0, 0, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
    graph.add_grasp_change(phi0, "grab", 0, 0);

    graspPhi0 = graph.add_robot_holding_cube_constraint(0, 1, 0, 0, 0.1);

    phi1 = graph.add_robot_to_point_displacement_constraint(1, 0, 1, np.array([0.0, 0.0, -0.2]));
    graph.add_robot_quat_linear_eq(1, 0, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
    graph.add_grasp_change(phi1, "release", 0, 0);

    phi2 = graph.add_robot_to_point_displacement_constraint(2, 1, 2, np.array([0.0, 0.0, -0.1]));
    graph.add_robot_quat_linear_eq(2, 1, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
    graph.add_grasp_change(phi2, "grab", 1, 2);

    graspPhi1 = graph.add_robot_holding_cube_constraint(2, 3, 1, 2, 0.1);

    phi3 = graph.add_robot_to_point_displacement_constraint(3, 1, 0, np.array([0.0, 0.0, -0.2]));
    graph.add_robot_quat_linear_eq(3, 1, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
    graph.add_grasp_change(phi3, "release", 1, 2);

    phi4 = graph.add_robot_to_point_displacement_constraint(4, 0, 1, np.array([0.0, 0.0, -0.5]));
    graph.add_robot_quat_linear_eq(4, 0, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))

    # GoC-MPC
    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = False,
                                    time_delta_cutoff = 0.0,
                                    phi_tolerance = 0.03)
    return env, graph, goc_mpc


def two_gripper_pick_and_pour():
    env = SimpleDrakeGym(["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"])

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"],
                               state_lower_bound, state_upper_bound)


    # RESET
    start_node = graph.structure.add_node()
    joint_agent_dim = graph.num_agents * graph.dim;
    home_position_1 = np.array([0.30, -0.2, 0.5, 0.0, -0.707071, -0.707071, 0.0,
                                1.50, -0.2, 0.5, 0.0, 0.707071, -0.707071, 0.0])
    graph.add_robots_linear_eq(0, np.eye(joint_agent_dim), home_position_1)

    # PITCHER
    pitcher_approach, pitcher_pick_up = graph.structure.add_nodes(2)
    graph.structure.add_edge(start_node, pitcher_approach, True)
    graph.structure.add_edge(pitcher_approach, pitcher_pick_up, True)

    graph.add_robot_to_point_alignment_cost(pitcher_approach,
                                            0, 0, np.array([0.0, 1.0, 1.0]),
                                            u_body_opt=np.array([1.0, 0.0, 0.0]),
                                            roll_ref_flat=True,
                                            w_flat=1.0)
    graph.add_robot_to_point_displacement_constraint(pitcher_approach, 0, 0, np.array([-0.20, 0.00, -0.05]));

    graph.add_robot_relative_rotation_constraint(pitcher_approach, pitcher_pick_up, 0, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());
    phi1 = graph.add_robot_to_point_displacement_constraint(pitcher_pick_up, 0, 0, np.array([-0.15, 0.00, -0.04]));
    graph.add_grasp_change(phi1, "grab", 0, 0);

    # CUP
    cup_approach, cup_pick_up = graph.structure.add_nodes(2)
    graph.structure.add_edge(start_node, cup_approach, True)
    graph.structure.add_edge(cup_approach, cup_pick_up, True)

    graph.add_robot_to_point_alignment_cost(cup_approach,
                                            1, 1, np.array([0.0, 0.0, 1.0]),
                                            u_body_opt=np.array([1.0, 0.0, 0.0]),
                                            roll_ref_flat=True,
                                            w_flat=1.0)
    graph.add_robot_to_point_displacement_cost(cup_approach, 1, 1, np.array([0.25, 0.0, -0.05]))

    graph.add_robot_relative_rotation_constraint(cup_approach, cup_pick_up, 1, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());
    phi2 = graph.add_robot_to_point_displacement_constraint(cup_pick_up, 1, 1, np.array([0.15, 0.00, -0.08]));
    graph.add_grasp_change(phi2, "grab", 1, 1);

    # BRING PITCHER AND CUP CLOSE TO EACH OTHER
    bring_close = graph.structure.add_node()
    graph.structure.add_edge(pitcher_pick_up, bring_close, True)
    graph.structure.add_edge(cup_pick_up, bring_close, True)

    graph.add_robot_holding_cube_constraint(pitcher_pick_up, bring_close, 0, 0, 0.25);
    graph.add_robot_holding_cube_constraint(cup_pick_up, bring_close, 1, 1, 0.25);

    graph.add_robot_relative_rotation_constraint(pitcher_pick_up, bring_close, 0, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());
    graph.add_robot_relative_rotation_constraint(cup_pick_up, bring_close, 1, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());

    graph.add_point_to_point_displacement_cost(bring_close, 0, 1, np.array([-0.1, 0.0, -0.12]));
    graph.add_point_linear_eq(bring_close, 0, np.array([[0.0, 0.0, 0.0],
                                                        [0.0, 0.0, 0.0],
                                                        [0.0, 0.0, 1.0]]), np.array([0.0, 0.0, 0.25]))

    # # OR OVER ANOTHER POINT IF WANTED:
    # # graph.add_point_to_point_displacement_cost(bring_close, 1, 2, np.array([0.0, -0.08, -0.20]));

    # POUR
    pour = graph.structure.add_node()
    graph.structure.add_edge(bring_close, pour, True)
    graph.add_robot_holding_cube_constraint(bring_close, pour, 0, 0, 0.25);
    graph.add_robot_holding_cube_constraint(bring_close, pour, 1, 1, 0.25);
    graph.add_robot_relative_displacement_constraint(bring_close, pour, 1, np.array([0.0, 0.0, 0.0]));
    graph.add_point_to_point_displacement_cost(pour, 0, 1, np.array([-0.01, 0.0, -0.1]));
    graph.add_robot_relative_rotation_constraint(bring_close, pour, 0,
                                                 RollPitchYaw(-np.pi/3, 0.0, 0.0).ToQuaternion());
    graph.make_node_unpassable(pour)


    # # PICK UP PITCHER AT ANGLE FROM SIDE
    # # graph.add_robot_to_point_alignment_cost(1, 0, 0, np.array([0.0, 0.0, 1.0])
    # #                                         # u_body_opt=np.array([1.0, 0.0, 0.0]),
    # #                                         # roll_ref_flat=True
    # #                                         )
    # # graph.add_robot_quat_linear_eq(1, 0, 0, np.array([0.05, 0.0, -0.05]));
    # graph.add_robot_relative_rotation_constraint(0, 1, 0, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());
    # phi2 = graph.add_robot_to_point_displacement_constraint(1, 0, 0, np.array([0.05, 0.0, -0.05]));
    # graph.add_grasp_change(phi2, "grab", 0, 0);

    # graspPhi0 = graph.add_robot_holding_cube_constraint(1, 3, 0, 0, 0.1);
    # graph.add_robot_relative_rotation_constraint(1, 3, 0, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());

    # # PICK UP CUP AT ANGLE FROM SIDE
    # # graph.add_robot_to_point_alignment_cost(2, 1, 2, np.array([0.0, 0.0, 1.0])
    # #                                         # u_body_opt=np.array([1.0, 0.0, 0.0]),
    # #                                         # roll_ref_flat=True
    # #                                         )
    # graph.add_robot_relative_rotation_constraint(0, 2, 1, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());
    # phi4 = graph.add_robot_to_point_displacement_cost(2, 1, 2, np.array([-0.05, 0.0, -0.05]));
    # graph.add_grasp_change(phi4, "grab", 1, 2);

    # graspPhi1 = graph.add_robot_holding_cube_constraint(2, 3, 1, 2, 0.1);
    # graph.add_robot_relative_rotation_constraint(2, 3, 1, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());

    # # BRING PITCHER AND CUP CLOSE TO EACH OTHER
    # graph.add_robot_to_point_displacement_cost(1, 0, 0, np.array([0.05, 0.0, -0.05]));
    # graph.add_robot_to_point_displacement_cost(2, 1, 2, np.array([-0.05, 0.0, -0.05]));
    # graph.add_point_to_point_displacement_cost(3, 0, 2, np.array([0.1, 0.0, -0.1]));
    # graph.add_point_linear_eq(3, 0, np.array([[0.0, 0.0, 0.0],
    #                                           [0.0, 0.0, 0.0],
    #                                           [0.0, 0.0, 1.0]]), np.array([0.0, 0.0, 0.3]))


    # # POUR PITCHER
    # graph.add_robot_holding_cube_constraint(3, 4, 0, 0, 0.1);
    # graph.add_robot_holding_cube_constraint(3, 4, 1, 2, 0.1);
    # graph.add_robot_relative_displacement_constraint(3, 4, 1, np.array([0.0, 0.0, 0.0]));
    # graph.add_point_to_point_displacement_cost(4, 0, 2, np.array([0.1, 0.0, -0.1]));
    # graph.add_robot_relative_rotation_constraint(3, 4, 0,
    #                                              RollPitchYaw(np.pi/3, 0.0, 0.0).ToQuaternion());
    # graph.make_node_unpassable(4)

    # GoC-MPC
    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = False,
                                    time_delta_cutoff = 0.0001,
                                    phi_tolerance = 0.03)
    return env, graph, goc_mpc


def two_gripper_fold_sheet():
    env = SimpleDrakeGym(["free_body_0", "free_body_1"], ["cube_0", "cube_1"])

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], ["cube_0", "cube_1"],
                               state_lower_bound, state_upper_bound)

    # RESET
    # start_node = graph.structure.add_node()
    joint_agent_dim = graph.num_agents * graph.dim;
    # home_position_1 = np.array([0.30, -0.2, 0.5, 0.0, -0.70701, -0.70701, 0.0,
    #                             -0.30, -0.2, 0.5, 0.0, 0.70701, -0.70701, 0.0])
    # graph.add_robots_linear_eq(0, np.eye(joint_agent_dim), home_position_1)

    # GRASP TWO CORNERS
    corner_approach, corner_pick_up = graph.structure.add_nodes(2)
    # graph.structure.add_edge(start_node, corner_approach, True)
    graph.structure.add_edge(corner_approach, corner_pick_up, True)

    # graph.add_robot_relative_rotation_constraint(start_node, corner_approach, 0, RollPitchYaw(0.0, 3*np.pi/8, 0.0).ToQuaternion())
    graph.add_robot_to_point_displacement_constraint(corner_approach, 0, 0, np.array([0.10, 0.15, -0.10]))

    # graph.add_robot_relative_rotation_constraint(start_node, corner_approach, 1, RollPitchYaw(0.0, -3*np.pi/8, 0.0).ToQuaternion())
    graph.add_robot_to_point_displacement_cost(corner_approach, 1, 1, np.array([-0.10, 0.15, -0.06]))

    graph.add_robot_relative_displacement_constraint(corner_approach, corner_pick_up, 0, np.array([0.0, 0.25, 0.05]))
    graph.add_robot_relative_displacement_constraint(corner_approach, corner_pick_up, 1, np.array([0.0, 0.25, 0.05]))

    trivial_phi1 = graph.add_robots_linear_eq(corner_pick_up, np.zeros((1, joint_agent_dim)), np.zeros((1,)))
    trivial_phi2 = graph.add_robots_linear_eq(corner_pick_up, np.zeros((1, joint_agent_dim)), np.zeros((1,)))

    graph.add_grasp_change(trivial_phi1, "grab", 0, 0);
    graph.add_grasp_change(trivial_phi2, "grab", 1, 1);

    # GoC-MPC
    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = False,
                                    time_delta_cutoff = 0.0,
                                    phi_tolerance = 0.03)
    return env, graph, goc_mpc


def two_gripper_assignable_move():
    env = SimpleDrakeGym(["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"])

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"],
                               state_lower_bound, state_upper_bound)

    graph.structure.add_nodes(2)
    graph.structure.add_edge(0, 1, True)

    r1 = graph.add_variable();
    cube = 1

    phi0 = graph.add_assignable_robot_to_point_displacement_constraint(0, r1, cube, np.array([0.0, 0.0, -0.1]))
    graph.add_assignable_robot_quat_linear_eq(0, r1, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
    graph.add_assignable_grasp_change(phi0, "grab", cube)

    graspPhi0 = graph.add_assignable_robot_holding_point_constraint(0, 1, r1, cube, 0.1)

    graph.add_assignable_robot_quat_linear_eq(1, r1, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
    graph.add_point_linear_eq(1, cube, np.eye(3), np.array([0.0, 0.0, 0.1]))

    # GoC-MPC
    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = True,
                                    time_delta_cutoff = 0.0,
                                    phi_tolerance = 0.03)
    return env, graph, goc_mpc


# def two_gripper_assignable_block_stacking():
#     env = SimpleDrakeGym(["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"])

#     state_lower_bound = -10.0
#     state_upper_bound =  10.0

#     symbolic_plant = env.plant.ToSymbolic()
#     graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"],
#                                state_lower_bound, state_upper_bound)

#     graph.structure.add_nodes(2)
#     graph.structure.add_edge(0, 1, True)

#     r1 = graph.add_variable();
#     # r2 = graph.add_variable();

#     phi0 = graph.add_assignable_robot_to_point_displacement_constraint(0, r1, 0, np.array([0.0, 0.0, -0.1]));
#     phi1 = graph.add_agent_quat_linear_eq(0, 0, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
#     graph.add_assignable_grasp_change(phi0, "grab", 0);

#     graspPhi0 = graph.add_robot_holding_cube_constraint(0, 1, 0, 0, 0.1);
#     # graspPhi0 = graph.add_assignable_robot_holding_point_constraint(0, 1, r1, 0, 0.1);

#     phi2 = graph.add_agent_quat_linear_eq(1, 0, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
#     phi3 = graph.add_point_to_point_displacement_constraint(1, 0, 1, np.array([0.0, 0.0, -0.1]), tol=0.001);
#     # graph.add_assignable_grasp_change(phi1, "release", 0);
#     graph.make_node_unpassable(1)

#     # phi2 = graph.add_assignable_robot_to_point_displacement_constraint(2, r2, 2, np.array([0.0, 0.0, -0.1]));
#     # graph.add_assignable_grasp_change(phi2, "grab", 2);

#     # graspPhi1 = graph.add_robot_holding_cube_constraint(2, 3, 1, 2, 0.1);

#     # phi3 = graph.add_assignable_robot_to_point_displacement_constraint(3, r2, 0, np.array([0.0, 0.0, -0.2]));
#     # graph.add_assignable_grasp_change(phi3, "release", 2);

#     # phi4 = graph.add_assignable_robot_to_point_displacement_constraint(4, r1, 1, np.array([0.0, 0.0, -0.5]));

#     # GoC-MPC
#     spline_spec = [Block.R(3), Block.SO3()]
#     goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
#                                     time_delta_cutoff = 0.001,)
#     return env, graph, goc_mpc


def two_gripper_rotate_in_place():
    env = SimpleDrakeGym(["free_body_0", "free_body_1"], ["cube_0"])

    state_lower_bound = -10.0
    tate_upper_bound =  10.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], ["cube_0"],
                               state_lower_bound, state_upper_bound)
    graph.structure.add_nodes(3)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(1, 2, True)

    # PICK UP CUBE
    phi0 = graph.add_robot_to_point_displacement_constraint(0, 0, 0, np.array([0.0, 0.0, -0.1]));
    graph.add_robot_quat_linear_eq(0, 0, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]));
    graph.add_grasp_change(phi0, "grab", 0, 0);

    graspPhi0 = graph.add_robot_holding_cube_constraint(0, 1, 0, 0, 0.1);

    # MOVE CUBE TO LOCATION
    phi2 = graph.add_point_linear_eq(1, 0, np.eye(3), np.array([0.0, 0.0, 0.3]))
    graph.add_robot_relative_rotation_constraint(0, 1, 0, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());

    graspPhi1 = graph.add_robot_holding_cube_constraint(1, 2, 0, 0, 0.1);

    # ROTATE WITHOUT MOVING POINT
    phi3 = graph.add_point_linear_eq(2, 0, np.eye(3), np.array([0.0, 0.0, 0.3]))
    phi4 = graph.add_robot_relative_rotation_constraint(1, 2, 0,
                                                        RollPitchYaw(-np.pi/4, 0.0, 0.0).ToQuaternion());

    # GoC-MPC
    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = False,
                                    time_delta_cutoff = 0.05,
                                    phi_tolerance = 0.05)
    return env, graph, goc_mpc


def two_gripper_rotation_test():
    meshcat = Meshcat(port=7002)

    # env and visualization
    env = SimpleDrakeGym(["free_body_0", "free_body_1"], [], meshcat=meshcat)

    state_lower_bound = -10.0
    tate_upper_bound =  10.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], [],
                               state_lower_bound, state_upper_bound)

    graph.structure.add_nodes(2)
    graph.structure.add_edge(0, 1, True)

    goal_position_1 = np.array([0.5, 0.1, 0.5, 0.707, 0.0, 0.707, 0.0,
                                1.5, 0.1, 0.5, 0.707, 0.0, 0.707, 0.0])
    goal_position_2 = np.array([0.5, 0.4, 0.8, 0.500, 0.5, 0.500, 0.5,
                                1.5, 0.4, 0.8, 0.500, 0.5, 0.500, 0.5])

    joint_agent_dim = graph.num_agents * graph.dim;
    phi0 = graph.add_robots_linear_eq(0, np.eye(joint_agent_dim), goal_position_1);
    phi0 = graph.add_robots_linear_eq(1, np.eye(joint_agent_dim), goal_position_2);

    # GoC-MPC
    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1)
    return env, graph, goc_mpc



def int_pair(arg_string):
    """
    Custom type function to parse a string into a pair of integers.
    Expects input in the format "int1,int2".
    """
    try:
        parts = arg_string.split(',')
        if len(parts) != 2:
            raise ValueError("Input must contain exactly two integers separated by a comma.")
        return int(parts[0]), int(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid integer pair format: {e}")




def block_disturbance(env, agent):
    # apply disturbance if the agent is grasping something (it should be grasping block 0)
    is_agent_grasping = any(filter(lambda g: g.robot_name == env._controlled_names[agent], env._grasps.values()))
    apply_disturbance = is_agent_grasping

    point = "cube_0"

    # disturbance sequence
    pos = env._get_model_q(point)

    new_pos = pos + np.array([-0.25, 0.5, 0.0])

    def disturbance(counter):
        if apply_disturbance:
            if counter == 0:
                env.release_grasp(point)
            elif counter == 1:
                env.release_grasp(point)
                env._set_model_q(point, new_pos)

    counter = 0
    while True:
        yield disturbance(counter)
        counter += 1


def pick_and_pour_disturbance(env, agent):
    # apply disturbance if any end-effector is grasping
    apply_disturbance = env.is_grasping(i=agent)

    raise NotImplementedError()

    cup = env.og_env.scene.object_registry("name", "block_3")

    # disturbance sequence
    pos0, orn0 = block.get_position_orientation()
    pose0 = np.concatenate([pos0, orn0])
    pose1 = np.array([-0.1, -0.3, pos0[2], *orn0])

    control_points = np.array([pose0, pose1])
    pose_seq = spline_interpolate_poses(control_points, num_steps=25)
    def disturbance(counter):
        if apply_disturbance:
            if counter < 20:
                if counter > 15:
                    # 15 - 20
                    print("last 5")
                    env.robots[agent].release_grasp_immediately()  # force robot to release the block
                else:
                    print("first 15")
                    pass  # do nothing for the first 15 steps
            elif counter < len(pose_seq) + 20:
                # 20 - pose_seq+20
                env.robots[agent].release_grasp_immediately()  # force robot to release the block
                pose = pose_seq[counter - 20]
                pos, orn = pose[:3], pose[3:]
                print("on traj: ", pos, orn)
                block.set_position_orientation(pos, orn)
                counter += 1
        else:
            print("NOT APPLYING")

    counter = 0
    while True:
        yield disturbance(counter)
        counter += 1


def apply_randomization(env, task_randomization_seed, task):
    if task_randomization_seed is not None:
        np.random.seed(task_randomization_seed)

        if "stack_blocks" in task:
            size = 0.5
            bounds_per_object = [(0.0, 0.0), (2.0, 2.0)]
        elif task == "pick_and_pour":
            size = 0.06
            bounds_per_object = [(0, -0.4), (-0.1, -0.3)]


        # Update to keep track of collisions
        new_positions = []
        for obj in env._passive_names:

            current_position = env._get_model_q(env._passive_names[0])
            current_z = current_position[2]

            collision_free = False
            while not collision_free:
                position = np.concatenate((np.random.uniform(*bounds_per_object), np.array([current_z])))

                collision_free = True
                for pos in new_positions:
                    if np.linalg.norm(np.array(position) - np.array(pos)) < size:
                        collision_free = False

            new_positions.append(position)

        passive_q = np.concatenate(new_positions)
        for i, name in enumerate(env._passive_names):
            env._set_model_q(name, passive_q[i*3:(i+1)*3])


def perform_task(task,
                 plan_builder,
                 task_randomization_seed=None,
                 disturbance_seq=None,
                 save_path=None):

    os.makedirs(save_path, exist_ok=True)

    env, graph, goc_mpc = plan_builder()

    dim = graph.dim
    num_agents = graph.num_agents

    observed_qs = []

    dt = goc_mpc.short_path_time_per_step

    # do it again?
    env._diagram.ForcedPublish(env._context)
    input("Continue?")

    total_cost = 0.0
    wp_sts = []
    timing_sts = []
    short_path_sts = []

    applied_disturbance = {
        stage: False for stage in goc_mpc.remaining_phases
    }
    stage_counter = {
        stage: 0 for stage in goc_mpc.remaining_phases
    }
    disturbance_funcs = []

    def update_disturbance_seq(completed_phases):
        if disturbance_seq is not None:
            for stage, disturbance in disturbance_seq.items():
                # if at a stage to apply a disturbance
                if (stage in completed_phases and
                    not applied_disturbance.get(stage, False) and
                    stage_counter[stage] >= disturbance.delay):

                    # set the disturbance sequence, the generator will yield and instantiate one disturbance function for each env.step until it is exhausted
                    disturbance_funcs.append(disturbance.func(env, disturbance.agent))
                    applied_disturbance[stage] = True

                elif (stage in completed_phases and
                      not applied_disturbance.get(stage, False)):
                    stage_counter[stage] += 1

    obs, _ = env.reset()
    goc_mpc.reset()

    print("APPLYING RANDOMIZATION")

    apply_randomization(env, task_randomization_seed, task)

    print("APPLIED RANDOMIZATION")

    # let settle
    for _ in range(20):
        qpos = obs[0][:num_agents * dim]
        obs, _, _, _, _ = env.step(qpos)

    disturbed = False

    step = 3
    n_steps = 0
    for k in range(0, 2000, step):
        x, x_dot = obs

        try:
            xi_h, _, _ = goc_mpc.step(k * dt, x, x_dot, teleport=False)
            wp_sts.append(goc_mpc.waypoint_mpc.get_last_solve_time())
            timing_sts.append(goc_mpc.timing_mpc.get_last_solve_time())
            short_path_sts.append(goc_mpc.short_path_mpc.get_last_solve_time())
        except RuntimeError as e:
            print(e)
            xi_h, _, _ = goc_mpc.last_cycle_short_path
            if xi_h.shape[0] > 1:
                xi_h = xi_h[1:]

        # if block:
        # go_up = graph.structure.add_node()
        # graph.structure.add_edge(prev, go_up, True)

        # graph.add_robot_to_point_displacement_constraint(go_up, agent, block, np.array([0.0, 0.0, -0.30]))

        # if k > 162:
        #     breakpoint()
        #     # detach grasp to see if backtracking is possible.
        #     assert "cube_0" in env._grasps

        #     if "cube_0" in env._grasps and not disturbed:
        #         env.release_grasp("cube_0")
        #         disturbed = True

        if len(goc_mpc.last_cycle_backtracked_phases) > 0:
            if task == "stack_blocks_sequence":
                points = list(env._grasps)
                for point in points:
                    env.release_grasp(point)

        qpos = xi_h[step].copy()

        # for ag in range(num_agents):
        #     if goc_mpc.timing_mpc.get_agent_spline_length(ag) == 1:
        #         qpos[7*ag + 2] += 0.01

        obs, rew, done, trunc, info = env.step(qpos, grasp_cmds=goc_mpc.last_grasp_commands)
        n_steps += 1

        if len(disturbance_funcs) > 0:
            for disturbance_func in disturbance_funcs:
                next(disturbance_func)
    
        old_poses = x[:num_agents*dim].reshape(num_agents, dim)
        agent_poses = qpos[:num_agents*dim].reshape(num_agents, dim)

        old_positions = old_poses[:, :3]
        agent_positions = agent_poses[:, :3]

        total_cost += np.sum(np.linalg.norm(agent_positions - old_positions, axis=1))

        update_disturbance_seq(goc_mpc.completed_phases)

        if len(goc_mpc.remaining_phases) == 0:
            break

    success = 1.0
    obj_positions = x[num_agents*dim:]
    for i in range(1, graph.num_objects):
        if obj_positions[3*(i-1) + 2] + 0.01 < obj_positions[3*i + 2]:
            continue
        else:
            success = 0.0

    sts = np.array(wp_sts) + np.array(timing_sts) + np.array(short_path_sts)

    print("Success", success)
    print("Mean Solve Time", np.mean(sts))
    print("Median Solve Time", np.median(sts))
    print("Max Solve Time", np.max(sts))
    print("Total Cost", total_cost)

    metrics = {
        'success': success,
        'avg_time': np.mean(sts),
        'median_time': np.median(sts),
        'max_time': np.max(sts),
        'total_cost': total_cost,
        'total_simulation_steps': n_steps,
        'waypoint_solve_times': wp_sts,
        'timing_solve_times': timing_sts,
        'short_path_solve_times': short_path_sts
    }

    # Save metrics to tasks dir
    metrics_dir = os.path.join(save_path, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = f'{task}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pkl'
    metrics_path = os.path.join(metrics_dir, metrics_file)
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)

    print("Finished Saving metrics to", metrics_path)
    return env, graph, goc_mpc



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='stack_blocks', help='task to perform')
    parser.add_argument('--seed_range', type=int_pair, default=(0, 1), help='seeds over which to evaluate')
    parser.add_argument('--save_path', type=str, help='path to save files and data')
    parser.add_argument('--use_cached_query', action='store_true', help='instead of querying the VLM, use the cached query')
    parser.add_argument('--use_disturbance', action='store_true', help='use a disturbance')
    parser.add_argument('--visualize', action='store_true', help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)')
    parser.add_argument('--randomize', '-r', action='store_true', help='randomize the setup environment for all tasks')
    args = parser.parse_args()

    # meshcat = Meshcat(port=8080)
    # env, graph, goc_mpc = n_gripper_n_block_stacking(n_grippers=3, n_blocks=5)

    # env, graph, goc_mpc = test_two_gripper_block_stacking(meshcat=meshcat)
    # env, graph, goc_mpc = n_gripper_n_block_stacking(n_grippers=3, n_blocks=5)
    # env, graph, goc_mpc = two_gripper_block_stacking()
    # env, graph, goc_mpc = two_gripper_pick_and_pour()
    # env, graph, goc_mpc = two_gripper_fold_sheet()
    # env, graph, goc_mpc = two_gripper_pick_and_pour()
    # env, graph, goc_mpc = two_gripper_assignable_move()

    stack_blocks_task = {
        'plan_builder': lambda: n_gripper_n_block_stacking(n_grippers=2, n_blocks=3),
        'disturbance_seq': {0: Disturbance(delay=2, func=block_disturbance, agent=0)},
    }

    # ABLATION
    stack_blocks_sequence_task = {
        'plan_builder': lambda: n_gripper_n_block_sequence_stacking(n_grippers=2, n_blocks=3),
        'disturbance_seq': {0: Disturbance(delay=2, func=block_disturbance, agent=0)},
    }

    stack_blocks_3x4_task = {
        'plan_builder': lambda: n_gripper_n_block_stacking(n_grippers=3, n_blocks=4),
        'disturbance_seq': {0: Disturbance(delay=2, func=block_disturbance, agent=0)},
    }

    pick_and_pour_task = {
        'plan_builder': two_gripper_pick_and_pour,
        'disturbance_seq': {0: Disturbance(delay=0, func=pick_and_pour_disturbance, agent=0)},
    }

    stack_blocks_2x2_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=2, n_blocks=2),
    }

    stack_blocks_2x5_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=2, n_blocks=5),
    }

    stack_blocks_2x8_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=2, n_blocks=8),
    }

    stack_blocks_2x11_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=2, n_blocks=11),
    }

    stack_blocks_2x12_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=2, n_blocks=12),
    }

    stack_blocks_3x5_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=3, n_blocks=5),
    }

    stack_blocks_3x8_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=3, n_blocks=8),
    }

    stack_blocks_3x11_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=3, n_blocks=11),
    }

    stack_blocks_3x12_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=3, n_blocks=12),
    }

    stack_blocks_4x5_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=4, n_blocks=5),
    }

    stack_blocks_4x8_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=4, n_blocks=8),
    }

    stack_blocks_4x11_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=4, n_blocks=11),
    }

    stack_blocks_4x12_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=4, n_blocks=12),
    }

    stack_blocks_5x5_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=5, n_blocks=5),
    }

    stack_blocks_5x8_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=5, n_blocks=8),
    }

    stack_blocks_5x11_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=5, n_blocks=11),
    }

    stack_blocks_5x12_task = {
        "plan_builder": lambda: n_gripper_n_block_stacking(n_grippers=5, n_blocks=12),
    }

    tasks = {
        "stack_blocks_2x2": stack_blocks_2x2_task,
        # "stack_blocks": stack_blocks_task,
        # "stack_blocks_sequence": stack_blocks_sequence_task,
        "stack_blocks_2x5": stack_blocks_2x5_task,
        "stack_blocks_2x8": stack_blocks_2x8_task,
        "stack_blocks_2x11": stack_blocks_2x11_task,
        "stack_blocks_2x12": stack_blocks_2x12_task,
        "stack_blocks_3x5": stack_blocks_3x5_task,
        "stack_blocks_3x8": stack_blocks_3x8_task,
        "stack_blocks_3x11": stack_blocks_3x11_task,
        "stack_blocks_3x12": stack_blocks_3x12_task,
        "stack_blocks_4x5": stack_blocks_4x5_task,
        "stack_blocks_4x8": stack_blocks_4x8_task,
        "stack_blocks_4x11": stack_blocks_4x11_task,
        "stack_blocks_4x12": stack_blocks_4x12_task,
        "stack_blocks_5x5": stack_blocks_5x5_task,
        "stack_blocks_5x8": stack_blocks_5x8_task,
        "stack_blocks_5x11": stack_blocks_5x11_task,
        "stack_blocks_5x12": stack_blocks_5x12_task,
        "pick_and_pour": pick_and_pour_task,
    }

    task_name = args.task
    task = tasks[args.task]

    if args.randomize:
        print(f"randomizing over seeds in range {args.seed_range}")
        for i in range(*args.seed_range):
            env, graph, goc_mpc = perform_task(task_name,
                                               task["plan_builder"],
                                               task_randomization_seed=i,
                                               disturbance_seq=task.get('disturbance_seq', None) if args.use_disturbance else None,
                                               save_path=os.path.join(args.save_path, args.task, str(i)))
    else:
        env, graph, goc_mpc = perform_task(task_name,
                                           task["plan_builder"],
                                           task_randomization_seed=None,
                                           disturbance_seq=task.get('disturbance_seq', None) if args.use_disturbance else None,
                                           save_path=os.path.join(args.save_path, args.task, "default"))


if __name__ == "__main__":
    main()
