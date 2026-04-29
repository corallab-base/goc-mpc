import os
import time
import imageio
import numpy as np
import mujoco as mj
import matplotlib.pyplot as plt

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
    env = SimpleDrakeGym(["pos_quat_0", "pos_quat_1"], [f"cube_{i}" for i in range(n_points)], meshcat=meshcat)

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    graph = GraphOfConstraints(["pos_quat_0", "pos_quat_1"], [f"cube_{i}" for i in range(n_points)],
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
    spline_spec = [Block.R(3), Block.SO3Quat()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = False,
                                    time_delta_cutoff = 0.0,
                                    phi_tolerance = 0.03)
    return env, graph, goc_mpc

def n_gripper_n_block_stacking(n_grippers=3, n_blocks=5, quat=np.array([0.0, 0.0, 1.0, 0.0])):
    agents = [f"pos_quat_{j}" for j in range(n_grippers)]
    objects = [f"cube_{i}" for i in range(n_blocks)]
    env = SimpleDrakeGym(agents, objects)

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    graph = GraphOfConstraints(agents, objects,
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

    pick1, place1 = add_pick_and_place(0, 0, 1)
    pick2, place2 = add_pick_and_place(1, 2, 0)
    pick3, place3 = add_pick_and_place(2, 3, 2)
    pick4, place4 = add_pick_and_place(0, 4, 3)

    graph.structure.add_edge(place1, place2, True)
    graph.structure.add_edge(place2, place3, True)
    graph.structure.add_edge(place3, place4, True)

    graph.structure.add_edge(place1, pick4, True)


    spline_spec = [Block.R(3), Block.SO3Quat()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = False,
                                    time_delta_cutoff = 0.0,
                                    phi_tolerance = 0.03)
    return env, graph, goc_mpc


def two_gripper_block_stacking():
    env = SimpleDrakeGym(["point_mass_0", "point_mass_1"], ["cube_0", "cube_1", "cube_2"])

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    graph = GraphOfConstraints(["point_mass_0", "point_mass_1"], ["cube_0", "cube_1", "cube_2"],
                               state_lower_bound, state_upper_bound)

    joint_agent_dim = graph.num_agents * graph.dim;

    graph.structure.add_nodes(6)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(2, 3, True)
    graph.structure.add_edge(1, 3, True)
    graph.structure.add_edge(1, 4, True)
    graph.structure.add_edge(3, 5, True)

    r1 = graph.add_variable()
    r2 = graph.add_variable()
    graph.add_variable_ineq_constraint(r1, r2)

    phi0 = graph.add_assignable_robot_to_point_displacement_constraint(0, r1, 0, np.array([0.0, 0.0, -0.1]));
    graph.add_assignable_grasp_change(phi0, "grab", 0);

    graspPhi0 = graph.add_assignable_robot_holding_point_constraint(0, 1, r1, 0, 0.2);

    phi1 = graph.add_assignable_robot_to_point_displacement_constraint(1, r1, 1, np.array([0.0, 0.0, -0.2]))
    graph.add_assignable_grasp_change(phi1, "release", 0);

    phi2 = graph.add_assignable_robot_to_point_displacement_constraint(2, r2, 2, np.array([0.0, 0.0, -0.1]));
    graph.add_assignable_grasp_change(phi2, "grab", 2);

    # r2 hold on to block 2 while moving to block 1
    graspPhi1 = graph.add_assignable_robot_holding_point_constraint(2, 3, r2, 2, 0.2);

    # r2 move 30 centimeters above block 1
    phi3 = graph.add_assignable_robot_to_point_displacement_constraint(3, r2, 1, np.array([0.0, 0.0, -0.3]))

    # after r2 is above block 1, release block 2
    graph.add_assignable_grasp_change(phi3, "release", 2);

    # ensure timing delay between placing block 0 and placing block 2
    graph.add_edge_min_tau_constraint(1, 3, 1.0)

    # move a safe distance away from blocks after placing them
    phi4 = graph.add_assignable_robot_to_point_displacement_constraint(4, r1, 0, np.array([0.5, 0.5, -0.5]));
    phi5 = graph.add_assignable_robot_to_point_displacement_constraint(5, r2, 2, np.array([-0.5, -0.5, -0.5]));

    # when the 0 stacked on 1 edge constraint is violated here, back track all the way to node 0
    stackedPhi0 = graph.add_edge_point_to_point_displacement_constraint(
        u=1, v=4, point_a=0, point_b=1,
        disp=np.array([0.0, 0.0, -0.1]),
        tol=np.array([0.1, 0.1, 0.5]))
    graph.add_backtrack_links(stackedPhi0, [0, 2])

    # when the 2 stacked on 0 edge constraint is violated here, back track all the way to node 2
    stackedPhi1 = graph.add_edge_point_to_point_displacement_constraint(
        u=3, v=5, point_a=2, point_b=0,
        disp=np.array([0.0, 0.0, -0.1]),
        tol=np.array([0.1, 0.1, 0.5]))
    graph.add_backtrack_links(stackedPhi1, [2])

    # forever attempt to move toward 4/5 so that backtracking can occur as
    # neccessary when disturbances are made at the end
    graph.make_node_unpassable(4)
    graph.make_node_unpassable(5)

    # GoC-MPC
    spline_spec = [Block.R(3)]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = False,
                                    time_delta_cutoff = 0.0,
                                    phi_tolerance = 0.03,
                                    max_acc = 1.0)
    return env, graph, goc_mpc


def two_gripper_block_stacking_real():
    env = SimpleDrakeGym(["point_mass_0", "point_mass_1"], ["cube_0", "cube_1", "cube_2"])

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    graph = GraphOfConstraints(["point_mass_0", "point_mass_1"], ["cube_0", "cube_1", "cube_2"],
                               state_lower_bound, state_upper_bound)

    joint_agent_dim = graph.num_agents * graph.dim;

    r1 = graph.add_variable()
    r2 = graph.add_variable()
    graph.add_variable_ineq_constraint(r1, r2)

    def add_grasp(robot, block):
        approach, pick_up = graph.structure.add_nodes(2)
        graph.structure.add_edge(approach, pick_up, True)

        graph.add_assignable_robot_to_point_displacement_constraint(approach, robot, block, np.array([0.0, 0.0, -0.5]));

        graph.add_edge_assignable_robot_to_point_displacement_constraint(
            u=approach, v=pick_up, var=robot, point_id=block,
            disp=np.array([0.0, 0.0, -0.25]),
            tol=np.array([0.08, 0.08, 0.3]))

        phi = graph.add_assignable_robot_to_point_displacement_constraint(pick_up, robot, block, np.array([0.0, 0.0, -0.16]));
        graph.add_assignable_grasp_change(phi, "grab", block);

        return approach, pick_up

    def add_release(robot, held_block, onto_block):
        # approach, release = graph.structure.add_nodes(2)
        release = graph.structure.add_node()
        # graph.structure.add_edge(approach, release, True)

        # graph.add_assignable_robot_to_point_displacement_constraint(approach, robot, block, np.array([0.0, 0.0, -0.2]))

        # keep holding between approach and putting down
        # graph.add_assignable_robot_holding_point_constraint(approach, release, robot, block, 0.2)

        phi = graph.add_assignable_robot_to_point_displacement_constraint(release, robot, onto_block, np.array([0.0, 0.0, -0.25]))
        graph.add_assignable_grasp_change(phi, "release", held_block)

        return None, release

    # grasp and release block 0
    approach_pick_up_0, pick_up_0 = add_grasp(r1, block=0)
    approach_release_0, release_0 = add_release(r1, held_block=0, onto_block=1)
    graspPhi0 = graph.add_assignable_robot_holding_point_constraint(pick_up_0, release_0, r1, 0, 0.2);

    graph.structure.add_edge(pick_up_0, release_0, True)

    # grasp and release block 0
    approach_pick_up_2, pick_up_2 = add_grasp(r2, block=2)
    approach_release_2, release_2 = add_release(r2, held_block=2, onto_block=0)
    graspPhi1 = graph.add_assignable_robot_holding_point_constraint(pick_up_2, release_2, r2, 2, 0.2);

    graph.structure.add_edge(pick_up_2, release_2, True)

    # reach release_2 no sooner than 1 second after release_0
    graph.structure.add_edge(release_0, release_2, True)
    graph.add_edge_min_tau_constraint(release_0, release_2, 1.0)

    # move a safe distance away from block 0 / 2 after placing it
    left_end = graph.structure.add_node()
    graph.structure.add_edge(release_0, left_end, True) # TODO: ADD CONDITIONS
    graph.structure.add_edge(release_2, left_end, True) # TODO: ADD CONDITIONS
    phi4 = graph.add_robot_pos_linear_eq(
        k=left_end, robot_id=0, A=np.eye(3), b=np.array([-0.5, 0.0, 0.5]));

    right_end = graph.structure.add_node()
    graph.structure.add_edge(release_0, right_end, True) # TODO: ADD CONDITIONS
    graph.structure.add_edge(release_2, right_end, True) # TODO: ADD CONDITIONS
    phi5 = graph.add_robot_pos_linear_eq(
        k=right_end, robot_id=1, A=np.eye(3), b=np.array([-0.5, -0.7, 0.5]));

    # when the 0 stacked on 1 edge constraint is violated here, back track all the way to node 0
    stackedPhi0 = graph.add_edge_point_to_point_displacement_constraint(
        u=release_0, v=left_end, point_a=0, point_b=1,
        disp=np.array([0.0, 0.0, -0.1]),
        tol=np.array([0.1, 0.1, 0.5]))
    stackedPhi1 = graph.add_edge_point_to_point_displacement_constraint(
        u=release_2, v=left_end, point_a=0, point_b=1,
        disp=np.array([0.0, 0.0, -0.1]),
        tol=np.array([0.1, 0.1, 0.5]))
    graph.add_backtrack_links(stackedPhi0, [approach_pick_up_0, approach_pick_up_2])
    graph.add_backtrack_links(stackedPhi1, [approach_pick_up_0, approach_pick_up_2])

    # when the 2 stacked on 0 edge constraint is violated here, back track all the way to node 2
    stackedPhi2 = graph.add_edge_point_to_point_displacement_constraint(
        u=release_2, v=right_end, point_a=2, point_b=0,
        disp=np.array([0.0, 0.0, -0.1]),
        tol=np.array([0.1, 0.1, 0.5]))
    graph.add_backtrack_links(stackedPhi2, [approach_pick_up_2])

    # forever attempt to move toward end points so that backtracking can occur as
    # neccessary when disturbances are made at the end
    graph.make_node_unpassable(left_end)
    graph.make_node_unpassable(right_end)

    # GoC-MPC
    spline_spec = [Block.R(3)]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = False,
                                    time_delta_cutoff = 0.0,
                                    phi_tolerance = 0.03,
                                    max_acc = 1.0)
    return env, graph, goc_mpc


def two_gripper_block_arranging_real():
    env = SimpleDrakeGym(["point_mass_0", "point_mass_1"], ["cube_0", "cube_1", "cube_2"])

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    graph = GraphOfConstraints(["point_mass_0", "point_mass_1"], ["cube_0", "cube_1", "cube_2"],
                               state_lower_bound, state_upper_bound)

    joint_agent_dim = graph.num_agents * graph.dim;

    r1 = graph.add_variable()
    r2 = graph.add_variable()
    graph.add_variable_ineq_constraint(r1, r2)

    def add_grasp(robot, block):
        approach, pick_up = graph.structure.add_nodes(2)
        graph.structure.add_edge(approach, pick_up, True)

        graph.add_assignable_robot_to_point_displacement_constraint(approach, robot, block, np.array([0.0, 0.0, -0.3]));

        aligned_phi = graph.add_edge_assignable_robot_to_point_displacement_constraint(
            u=approach, v=pick_up, var=robot, point_id=block,
            disp=np.array([0.0, 0.0, -0.25]),
            tol=np.array([0.1, 0.1, 0.3]))

        phi = graph.add_assignable_robot_to_point_displacement_constraint(pick_up, robot, block, np.array([0.0, 0.0, -0.17]));
        graph.add_assignable_grasp_change(phi, "grab", block);

        return approach, pick_up

    def add_release(robot, held_block, relative_to_block, displacement):
        # approach, release = graph.structure.add_nodes(2)
        release = graph.structure.add_node()
        # graph.structure.add_edge(approach, release, True)

        # graph.add_assignable_robot_to_point_displacement_constraint(approach, robot, block, np.array([0.0, 0.0, -0.2]))

        # keep holding between approach and putting down
        # graph.add_assignable_robot_holding_point_constraint(approach, release, robot, block, 0.2)

        phi = graph.add_assignable_robot_to_point_displacement_constraint(release, robot, relative_to_block, displacement)
        graph.add_assignable_grasp_change(phi, "release", held_block)

        return None, release

    # grasp and release block 0
    approach_pick_up_0, pick_up_0 = add_grasp(r1, block=0)
    _, release_0 = add_release(r1, held_block=0, relative_to_block=1, displacement=np.array([-0.15, -0.10, -0.21]))
    grasp_phi_0 = graph.add_assignable_robot_holding_point_constraint(pick_up_0, release_0, r1, 0, 0.2);

    graph.structure.add_edge(pick_up_0, release_0, True)

    # grasp and release block 0
    approach_pick_up_2, pick_up_2 = add_grasp(r2, block=2)
    _, release_2 = add_release(r2, held_block=2, relative_to_block=1, displacement=np.array([0.15, 0.10, -0.21]))
    grasp_phi_1 = graph.add_assignable_robot_holding_point_constraint(pick_up_2, release_2, r2, 2, 0.2);

    graph.structure.add_edge(pick_up_2, release_2, True)

    # reach release_2 no sooner than 1 second after release_0
    graph.structure.add_edge(release_0, pick_up_2, True)
    graph.structure.add_edge(release_0, release_2, True)
    graph.add_edge_min_tau_constraint(release_0, release_2, 3.0)

    # move a safe distance away from block 0 / 2 after placing it
    left_end = graph.structure.add_node()
    graph.structure.add_edge(release_0, left_end, True) # TODO: ADD CONDITIONS
    graph.structure.add_edge(release_2, left_end, True) # TODO: ADD CONDITIONS
    phi4 = graph.add_robot_pos_linear_eq(
        k=left_end, robot_id=0, A=np.eye(3), b=np.array([-0.5, 0.0, 0.5]));

    right_end = graph.structure.add_node()
    graph.structure.add_edge(release_0, right_end, True) # TODO: ADD CONDITIONS
    graph.structure.add_edge(release_2, right_end, True) # TODO: ADD CONDITIONS
    phi5 = graph.add_robot_pos_linear_eq(
        k=right_end, robot_id=1, A=np.eye(3), b=np.array([-0.5, -0.7, 0.5]));

    # when the 0 stacked on 1 edge constraint is violated here, back track all the way to node 0
    arrangedPhi0 = graph.add_edge_point_to_point_displacement_constraint(
        u=release_0, v=left_end, point_a=0, point_b=1,
        disp=np.array([-0.15, -0.10, 0.0]),
        tol=np.array([0.2, 0.2, 0.5]))
    arrangedPhi1 = graph.add_edge_point_to_point_displacement_constraint(
        u=release_2, v=left_end, point_a=0, point_b=1,
        disp=np.array([-0.15, -0.10, 0.0]),
        tol=np.array([0.2, 0.2, 0.5]))
    graph.add_manual_backtrack_links(arrangedPhi0, [approach_pick_up_0, pick_up_0, release_0])

    # when the 2 stacked on 0 edge constraint is violated here, back track all the way to node 2
    arrangedPhi2 = graph.add_edge_point_to_point_displacement_constraint(
        u=release_2, v=right_end, point_a=2, point_b=0,
        disp=np.array([0.15, 0.10, 0.0]),
        tol=np.array([0.2, 0.2, 0.5]))
    graph.add_manual_backtrack_links(arrangedPhi2, [approach_pick_up_2, pick_up_2, release_2])

    # forever attempt to move toward end points so that backtracking can occur as
    # neccessary when disturbances are made at the end
    graph.make_node_unpassable(left_end)
    graph.make_node_unpassable(right_end)

    # GoC-MPC
    spline_spec = [Block.R(3)]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec,
                                    time_delta_cutoff = 0.0,
                                    short_path_time_per_step = 0.1,
                                    phi_tolerance = 0.03,
                                    # max_vel = 0.05,  # maximum velocity for every joint
                                    max_acc = 1.00,  # maximum acceleration for every joint
                                    # max_jerk = 0.05 # maximum jerk for every joint
                                )
    return env, graph, goc_mpc


def two_gripper_fold_sheet():
    env = SimpleDrakeGym(["pos_quat_0", "pos_quat_1"], ["cube_0", "cube_1"])

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    graph = GraphOfConstraints(["pos_quat_0", "pos_quat_1"], ["cube_0", "cube_1"],
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
    spline_spec = [Block.R(3), Block.SO3Quat()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = False,
                                    time_delta_cutoff = 0.0,
                                    phi_tolerance = 0.03)
    return env, graph, goc_mpc


def two_gripper_assignable_move():
    env = SimpleDrakeGym(["pos_quat_0", "pos_quat_1"], ["cube_0", "cube_1", "cube_2"])

    state_lower_bound = -10.0
    state_upper_bound =  10.0

    graph = GraphOfConstraints(["pos_quat_0", "pos_quat_1"], ["cube_0", "cube_1", "cube_2"],
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
    spline_spec = [Block.R(3), Block.SO3Quat()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = True,
                                    time_delta_cutoff = 0.0,
                                    phi_tolerance = 0.03)
    return env, graph, goc_mpc


# def two_gripper_assignable_block_stacking():
#     env = SimpleDrakeGym(["pos_quat_0", "pos_quat_1"], ["cube_0", "cube_1", "cube_2"])

#     state_lower_bound = -10.0
#     state_upper_bound =  10.0

#     symbolic_plant = env.plant.ToSymbolic()
#     graph = GraphOfConstraints(symbolic_plant, ["pos_quat_0", "pos_quat_1"], ["cube_0", "cube_1", "cube_2"],
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
#     spline_spec = [Block.R(3), Block.SO3Quat()]
#     goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
#                                     time_delta_cutoff = 0.001,)
#     return env, graph, goc_mpc


def two_gripper_rotate_in_place():
    env = SimpleDrakeGym(["pos_quat_0", "pos_quat_1"], ["cube_0"])

    state_lower_bound = -10.0
    tate_upper_bound =  10.0

    graph = GraphOfConstraints(["pos_quat_0", "pos_quat_1"], ["cube_0"],
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
    spline_spec = [Block.R(3), Block.SO3Quat()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = False,
                                    time_delta_cutoff = 0.05,
                                    phi_tolerance = 0.05)
    return env, graph, goc_mpc


def two_gripper_rotation_test():
    meshcat = Meshcat(port=7002)

    # env and visualization
    env = SimpleDrakeGym(["pos_quat_0", "pos_quat_1"], [], meshcat=meshcat)

    state_lower_bound = -10.0
    tate_upper_bound =  10.0

    graph = GraphOfConstraints(["pos_quat_0", "pos_quat_1"], [],
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
    spline_spec = [Block.R(3), Block.SO3Quat()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1)
    return env, graph, goc_mpc



def main():
    # meshcat = Meshcat(port=8080)

    # env, graph, goc_mpc = test_two_gripper_block_stacking(meshcat=meshcat)
    # env, graph, goc_mpc = n_gripper_n_block_stacking(n_grippers=3, n_blocks=5)
    # env, graph, goc_mpc = two_gripper_block_stacking()
    # env, graph, goc_mpc = two_gripper_block_stacking_real()
    env, graph, goc_mpc = two_gripper_block_arranging_real()
    # env, graph, goc_mpc = two_gripper_pick_and_pour()
    # env, graph, goc_mpc = two_gripper_fold_sheet()
    # env, graph, goc_mpc = two_gripper_pick_and_pour()
    # env, graph, goc_mpc = two_gripper_assignable_move()

    dim = graph.dim
    num_agents = graph.num_agents

    observed_qs = []

    dt = goc_mpc.short_path_time_per_step

    # do it again?
    env._diagram.ForcedPublish(env._context)
    input("Continue?")

    wp_sts = []
    timing_sts = []
    short_path_sts = []

    while True:
        # obs, _ = env.reset(
        #     q0=np.array([ 0.1332869 ,  0.45878515,  0.07459005,  0.25109156,  0.66010387,
        #                   0.66245582,  0.24973625,  2.26091005,  0.48135286,  0.10280809,
        #                   0.25161163,  0.66336338, -0.65905311, -0.24957888,  0.25      ,
        #                   0.5       ,  0.02496069,  1.25      ,  0.5       ,  0.02496069,
        #                   2.19099191,  0.49969403,  0.03780399])
        # )
        obs, _ = env.reset()
        goc_mpc.reset()

        # let settle
        for _ in range(20):
            qpos = obs[0][:num_agents * dim]
            obs, _, _, _, _ = env.step(qpos)

        # # for debugging, get assignments and pass a few nodes.
        # x, x_dot = obs
        # goc_mpc._solve_for_waypoints(x)
        # assignments = goc_mpc.waypoint_mpc.view_assignments()
        # goc_mpc.pass_node(0, assignments)
        # goc_mpc.pass_node(2, assignments)

        disturbed = False
        disturbed2 = False

        env._meshcat.StartRecording()

        t = 0.0
        teleport = False

        step = 1
        for k in range(0, 2000, step):
            x, x_dot = obs

            try:
                xi_h, xi_dot_h, times_h = goc_mpc.step(t, x, x_dot, teleport=teleport)

                wp_sts.append(goc_mpc.waypoint_mpc.get_last_solve_time())
                timing_sts.append(goc_mpc.timing_mpc.get_last_solve_time())
                short_path_sts.append(goc_mpc.short_path_mpc.get_last_solve_time())
            except RuntimeError as e:
                print(e)
                xi_h, _, _ = goc_mpc.last_cycle_short_path
                if xi_h.shape[0] > 1:
                    xi_h = xi_h[1:]

            # get target point on xi_h and associated time
            qpos = xi_h[step]
            time_delta = times_h[step]

            # if not disturbed and k > 1000:
            #     i = 2
            #     name = env._passive_names[i]
            #     q_slice = env._passive_q_slices[i]
            #     new_q = obs[0][q_slice] + np.array([0.5, 0.0, 0.0])
            #     env._set_model_q(name, new_q)
            #     disturbed = True

            if not disturbed2 and k > 600:
                i = 2
                name = env._passive_names[i]
                q_slice = env._passive_q_slices[i]
                new_q = obs[0][q_slice] + np.array([0.5, 0.0, 0.0])
                env._set_model_q(name, new_q)

                i = 0
                name = env._passive_names[i]
                q_slice = env._passive_q_slices[i]
                new_q = obs[0][q_slice] + np.array([0.5, 0.0, 0.0])
                env._set_model_q(name, new_q)

                disturbed2 = True

            #     breakpoint()

            # jump to target point
            obs, rew, done, trunc, info = env.step(qpos, grasp_cmds=goc_mpc.last_grasp_commands)

            # advance time by the associated time delta
            t += time_delta

            if len(goc_mpc.remaining_phases) == 0:
                print("Finished!")
                break

            if teleport:
                time.sleep(1.0)

        sts = np.array(wp_sts) + np.array(timing_sts) + np.array(short_path_sts)
        print("Mean Solve Time", np.mean(sts))
        print("Median Solve Time", np.median(sts))
        print("Max Solve Time", np.max(sts))

        env._meshcat.StopRecording()

        resp = input("Repeat?")
        if resp == 'q':
            break




if __name__ == "__main__":
    main()
