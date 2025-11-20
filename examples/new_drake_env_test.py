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

    pick1, place1 = add_pick_and_place(0, 0, 1)
    pick2, place2 = add_pick_and_place(1, 2, 0)
    pick3, place3 = add_pick_and_place(2, 3, 2)
    pick4, place4 = add_pick_and_place(0, 4, 3)

    graph.structure.add_edge(place1, place2, True)
    graph.structure.add_edge(place2, place3, True)
    graph.structure.add_edge(place3, place4, True)

    graph.structure.add_edge(place1, pick4, True)


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

def two_gripper_assignable_block_stacking():
    env = SimpleDrakeGym(["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"])

    state_lower_bound = -100.0
    state_upper_bound =  100.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"],
                               state_lower_bound, state_upper_bound)
    #graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], ["cube_0"],
     #                         state_lower_bound, state_upper_bound)

    # 0: Start
    # 1: Pick cube_0
    # 2: Place cube_0 on cube_1
    graph.structure.add_nodes(3)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(1, 2, True)

    # Home pose at node 0 (same as pick_and_pour demo)
    joint_agent_dim = graph.num_agents * graph.dim
    home_position = np.array([
        0.30, -0.2, 0.5, 0.0, -0.707071, -0.707071, 0.0,   # free_body_0
        1.50, -0.2, 0.5, 0.0,  0.707071, -0.707071, 0.0    # free_body_1
    ])
    graph.add_robots_linear_eq(0, np.eye(joint_agent_dim), home_position)

    # Which robot does the job
    r1 = graph.add_variable()

    #phi0 = graph.add_robot_to_point_displacement_constraint(0, 0, 0, np.array([0.0, 0.0, -0.1]))
    phi_pick = graph.add_assignable_robot_to_point_displacement_constraint(
            1,  # Node Index
            r1, # Robot variable
            0,  # Cube Index
            np.array([0.0, 0.0, -0.1])
    )

    # Grab the block once the robot gets there
    graph.add_assignable_grasp_change(phi_pick, "grab", 0)

    # Hold the block as it is moved
    phi_hold = graph.add_assignable_robot_holding_point_constraint(
            1,  # Start Node
            2,  # End Node
            r1,
            0,  # Cube Index
            0.1
    )

    # The same robot that is holding cube_0 should place it on cube_1
    phi_place = graph.add_assignable_robot_to_point_displacement_constraint(
            2,  # Node Index
            r1,
            1,  # Cube Index
            np.array([0.0, 0.0, -0.1])
    )

    graph.add_assignable_grasp_change(phi_place, "release", 0);

    # phi1 = graph.add_assignable_robot_to_point_displacement_constraint(0, r2, 2, np.array([0.0, 0.0, -0.1]))

    #phi1 = graph.add_robot_quat_linear_eq(0, 0, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))

    # Grab cube_0 (which ever robot is assigned)
        #graspPhi0 = graph.add_assignable_robot_holding_point_constraint(0, 1, r1, 0, 0.1)

    #phi2 = graph.add_robot_quat_linear_eq(1, 0, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
    #phi3 = graph.add_point_to_point_displacement_constraint(1, 0, 1, np.array([0.0, 0.0, -0.1]), tol=0.001)
    #graph.add_assignable_grasp_change(phi1, "release", 0)
    #graph.make_node_unpassable(1)
        # graph.add_assignable_grasp_change(phi2, "grab", 2);

    # graspPhi1 = graph.add_robot_holding_cube_constraint(2, 3, 1, 2, 0.1);

    # phi3 = graph.add_assignable_robot_to_point_displacement_constraint(3, r2, 0, np.array([0.0, 0.0, -0.2]));

    # phi4 = graph.add_assignable_robot_to_point_displacement_constraint(4, r1, 1, np.array([0.0, 0.0, -0.5]));

    # GoC-MPC
    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(
            graph,
            spline_spec,
            short_path_time_per_step = 0.1,
            time_delta_cutoff = 0.001,
            phi_tolerance = 0.03
    )
    return env, graph, goc_mpc


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


def two_gripper_fold_sweater():
    env = SimpleDrakeGym(["free_body_0", "free_body_1"], ["cube_0", "cube_1"])

    state_lower_bound = -100.0
    state_upper_bound =  100.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], ["cube_0", "cube_1"],
                               state_lower_bound, state_upper_bound)
    graph.structure.add_nodes(1)

    # home_position_1 = np.array([0.30, -0.2, 0.5, 0.0, 0.0, 1.0, 0.0,
    #                            -0.30, -0.2, 0.5, 0.0, 0.0, 1.0, 0.0])
    # phi0 = graph.add_robots_linear_eq(0, np.eye(joint_agent_dim), home_position_1)

    phi1 = graph.add_robot_above_cube_constraint(0, 0, 0, 0.18)
    # graph.add_grasp_change(phi1, "grab", 0, 0)

    goal_position_1 = np.array([0, 0.707, -0.707, 0])
    graph.add_robot_quat_linear_eq(0, 0, np.eye(4), goal_position_1)

    # GoC-MPC
    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1, time_delta_cutoff = 0.01)
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



def main():
    # meshcat = Meshcat(port=8080)

    # env, graph, goc_mpc = test_two_gripper_block_stacking(meshcat=meshcat)
    # env, graph, goc_mpc = n_gripper_n_block_stacking(n_grippers=3, n_blocks=5)
    # env, graph, goc_mpc = two_gripper_block_stacking()
    # env, graph, goc_mpc = two_gripper_pick_and_pour()
    env, graph, goc_mpc = two_gripper_assignable_block_stacking()
    # env, graph, goc_mpc = two_gripper_rotate_in_place()

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
            qpos = obs[0][:graph.num_agents * graph.dim]
            obs, _, _, _, _ = env.step(qpos)

        # # for debugging, get assignments and pass a few nodes.
        # x, x_dot = obs
        # goc_mpc._solve_for_waypoints(x)
        # assignments = goc_mpc.waypoint_mpc.view_assignments()
        # goc_mpc.pass_node(0, assignments)
        # goc_mpc.pass_node(2, assignments)

        disturbed = False

        env._meshcat.StartRecording()

        step = 3
        for k in range(0, 2000, step):
            x, x_dot = obs

            # if k % 300 == 0:
            #     breakpoint()
            # time.sleep(0.5)

            try:
                xi_h, _, _ = goc_mpc.step(k * dt, x, x_dot, teleport=False)
                wp_sts.append(goc_mpc.waypoint_mpc.get_last_solve_time())
                timing_sts.append(goc_mpc.timing_mpc.get_last_solve_time())
                short_path_sts.append(goc_mpc.short_path_mpc.get_last_solve_time())
            except RuntimeError as e:
                print(e)
                breakpoint()
                xi_h, _, _ = goc_mpc.last_cycle_short_path
                if xi_h.shape[0] > 1:
                    xi_h = xi_h[1:]

            # if len(goc_mpc.remaining_phases) < 2:
            #     breakpoint()

            # print("real cube 0 q:", x[6:9])
            # ag0_next_node = goc_mpc.timing_mpc.get_agent_spline_nodes(0)[0]
            # print("agent 0 next spline node:", ag0_next_node)
            # print("agent 0 next goal:", goc_mpc.waypoint_mpc.view_waypoints()[ag0_next_node, 6:9])

            # if k > 162:
                # breakpoint()
                # # detach grasp to see if backtracking is possible.
                # assert "cube_0" in env._grasps

                # if "cube_0" in env._grasps and not disturbed:
                #     env.release_grasp("cube_0")
                #     disturbed = True

            if len(goc_mpc.last_cycle_backtracked_phases) > 0:
                breakpoint()

            # if k % 200 == 0:
            #     visualize_last_cycle(goc_mpc)
                # fig = visualize_last_cycle(goc_mpc)
                # breakpoint()
                # input("Continue?")
                # plt.close(fig)

            qpos = xi_h[step]
            obs, rew, done, trunc, info = env.step(qpos, grasp_cmds=goc_mpc.last_grasp_commands)

        sts = np.array(wp_sts) + np.array(timing_sts) + np.array(short_path_sts)
        print("Mean Solve Time", np.mean(sts))
        print("Median Solve Time", np.median(sts))
        print("Max Solve Time", np.max(sts))

        breakpoint()

        env._meshcat.StopRecording()

        resp = input("Repeat?")
        if resp == 'q':
            break




if __name__ == "__main__":
    main()
