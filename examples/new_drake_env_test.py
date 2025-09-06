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


def two_gripper_block_stacking():
    env = SimpleDrakeGym(["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"])

    state_lower_bound = -100.0
    state_upper_bound =  100.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"],
                               state_lower_bound, state_upper_bound)

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

# def two_gripper_assignable_block_stacking():
#     env = SimpleDrakeGym(["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"])

#     state_lower_bound = -100.0
#     state_upper_bound =  100.0

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

def two_gripper_pick_and_pour():
    env = SimpleDrakeGym(["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"])

    state_lower_bound = -100.0
    state_upper_bound =  100.0

    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], ["cube_0", "cube_1", "cube_2"],
                               state_lower_bound, state_upper_bound)
    graph.structure.add_nodes(5) # 5
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(0, 2, True)
    graph.structure.add_edge(1, 3, True)
    graph.structure.add_edge(2, 3, True)
    graph.structure.add_edge(3, 4, True)

    # RESET
    joint_agent_dim = graph.num_agents * graph.dim;
    home_position_1 = np.array([0.30, -0.2, 0.5, 0.0, -0.707071, -0.707071, 0.0,
                                1.50, -0.2, 0.5, 0.0, 0.707071, -0.707071, 0.0])
    graph.add_robots_linear_eq(0, np.eye(joint_agent_dim), home_position_1)

    # PICK UP PITCHER AT ANGLE FROM SIDE
    # graph.add_robot_to_point_alignment_cost(1, 0, 0, np.array([0.0, 0.0, 1.0])
    #                                         # u_body_opt=np.array([1.0, 0.0, 0.0]),
    #                                         # roll_ref_flat=True
    #                                         )
    # graph.add_robot_quat_linear_eq(1, 0, 0, np.array([0.05, 0.0, -0.05]));
    graph.add_robot_relative_rotation_constraint(0, 1, 0, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());
    phi2 = graph.add_robot_to_point_displacement_constraint(1, 0, 0, np.array([0.05, 0.0, -0.05]));
    graph.add_grasp_change(phi2, "grab", 0, 0);

    graspPhi0 = graph.add_robot_holding_cube_constraint(1, 3, 0, 0, 0.1);
    graph.add_robot_relative_rotation_constraint(1, 3, 0, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());

    # PICK UP CUP AT ANGLE FROM SIDE
    # graph.add_robot_to_point_alignment_cost(2, 1, 2, np.array([0.0, 0.0, 1.0])
    #                                         # u_body_opt=np.array([1.0, 0.0, 0.0]),
    #                                         # roll_ref_flat=True
    #                                         )
    graph.add_robot_relative_rotation_constraint(0, 2, 1, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());
    phi4 = graph.add_robot_to_point_displacement_cost(2, 1, 2, np.array([-0.05, 0.0, -0.05]));
    graph.add_grasp_change(phi4, "grab", 1, 2);

    graspPhi1 = graph.add_robot_holding_cube_constraint(2, 3, 1, 2, 0.1);
    graph.add_robot_relative_rotation_constraint(2, 3, 1, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());

    # BRING PITCHER AND CUP CLOSE TO EACH OTHER
    graph.add_robot_to_point_displacement_cost(1, 0, 0, np.array([0.05, 0.0, -0.05]));
    graph.add_robot_to_point_displacement_cost(2, 1, 2, np.array([-0.05, 0.0, -0.05]));
    graph.add_point_to_point_displacement_cost(3, 0, 2, np.array([0.1, 0.0, -0.1]));
    graph.add_point_linear_eq(3, 0, np.array([[0.0, 0.0, 0.0],
                                              [0.0, 0.0, 0.0],
                                              [0.0, 0.0, 1.0]]), np.array([0.0, 0.0, 0.3]))


    # POUR PITCHER
    graph.add_robot_holding_cube_constraint(3, 4, 0, 0, 0.1);
    graph.add_robot_holding_cube_constraint(3, 4, 1, 2, 0.1);
    graph.add_robot_relative_displacement_constraint(3, 4, 1, np.array([0.0, 0.0, 0.0]));
    graph.add_point_to_point_displacement_cost(4, 0, 2, np.array([0.1, 0.0, -0.1]));
    graph.add_robot_relative_rotation_constraint(3, 4, 0,
                                                 RollPitchYaw(np.pi/3, 0.0, 0.0).ToQuaternion());
    graph.make_node_unpassable(4)

    # GoC-MPC
    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
                                    solve_for_waypoints_once = False,
                                    time_delta_cutoff = 0.0001,
                                    phi_tolerance = 0.03)
    return env, graph, goc_mpc

def two_gripper_rotate_in_place():
    env = SimpleDrakeGym(["free_body_0", "free_body_1"], ["cube_0"])

    state_lower_bound = -100.0
    state_upper_bound =  100.0

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
    graph.add_agent_quat_linear_eq(0, 0, np.eye(4), goal_position_1)

    # GoC-MPC
    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1, time_delta_cutoff = 0.01)
    return env, graph, goc_mpc


def two_gripper_rotation_test():
    meshcat = Meshcat(port=7002)

    # env and visualization
    env = SimpleDrakeGym(["free_body_0", "free_body_1"], [], meshcat=meshcat)

    state_lower_bound = -100.0
    state_upper_bound =  100.0

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
    # env, graph, goc_mpc = two_gripper_block_stacking()
    env, graph, goc_mpc = two_gripper_pick_and_pour()
    # env, graph, goc_mpc = two_gripper_assignable_block_stacking()
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
            qpos = obs[0][:14]
            obs, _, _, _, _ = env.step(qpos)

        # # for debugging, get assignments and pass a few nodes.
        # x, x_dot = obs
        # goc_mpc._solve_for_waypoints(x)
        # assignments = goc_mpc.waypoint_mpc.view_assignments()
        # goc_mpc.pass_node(0, assignments)
        # goc_mpc.pass_node(2, assignments)

        for k in range(2000):
            x, x_dot = obs

            # if k % 500 == 0:
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

            # if k == 250:
            #     # detach grasp to see if backtracking is possible.
            #     assert "cube_0" in env._grasps
            #     env.release_grasp("cube_0")

            # if k % 200 == 0:
            #     visualize_last_cycle(goc_mpc)
                # fig = visualize_last_cycle(goc_mpc)
                # breakpoint()
                # input("Continue?")
                # plt.close(fig)

            qpos = xi_h[0]
            obs, rew, done, trunc, info = env.step(qpos, grasp_cmds=goc_mpc.last_grasp_commands)

        resp = input("Repeat?")
        if resp == 'q':
            break




if __name__ == "__main__":
    main()
