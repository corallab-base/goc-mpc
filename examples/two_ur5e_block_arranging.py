"""Two UR5e block-arranging example
====================================
Two UR5e arms (each with a Robotiq 2F-85 gripper) at positions (0,0,0) and
(0,0.8,0) collaborate to rearrange three cubes.

The graph-of-constraints plan (do_block_arranging) is adapted from
ros_workspaces/test_ws/src/goc_demo/goc_demo/plans/two_robot_plans.py.

Robot 0 picks up cube 0 and places it relative to cube 1.
Robot 1 picks up cube 2 and places it relative to cube 1.
Both then return to safe home positions.  The waypoint solver assigns which
physical robot handles which task via assignable constraints.
"""

import time
import numpy as np

from goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC, WaypointSolver, WaypointObjective
from goc_mpc.splines import Block
from goc_mpc.systems import TwoUR5eDiffIKEnv

from pydrake.math import eq


# -----------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------

def build_graph(home_pos_0: np.ndarray, home_pos_1: np.ndarray):
    """Build the block-arranging GoC graph for two UR5e arms and three cubes.

    home_pos_0 / home_pos_1 are the EE positions of each arm in the home
    configuration (used as safe return positions at the end of the task).
    """
    robot_spec  = [Block.R(3)]
    object_spec = [Block.R(3)]

    graph = GraphOfConstraints(
        [robot_spec, robot_spec],
        [object_spec, object_spec, object_spec],
        state_lower_bound=-10.0,
        state_upper_bound=10.0,
        robot_names=["ur5e_0_ee", "ur5e_1_ee"],
        object_names=["cube_0", "cube_1", "cube_2"],
    )

    # Two assignable variables — the waypoint solver picks which robot does which task
    r1 = graph.add_variable()
    r2 = graph.add_variable()
    graph.add_variable_ineq_constraint(r1, r2)

    def add_grasp(robot_var, block_id):
        """Approach above the block, then descend to grasp height."""
        approach, pick_up = graph.structure.add_nodes(2)
        graph.structure.add_edge(approach, pick_up, True)

        # Approach: EE 30 cm above block
        graph.add_assignable_robot_to_point_displacement_constraint(
            approach, robot_var, block_id, np.array([0.0, 0.0, -0.30]))
        # Pick: EE 19 cm above block; trigger grasp
        phi = graph.add_assignable_robot_to_point_displacement_constraint(
            pick_up, robot_var, block_id, np.array([0.0, 0.0, -0.19]))
        graph.add_assignable_grasp_change(phi, "grab", block_id)

        return approach, pick_up

    def add_release(robot_var, held_block_id, ref_block_id, displacement):
        """Place the held block at ref_block + displacement and release."""
        release = graph.structure.add_node()
        phi = graph.add_assignable_robot_to_point_displacement_constraint(
            release, robot_var, ref_block_id, displacement)
        graph.add_assignable_grasp_change(phi, "release", held_block_id)
        return release

    # --- Robot r1: grasp cube 0, place relative to cube 1 ---
    approach_0, pick_up_0 = add_grasp(r1, block_id=0)
    release_0 = add_release(r1, held_block_id=0, ref_block_id=1,
                            displacement=np.array([-0.10, -0.15, -0.21]))
    graph.structure.add_edge(pick_up_0, release_0, True)

    # --- Robot r2: grasp cube 2, place relative to cube 1 ---
    approach_2, pick_up_2 = add_grasp(r2, block_id=2)
    release_2 = add_release(r2, held_block_id=2, ref_block_id=1,
                            displacement=np.array([0.10, 0.15, -0.21]))
    graph.structure.add_edge(pick_up_2, release_2, True)

    # r2 task cannot start until r1 has placed its block (+ 3 s margin)
    graph.structure.add_edge(release_0, pick_up_2, True)
    graph.structure.add_edge(release_0, release_2, True)
    graph.add_edge_min_tau_constraint(release_0, release_2, 3.0)

    # --- End nodes: both arms return to their home positions ---
    left_end = graph.structure.add_node()
    graph.structure.add_edge(release_0, left_end, True)
    graph.structure.add_edge(release_2, left_end, True)
    graph.add_robot_pos_linear_eq(k=left_end, robot_id=0,
                                  A=np.eye(3), b=home_pos_0)

    right_end = graph.structure.add_node()
    graph.structure.add_edge(release_0, right_end, True)
    graph.structure.add_edge(release_2, right_end, True)
    graph.add_robot_pos_linear_eq(k=right_end, robot_id=1,
                                  A=np.eye(3), b=home_pos_1)

    # Verify placement with edge constraints; backtrack if cubes have moved
    arranged_phi_0 = graph.add_edge_point_to_point_displacement_constraint(
        u=release_0, v=left_end, point_a=0, point_b=1,
        disp=np.array([-0.10, -0.15, 0.0]),
        tol=np.array([0.05, 0.05, 0.5]))
    arranged_phi_1 = graph.add_edge_point_to_point_displacement_constraint(
        u=release_2, v=left_end, point_a=0, point_b=1,
        disp=np.array([-0.10, -0.15, 0.0]),
        tol=np.array([0.05, 0.05, 0.5]))
    graph.add_manual_backtrack_links(
        arranged_phi_0, [approach_0, pick_up_0, release_0])

    arranged_phi_2 = graph.add_edge_point_to_point_displacement_constraint(
        u=release_2, v=right_end, point_a=2, point_b=0,
        disp=np.array([0.10, 0.15, 0.0]),
        tol=np.array([0.05, 0.05, 0.5]))
    graph.add_manual_backtrack_links(
        arranged_phi_2, [approach_2, pick_up_2, release_2])

    # Keep trying to reach end nodes so backtracking can occur on disturbances
    # graph.make_node_unpassable(left_end)
    # graph.make_node_unpassable(right_end)

    return graph


def build_goc_mpc(graph: GraphOfConstraints):
    spline_spec = [Block.R(3)]
    return GraphOfConstraintsMPC(
        graph,
        spline_spec,
        waypoint_solver=WaypointSolver.kGurobi,
        waypoint_objective=WaypointObjective.kMinMaxL2,
        waypoint_enforce_rigidity=False,
        time_delta_cutoff=0.3,
        short_path_time_per_step=0.1,
        phi_tolerance=0.05,
        linear_interpolation=True,
    )


# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------

def obs_to_state(obs):
    """Flatten the environment observation into (x, x_dot) for goc-mpc."""
    # obs = ((ee0_pos, ee0_vel), (ee1_pos, ee1_vel),
    #         (c0_pos, c0_vel), (c1_pos, c1_vel), (c2_pos, c2_vel))
    x     = np.concatenate([o[0] for o in obs])
    x_dot = np.concatenate([o[1] for o in obs])
    return x, x_dot


def main():
    env = TwoUR5eDiffIKEnv(n_substeps=50)
    env.launch_viewer(port=7000)

    obs, _ = env.reset()
    (home_ee_pos_0, _), (home_ee_pos_1, _) = obs[0], obs[1]
    print(f"Home EE robot 0: {home_ee_pos_0}")
    print(f"Home EE robot 1: {home_ee_pos_1}")

    graph   = build_graph(home_ee_pos_0, home_ee_pos_1)
    goc_mpc = build_goc_mpc(graph)

    input("Press Enter to start...")

    while True:
        obs, _ = env.reset()
        goc_mpc.reset()

        t = 0.0
        xi_h = xi_dot_h = times_h = None

        for k in range(3000):
            x, x_dot = obs_to_state(obs)

            try:
                xi_h, xi_dot_h, times_h = goc_mpc.step(t, x, x_dot)
            except (RuntimeError, TypeError) as e:
                print(f"  goc-mpc step failed: {e}")

            # Apply any grasp commands generated during this step
            if goc_mpc.last_grasp_commands:
                print(f"  Grasp commands: {goc_mpc.last_grasp_commands}")
                env.apply_grasp_commands(goc_mpc.last_grasp_commands)

            if xi_h is None:
                time.sleep(0.01)
                continue

            # xi_h shape: (short_path_length, 6)  — first 3 cols = arm 0, next 3 = arm 1
            target_pos_0 = xi_h[1, 0:3]
            target_pos_1 = xi_h[1, 3:6]
            target_vel_0 = xi_dot_h[1, 0:3] if xi_dot_h is not None else None
            target_vel_1 = xi_dot_h[1, 3:6] if xi_dot_h is not None else None
            time_delta   = times_h[1]

            obs, _rew, _done, _trunc, _info = env.step(
                target_pos_0, target_pos_1, target_vel_0, target_vel_1)

            time.sleep(0.05)
            t += time_delta

            if len(goc_mpc.remaining_phases) == 0:
                print("All phases completed.")
                break

        resp = input("Repeat? [q to quit] ")
        if resp.strip().lower() == "q":
            break

    env.close_viewer()


if __name__ == "__main__":
    main()
