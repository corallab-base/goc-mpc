"""Validation script for `goc_mpc.agent_link_names`/`agent_workspace_tracks`
(po_goc_mpc's `draw_mpc_overlay` refactor motivated these -- see that
package's `experiments/drive_loop.py`): confirms that splitting a flat
multi-agent row array and resolving each agent's own workspace point(s)
through `GraphOfConstraints.link_pose` gives DISTINCT, individually correct
points per (agent, link) -- in particular that a single agent with a
registered "" (body) fk_fn AND a registered "hand" fk_fn gets two genuinely
different Cartesian tracks out of the SAME row array, not the same value
twice or a garbage row-slice -- and that it accepts BOTH row widths a real
caller hands it: `graph.total_dim` (agent + object columns, e.g.
`WaypointMPC.view_waypoints()`) and `graph.agent_col_offsets[-1]`
(agent-only, e.g. `GraphOfConstraintsMPC.step`'s own returned `xi_h`, which
carries no object columns even when the graph has objects).

2 agents + 1 object:
  agent 0: [Block.R(2), Block.Torus(1)]  -- a plain (x, y, yaw) mobile base;
    no fk_fn registered, so link_pose("") falls back to the built-in
    RobotKind (kPosYaw) dispatch -- exercises the "no registration needed"
    common case agent_link_names/agent_workspace_tracks are meant to cover
    for free.
  agent 1: [Block.R(2), Block.Torus(1), Block.R(3), Block.SO3Quat()] -- a
    G1-with-active-arm-shaped agent (base x, y, yaw + a relative hand
    pos/quat tail). Registers BOTH "" (a trivial (x, y, STANDING_Z) body fk,
    mirroring po_goc_mpc's build_base_fk) and "hand" (a trivial "relative
    hand pos, offset into world" fk) -- the row-slicing fallback would be
    wrong here (this agent's leading 3 columns are (x, y, yaw), not a
    Cartesian point), which is exactly why both links need registering.
  object 0: [Block.R(2)] -- a plain 2D point, present ONLY so
    graph.agent_col_offsets[-1] < graph.total_dim (otherwise the two row
    widths this script means to distinguish would coincide).

Run directly: python examples/test_agent_workspace_tracks.py
"""

import numpy as np

from goc_mpc import GraphOfConstraints
from goc_mpc import agent_link_names, agent_workspace_tracks
from goc_mpc._ext.configuration_spline import Block

STANDING_Z = 0.8
N_ROWS = 5


def body_fk(q):
    """(x, y, yaw, *_) -> (pos=(x, y, STANDING_Z), rot=identity) -- same
    shape of trick po_goc_mpc's build_base_fk uses for a real G1."""
    x, y = q[0], q[1]
    return np.array([x, y, STANDING_Z]), np.eye(3)


def hand_fk(q):
    """(x, y, yaw, hand_pos_rel(3), *_) -> hand_pos_rel offset by the base
    (x, y) -- deliberately simple (no real rotation composition), just
    needs to be a DIFFERENT function of the row than body_fk above."""
    x, y = q[0], q[1]
    hand_rel = q[3:6]
    return np.array([x, y, 0.0]) + hand_rel, np.eye(3)


def build_graph() -> GraphOfConstraints:
    agent0_spec = [Block.R(2), Block.Torus(1)]
    agent1_spec = [Block.R(2), Block.Torus(1), Block.R(3), Block.SO3Quat()]
    object_spec = [Block.R(2)]
    graph = GraphOfConstraints(
        [agent0_spec, agent1_spec], [object_spec],
        state_lower_bound=-20.0, state_upper_bound=20.0,
        robot_names=["base_only", "base_plus_arm"], object_names=["target_point"],
    )
    graph.set_robot_fk(1, "", body_fk)
    graph.set_robot_fk(1, "hand", hand_fk)
    return graph


def test_agent_link_names():
    graph = build_graph()
    assert agent_link_names(graph, 0) == [""]
    assert agent_link_names(graph, 1) == ["", "hand"]
    print("agent_link_names: OK")


def test_agent_workspace_tracks():
    graph = build_graph()
    agent_width = graph.agent_col_offsets[-1]
    assert agent_width < graph.total_dim, "test graph needs a real object to exercise both row widths"

    rng = np.random.default_rng(0)
    full_rows = rng.uniform(-2.0, 2.0, size=(N_ROWS, graph.total_dim))
    agent_only_rows = full_rows[:, :agent_width]  # e.g. xi_h -- no object columns

    # agent 0: no registration -- exercises the built-in fallback, just
    # needs to run without raising and return the right shape/keys, and
    # agree between the full-width and agent-only-width call forms (object
    # columns are irrelevant to agent 0's own pose either way).
    tracks0_full = agent_workspace_tracks(graph, full_rows, 0)
    tracks0_agent_only = agent_workspace_tracks(graph, agent_only_rows, 0)
    assert list(tracks0_full.keys()) == [""]
    assert tracks0_full[""].shape == (N_ROWS, graph.workspace_dim)
    assert np.allclose(tracks0_full[""], tracks0_agent_only[""])

    # agent 1: both registered links should come back, each matching its
    # OWN registered fk_fn evaluated independently against agent 1's own
    # row slice -- and the two should be genuinely different points, not a
    # duplicated body-only value. Checked against BOTH row widths.
    for rows in (full_rows, agent_only_rows):
        tracks1 = agent_workspace_tracks(graph, rows, 1)
        assert list(tracks1.keys()) == ["", "hand"]

        lo, hi = graph.agent_col_offsets[1], graph.agent_col_offsets[2]
        for i, row in enumerate(rows):
            q1 = row[lo:hi]
            expected_body, _ = body_fk(q1)
            expected_hand, _ = hand_fk(q1)
            assert np.allclose(tracks1[""][i], expected_body, atol=1e-9)
            assert np.allclose(tracks1["hand"][i], expected_hand, atol=1e-9)
            assert not np.allclose(tracks1[""][i], tracks1["hand"][i]), \
                "body and hand tracks must not coincide -- they're different fk_fns"

    print("agent_workspace_tracks: OK "
          f"({N_ROWS} rows, both full-width and agent-only-width, "
          "agent 1 body/hand tracks distinct and correct)")


def test_width_mismatch_raises():
    graph = build_graph()
    agent_width = graph.agent_col_offsets[-1]
    # total_dim - 1 is neither total_dim nor agent_width (object_spec is
    # 2-wide here, so the two valid widths are >= 2 apart).
    bad_rows = np.zeros((3, graph.total_dim - 1))
    assert bad_rows.shape[1] != agent_width
    try:
        agent_workspace_tracks(graph, bad_rows, 0)
    except ValueError:
        print("width-mismatch ValueError: OK")
        return
    raise AssertionError("expected ValueError for a rows width that is neither valid width")


if __name__ == "__main__":
    test_agent_link_names()
    test_agent_workspace_tracks()
    test_width_mismatch_raises()
    print("All agent_workspace_tracks checks passed.")
