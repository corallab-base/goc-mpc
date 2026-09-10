"""`DpMasterWaypointSolver`: a `waypoint_mpc=`-compatible solver
(`GraphOfConstraintsMPC(graph, splines, waypoint_mpc=...)`) that resolves
the DISCRETE routing / assignment / per-node analytic-IK branch problem
exactly (`dp_master.solve_dp_master`) and then MATERIALISES the waypoint
matrix W from that discrete solution via `apply_projections` -- rather than
searching W continuously the way `EvolutionaryWaypointSolver` does.

Only meaningful for a graph whose W is (mostly) a pushforward of the
discrete solution: every routing-relevant agent config pinned by a
projection (an analytic-IK `ProjOperator`, gated stationary / rigid-carry,
literal / relational object pins). A genuinely-free waypoint column comes
out at its `warm_start_wp` value -- use the evolutionary solver there.

Scope / not-yet:
  * `remaining_vertices` is honoured only coarsely -- the whole graph is
    re-solved every cycle and a passed node keeps its previously written
    row (fine for a deterministic solve on a static scene; a backtrack or a
    moving target needs the finer anchor handling `EvolutionaryWaypointSolver.
    _compute_anchor` already does and `solve_dp_master` does not yet).
  * MILP-style `view_Z` / `view_t` are not provided (nothing in
    GraphOfConstraintsMPC's own waypoint_mpc calls needs them).

The row-layout / anchor helpers below mirror `EvolutionaryWaypointSolver`'s
(evolutionary_waypoint_solver/mpc.py) -- duplicated rather than shared to
keep this experimental solver off that committed path.
"""

import time

import jax.numpy as jnp
import numpy as np

from ..evolutionary_waypoint_solver.problem import AnchorState, apply_anchor, jit_apply_projections
from ..evolutionary_waypoint_solver.spec import (
    _agent_widths, _object_slot_width, _object_widths, _slot_width,
    build_graph_ordering_problem)
from .dp_master import solve_dp_master
from .structure import entry_owner, node_candidates, node_instances, warm_start_wp


class DpMasterWaypointSolver:
    def __init__(self, graph, splines, *, objective="avg", edge_cost_fn=None,
                 wp_bounds=(-10.0, 10.0), **kwargs):
        self._graph = graph
        self._splines = splines
        self._objective = objective if objective in ("avg", "minmax", "makespan") else "avg"
        self._edge_cost_fn = edge_cost_fn
        self._wp_bounds = wp_bounds
        self._dp_kwargs = {k: kwargs.pop(k) for k in
                           ("max_assign_combos", "max_orders", "max_branch_combos")
                           if k in kwargs}
        # "python" (default) -> dp_master.solve_dp_master; "jax" ->
        # dp_master_jax.make_dp_master_jax, the build-once jitted grid solver
        # (needs a jnp-traceable edge_cost_fn). Errors from the jax build
        # propagate -- no silent fallback, so a profiling run fails loudly.
        self._dp_backend = kwargs.pop("dp_backend", "python")
        self._jax_solver = None
        # Evolutionary-shaped kwargs (pop_size / n_gen / outer_iters / ...)
        # ride in via every experiment's WAYPOINT_KWARGS; inert here, ignored
        # rather than raised on (matches how MILP drops the ones it can't use).
        self._ignored_kwargs = sorted(kwargs)

        n_nodes = graph.structure.num_nodes
        agents_width = graph.agent_col_offsets[-1]
        self._waypoints = np.zeros((n_nodes, graph.total_dim))
        self._waypoints[:, agents_width:] = np.nan
        self._assignments = -np.ones(graph.num_phis, dtype=int)
        self._var_assignments = -np.ones(graph.num_variables, dtype=int)
        self._t_by_node_id = np.zeros(n_nodes)
        self._last_solve_time = 0.0
        self._problem = None

    # -- row layout / anchor (mirrors EvolutionaryWaypointSolver) ---------

    def _ensure_built(self, x0):
        if self._problem is not None:
            return
        graph = self._graph
        aw = _agent_widths(graph)
        sw = _slot_width(aw)
        off = graph.agent_col_offsets
        x0_per_agent = np.zeros((graph.num_agents, sw))
        for k in range(graph.num_agents):
            x0_per_agent[k, :aw[k]] = x0[off[k]:off[k + 1]]
        self._agent_widths, self._slot_width = aw, sw
        self._object_widths = _object_widths(graph)
        self._object_slot_width = _object_slot_width(self._object_widths)
        self._agent_offsets = off
        self._object_offsets = graph.object_col_offsets
        self._problem = build_graph_ordering_problem(
            graph, x0_per_agent, self._wp_bounds,
            objective=self._objective, edge_cost_fn=self._edge_cost_fn)
        if self._dp_backend == "jax":
            from .dp_master_jax import make_dp_master_jax
            self._jax_solver = make_dp_master_jax(
                self._problem, objective=self._objective,
                edge_cost_fn=getattr(self._problem, "edge_cost_fn", self._edge_cost_fn),
                **self._dp_kwargs)

    def _to_padded_row(self, packed_row):
        aw, sw = self._agent_widths, self._slot_width
        ow, osw = self._object_widths, self._object_slot_width
        ao, oo = self._agent_offsets, self._object_offsets
        agents_width = len(aw) * sw
        out = np.zeros(agents_width + len(ow) * osw)
        for k, w in enumerate(aw):
            out[k * sw:k * sw + w] = packed_row[ao[k]:ao[k] + w]
        for k, w in enumerate(ow):
            c0 = agents_width + k * osw
            out[c0:c0 + w] = packed_row[oo[k]:oo[k] + w]
        return out

    def _to_packed_row(self, padded_row):
        aw, sw = self._agent_widths, self._slot_width
        ow, osw = self._object_widths, self._object_slot_width
        ao, oo = self._agent_offsets, self._object_offsets
        agents_width = len(aw) * sw
        out = np.zeros(self._graph.total_dim)
        for k, w in enumerate(aw):
            out[ao[k]:ao[k] + w] = padded_row[k * sw:k * sw + w]
        for k, w in enumerate(ow):
            c0 = agents_width + k * osw
            out[oo[k]:oo[k] + w] = padded_row[c0:c0 + w]
        return out

    def _compute_anchor(self, remaining_vertices):
        problem = self._problem
        remaining = set(remaining_vertices)
        node_active = np.array([n in remaining for n in range(problem.n_nodes)], dtype=bool)
        anchor_wp = np.zeros((problem.n_nodes, problem.state_dim))
        for n in range(problem.n_nodes):
            if n not in remaining:
                anchor_wp[n, :] = self._to_padded_row(np.nan_to_num(self._waypoints[n], nan=0.0))
        var_nodes = {}
        for node, (kind, val) in problem.instance_list:
            if kind == "var":
                var_nodes.setdefault(val, []).append(node)
        var_committed = np.zeros(problem.n_variables, dtype=bool)
        var_anchor = np.zeros(problem.n_variables, dtype=np.int32)
        for var_id, slot in problem.var_id_to_slot.items():
            if any(node not in remaining for node in var_nodes.get(var_id, [])):
                var_committed[slot] = True
                var_anchor[slot] = self._var_assignments[var_id]
        return AnchorState(node_active=jnp.asarray(node_active),
                           anchor_wp=jnp.asarray(anchor_wp),
                           var_committed=jnp.asarray(var_committed),
                           var_anchor=jnp.asarray(var_anchor))

    # -- solve ----------------------------------------------------------

    def warmup(self, remaining_vertices, x0):
        """First solve -- pays the one-time build + jit_apply_projections
        compile so the first timed solve() doesn't. Returns the elapsed
        time (drive_loop prints it)."""
        t0 = time.perf_counter()
        self.solve(remaining_vertices, x0)
        return time.perf_counter() - t0

    def solve(self, remaining_vertices, x0) -> bool:
        t0 = time.perf_counter()
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        assert x0.size == self._graph.total_dim
        self._ensure_built(x0)
        problem = self._problem
        remaining = set(remaining_vertices)
        params = np.asarray(self._graph.view_param_values())
        x0_full = self._to_padded_row(x0)
        anchor = self._compute_anchor(remaining_vertices)

        wp_template = warm_start_wp(problem, x0_full)
        dim = problem.dim
        x0_by_agent = {a: np.pad(x0_full[a * dim:a * dim + dim],
                                 (a * dim, problem.state_dim - a * dim - dim))
                       for a in range(problem.n_agents)}

        if self._dp_backend == "jax":
            r = self._jax_solver.run_vec(
                params, wp_template, x0_full, x0_by_agent,
                node_active=np.asarray(anchor.node_active, dtype=bool),
                var_committed=np.asarray(anchor.var_committed, dtype=bool),
                var_anchor=np.asarray(anchor.var_anchor, dtype=int))
        else:
            cands = node_candidates(problem, wp_template, params, allow_unresolved=True,
                                    active_nodes=remaining)
            inst = node_instances(problem)
            r = solve_dp_master(problem, cands, wp_template, x0_by_agent, inst,
                                ordering_edges=problem.ordering_edges,
                                edge_cost_fn=getattr(problem, "edge_cost_fn", None),
                                objective=self._objective, x0_full=x0_full, anchor=anchor,
                                **self._dp_kwargs)
        if r["status"] != "OPTIMAL":
            self._last_solve_time = time.perf_counter() - t0
            return False

        # -- materialise W from the discrete solution --------------------
        assign = np.zeros((1, problem.n_variables, problem.n_agents))
        for slot, agent in r["assignment"].items():
            assign[0, slot, int(agent)] = 1.0
        aux = np.zeros((1, problem.n_cond_vars))
        for k, v in r.get("aux", {}).items():
            aux[0, k] = float(v)
        owner_variable = np.array(
            [int(r["assignment"].get(s, 0)) for s in range(problem.n_variables)], dtype=int)
        proj_branch = np.zeros(problem.n_branch)
        for e in problem.projections:
            if e.discrete_params <= 1:
                continue
            # `branch` is keyed by (node, owner); for a dynamic (var_agent_q)
            # entry the owner is the agent this result's assignment bound its
            # variable to (owner_variable), matching how solve_dp_master wrote it.
            idx = int(r["branch"].get((int(e.write_node), entry_owner(problem, e, owner_variable)), 0))
            proj_branch[e.branch_slice.start + idx] = 1.0
        t_vec = np.array([r["time"].get(n, 0.0) for n in range(problem.n_nodes)], dtype=float)

        W = np.asarray(jit_apply_projections(problem)(
            jnp.asarray(wp_template[None]), jnp.zeros((1, problem.n_psi)),
            jnp.asarray(proj_branch[None]), jnp.asarray(params),
            jnp.asarray(assign), jnp.asarray(aux),
            jnp.asarray(t_vec[None]), anchor.node_active,
            jnp.asarray(x0_full)))[0]
        _, wp_frozen, _ = apply_anchor(problem, jnp.asarray(assign), jnp.asarray(W[None]),
                                       anchor, jnp.asarray(x0_full))
        W = np.asarray(wp_frozen)[0]

        node_rank = np.asarray(problem._decode_node_rank(
            owner_variable, aux[0], t_vec, np.asarray(anchor.node_active)))

        for node in remaining:
            self._waypoints[node, :] = self._to_packed_row(W[node, :])
            self._t_by_node_id[node] = float(node_rank[node])

        for var_id, slot in problem.var_id_to_slot.items():
            if not np.asarray(anchor.var_committed)[slot]:
                self._var_assignments[var_id] = int(r["assignment"].get(slot, 0))
        for phi_id in range(self._graph.num_phis):
            if phi_id in self._graph.phi_to_variable_map:
                self._assignments[phi_id] = self._var_assignments[self._graph.phi_to_variable_map[phi_id]]
            elif phi_id in self._graph.phi_to_static_assignment_map:
                self._assignments[phi_id] = self._graph.phi_to_static_assignment_map[phi_id]
            else:
                self._assignments[phi_id] = -1

        self._last_solve_time = time.perf_counter() - t0
        return True

    # -- views --------------------------------------------------------

    def view_waypoints(self):
        return self._waypoints

    def view_object_waypoints(self):
        return self._waypoints[:, self._graph.agent_col_offsets[-1]:]

    def view_assignments(self):
        return self._assignments

    def view_var_assignments(self):
        return self._var_assignments

    def view_t_by_node(self):
        return self._t_by_node_id

    def get_last_solve_time(self):
        return self._last_solve_time
