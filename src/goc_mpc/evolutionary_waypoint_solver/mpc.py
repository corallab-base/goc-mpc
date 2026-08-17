"""EvolutionaryWaypointSolver: a GraphWaypointMPC-compatible (duck-typed)
wrapper around GraphOrderingSpec + solver.run_lamarckian_al.
`GraphOfConstraintsMPC` (goc_mpc.py) accepts any object satisfying this
protocol via its `waypoint_mpc=` constructor argument, with no isinstance
check, so no changes are needed there.

Output buffer shapes/conventions match GraphWaypointMPC's exactly:
  _waypoints:       (num_nodes, graph.total_dim) -- agent columns followed by
                     object columns, one row per node, mirroring
                     MILP's per-node joint configuration `W`
                     (graph_of_constraints.cpp:79's `total_dim`). view_
                     waypoints() keeps returning this whole row (matching
                     GraphWaypointMPC's contract, which callers already slice
                     down to the agent columns themselves -- see goc_mpc.py's
                     step()); view_object_waypoints() returns the object-
                     column slice as a convenience view. A node not yet
                     covered by any solve() call (still at construction-time
                     init) reads as NaN; every node in remaining_vertices as
                     of the last solve() call carries that solve's real,
                     GA-solved value, whether or not any constraint
                     referenced it there (e.g. an un-held object's value at
                     a node tied to it only via the auto-generated "stay
                     stationary" invariant, GraphOrderingSpec._resolve_
                     stationary_objects, not a NaN sentinel).
  _assignments:     (num_phis,)
  _var_assignments: (num_variables,)
  _t_by_node_id:    (num_graph_nodes,)
so GraphOfConstraintsMPC.step()/_solve_for_waypoints/_solve_for_timing
consume it identically to MILPWaypointMPC.

The underlying JAX kernel models a full per-node configuration (every
agent's AND every referenced object's position, one row per node -- see
spec.py's module docstring), the same shape of information MILPWaypointMPC's
joint `W` carries, just solved via GA+AL instead of MILP.

Build once, reuse forever, remaining_vertices as a runtime input
------------------------------------------------------------------
Unlike MILPWaypointMPC (which rebuilds a SubgraphOfConstraints fresh every
solve from remaining_vertices), GraphOrderingSpec/GraphOrderingRelaxed now
span the WHOLE graph unconditionally (see spec.py's class docstring) --
their shape (n_var, n_eq_constr, n_ieq_constr, ...) is fixed for the
lifetime of a GraphOfConstraints, independent of remaining_vertices. So
`_ensure_built` constructs `_spec`/`_problem`/`_ga_fn` exactly ONCE, on the
first solve()/warmup() call, and every later call reuses them unconditionally
-- no cache-key/shape-key bookkeeping, since the shape can never mismatch
again. `add_python_constraint` must be called before that first call (raises
otherwise): constraints are baked into `_problem`'s fixed structure once,
not re-applied per solve.

remaining_vertices' actual effect is now a per-call runtime input instead of
a structural one: `_compute_anchor` builds a `problem.AnchorState` from
remaining_vertices plus this solver's own persisted `_waypoints`/
`_var_assignments` (a passed node's row, or a committed variable's
assignment, is exactly the last value THIS solver wrote there while it was
still active, and is left untouched by the write-back loops below once it's
no longer in remaining_vertices -- the same "frozen last-committed value"
MILP substitutes via `previous_X.row(u_node)`). `problem.apply_anchor`/
kernel.py's `node_active` argument then splice that in wherever a constraint
or routing decision reads a node's position or a variable's owner: a node/
edge constraint anchored at an already-passed node stays correctly enforced
against that known constant instead of silently disappearing (the bug this
design fixes -- see spec.py's module docstring), while ordering edges
touching a passed node are (correctly) dropped, since precedence against an
already-visited node is moot.

A passed node's "known constant" is one of TWO choices, not one: `anchor_wp`
above (the frozen, last-committed-while-active value -- "the planned
position") or this call's real state (the full-configuration `x0` given to
solve()/warmup(), read straight off by problem.apply_anchor -- "the live
position"). Which one a given EDGE constraint's `u`-side residual reads is a
per-constraint choice made where the constraint is defined
(GraphOfConstraints.add_edge_constraint(..., live=True) --
graph_of_constraints.hpp's docstring) -- GraphOrderingSpec reads
`graph.live_edge_phis` straight off the graph itself, not something this
class threads through as a constructor argument (there is nothing for a
caller here to legitimately override; see GraphOrderingSpec.__init__'s own
docstring comment), nor something this class rebinds or tracks per node --
there is no "rewrite
this row because the node just passed" step anywhere here, deliberately: a
constraint that wants the mobile real state (e.g. a rigid-transport edge,
where the *real* carried-object position should keep tracking the *real*
carrying-agent position even after the grasp node's nominal pass point, not
the grasp node's originally-planned position) should be registered live;
anything else defaults to frozen, matching `main` branch's
DeferredEdgeOp.waypoint_builder x_u argument (previous_X.row(u_node)) with no
artificial rebind hack layered on top. Node constraints have no live/frozen
choice to make at all (graph_of_constraints.hpp's live_edge_phis comment) --
GraphOrderingSpec._resolve_symbolic_constraints hardcodes "frozen" for every
one of them.

Because AnchorState is a fixed-shape pytree (n_nodes/n_variables never
change), passing a different `anchor` value on each call -- like `x0` -- adds
no retracing, only a cheap-in-Python `_compute_anchor` call. Population reuse
(`_carry`) is therefore unconditional after the first build too: `solve()`
always resumes from the previous call's population/AL-state via
`carry_from_population`, which re-evaluates F/CV/best fresh under the
*current* x0 and anchor before continuing (a single cheap batched forward
pass, not a full n_gen search). Call `warmup()` once up front to pay the
first trace/compile outside of a timed run and seed `_carry` from that run's
own result.
"""

import time

import jax
import numpy as np
import jax.numpy as jnp

from .spec import GraphOrderingSpec
from .problem import AnchorState
from .solver import (
    run_lamarckian_al,
    build_lamarckian_ga,
    build_initial_carry_fn,
    carry_from_population,
)

# Params that only affect *building the initial carry* (a fresh random
# population, or re-wrapping a reused one) -- as opposed to the GA stepper
# built by build_lamarckian_ga, which no longer takes them. Split out of
# **lamarckian_kwargs in __init__ so each downstream call only ever receives
# the subset of kwargs the function it's calling actually accepts.
_CARRY_KWARGS = ("rho0", "n_seed_individuals", "seed_jitter_t", "seed_jitter_wp_frac")


class EvolutionaryWaypointSolver:
    def __init__(self, graph, splines, objective="avg", edge_cost_fn=None,
                 wp_bounds=(-10.0, 10.0), pop_size=30, n_gen=60,
                 **lamarckian_kwargs):
        self._graph = graph
        self._splines = splines
        self._objective = objective
        self._edge_cost_fn = edge_cost_fn
        self._wp_bounds = wp_bounds
        self._pop_size = pop_size
        self._n_gen = n_gen
        # `seed` selects the PRNGKey for a from-scratch initial carry (only
        # used when no previous population is available to resume from).
        self._seed = lamarckian_kwargs.pop("seed", 1)
        self._carry_kwargs = {k: lamarckian_kwargs.pop(k) for k in _CARRY_KWARGS
                               if k in lamarckian_kwargs}
        self._lamarckian_kwargs = lamarckian_kwargs
        self._python_constraints = []  # (node, fn, kind, name)

        num_nodes = graph.structure.num_nodes
        agents_width = graph.num_agents * graph.dim
        # One row per node -- agent columns then object columns,
        # mirroring MILP's per-node joint configuration `W` (see module
        # docstring). Object columns start NaN (never-referenced sentinel,
        # see view_object_waypoints()); agent columns start 0, overwritten
        # with a real depot value before ever being read (solve()'s
        # per-active-node reset, the only writer -- see module docstring).
        self._waypoints = np.zeros((num_nodes, graph.total_dim))
        self._waypoints[:, agents_width:] = np.nan
        self._assignments = -np.ones(graph.num_phis, dtype=int)
        self._var_assignments = -np.ones(graph.num_variables, dtype=int)
        self._t_by_node_id = np.zeros(num_nodes)
        self._last_solve_time = 0.0

        # Built once, on the first solve()/warmup() call -- see module
        # docstring -- and never rebuilt again.
        self._spec = None
        self._problem = None
        self._ga_fn = None
        self._carry = None
        self._last_solve_was_warm = False
        self._last_population_reused = False
        self._last_compile_time = 0.0

    def add_python_constraint(self, node, fn, kind="eq", name=None):
        """Registers fn(wp_row) -> (k,) on `node`, baked into `_problem`'s
        structure the first time it's built (the first solve()/warmup()
        call). Must be called before that first call -- structure is built
        once and reused for this solver's whole lifetime (see module
        docstring), so there's no later point at which a newly-registered
        constraint could take effect."""
        if self._spec is not None:
            raise RuntimeError(
                "add_python_constraint() called after the first solve()/warmup() -- "
                "structure is built once, from every constraint registered up to that "
                "point, and reused for the solver's whole lifetime; register all "
                "python constraints before the first solve()/warmup() call")
        self._python_constraints.append((node, fn, kind, name))

    def _agent_depot(self, x0):
        """Slices `x0` down to the agent-only depot prefix (num_agents*dim)
        this solver's kernel/routing machinery expects. `x0` may already be
        exactly that size (e.g. goc-mpc/examples/test_evolutionary_object_
        constraints.py's direct solver.solve() calls) or may be the full
        graph.total_dim state -- agents followed by objects -- that
        GraphOfConstraintsMPC.step() always passes (it asserts x.size ==
        graph.total_dim before calling waypoint_mpc.solve()/warmup()).
        Object positions are never a depot concept for this solver: they're
        determined purely by symbolic node/edge constraints (see spec.py),
        so any trailing object-state in `x0` is simply not needed here."""
        graph = self._graph
        return np.asarray(x0)[: graph.num_agents * graph.dim]

    def _object_depot(self, x0):
        """Slices `x0` down to the object-state suffix (num_objects *
        non_robot_dim) when `x0` is the full graph.total_dim vector
        GraphOfConstraintsMPC.step() passes (agents then objects -- see
        _agent_depot). Returns None when `x0` is only agent-depot-sized
        (e.g. a direct low-level solve()/warmup() caller with no object
        state to give), since there's then no real object state to report."""
        graph = self._graph
        x0 = np.asarray(x0)
        agents_width = graph.num_agents * graph.dim
        objects_width = graph.num_objects * graph.non_robot_dim
        if objects_width > 0 and x0.size == agents_width + objects_width:
            return x0[agents_width:agents_width + objects_width]
        return None

    def _full_x0(self, x0):
        """Builds the full-configuration runtime x0 (graph.total_dim,) this
        solver threads through solver.py/problem.py (agent_depot, apply_anchor):
        the agent depot always, plus the object depot when the caller
        supplied one (see _object_depot). Zero-filled -- never NaN, to keep
        the JAX-traced path free of NaNs that could otherwise poison
        autodiff even through masked-out branches -- for any object column
        no x0 this solver has ever seen covers (e.g. a direct low-level
        caller that only ever passes an agent-depot-sized x0, see
        examples/test_evolutionary_object_constraints.py)."""
        graph = self._graph
        agents_width = graph.num_agents * graph.dim
        row = np.zeros(graph.total_dim)
        row[:agents_width] = self._agent_depot(x0)
        obj_flat = self._object_depot(x0)
        if obj_flat is not None:
            row[agents_width:] = obj_flat
        return row

    def _ensure_built(self, x0_flat):
        if self._spec is not None:
            return
        graph = self._graph
        x0_per_agent = np.asarray(x0_flat).reshape(graph.num_agents, graph.dim)

        spec = GraphOrderingSpec.from_graph_of_constraints(
            graph, x0_per_agent, self._wp_bounds,
            objective=self._objective, edge_cost_fn=self._edge_cost_fn)
        for node, fn, kind, name in self._python_constraints:
            spec.add_python_constraint(node, fn, kind=kind, name=name)
        problem = spec.build_problem()

        self._spec = spec
        self._problem = problem
        self._ga_fn = build_lamarckian_ga(problem, self._pop_size, self._n_gen, **self._lamarckian_kwargs)

    def _compute_anchor(self, remaining_vertices):
        """Builds this solve()/warmup() call's AnchorState from
        remaining_vertices plus this solver's own persisted _waypoints/
        _var_assignments -- see module docstring. Cheap (small numpy loops
        over spec's static, graph-global structure), not part of any
        JIT-traced path. Only ever populates anchor_wp -- the "frozen,
        planned" reading; the "live, real state" reading is `x0` itself,
        supplied straight to problem.apply_anchor by the caller (solve()/
        warmup()), not something this method builds."""
        graph, spec = self._graph, self._spec
        remaining_set = set(remaining_vertices)

        node_active = np.array([node in remaining_set for node in spec._node_list], dtype=bool)

        anchor_wp = np.zeros((spec.n_nodes, spec.state_dim))
        for node in spec._node_list:
            if node in remaining_set:
                continue
            anchor_wp[node, :] = np.nan_to_num(self._waypoints[node], nan=0.0)

        var_nodes = {}
        for node, (kind, val) in spec._instance_list:
            if kind == "var":
                var_nodes.setdefault(val, []).append(node)

        var_committed = np.zeros(spec.n_variables, dtype=bool)
        var_anchor = np.zeros(spec.n_variables, dtype=np.int32)
        for var_id, slot in spec._var_id_to_slot.items():
            nodes = var_nodes.get(var_id, [])
            if any(node not in remaining_set for node in nodes):
                assert self._var_assignments[var_id] != -1, (
                    f"variable {var_id} has an already-passed instance but was never assigned")
                var_committed[slot] = True
                var_anchor[slot] = self._var_assignments[var_id]

        return AnchorState(
            node_active=jnp.asarray(node_active),
            anchor_wp=jnp.asarray(anchor_wp),
            var_committed=jnp.asarray(var_committed),
            var_anchor=jnp.asarray(var_anchor),
        )

    def warmup(self, remaining_vertices, x0):
        """Builds (on the first call) and JIT-compiles the GA (against the
        given x0/remaining_vertices, though the compiled GA remains valid
        for any x0/remaining_vertices afterwards -- see module docstring)
        and seeds `_carry` with the resulting (already optimized)
        population, so this cost is paid once up front instead of on the
        first timed solve() -- which then also starts from a genuinely warm
        population rather than a random one. Returns the compile time."""
        x0_flat = self._agent_depot(x0)
        self._ensure_built(x0_flat)
        x0_arr = jnp.asarray(self._full_x0(x0))
        anchor = self._compute_anchor(remaining_vertices)

        init_fn = build_initial_carry_fn(
            self._problem, self._pop_size, anchor, **self._carry_kwargs)

        start = time.perf_counter()
        carry_in = init_fn(jax.random.PRNGKey(self._seed))
        carry_out = self._ga_fn(carry_in, x0_arr, anchor)
        jax.block_until_ready(carry_out)
        compile_time = time.perf_counter() - start

        self._carry = carry_out
        self._last_compile_time = compile_time
        return compile_time

    def solve(self, remaining_vertices, x0) -> bool:
        start = time.perf_counter()
        graph = self._graph
        x0_flat = self._agent_depot(x0)
        was_already_built = self._spec is not None
        self._ensure_built(x0_flat)
        spec, problem, ga_fn = self._spec, self._problem, self._ga_fn
        x0_arr = jnp.asarray(self._full_x0(x0))
        remaining_set = set(remaining_vertices)

        anchor = self._compute_anchor(remaining_vertices)
        # numpy-side copies for the write-back loops below (cheap, small
        # arrays -- avoids repeated jnp->python scalar coercion per instance).
        var_committed_np = np.asarray(anchor.var_committed)
        var_anchor_np = np.asarray(anchor.var_anchor)

        self._last_solve_was_warm = was_already_built

        if self._carry is not None:
            # Resume the previous solve's population *and* AL multiplier
            # state (mu/lam/rho) instead of cold-random-initializing and
            # resetting the constraint penalty from scratch -- see
            # carry_from_population's docstring. F/CV/best are always
            # freshly re-evaluated under the *current* x0/anchor, since the
            # cached carry's own were scored under whatever was current on
            # the call that produced them.
            prev_X, prev_mu, prev_lam, prev_rho, prev_key = (
                self._carry[0], self._carry[1], self._carry[2], self._carry[3], self._carry[6])
            init_carry = carry_from_population(
                problem, prev_X, x0_arr, prev_key, anchor, mu=prev_mu, lam=prev_lam, rho=prev_rho,
                pop_size=self._pop_size, **self._carry_kwargs_for("rho0"))
            self._last_population_reused = True
        else:
            init_carry = None
            self._last_population_reused = False

        result = run_lamarckian_al(problem, anchor, pop_size=self._pop_size, n_gen=self._n_gen,
                                    seed=self._seed, _ga_fn=ga_fn, _init_carry=init_carry,
                                    x0=x0_arr, **self._lamarckian_kwargs, **self._carry_kwargs)

        self._carry = result.pop

        assign, _cond_binary, t, wp = problem._extract_single(np.asarray(result.X))
        wp = np.asarray(wp)
        owner_variable = (np.argmax(np.asarray(assign), axis=-1) if spec.n_variables > 0
                          else np.zeros(0, dtype=int))

        # Copy the whole solved row through unconditionally for every
        # remaining node -- agent AND object columns alike, exactly like
        # MILPWaypointMPC.view_waypoints() (which likewise always returns
        # its whole solved W row, not filtered by which agent/object "owns"
        # each node). Downstream readers only ever index a node's row for an
        # agent/object a constraint or graph.get_agent_paths actually routes
        # through, so a column at a node that never references it is simply
        # never read, whatever it holds. Restricted to remaining_vertices
        # (spec._var_id_to_slot's write-back below is too, for the same
        # reason) so an already-passed node's frozen anchor is never
        # overwritten with this solve's masked-out/inert GA output. A node
        # id is directly its own row index into wp/t (spec.py gives every
        # graph node a row, matching MILP), so no node-id -> row-index
        # conversion is needed anywhere below.
        for node in remaining_set:
            self._waypoints[node, :] = wp[node, :]

        # Report the DECODED visiting-order rank, not the raw priority `t`:
        # `t` only carries a meaningful order after kernel.py's topological
        # decode (see kernel.py's module docstring) -- writing back the raw
        # value here would silently reintroduce the exact bug that decoder
        # exists to fix, one level up (GraphOfConstraints.get_agent_paths
        # sorts nodes by view_t_by_node() too, and needs the same
        # already-topologically-valid order the route-cost computation
        # actually used). One value per node (kernel.py), so no
        # per-instance collapsing is needed here -- just index it directly.
        node_rank = np.asarray(problem._decode_node_rank(
            owner_variable, np.asarray(_cond_binary), t, np.asarray(anchor.node_active)))
        for node in remaining_set:
            self._t_by_node_id[node] = float(node_rank[node])

        for var_id, slot in spec._var_id_to_slot.items():
            if var_committed_np[slot]:
                continue  # already correctly frozen; don't overwrite with this solve's inert output
            self._var_assignments[var_id] = int(owner_variable[slot])

        for phi_id in range(graph.num_phis):
            if phi_id in graph.phi_to_variable_map:
                var_id = graph.phi_to_variable_map[phi_id]
                self._assignments[phi_id] = self._var_assignments[var_id]
            elif phi_id in graph.phi_to_static_assignment_map:
                self._assignments[phi_id] = graph.phi_to_static_assignment_map[phi_id]
            else:
                self._assignments[phi_id] = -1

        self._last_solve_time = time.perf_counter() - start
        return True

    def _carry_kwargs_for(self, *names):
        return {k: self._carry_kwargs[k] for k in names if k in self._carry_kwargs}

    def view_waypoints(self):
        return self._waypoints

    def view_object_waypoints(self):
        """(num_nodes, num_objects * non_robot_dim) solved object positions --
        a view onto _waypoints' object-column slice (see view_waypoints()),
        matching its node-indexed convention. NaN for any node not yet
        covered by any solve() call; every node in the last solve()'s
        remaining_vertices carries a real, GA-solved value regardless of
        whether any constraint referenced it there."""
        agents_width = self._graph.num_agents * self._graph.dim
        return self._waypoints[:, agents_width:]

    def view_assignments(self):
        return self._assignments

    def view_var_assignments(self):
        return self._var_assignments

    def view_t_by_node(self):
        return self._t_by_node_id

    def get_last_solve_time(self):
        return self._last_solve_time

    def was_last_solve_warm(self):
        """True if the last solve() reused the already-compiled GA (i.e. it
        wasn't the first solve()/warmup() call ever made on this solver
        instance -- x0/remaining_vertices may differ freely either way,
        since the compiled GA is valid for any of either, see module
        docstring)."""
        return self._last_solve_was_warm

    def was_last_population_reused(self):
        """True if the last solve() started from a previous carry's
        population (refreshed under the current x0/anchor via
        carry_from_population) rather than a cold random one."""
        return self._last_population_reused

    def get_last_compile_time(self):
        """Compile time from the last warmup() call (0.0 if warmup() was
        never called)."""
        return self._last_compile_time
