"""ProjOperator: an analytic-elimination hint for one symbolic node/edge
constraint (GraphOfConstraints.add_constraint/add_edge_constraint's `proj=`
kwarg) -- lets the evolutionary waypoint solver (spec.py's
_resolve_projections) replace that constraint's soft AL residual with an
EXACT substitution (analytic IK, a fixed grasp offset, ...) instead of
searching its pinned columns continuously under a penalty. See spec.py's
module docstring and _resolve_projections' own docstring for how this is
compiled and applied.

Not core to GraphOfConstraints -- the C++ side only ever carries this as an
opaque Python object (see graph_of_constraints.hpp's phi_to_projection doc
comment). MILPWaypointSolver and any other consumer never look at it, and
keep compiling the constraint's Formula as an ordinary residual regardless
of whether a proj is attached."""

import dataclasses
from collections.abc import Callable


@dataclasses.dataclass(frozen=True)
class ProjOperator:
    """
    pins: the placeholder array this projection ASSIGNS -- e.g.
        `graph.agent_q(0)[0:3]`, or the whole `graph.agent_q(0)`. Every
        component must be a plain, static agent_q(k)/object_q(k) placeholder
        (u_/v_-prefixed for an edge constraint's pin side) -- the same
        static-column case _make_row_resolver already resolves for an
        ordinary constraint's placeholders; an FK or var_agent_q(...)
        placeholder can't be pinned this way (there's no decision-variable
        column to write a value into).

    reads: placeholder arrays `func` reads, in call order -- e.g.
        `(graph.object_q(0)[0:3],)`. Each may be a static agent_q/object_q
        slice, an agent_link_pos/_rot FK call, or a param(id) -- anything
        the ordinary row_resolver already knows how to read. `pins` union
        every array in `reads` must cover every free variable the
        constraint's Formula references; add_constraint/add_edge_constraint
        raise otherwise, so no placeholder is ever silently unaccounted for.

        A `reads` array MAY reference a column another projection `pins`:
        _resolve_projections orders the projections so the writer runs
        first, and apply_projections threads the running wp through, so
        `func` sees the already-substituted value (e.g. an analytic-IK
        projection reading an object column a preceding grasp/stationary
        projection just pinned). A dependency cycle between two projections
        raises -- that is an implicit relation, not an elimination.

        Leave `reads=()` when `func` needs no row input at all -- e.g. a
        Pick target that's a literal, spec-build-time-fixed pose baked
        directly into `func`'s own closure. This is also the ONLY case
        (together with continuous_params=0) `_resolve_projections` can
        precompute into a plain lookup table at spec-build time rather than
        calling `func` on every batched pass -- see its docstring.

    continuous_params: how many extra free reals `func` needs beyond
        `reads` (e.g. a redundant arm's self-motion parameter). 0 for an
        exactly-determined system -- e.g. UR5e's 6-DOF closed-form IK has no
        leftover continuous freedom, only a discrete branch choice.

    discrete_params: cardinality of a single flat branch selector `func` is
        evaluated at (e.g. 8 for UR5e's IK branches). 1 if there's no
        discrete choice at all (a trivial elimination, e.g. a fixed grasp
        offset with a unique solution).

    psi_bounds: (lo, hi) box for the continuous_params block, in `func`'s
        own units. Ignored when continuous_params == 0.

    func: `(*read_values, psi, branch) -> value`, all UNBATCHED (a single
        population member) -- _resolve_projections vmaps it over the
        population itself, the same convention every other compiled
        constraint in this module follows (see spec.py's
        _batch_symbolic_constraint_fn). `read_values` are plain arrays
        matching each `reads` entry's shape, in order; `psi` is a
        `(continuous_params,)` array; `branch` is a scalar int32 in
        `[0, discrete_params)`; the return matches `pins`' flat width.
        Built from ordinary jax.numpy (or, when `reads=()` and
        continuous_params == 0, plain numpy -- see `reads`' docstring)
        array ops -- no side effects, since local refinement differentiates
        through it whenever it isn't tabled.
    """
    pins: object
    reads: tuple = ()
    continuous_params: int = 0
    discrete_params: int = 1
    psi_bounds: tuple = (-1.0, 1.0)
    func: Callable = None

    def __post_init__(self):
        if self.continuous_params < 0:
            raise ValueError("ProjOperator.continuous_params must be >= 0")
        if self.discrete_params < 1:
            raise ValueError(
                "ProjOperator.discrete_params must be >= 1 (1 means no discrete "
                "branch choice, e.g. a trivial single-solution elimination)")
        if self.func is None:
            raise ValueError("ProjOperator.func is required")
