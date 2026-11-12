#pragma once

#include "collision_checker.hpp"
#include "initialization_oracle.hpp"

#include <map>
#include <memory>

// An initialization oracle that solves the graph's constraints by
// propagation, one assignment at a time.
//
// For an assignment, every remaining node's constraints, the holds and the
// objects' stationarity become relations over scalar unknowns: each
// object's state components, each visiting agent's configuration
// components and its FK links' pose components (position, then the
// row-major rotation) at each node. Affine equalities and single-unknown
// inequalities (bounds) are read; anything else is only checked. Then,
// until nothing is left to set:
//   1. an equality with one unknown left sets it;
//   2. an agent at a node with one FK link's full pose known and an
//      IkSolver gets one of that solver's configurations that is
//      collision-free (collision_checker, against the environment and the
//      agents already placed at the node) -- a branch choice, nearest the
//      agent's x0 first -- and its other links follow by FK; every
//      collision-free solution goes to the guess's candidates;
//   3. otherwise the earliest node's object still entangled in an
//      equality gets a region choice for its unknown components: its
//      previous-guess or x0 value clamped a little inside their bounds,
//      then `samples` Halton points in those bounds, nearest that value
//      first; failing that, an agent whose links are entangled is set to its
//      reference (the previous guess there, else x0).
// Whatever is still unset takes its reference value.
//
// A failure -- no IK solution, or every branch colliding -- is traced to the
// choices it was derived from, and the search backjumps
// (conflict-directed): the most recent such choice takes its next option
// (region options in parallel batches of `threads`), an exhausted one passes
// its accumulated conflict on to the next most recent. A failure derived
// from no choice and through exact relations only, with a complete IK
// solver, or one left after exhausting only branch choices, proves its
// (variable, agent) pairs infeasible together; those become cuts.
//
// Assignments are enumerated in preference order (previous solve's agent,
// then the previous guess's, then by agent id) within the graph's candidate,
// committed and inequality restrictions and one role per agent per node, up
// to max_assignments; the first one whose guess violates nothing is
// returned, else the least violated.
class PropagationOracle : public InitializationOracle {
public:
	// ik_solvers per agent id (an agent without one is never set by IK);
	// collision_checker null: nothing collides. propagations bounds the runs
	// per assignment. threads > 1 needs C++ (GIL-free) IK solvers and
	// checker.
	explicit PropagationOracle(std::map<int, std::shared_ptr<IkSolver>> ik_solvers,
				   std::shared_ptr<CollisionChecker> collision_checker = nullptr, int max_assignments = 64,
				   int samples = 64, int propagations = 1024, int threads = 1, double tolerance = 1e-6);

	InitialGuess Initialize(const InitializationRequest& request) const override;

	const std::map<int, std::shared_ptr<IkSolver>>& ik_solvers() const { return _ik_solvers; }
	const std::shared_ptr<CollisionChecker>& collision_checker() const { return _collision_checker; }
	int max_assignments() const { return _max_assignments; }
	int samples() const { return _samples; }
	int propagations() const { return _propagations; }
	int threads() const { return _threads; }

private:
	std::map<int, std::shared_ptr<IkSolver>> _ik_solvers;
	std::shared_ptr<CollisionChecker> _collision_checker;
	int _max_assignments, _samples, _propagations, _threads;
	// Equalities hold to within this; a static hold's to its 1e-3 band.
	double _tolerance;
};
