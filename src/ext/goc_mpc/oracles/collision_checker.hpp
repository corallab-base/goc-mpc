#pragma once

#include <Eigen/Core>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "../agent_collision_model.hpp"
#include "../obstacle_set.hpp"

struct GraphOfConstraints;

// Collision checking for an initialization oracle: whether an agent's
// configuration at a node collides with the environment or with the other
// agents already placed there.

// The other agents at the node: (agent, configuration).
struct CollisionScene {
	int node = -1;
	std::vector<std::pair<int, Eigen::VectorXd>> others;
};

struct CollisionResult {
	bool free = true;
	// What it hit, when not free.
	std::string hit;
};

class CollisionChecker {
public:
	virtual ~CollisionChecker() = default;

	// One result per configuration of `agent`, each against the scene.
	// Called concurrently from several threads when the oracle runs with
	// threads > 1.
	virtual std::vector<CollisionResult> CheckBatch(const GraphOfConstraints& graph, int agent,
							const std::vector<Eigen::VectorXd>& configurations,
							const CollisionScene& scene) const = 0;
};

class VampEnvironment;
class VampRobot;
struct VampFrame;

// Collision checks through VAMP. An agent named in `robots` (agent -> a
// compiled VAMP robot, see VampRobot::Names()) is checked by that robot's
// generated kernel in its base frame (the pose its agent_collision_spec
// welds it at): its spheres against the obstacle set's spheres and boxes,
// each grown by its margin, and against the scene agents' spheres, plus its
// own self-collision. That robot's spheres must be the spec's (checked on
// first use). Any other agent with a spec has its spheres -- through the
// same Drake-plant sphere model GraphShortPathMPC uses -- queried against
// the same shapes, without self-collision. An agent with neither has no
// spheres and never collides. `inflation` grows every sphere.
class VampCollisionChecker : public CollisionChecker {
public:
	// Copies the obstacle set's spheres and boxes.
	explicit VampCollisionChecker(const ObstacleSet& obstacles, double inflation = 0.0,
				      std::map<int, std::string> robots = {});
	~VampCollisionChecker() override;

	std::vector<CollisionResult> CheckBatch(const GraphOfConstraints& graph, int agent,
						const std::vector<Eigen::VectorXd>& configurations,
						const CollisionScene& scene) const override;

	double inflation() const { return _inflation; }
	const std::map<int, std::string>& robots() const { return _robots; }

private:
	struct Sphere {
		Eigen::Vector3d center;
		double radius;
		int index;
	};
	// A VAMP robot placed for one (graph, agent).
	struct Placed {
		std::shared_ptr<const VampRobot> robot;
		std::shared_ptr<const VampFrame> base_from_world;
		// The spec sphere each kernel sphere is.
		std::vector<int> spec_index;
	};
	std::vector<Sphere> Spheres(const GraphOfConstraints& graph, int agent, const Eigen::VectorXd& q) const;
	const Placed& Place(const GraphOfConstraints& graph, int agent) const;

	struct Shapes;
	std::unique_ptr<Shapes> _shapes;
	double _inflation;
	std::map<int, std::string> _robots;
	mutable std::mutex _mutex;
	// Idle sphere models per (graph, agent): a model's Eval isn't
	// reentrant, so each concurrent caller takes its own.
	mutable std::map<std::pair<const GraphOfConstraints*, int>,
			 std::vector<std::unique_ptr<sqp_short_path::AgentCollisionModel>>>
		_pool;
	mutable std::map<std::pair<const GraphOfConstraints*, int>, Placed> _placed;
};
