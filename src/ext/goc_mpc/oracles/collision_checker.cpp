#include "collision_checker.hpp"

#include <Eigen/Geometry>
#include <fmt/format.h>

#include "../graph_of_constraints.hpp"
#include "vamp_environment.hpp"

struct VampCollisionChecker::Shapes {
	std::vector<VampEnvironment::Box> boxes;
	std::vector<VampEnvironment::Ball> balls;
};

namespace {

std::array<float, 3> F3(const Eigen::Vector3d& v) {
	return {static_cast<float>(v.x()), static_cast<float>(v.y()), static_cast<float>(v.z())};
}

}  // namespace

VampCollisionChecker::VampCollisionChecker(const ObstacleSet& obstacles, double inflation,
					   std::map<int, std::string> robots)
	: _shapes(std::make_unique<Shapes>()), _inflation(inflation), _robots(std::move(robots)) {
	const auto& all = obstacles.obstacles();
	for (size_t i = 0; i < all.size(); ++i) {
		const Obstacle& o = all[i];
		const auto f = [&](int k) { return static_cast<float>(o.params(k)); };
		const int n = static_cast<int>(o.params.size());
		// A sphere grown by `inflation` reaches as far as the shape grown by it.
		const float margin = static_cast<float>(o.margin + inflation);
		if (o.kind == ObstacleKind::kSphere && n == 4)
			_shapes->balls.push_back({{f(0), f(1), f(2)}, f(3), margin, fmt::format("obstacle {} (sphere)", i)});
		else if (o.kind == ObstacleKind::kBox && n == 6)
			_shapes->boxes.push_back(
				{{f(0), f(1), f(2)}, {f(3), f(4), f(5)}, margin, fmt::format("obstacle {} (box)", i)});
		else
			throw std::invalid_argument("VampCollisionChecker: obstacles must be 3D spheres and boxes");
	}
	for (const auto& [agent, name] : _robots) VampRobot::Make(name);  // throws for an unknown name
}

VampCollisionChecker::~VampCollisionChecker() = default;

std::vector<VampCollisionChecker::Sphere> VampCollisionChecker::Spheres(const GraphOfConstraints& graph, int agent,
									  const Eigen::VectorXd& q) const {
	const auto spec = graph.agent_collision_specs.find(agent);
	if (spec == graph.agent_collision_specs.end()) return {};
	const std::pair<const GraphOfConstraints*, int> key{&graph, agent};
	std::unique_ptr<sqp_short_path::AgentCollisionModel> model;
	{
		std::lock_guard<std::mutex> lock(_mutex);
		auto& idle = _pool[key];
		if (!idle.empty()) {
			model = std::move(idle.back());
			idle.pop_back();
		}
	}
	if (!model)
		model = sqp_short_path::MakeDrakePlantCollisionModel(spec->second, 3, graph.robot_tangent_dim(agent));
	std::vector<Sphere> out;
	const auto spheres = model->Eval(q);
	for (size_t k = 0; k < spheres.size(); ++k) out.push_back({spheres[k].center, spheres[k].radius, static_cast<int>(k)});
	std::lock_guard<std::mutex> lock(_mutex);
	_pool[key].push_back(std::move(model));
	return out;
}

const VampCollisionChecker::Placed& VampCollisionChecker::Place(const GraphOfConstraints& graph, int agent) const {
	const std::pair<const GraphOfConstraints*, int> key{&graph, agent};
	{
		std::lock_guard<std::mutex> lock(_mutex);
		const auto it = _placed.find(key);
		if (it != _placed.end()) return it->second;
	}
	const auto spec = graph.agent_collision_specs.find(agent);
	if (spec == graph.agent_collision_specs.end())
		throw std::invalid_argument(fmt::format("VampCollisionChecker: agent {} has a VAMP robot but no collision spec", agent));
	Placed placed;
	const std::string& name = _robots.at(agent);
	placed.robot = VampRobot::Make(name);
	const Eigen::Vector4d& w = spec->second.base_quaternion_wxyz;
	const Eigen::Matrix3d R = Eigen::Quaterniond(w(0), w(1), w(2), w(3)).normalized().toRotationMatrix().transpose();
	const Eigen::Vector3d t = -R * spec->second.base_translation;
	auto frame = std::make_shared<VampFrame>();
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) frame->rotation[3 * i + j] = static_cast<float>(R(i, j));
		frame->translation[i] = static_cast<float>(t(i));
	}
	placed.base_from_world = frame;

	// The kernel's spheres must be the spec's: match them at a few
	// configurations, by radius and base-frame center.
	const int d = placed.robot->dimension();
	if (graph.robot_tangent_dim(agent) != d)
		throw std::invalid_argument(fmt::format("VampCollisionChecker: agent {} has {} joints, robot \"{}\" {}", agent,
							graph.robot_tangent_dim(agent), name, d));
	const std::vector<Eigen::VectorXd> probes = {Eigen::VectorXd::Zero(d), Eigen::VectorXd::LinSpaced(d, -1.3, 0.9),
						     Eigen::VectorXd::LinSpaced(d, 2.1, -0.4)};
	for (size_t p = 0; p < probes.size(); ++p) {
		const auto spec_spheres = Spheres(graph, agent, probes[p]);
		const std::vector<float> q(probes[p].data(), probes[p].data() + d);
		const auto kernel_spheres = placed.robot->Spheres(q);
		if (spec_spheres.size() != kernel_spheres.size())
			throw std::invalid_argument(fmt::format("VampCollisionChecker: robot \"{}\" has {} spheres, agent {}'s spec {}",
								name, kernel_spheres.size(), agent, spec_spheres.size()));
		std::vector<int> match(kernel_spheres.size(), -1);
		for (size_t i = 0; i < kernel_spheres.size(); ++i) {
			const Eigen::Vector3d c(kernel_spheres[i][0], kernel_spheres[i][1], kernel_spheres[i][2]);
			for (const Sphere& s : spec_spheres)
				if (std::abs(s.radius - kernel_spheres[i][3]) < 1e-5 &&
				    (R * s.center + t - c).norm() < 1e-4 &&
				    std::find(match.begin(), match.end(), s.index) == match.end()) {
					match[i] = s.index;
					break;
				}
			if (match[i] < 0)
				throw std::invalid_argument(
					fmt::format("VampCollisionChecker: robot \"{}\" sphere {} matches none of agent {}'s spec spheres "
						    "(regenerate the robot from the spec's sphere set)",
						    name, i, agent));
		}
		if (p == 0) placed.spec_index = match;
		else if (match != placed.spec_index)
			throw std::invalid_argument(fmt::format("VampCollisionChecker: robot \"{}\" spheres follow different links "
								"than agent {}'s spec",
								name, agent));
	}
	std::lock_guard<std::mutex> lock(_mutex);
	return _placed.emplace(key, std::move(placed)).first->second;
}

std::vector<CollisionResult> VampCollisionChecker::CheckBatch(const GraphOfConstraints& graph, int agent,
							      const std::vector<Eigen::VectorXd>& configurations,
							      const CollisionScene& scene) const {
	if (graph.workspace_dim != 3) throw std::runtime_error("VampCollisionChecker: needs a 3D workspace");
	auto body = [&](int a, int k) {
		const auto& spheres = graph.agent_collision_specs.at(a).spheres;
		return k >= 0 && k < static_cast<int>(spheres.size()) ? spheres[k].body : std::string("?");
	};
	// The scene agents' spheres join the obstacles' balls.
	std::vector<VampEnvironment::Ball> balls = _shapes->balls;
	for (const auto& [other, q] : scene.others)
		for (const Sphere& t : Spheres(graph, other, q))
			balls.push_back({F3(t.center), static_cast<float>(t.radius + 2 * _inflation), 0.0f,
					 fmt::format("agent {} {}", other, body(other, t.index))});
	std::vector<CollisionResult> results(configurations.size());

	if (_robots.count(agent)) {
		const Placed& placed = Place(graph, agent);
		const VampEnvironment environment(_shapes->boxes, balls, *placed.base_from_world);
		const int d = placed.robot->dimension();
		std::vector<float> flat;
		flat.reserve(configurations.size() * d);
		for (const Eigen::VectorXd& q : configurations)
			for (int j = 0; j < d; ++j) flat.push_back(static_cast<float>(q(j)));
		const auto checks = placed.robot->Check(environment, flat);
		for (size_t c = 0; c < checks.size(); ++c) {
			if (checks[c].free) continue;
			const std::string what = body(agent, placed.spec_index.at(checks[c].sphere));
			results[c] = {false, checks[c].hit.empty()
						     ? fmt::format("agent {} {} vs {} (self)", agent, what,
								   body(agent, placed.spec_index.at(checks[c].other)))
						     : fmt::format("agent {} {} vs {}", agent, what, checks[c].hit)};
		}
		return results;
	}

	const VampEnvironment environment(_shapes->boxes, balls);
	std::vector<float> x, y, z, r;
	std::vector<std::pair<int, int>> owner;  // (configuration, sphere index)
	for (size_t c = 0; c < configurations.size(); ++c)
		for (const Sphere& s : Spheres(graph, agent, configurations[c])) {
			const auto p = F3(s.center);
			x.push_back(p[0]), y.push_back(p[1]), z.push_back(p[2]);
			r.push_back(static_cast<float>(s.radius));
			owner.emplace_back(static_cast<int>(c), s.index);
		}
	if (x.empty()) return results;
	const auto hits = environment.FirstHits(x, y, z, r);
	for (size_t i = 0; i < x.size(); ++i) {
		CollisionResult& result = results[owner[i].first];
		if (result.free && !hits[i].empty())
			result = {false, fmt::format("agent {} {} vs {}", agent, body(agent, owner[i].second), hits[i])};
	}
	return results;
}
