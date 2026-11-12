#include "vamp_environment.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

#include <vamp/collision/environment.hh>
#include <vamp/collision/factory.hh>
#include <vamp/collision/validity.hh>
#include <vamp/vector.hh>

#include "vamp_robots/ur5e_2f85.hh"

namespace {

using Lanes = vamp::FloatVector<>;
constexpr std::size_t kLanes = vamp::FloatVectorWidth;
using Vec3 = std::array<float, 3>;

Vec3 Apply(const VampFrame& f, const Vec3& p) {
	Vec3 out;
	for (int i = 0; i < 3; ++i)
		out[i] = f.rotation[3 * i] * p[0] + f.rotation[3 * i + 1] * p[1] + f.rotation[3 * i + 2] * p[2] +
			 f.translation[i];
	return out;
}

// A box (world-axis-aligned, center c and half extents h) in frame f.
vamp::collision::Cuboid<float> FrameBox(const VampFrame& f, const Vec3& c, const Vec3& h, const std::string& name) {
	const Vec3 p = Apply(f, c);
	const auto& R = f.rotation;  // column k: world axis k in the frame
	vamp::collision::Cuboid<float> box(p[0], p[1], p[2], R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8], h[0],
					   h[1], h[2]);
	box.name = name;
	return box;
}

// Shapes sharing one margin: a query sphere's radius is grown by it.
struct MarginGroup {
	float margin;
	vamp::collision::Environment<Lanes> environment;
};

}  // namespace

struct VampEnvironment::Impl {
	// For FirstHits: the shapes as given, grouped by margin.
	std::vector<MarginGroup> groups;
	// For the robot kernels, which take no margins: each shape grown by its
	// margin exactly (a box's Minkowski sum with a ball is three slabs, a
	// capsule per edge and a ball per corner).
	vamp::collision::Environment<Lanes> grown;
};

VampEnvironment::VampEnvironment(const std::vector<Box>& boxes, const std::vector<Ball>& balls, const VampFrame& frame)
	: _impl(std::make_unique<Impl>()) {
	std::map<float, vamp::collision::Environment<float>> by_margin;
	vamp::collision::Environment<float> grown;
	for (const Box& b : boxes) {
		by_margin[b.margin].cuboids.push_back(FrameBox(frame, b.center, b.half_extents, b.name));
		const float m = b.margin;
		const Vec3& h = b.half_extents;
		if (m <= 0.0f) {
			grown.cuboids.push_back(FrameBox(frame, b.center, h, b.name));
			continue;
		}
		for (int k = 0; k < 3; ++k) {
			Vec3 slab = h;
			slab[k] += m;
			grown.cuboids.push_back(FrameBox(frame, b.center, slab, b.name));
		}
		for (int k = 0; k < 3; ++k) {  // edges along axis k
			const int i = (k + 1) % 3, j = (k + 2) % 3;
			for (float si : {-1.0f, 1.0f})
				for (float sj : {-1.0f, 1.0f}) {
					Vec3 a = b.center, e = b.center;
					a[i] = e[i] = b.center[i] + si * h[i];
					a[j] = e[j] = b.center[j] + sj * h[j];
					a[k] -= h[k], e[k] += h[k];
					const Vec3 pa = Apply(frame, a), pe = Apply(frame, e);
					auto edge = vamp::collision::factory::capsule::endpoints::flat(pa[0], pa[1], pa[2], pe[0],
												      pe[1], pe[2], m);
					edge.name = b.name;
					grown.capsules.push_back(edge);
				}
		}
		for (float sx : {-1.0f, 1.0f})
			for (float sy : {-1.0f, 1.0f})
				for (float sz : {-1.0f, 1.0f}) {
					const Vec3 p = Apply(frame, {b.center[0] + sx * h[0], b.center[1] + sy * h[1],
								     b.center[2] + sz * h[2]});
					vamp::collision::Sphere<float> corner(p[0], p[1], p[2], m);
					corner.name = b.name;
					grown.spheres.push_back(corner);
				}
	}
	for (const Ball& s : balls) {
		const Vec3 p = Apply(frame, s.center);
		vamp::collision::Sphere<float> sphere(p[0], p[1], p[2], s.radius);
		sphere.name = s.name;
		by_margin[s.margin].spheres.push_back(sphere);
		vamp::collision::Sphere<float> big(p[0], p[1], p[2], s.radius + s.margin);
		big.name = s.name;
		grown.spheres.push_back(big);
	}
	for (auto& [margin, environment] : by_margin) {
		environment.sort();
		_impl->groups.push_back({margin, vamp::collision::Environment<Lanes>(environment)});
	}
	grown.sort();
	_impl->grown = vamp::collision::Environment<Lanes>(grown);
}

VampEnvironment::~VampEnvironment() = default;
VampEnvironment::VampEnvironment(VampEnvironment&&) noexcept = default;
VampEnvironment& VampEnvironment::operator=(VampEnvironment&&) noexcept = default;

std::vector<std::string> VampEnvironment::FirstHits(const std::vector<float>& x, const std::vector<float>& y,
						    const std::vector<float>& z, const std::vector<float>& r) const {
	const std::size_t n = x.size();
	std::vector<std::string> hits(n);
	for (std::size_t begin = 0; begin < n; begin += kLanes) {
		// One query sphere per lane; a short last chunk repeats its last sphere.
		alignas(Lanes::S::Alignment) std::array<float, kLanes> lx, ly, lz, lr;
		for (std::size_t k = 0; k < kLanes; ++k) {
			const std::size_t i = std::min(begin + k, n - 1);
			lx[k] = x[i], ly[k] = y[i], lz[k] = z[i], lr[k] = r[i];
		}
		const Lanes vx(lx.data(), true), vy(ly.data(), true), vz(lz.data(), true), vr(lr.data(), true);
		for (const MarginGroup& group : _impl->groups) {
			if (!vamp::sphere_environment_in_collision(group.environment, vx, vy, vz, vr + group.margin)) continue;
			// Some lane hit: find which, one sphere broadcast to every lane.
			for (std::size_t k = 0; k < kLanes && begin + k < n; ++k) {
				if (!hits[begin + k].empty()) continue;
				const auto names = vamp::sphere_environment_get_collisions(group.environment, Lanes::fill(lx[k]),
											    Lanes::fill(ly[k]), Lanes::fill(lz[k]),
											    Lanes::fill(lr[k] + group.margin));
				if (!names.empty()) hits[begin + k] = names.front();
			}
		}
	}
	return hits;
}

const void* VampRobot::KernelEnvironment(const VampEnvironment& environment) { return &environment._impl->grown; }

namespace {

template <typename Robot>
class Kernel final : public VampRobot {
public:
	using Block = typename Robot::template ConfigurationBlock<kLanes>;

	int dimension() const override { return static_cast<int>(Robot::dimension); }

	std::vector<std::array<float, 4>> Spheres(const std::vector<float>& q) const override {
		Block block;
		for (std::size_t j = 0; j < Robot::dimension; ++j) block[j] = Lanes::fill(q.at(j));
		typename Robot::template Spheres<kLanes> out;
		Robot::template sphere_fk<kLanes>(block, out);
		std::vector<std::array<float, 4>> spheres(Robot::n_spheres);
		for (std::size_t i = 0; i < Robot::n_spheres; ++i)
			spheres[i] = {out.x[i].to_array()[0], out.y[i].to_array()[0], out.z[i].to_array()[0],
				      out.r[i].to_array()[0]};
		return spheres;
	}

	std::vector<Result> Check(const VampEnvironment& environment,
				  const std::vector<float>& configurations) const override {
		const auto& env = *static_cast<const vamp::collision::Environment<Lanes>*>(KernelEnvironment(environment));
		const std::size_t d = Robot::dimension, n = configurations.size() / d;
		std::vector<Result> results(n);
		for (std::size_t begin = 0; begin < n; begin += kLanes) {
			// One configuration per lane; a short last chunk repeats its last.
			Block block;
			for (std::size_t j = 0; j < d; ++j) {
				alignas(Lanes::S::Alignment) std::array<float, kLanes> lane;
				for (std::size_t k = 0; k < kLanes; ++k) lane[k] = configurations[std::min(begin + k, n - 1) * d + j];
				block[j] = Lanes(lane.data(), true);
			}
			if (Robot::template fkcc<kLanes>(env, block)) continue;
			for (std::size_t c = begin; c < std::min(begin + kLanes, n); ++c) {
				Block one;
				for (std::size_t j = 0; j < d; ++j) one[j] = Lanes::fill(configurations[c * d + j]);
				if (Robot::template fkcc<kLanes>(env, one)) continue;
				Result& result = results[c];
				result.free = false;
				const auto debug = Robot::template fkcc_debug<kLanes>(env, one);
				for (std::size_t s = 0; s < debug.first.size() && result.sphere < 0; ++s)
					if (!debug.first[s].empty()) result.sphere = static_cast<int>(s), result.hit = debug.first[s].front();
				if (result.sphere < 0 && !debug.second.empty())
					result.sphere = static_cast<int>(debug.second.front().first),
					result.other = static_cast<int>(debug.second.front().second);
			}
		}
		return results;
	}
};

}  // namespace

std::vector<std::string> VampRobot::Names() { return {"ur5e_2f85"}; }

std::unique_ptr<VampRobot> VampRobot::Make(const std::string& name) {
	if (name == "ur5e_2f85") return std::make_unique<Kernel<vamp::robots::UR5e2F85>>();
	throw std::invalid_argument("VampRobot: no compiled robot \"" + name + "\"");
}
