#include "repulsive_refiner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>

#include <fmt/format.h>

#include "graph_of_constraints.hpp"

namespace repulsive_refinement {

namespace {

using BlockType = CubicConfigurationSpline::Block::Type;
using spacetime_repulsion::AgentSamples;

double WrapPi(double a) {
	const double two_pi = 2.0 * M_PI;
	double r = std::fmod(a + M_PI, two_pi);
	if (r <= 0.0) r += two_pi;
	return r - M_PI;
}

struct BlockLayout {
	BlockType type;
	int offset, size;
};

// An agent's spline layout, and for every block and base knot interval k the
// knots (lo, hi) of the Hermite piece that bridges k.
struct AgentLayout {
	std::vector<BlockLayout> blocks;
	int dim = 0;
	std::vector<std::vector<int>> lo, hi;  // [block][k]
};

AgentLayout MakeLayout(const CubicConfigurationSpline::Spec& spec, const AgentKnots& knots, int agent) {
	AgentLayout out;
	for (const auto& b : spec) {
		if (b.type != BlockType::R && b.type != BlockType::Torus)
			throw std::invalid_argument(
				fmt::format("RepulsiveRefiner: agent {} has a block other than R or Torus", agent));
		out.blocks.push_back({b.type, out.dim, b.size});
		out.dim += b.size;
	}
	const int n = static_cast<int>(knots.times.size());
	if (n < 2) throw std::invalid_argument(fmt::format("RepulsiveRefiner: agent {} has fewer than 2 knots", agent));
	if (knots.wps.rows() != n || knots.wps.cols() != out.dim || knots.vs.rows() != n || knots.vs.cols() != out.dim)
		throw std::invalid_argument(fmt::format("RepulsiveRefiner: agent {}'s knot arrays don't match its spec", agent));
	const int nb = static_cast<int>(out.blocks.size());
	if (knots.active.size() && (knots.active.rows() != n || knots.active.cols() != nb))
		throw std::invalid_argument(fmt::format("RepulsiveRefiner: agent {}'s active mask is not n x blocks", agent));
	out.lo.assign(nb, std::vector<int>(n - 1));
	out.hi.assign(nb, std::vector<int>(n - 1));
	for (int bi = 0; bi < nb; ++bi) {
		auto active = [&](int i) { return i == 0 || i == n - 1 || !knots.active.size() || knots.active(i, bi) != 0; };
		int last = 0;
		for (int k = 0; k < n - 1; ++k) {
			if (active(k)) last = k;
			out.lo[bi][k] = last;
		}
		int next = n - 1;
		for (int k = n - 2; k >= 0; --k) {
			if (active(k + 1)) next = k + 1;
			out.hi[bi][k] = next;
		}
	}
	return out;
}

struct KnotGradient {
	Eigen::MatrixXd wps, vs;
	Eigen::VectorXd times;
};

// Where a sample sits: in base knot interval k, at fraction u of it; or, for
// k < 0, on the parked tail after the last knot, at fraction u of the way from
// the last knot time to the horizon. Samples move with the knot times.
struct SamplePoint {
	int k;
	double u;
	int m = 1;  // samples in interval k
};

double SampleTime(const AgentKnots& K, const SamplePoint& p, double horizon) {
	const int n = static_cast<int>(K.times.size());
	if (p.k < 0) return K.times(n - 1) + p.u * (horizon - K.times(n - 1));
	return K.times(p.k) + p.u * (K.times(p.k + 1) - K.times(p.k));
}

// q at sample `p`. With `gq` (dE/dq there) and `gt` (dE/d(sample time)), adds
// their pullback to `grad`.
void Sample(const AgentKnots& K, const AgentLayout& L, const SamplePoint& p, double horizon, double* q,
	    const double* gq, double gt, KnotGradient* grad) {
	const int n = static_cast<int>(K.times.size());
	if (p.k < 0 || (p.k == n - 2 && p.u >= 1.0)) {
		for (int d = 0; d < L.dim; ++d) {
			q[d] = K.wps(n - 1, d);
			if (gq) grad->wps(n - 1, d) += gq[d];
		}
		if (gq) grad->times(n - 1) += gt * (p.k < 0 ? 1.0 - p.u : 1.0);
		return;
	}
	const int k = p.k;
	const double t = SampleTime(K, p, horizon);
	double sensitivity = 0.0;  // dE/dq . dq/dt
	for (size_t bi = 0; bi < L.blocks.size(); ++bi) {
		const BlockLayout& b = L.blocks[bi];
		const int s = L.lo[bi][k], e = L.hi[bi][k];
		const double tau = K.times(e) - K.times(s);
		const double u = (t - K.times(s)) / tau;
		const double u2 = u * u, u3 = u2 * u;
		const double h01 = -2.0 * u3 + 3.0 * u2, h10 = u3 - 2.0 * u2 + u, h11 = u3 - u2;
		const double d01 = -6.0 * u2 + 6.0 * u, d10 = 3.0 * u2 - 4.0 * u + 1.0, d11 = 3.0 * u2 - 2.0 * u;
		for (int j = 0; j < b.size; ++j) {
			const int d = b.offset + j;
			const double x0 = K.wps(s, d), x1 = K.wps(e, d), v0 = K.vs(s, d), v1 = K.vs(e, d);
			const double disp = b.type == BlockType::Torus ? WrapPi(x1 - x0) : x1 - x0;
			const double val = x0 + h01 * disp + tau * (h10 * v0 + h11 * v1);
			q[d] = b.type == BlockType::Torus ? WrapPi(val) : val;
			if (!gq) continue;
			const double g = gq[d];
			grad->wps(s, d) += g * (1.0 - h01);
			grad->wps(e, d) += g * h01;
			grad->vs(s, d) += g * tau * h10;
			grad->vs(e, d) += g * tau * h11;
			const double dq_du = d01 * disp + tau * (d10 * v0 + d11 * v1);
			const double dq_dtau = h10 * v0 + h11 * v1;
			// With the sample time held: u and tau move with the piece's knots.
			grad->times(s) += g * (dq_du * (u - 1.0) / tau - dq_dtau);
			grad->times(e) += g * (-dq_du * u / tau + dq_dtau);
			sensitivity += g * dq_du / tau;
		}
	}
	if (gq) {
		// The sample time itself moves with interval k's knots.
		grad->times(k) += (sensitivity + gt) * (1.0 - p.u);
		grad->times(k + 1) += (sensitivity + gt) * p.u;
	}
}

// The speed penalty over agent `K`'s interval samples (see
// Options::speed_weight), each weighted by its share of its interval's
// duration. With `grad`, adds its gradient.
double SpeedPenalty(const AgentKnots& K, const AgentLayout& L, const std::vector<SamplePoint>& points, double horizon,
		    double max_speed, double weight, KnotGradient* grad) {
	double total = 0.0;
	std::vector<double> qd(L.dim), dqd_du(L.dim), dqd_dtau(L.dim);
	for (const SamplePoint& p : points) {
		if (p.k < 0 || p.u >= 1.0) continue;
		const int k = p.k;
		const double t = SampleTime(K, p, horizon);
		const double w = (K.times(k + 1) - K.times(k)) / p.m;
		for (size_t bi = 0; bi < L.blocks.size(); ++bi) {
			const BlockLayout& b = L.blocks[bi];
			const int s = L.lo[bi][k], e = L.hi[bi][k];
			const double tau = K.times(e) - K.times(s);
			const double u = (t - K.times(s)) / tau;
			const double u2 = u * u;
			const double d01 = -6.0 * u2 + 6.0 * u, d10 = 3.0 * u2 - 4.0 * u + 1.0, d11 = 3.0 * u2 - 2.0 * u;
			const double e01 = -12.0 * u + 6.0, e10 = 6.0 * u - 4.0, e11 = 6.0 * u - 2.0;
			double speed2 = 0.0;
			for (int j = 0; j < b.size; ++j) {
				const int d = b.offset + j;
				const double x0 = K.wps(s, d), x1 = K.wps(e, d), v0 = K.vs(s, d), v1 = K.vs(e, d);
				const double disp = b.type == BlockType::Torus ? WrapPi(x1 - x0) : x1 - x0;
				qd[j] = d01 * disp / tau + d10 * v0 + d11 * v1;
				dqd_du[j] = e01 * disp / tau + e10 * v0 + e11 * v1;
				dqd_dtau[j] = -d01 * disp / (tau * tau);
				speed2 += qd[j] * qd[j];
			}
			const double speed = std::sqrt(speed2);
			const double excess = speed - max_speed;
			if (!(excess > 0.0) || speed == 0.0) continue;
			total += weight * w * excess * excess;
			if (!grad) continue;
			// d(penalty)/d(qd_j) = weight * w * 2 excess * qd_j / speed.
			const double c = weight * w * 2.0 * excess / speed;
			double sensitivity = 0.0;  // d/d(sample time)
			for (int j = 0; j < b.size; ++j) {
				const int d = b.offset + j;
				const double g = c * qd[j];
				grad->wps(s, d) -= g * d01 / tau;
				grad->wps(e, d) += g * d01 / tau;
				grad->vs(s, d) += g * d10;
				grad->vs(e, d) += g * d11;
				grad->times(s) += g * (dqd_du[j] * (u - 1.0) / tau - dqd_dtau[j]);
				grad->times(e) += g * (-dqd_du[j] * u / tau + dqd_dtau[j]);
				sensitivity += g * dqd_du[j] / tau;
			}
			grad->times(k) += sensitivity * (1.0 - p.u);
			grad->times(k + 1) += sensitivity * p.u;
			// The weight w moves with interval k's length.
			const double dw = weight * excess * excess / p.m;
			grad->times(k + 1) += dw;
			grad->times(k) -= dw;
		}
	}
	return total;
}

// Keeps every knot gap at least `min_dt` and the last knot at most `upper`,
// moving only knots that break that.
void ProjectTimes(Eigen::VectorXd& t, double min_dt, double upper) {
	double floor = -std::numeric_limits<double>::infinity();
	const int n = static_cast<int>(t.size());
	for (int k = 0; k < n; ++k) {
		const double y = t(k) - min_dt * k;
		floor = std::max(floor, y);
		if (floor > y) t(k) = floor + min_dt * k;
	}
	if (t(n - 1) > upper) {
		t(n - 1) = upper;
		for (int k = n - 2; k >= 1 && t(k) > t(k + 1) - min_dt; --k) t(k) = t(k + 1) - min_dt;
	}
}

// The free knot entries, flattened.
struct Variables {
	struct Entry {
		int agent, kind, row, col;  // kind: 0 wps, 1 vs, 2 times
	};
	std::vector<Entry> entries;
	Eigen::VectorXd metric;

	Variables(const std::vector<AgentKnots>& knots, const Eigen::Vector3d& scale) {
		auto free = [](const Eigen::Matrix<bool, Eigen::Dynamic, 1>& m, int i) { return m.size() > i && m(i); };
		for (int a = 0; a < static_cast<int>(knots.size()); ++a) {
			const AgentKnots& K = knots[a];
			for (int i = 0; i < K.wps.rows(); ++i)
				if (free(K.wps_free, i))
					for (int j = 0; j < K.wps.cols(); ++j) entries.push_back({a, 0, i, j});
			for (int i = 0; i < K.vs.rows(); ++i)
				if (free(K.vs_free, i))
					for (int j = 0; j < K.vs.cols(); ++j) entries.push_back({a, 1, i, j});
			for (int i = 0; i < K.times.size(); ++i)
				if (free(K.times_free, i)) entries.push_back({a, 2, i, 0});
		}
		metric.resize(entries.size());
		for (size_t i = 0; i < entries.size(); ++i) metric(i) = scale(entries[i].kind);
	}

	int size() const { return static_cast<int>(entries.size()); }

	static double& Ref(std::vector<AgentKnots>& knots, const Entry& e) {
		AgentKnots& K = knots[e.agent];
		return e.kind == 0 ? K.wps(e.row, e.col) : e.kind == 1 ? K.vs(e.row, e.col) : K.times(e.row);
	}

	Eigen::VectorXd Pack(const std::vector<AgentKnots>& knots) const {
		Eigen::VectorXd x(size());
		auto& k = const_cast<std::vector<AgentKnots>&>(knots);
		for (int i = 0; i < size(); ++i) x(i) = Ref(k, entries[i]);
		return x;
	}

	void Unpack(const Eigen::VectorXd& x, std::vector<AgentKnots>& knots) const {
		for (int i = 0; i < size(); ++i) Ref(knots, entries[i]) = x(i);
	}

	Eigen::VectorXd PackGradient(const std::vector<KnotGradient>& g) const {
		Eigen::VectorXd out(size());
		for (int i = 0; i < size(); ++i) {
			const Entry& e = entries[i];
			const KnotGradient& G = g[e.agent];
			out(i) = e.kind == 0 ? G.wps(e.row, e.col) : e.kind == 1 ? G.vs(e.row, e.col) : G.times(e.row);
		}
		return out;
	}
};

}  // namespace

Refiner::Refiner(const GraphOfConstraints& graph, const ObstacleSet* obstacles, Options options)
    : _graph(&graph), _obstacles(obstacles), _options(std::move(options)) {
	spacetime_repulsion::CheckKernel(_options.kernel);
	if (!(_options.repulsion_weight > 0.0)) throw std::invalid_argument("RepulsiveRefiner: repulsion_weight must be > 0");
	if (!(_options.sample_dt > 0.0)) throw std::invalid_argument("RepulsiveRefiner: sample_dt must be > 0");
	if (!(_options.max_step > 0.0)) throw std::invalid_argument("RepulsiveRefiner: max_step must be > 0");
	if (_options.speed_weight < 0.0 || _options.max_speed < 0.0)
		throw std::invalid_argument("RepulsiveRefiner: speed_weight and max_speed must be >= 0");
	const int wd = graph.workspace_dim;
	for (int a = 0; a < graph.num_agents; ++a) {
		const int tan = graph.robot_tangent_dim(a), amb = graph.robot_ambient_dim(a);
		const auto spec = graph.agent_collision_specs.find(a);
		if (spec != graph.agent_collision_specs.end()) {
			_models.push_back(sqp_short_path::MakeDrakePlantCollisionModel(spec->second, wd, tan));
		} else {
			_models.push_back(sqp_short_path::MakeTrivialCollisionModel(wd, tan, amb));
		}
		const auto spheres = _models.back()->Eval(Eigen::VectorXd::Zero(amb));
		Eigen::VectorXd r(spheres.size());
		for (size_t k = 0; k < spheres.size(); ++k) r(k) = spheres[k].radius;
		if (_models.back()->is_trivial())
			r.setConstant(_options.agent_radii.size() > a ? _options.agent_radii(a) : 0.0);
		_radii.push_back(r);
	}
}

Refiner::~Refiner() = default;

Report Refiner::Refine(std::vector<AgentKnots>& knots) const { return Run(knots, nullptr, nullptr); }

std::pair<double, Eigen::VectorXd> Refiner::EnergyAndGradient(const std::vector<AgentKnots>& knots) const {
	std::vector<AgentKnots> copy = knots;
	std::pair<double, Eigen::VectorXd> out;
	Run(copy, &out.first, &out.second);
	return out;
}

Report Refiner::Run(std::vector<AgentKnots>& knots, double* energy_out, Eigen::VectorXd* gradient_out) const {
	const auto start = std::chrono::steady_clock::now();
	const int A = _graph->num_agents;
	if (static_cast<int>(knots.size()) != A)
		throw std::invalid_argument(fmt::format("RepulsiveRefiner: {} knot sets for {} agents", knots.size(), A));
	const int wd = _graph->workspace_dim;
	const Options& opt = _options;

	std::vector<AgentLayout> layouts;
	double t_end = 0.0;
	for (int a = 0; a < A; ++a) {
		layouts.push_back(MakeLayout(_graph->_robot_specs[a], knots[a], a));
		t_end = std::max(t_end, knots[a].times(knots[a].times.size() - 1));
	}
	const double horizon = t_end + opt.horizon_margin;
	// Each knot interval gets samples at fixed fractions, about sample_dt
	// apart at the start, then the last knot, then the parked tail.
	std::vector<std::vector<SamplePoint>> points(A);
	for (int a = 0; a < A; ++a) {
		const Eigen::VectorXd& t = knots[a].times;
		const int n = static_cast<int>(t.size());
		for (int k = 0; k + 1 < n; ++k) {
			const int m = std::max(1, static_cast<int>(std::ceil((t(k + 1) - t(k)) / opt.sample_dt)));
			for (int j = 0; j < m; ++j) points[a].push_back({k, static_cast<double>(j) / m, m});
		}
		points[a].push_back({n - 2, 1.0});
		const int m = std::max(1, static_cast<int>(std::ceil((horizon - t(n - 1)) / opt.sample_dt)));
		for (int j = 1; j <= m; ++j) points[a].push_back({-1, static_cast<double>(j) / m});
	}

	Report report;
	std::vector<AgentSamples> samples(A);
	// Per agent and sample, J_k^T for every sphere k (tangent x workspace).
	std::vector<std::vector<Eigen::MatrixXd>> jacobians(A);

	using Clock = std::chrono::steady_clock;
	auto since = [](Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); };
	auto sample_all = [&](const std::vector<AgentKnots>& K, bool with_jac) {
		const auto t0 = Clock::now();
		for (int a = 0; a < A; ++a) {
			const int S = static_cast<int>(_radii[a].size());
			const int T = static_cast<int>(points[a].size());
			samples[a].times.resize(T);
			samples[a].centers.resize(T * S, wd);
			samples[a].radii = _radii[a];
			if (with_jac) jacobians[a].resize(T * S);
			Eigen::VectorXd q(layouts[a].dim);
			for (int s = 0; s < T; ++s) {
				samples[a].times(s) = SampleTime(K[a], points[a][s], horizon);
				Sample(K[a], layouts[a], points[a][s], horizon, q.data(), nullptr, 0.0, nullptr);
				const auto spheres = _models[a]->Eval(q);
				for (int k = 0; k < S; ++k) {
					samples[a].centers.row(s * S + k) = spheres[k].center.head(wd).transpose();
					if (with_jac) jacobians[a][s * S + k] = spheres[k].jac.transpose();
				}
			}
		}
		report.time_sampling += since(t0);
	};

	const double obstacle_ratio = opt.obstacle_weight / opt.repulsion_weight;
	// `resample` false reuses samples already taken at K (with Jacobians when
	// a gradient is asked for).
	auto energy = [&](const std::vector<AgentKnots>& K, std::vector<KnotGradient>* grad, int* pairs,
			  bool resample = true) {
		if (resample) sample_all(K, grad != nullptr);
		++report.evaluations;
		auto t0 = Clock::now();
		const auto r = spacetime_repulsion::Evaluate(samples, _obstacles, opt.kernel, obstacle_ratio, grad != nullptr);
		report.time_energy += since(t0);
		t0 = Clock::now();
		if (pairs) *pairs = r.pairs;
		double speed = 0.0;
		if (opt.speed_weight > 0.0)
			for (int a = 0; a < A; ++a)
				speed += SpeedPenalty(K[a], layouts[a], points[a], horizon, opt.max_speed, opt.speed_weight, nullptr);
		if (grad) {
			grad->assign(A, KnotGradient{});
			for (int a = 0; a < A; ++a) {
				const AgentKnots& Ka = K[a];
				KnotGradient& G = (*grad)[a];
				G.wps = Eigen::MatrixXd::Zero(Ka.wps.rows(), Ka.wps.cols());
				G.vs = Eigen::MatrixXd::Zero(Ka.vs.rows(), Ka.vs.cols());
				G.times = Eigen::VectorXd::Zero(Ka.times.size());
				const int S = static_cast<int>(_radii[a].size());
				const int T = static_cast<int>(points[a].size());
				Eigen::VectorXd q(layouts[a].dim), gq(layouts[a].dim);
				for (int s = 0; s < T; ++s) {
					gq.setZero();
					for (int k = 0; k < S; ++k)
						gq += opt.repulsion_weight * jacobians[a][s * S + k] * r.gradient[a].row(s * S + k).transpose();
					const double gt = opt.repulsion_weight * r.time_gradient[a](s);
					if (gq.isZero(0.0) && gt == 0.0) continue;
					Sample(Ka, layouts[a], points[a][s], horizon, q.data(), gq.data(), gt, &G);
				}
				if (opt.speed_weight > 0.0)
					SpeedPenalty(Ka, layouts[a], points[a], horizon, opt.max_speed, opt.speed_weight, &G);
			}
		}
		report.time_pullback += since(t0);
		return opt.repulsion_weight * r.inter_agent + opt.obstacle_weight * r.obstacle + speed;
	};
	auto clearance = [&]() {
		const auto t0 = Clock::now();
		const auto c = spacetime_repulsion::MeasureClearance(samples, _obstacles);
		report.time_clearance += since(t0);
		return c;
	};
	auto clear = [&](const spacetime_repulsion::Clearance& c) {
		return c.inter_agent >= opt.target_clearance && c.obstacle >= opt.target_clearance;
	};

	const Variables vars(knots, opt.metric_scale);
	std::vector<AgentKnots> work = knots;
	std::vector<KnotGradient> kg;
	// Samples with Jacobians: the clearance check needs only the samples, and
	// the first gradient reuses them.
	sample_all(work, true);
	report.clearance_initial = clearance();
	auto finish = [&](const std::string& reason, double E_final, bool resample) {
		report.energy_final = E_final;
		if (resample) sample_all(work, false);
		report.clearance_final = resample ? clearance() : report.clearance_initial;
		report.stop_reason = reason;
		report.time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
		return report;
	};
	if (!energy_out && (vars.size() == 0 || clear(report.clearance_initial))) {
		// Nothing to do: the energy isn't evaluated.
		report.energy_initial = std::numeric_limits<double>::quiet_NaN();
		return finish(vars.size() == 0 ? "no free knots" : "clear", report.energy_initial, false);
	}
	double E = energy(work, &kg, &report.pairs_initial, false);
	report.energy_initial = E;
	report.pairs_final = report.pairs_initial;
	if (energy_out) {
		*energy_out = E;
		*gradient_out = vars.PackGradient(kg);
		return finish("evaluated", E, false);
	}
	if (E == 0.0) return finish("zero energy", E, false);

	Eigen::VectorXd x = vars.Pack(work), g = vars.PackGradient(kg);
	const Eigen::VectorXd x_start = x;
	std::deque<std::pair<Eigen::VectorXd, Eigen::VectorXd>> history;  // (s, y)

	auto apply = [&](const Eigen::VectorXd& xv, std::vector<AgentKnots>& K) {
		vars.Unpack(xv, K);
		for (AgentKnots& Ka : K) ProjectTimes(Ka.times, opt.min_knot_dt, horizon - opt.min_knot_dt);
	};

	std::string reason = "max iterations";
	std::vector<AgentKnots> trial = work;
	for (int it = 0; it < opt.max_iterations; ++it) {
		// L-BFGS direction under the metric.
		Eigen::VectorXd d = g;
		std::vector<double> alphas(history.size());
		for (int i = static_cast<int>(history.size()) - 1; i >= 0; --i) {
			const auto& [s, y] = history[i];
			alphas[i] = s.dot(d) / y.dot(s);
			d -= alphas[i] * y;
		}
		double gamma = 1.0;
		if (!history.empty()) {
			const auto& [s, y] = history.back();
			gamma = s.dot(y) / y.dot(vars.metric.cwiseProduct(y));
		}
		d = gamma * vars.metric.cwiseProduct(d);
		for (size_t i = 0; i < history.size(); ++i) {
			const auto& [s, y] = history[i];
			const double beta = y.dot(d) / y.dot(s);
			d += s * (alphas[i] - beta);
		}
		d = -d;
		double slope = g.dot(d);
		if (!(slope < 0.0)) {
			history.clear();
			d = -vars.metric.cwiseProduct(g);
			slope = g.dot(d);
		}
		const double largest = d.cwiseAbs().maxCoeff();
		double alpha = std::min(1.0, opt.max_step / std::max(largest, 1e-300));
		bool accepted = false;
		double E_trial = E;
		Eigen::VectorXd x_trial;
		for (int bt = 0; bt <= opt.max_backtracks; ++bt) {
			apply(x + alpha * d, trial);
			x_trial = vars.Pack(trial);
			E_trial = energy(trial, nullptr, nullptr);
			if (std::isfinite(E_trial) && E_trial <= E + opt.armijo * alpha * slope) {
				accepted = true;
				break;
			}
			alpha *= 0.5;
		}
		if (!accepted) {
			if (!history.empty()) {
				history.clear();
				continue;
			}
			reason = "line search failed";
			break;
		}
		work = trial;
		std::vector<KnotGradient> kg_new;
		E = energy(work, &kg_new, &report.pairs_final);
		const Eigen::VectorXd g_new = vars.PackGradient(kg_new);
		const Eigen::VectorXd s = x_trial - x, y = g_new - g;
		if (s.dot(y) > 1e-12 * s.norm() * y.norm()) {
			history.emplace_back(s, y);
			if (static_cast<int>(history.size()) > opt.memory) history.pop_front();
		}
		x = x_trial;
		g = g_new;
		report.iterations = it + 1;
		if (clear(clearance())) {
			reason = "clear";
			break;
		}
		if (E == 0.0) {
			reason = "zero energy";
			break;
		}
	}
	report.moved = (x - x_start).cwiseAbs().maxCoeff() > 0.0;
	bool finite = true;
	for (const AgentKnots& K : work)
		finite = finite && K.wps.allFinite() && K.vs.allFinite() && K.times.allFinite();
	if (!finite) {
		work = knots;
		E = report.energy_initial;
		report.moved = false;
		reason = "non-finite result";
	}
	knots = work;
	return finish(reason, E, true);
}

}  // namespace repulsive_refinement
