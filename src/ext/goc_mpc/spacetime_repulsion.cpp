#include "spacetime_repulsion.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace spacetime_repulsion {

namespace {

constexpr int kMaxDim = 4;  // workspace_dim + 1
constexpr double kEps = 1e-12;

// The windowed power, and its first two derivatives, at gap g.
void WindowedPower(double g, double p, double c, double* f, double* f1, double* f2) {
	const double w = 1.0 - g * g / (c * c);
	const double gp = std::pow(g, -p);
	*f = gp * w * w;
	*f1 = -p * gp / g * w * w - 4.0 * gp * g * w / (c * c);
	if (f2) {
		const double a1 = p * (p + 1.0) * gp / (g * g) * w * w + 4.0 * p * gp * w / (c * c);
		const double b1 = -4.0 * (1.0 - p) * gp * w / (c * c) + 8.0 * gp * g * g / (c * c * c * c);
		*f2 = a1 + b1;
	}
}

struct Sdf {
	double value;
	double grad[3];
};

// Signed distance from workspace point `x` (dim entries) to an obstacle,
// minus its margin, with the gradient: the same sphere/box distances the
// short-path solver uses.
Sdf ObstacleSdf(const double* x, int dim, const Obstacle& o) {
	Sdf out{0.0, {0.0, 0.0, 0.0}};
	const Eigen::VectorXd& prm = o.params;
	if (o.kind == ObstacleKind::kSphere) {
		double d2 = 0.0;
		for (int i = 0; i < dim; ++i) d2 += (x[i] - prm(i)) * (x[i] - prm(i));
		const double d = std::sqrt(d2 + 1e-8);
		for (int i = 0; i < dim; ++i) out.grad[i] = (x[i] - prm(i)) / d;
		out.value = d - prm(dim) - o.margin;
		return out;
	}
	double q[3], outside2 = 0.0, m = -std::numeric_limits<double>::infinity();
	int kstar = 0;
	for (int i = 0; i < dim; ++i) {
		q[i] = std::abs(x[i] - prm(i)) - prm(dim + i);
		outside2 += std::max(q[i], 0.0) * std::max(q[i], 0.0);
		if (q[i] > m) {
			m = q[i];
			kstar = i;
		}
	}
	const double outside = std::sqrt(outside2 + 1e-8);
	out.value = outside + std::min(m, 0.0) - o.margin;
	if (outside2 > 1e-12) {
		for (int i = 0; i < dim; ++i)
			if (q[i] > 0.0) out.grad[i] = (x[i] >= prm(i) ? 1.0 : -1.0) * q[i] / outside;
	} else {
		out.grad[kstar] = x[kstar] >= prm(kstar) ? 1.0 : -1.0;
	}
	return out;
}

// sin(theta)^alpha between unit tangent `t` and `v`, and its derivatives with
// respect to v and t.
struct SinPower {
	double value;
	double dv[kMaxDim], dt[kMaxDim];
};

SinPower SinPow(const double* v, const double* t, int D, double alpha) {
	SinPower out{};
	double ww = kEps, u = 0.0;
	for (int i = 0; i < D; ++i) {
		ww += v[i] * v[i];
		u += v[i] * t[i];
	}
	double s2 = 1.0 - u * u / ww;
	double ds = 1.0;
	if (s2 <= 0.0 || s2 >= 1.0) {
		s2 = std::clamp(s2, 0.0, 1.0);
		ds = 0.0;
	}
	double S = s2, dS = ds;
	if (alpha != 2.0) {
		const double h = 0.5 * alpha;
		S = std::pow(s2 + kEps, h);
		dS = ds * h * std::pow(s2 + kEps, h - 1.0);
	}
	out.value = S;
	for (int i = 0; i < D; ++i) {
		out.dv[i] = dS * (-2.0 * u * t[i] / ww + 2.0 * u * u * v[i] / (ww * ww));
		out.dt[i] = dS * (-2.0 * u * v[i] / ww);
	}
	return out;
}

// One agent's space-time edges, for every sphere: edge (s, k) joins samples s
// and s + 1 of sphere k.
struct Edges {
	int num_edges = 0, num_spheres = 0;
	std::vector<double> mid, tangent, length;  // (edge * num_spheres + k) * D + i
	// Per edge index: centre and radius of a ball holding every sphere's
	// midpoint (spatial part), grown by the largest sphere radius.
	std::vector<double> bound_center, bound_radius;
};

Edges MakeEdges(const AgentSamples& a, double c, int wd) {
	const Eigen::VectorXd& t = a.times;
	const int D = wd + 1;
	Edges e;
	e.num_spheres = static_cast<int>(a.radii.size());
	e.num_edges = static_cast<int>(t.size()) - 1;
	const int S = e.num_spheres, n = e.num_edges * S;
	e.mid.resize(n * D);
	e.tangent.resize(n * D);
	e.length.resize(n);
	e.bound_center.assign(e.num_edges * wd, 0.0);
	e.bound_radius.assign(e.num_edges, 0.0);
	const double rmax = S ? a.radii.maxCoeff() : 0.0;
	for (int s = 0; s < e.num_edges; ++s) {
		for (int k = 0; k < S; ++k) {
			const int idx = s * S + k;
			double ev[kMaxDim], l2 = 0.0;
			for (int i = 0; i < wd; ++i) {
				const double x0 = a.centers(s * S + k, i), x1 = a.centers((s + 1) * S + k, i);
				e.mid[idx * D + i] = 0.5 * (x0 + x1);
				ev[i] = x1 - x0;
			}
			e.mid[idx * D + wd] = 0.5 * c * (t(s) + t(s + 1));
			ev[wd] = c * (t(s + 1) - t(s));
			for (int i = 0; i < D; ++i) l2 += ev[i] * ev[i];
			const double l = std::sqrt(l2 + kEps);
			e.length[idx] = l;
			for (int i = 0; i < D; ++i) e.tangent[idx * D + i] = ev[i] / l;
			for (int i = 0; i < wd; ++i) e.bound_center[s * wd + i] += e.mid[idx * D + i] / S;
		}
		double r = 0.0;
		for (int k = 0; k < S; ++k) {
			double d2 = 0.0;
			for (int i = 0; i < wd; ++i) {
				const double d = e.mid[(s * S + k) * D + i] - e.bound_center[s * wd + i];
				d2 += d * d;
			}
			r = std::max(r, std::sqrt(d2));
		}
		e.bound_radius[s] = r + rmax;
	}
	return e;
}

// Adds d(energy)/d(edge endpoints) to the samples' gradients: `dmid` and
// `dtan`/`dlen` are this edge's own derivatives. The endpoints' last
// coordinate is c * t, so its derivative goes to `grad_t` times c.
void ScatterEdge(Eigen::MatrixXd& grad, Eigen::VectorXd& grad_t, int s, int k, int S, int wd, double c,
		 const double* dmid, const double* dtan, double dlen, const double* tangent, double length,
		 double weight) {
	const int D = wd + 1;
	double tdot = 0.0;
	for (int i = 0; i < D; ++i) tdot += tangent[i] * dtan[i];
	for (int i = 0; i < D; ++i) {
		// d/d(edge vector) = (I - t t^T) dtan / l + dlen t
		const double de = (dtan[i] - tangent[i] * tdot) / length + dlen * tangent[i];
		const double g0 = weight * (0.5 * dmid[i] - de), g1 = weight * (0.5 * dmid[i] + de);
		if (i < wd) {
			grad(s * S + k, i) += g0;
			grad((s + 1) * S + k, i) += g1;
		} else {
			grad_t(s) += c * g0;
			grad_t(s + 1) += c * g1;
		}
	}
}

}  // namespace

void CheckKernel(const Kernel& k) {
	if (!(k.alpha >= 0.0)) throw std::invalid_argument("spacetime_repulsion: alpha must be >= 0");
	if (!(k.beta > k.alpha)) throw std::invalid_argument("spacetime_repulsion: need beta > alpha");
	if (!(k.time_scale > 0.0)) throw std::invalid_argument("spacetime_repulsion: time_scale must be > 0");
	if (!(0.0 < k.relax && k.relax < k.cutoff))
		throw std::invalid_argument("spacetime_repulsion: need 0 < relax < cutoff");
	if (!(k.relax <= k.reference_gap && k.reference_gap < k.cutoff))
		throw std::invalid_argument("spacetime_repulsion: need relax <= reference_gap < cutoff");
}

Radial::Radial(const Kernel& k) : p(k.beta - k.alpha), cutoff(k.cutoff), relax(k.relax) {
	CheckKernel(k);
	WindowedPower(relax, p, cutoff, &h0, &h1, &h2);
	double f, f1;
	WindowedPower(k.reference_gap, p, cutoff, &f, &f1, nullptr);
	inv_force = 1.0 / (-f1);
}

double Radial::value(double g, double* slope) const {
	if (g >= cutoff) {
		*slope = 0.0;
		return 0.0;
	}
	if (g > relax) {
		double f, f1;
		WindowedPower(g, p, cutoff, &f, &f1, nullptr);
		*slope = f1 * inv_force;
		return f * inv_force;
	}
	const double off = g - relax;
	*slope = (h1 + h2 * off) * inv_force;
	return (h0 + h1 * off + 0.5 * h2 * off * off) * inv_force;
}

Result Evaluate(const std::vector<AgentSamples>& agents, const ObstacleSet* obstacles, const Kernel& kernel,
		double obstacle_weight, bool gradient) {
	Result out;
	const int A = static_cast<int>(agents.size());
	if (A == 0) return out;
	const int wd = static_cast<int>(agents[0].centers.cols());
	const int D = wd + 1;
	if (D > kMaxDim) throw std::invalid_argument("spacetime_repulsion: workspace_dim must be 2 or 3");
	const double c = kernel.time_scale;
	const Radial radial(kernel);

	std::vector<Edges> edges;
	edges.reserve(A);
	std::vector<Eigen::VectorXd> tm(A);
	for (int a = 0; a < A; ++a) {
		const AgentSamples& ag = agents[a];
		if (ag.times.size() < 2 || ag.centers.rows() != ag.times.size() * ag.radii.size())
			throw std::invalid_argument("spacetime_repulsion: an agent needs >= 2 samples, one row per sample and sphere");
		edges.push_back(MakeEdges(ag, c, wd));
		const int E = static_cast<int>(ag.times.size()) - 1;
		tm[a].resize(E);
		for (int s = 0; s < E; ++s) tm[a](s) = 0.5 * c * (ag.times(s) + ag.times(s + 1));
	}
	if (gradient) {
		out.gradient.resize(A);
		out.time_gradient.resize(A);
		for (int a = 0; a < A; ++a) {
			out.gradient[a] = Eigen::MatrixXd::Zero(agents[a].centers.rows(), wd);
			out.time_gradient[a] = Eigen::VectorXd::Zero(agents[a].times.size());
		}
	}

	double inter = 0.0;
	for (int a = 0; a < A; ++a) {
		const Edges& ea = edges[a];
		const int Sa = ea.num_spheres;
		for (int b = a + 1; b < A; ++b) {
			const Edges& eb = edges[b];
			const int Sb = eb.num_spheres;
			if (Sa == 0 || Sb == 0) continue;
			const double reach = kernel.cutoff + agents[a].radii.maxCoeff() + agents[b].radii.maxCoeff();
			const int Ea = ea.num_edges, Eb = eb.num_edges;
			int lo = 0;
			for (int I = 0; I < Ea; ++I) {
				while (lo < Eb && tm[b](lo) < tm[a](I) - reach) ++lo;
				for (int J = lo; J < Eb && tm[b](J) <= tm[a](I) + reach; ++J) {
					const double dt = tm[b](J) - tm[a](I);
					double bd2 = dt * dt;
					for (int i = 0; i < wd; ++i) {
						const double d = ea.bound_center[I * wd + i] - eb.bound_center[J * wd + i];
						bd2 += d * d;
					}
					const double bound = ea.bound_radius[I] + eb.bound_radius[J] + kernel.cutoff;
					if (bd2 > bound * bound) continue;
					for (int k = 0; k < Sa; ++k) {
						const int ia = I * Sa + k;
						const double* mx = &ea.mid[ia * D];
						const double* tx = &ea.tangent[ia * D];
						const double lx = ea.length[ia];
						for (int kk = 0; kk < Sb; ++kk) {
							const int ib = J * Sb + kk;
							const double* my = &eb.mid[ib * D];
							double v[kMaxDim], w = kEps;
							for (int i = 0; i < D; ++i) {
								v[i] = my[i] - mx[i];
								w += v[i] * v[i];
							}
							const double R = agents[a].radii(k) + agents[b].radii(kk);
							const double dist = std::sqrt(w);
							if (dist - R >= kernel.cutoff) continue;
							const double* ty = &eb.tangent[ib * D];
							const double ly = eb.length[ib];
							double dr;
							const double rho = radial.value(dist - R, &dr);
							const SinPower sx = SinPow(v, tx, D, kernel.alpha);
							const SinPower sy = SinPow(v, ty, D, kernel.alpha);
							const double L = lx * ly, num = sx.value + sy.value;
							inter += L * num * rho;
							++out.pairs;
							if (!gradient) continue;
							double dv[kMaxDim], dtx[kMaxDim], dty[kMaxDim];
							for (int i = 0; i < D; ++i) {
								dv[i] = L * ((sx.dv[i] + sy.dv[i]) * rho + num * dr * v[i] / dist);
								dtx[i] = L * sx.dt[i] * rho;
								dty[i] = L * sy.dt[i] * rho;
							}
							// v = mid_y - mid_x
							double dmx[kMaxDim], dmy[kMaxDim];
							for (int i = 0; i < D; ++i) {
								dmx[i] = -dv[i];
								dmy[i] = dv[i];
							}
							ScatterEdge(out.gradient[a], out.time_gradient[a], I, k, Sa, wd, c, dmx, dtx,
								    ly * num * rho, tx, lx, 1.0 / c);
							ScatterEdge(out.gradient[b], out.time_gradient[b], J, kk, Sb, wd, c, dmy, dty,
								    lx * num * rho, ty, ly, 1.0 / c);
						}
					}
				}
			}
		}
	}
	out.inter_agent = inter / c;

	double obs = 0.0;
	if (obstacles != nullptr && !obstacles->obstacles().empty()) {
		for (int a = 0; a < A; ++a) {
			const Edges& ea = edges[a];
			const int S = ea.num_spheres;
			for (int s = 0; s < ea.num_edges; ++s) {
				for (int k = 0; k < S; ++k) {
					const int idx = s * S + k;
					const double* m = &ea.mid[idx * D];
					for (const Obstacle& o : obstacles->obstacles()) {
						if (static_cast<int>(o.params.size()) != (o.kind == ObstacleKind::kBox ? 2 * wd : wd + 1))
							throw std::invalid_argument(
								"spacetime_repulsion: obstacle dimension doesn't match the workspace");
						const Sdf sdf = ObstacleSdf(m, wd, o);
						const double g = sdf.value - agents[a].radii(k);
						if (g >= kernel.cutoff) continue;
						double dr;
						const double rho = radial.value(g, &dr);
						obs += ea.length[idx] * rho;
						if (!gradient) continue;
						double dmid[kMaxDim] = {}, dtan[kMaxDim] = {};
						for (int i = 0; i < wd; ++i) dmid[i] = ea.length[idx] * dr * sdf.grad[i];
						ScatterEdge(out.gradient[a], out.time_gradient[a], s, k, S, wd, c, dmid, dtan, rho,
							    &ea.tangent[idx * D], ea.length[idx], obstacle_weight / c);
					}
				}
			}
		}
	}
	out.obstacle = obs / c;
	return out;
}

Clearance MeasureClearance(const std::vector<AgentSamples>& agents, const ObstacleSet* obstacles) {
	Clearance out{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
	const int A = static_cast<int>(agents.size());
	if (A == 0) return out;
	const int wd = static_cast<int>(agents[0].centers.cols());
	// Every sphere of `ag` at each of `ts` (increasing), linear between its
	// own samples and held outside them: row (i * S + k).
	auto resample = [&](const AgentSamples& ag, const std::vector<double>& ts, Eigen::MatrixXd& P) {
		const int S = static_cast<int>(ag.radii.size()), n = static_cast<int>(ag.times.size());
		P.resize(ts.size() * S, wd);
		int j = 0;
		for (size_t i = 0; i < ts.size(); ++i) {
			while (j + 2 < n && ag.times(j + 1) <= ts[i]) ++j;
			const double span = ag.times(j + 1) - ag.times(j);
			const double f = std::clamp(span > 0.0 ? (ts[i] - ag.times(j)) / span : 0.0, 0.0, 1.0);
			for (int k = 0; k < S; ++k)
				P.row(i * S + k) = (1.0 - f) * ag.centers.row(j * S + k) + f * ag.centers.row((j + 1) * S + k);
		}
	};
	std::vector<double> ts;
	Eigen::MatrixXd Pa, Pb;
	for (int a = 0; a < A; ++a) {
		for (int b = a + 1; b < A; ++b) {
			const AgentSamples &pa = agents[a], &pb = agents[b];
			ts.assign(pa.times.data(), pa.times.data() + pa.times.size());
			ts.insert(ts.end(), pb.times.data(), pb.times.data() + pb.times.size());
			std::sort(ts.begin(), ts.end());
			ts.erase(std::unique(ts.begin(), ts.end()), ts.end());
			resample(pa, ts, Pa);
			resample(pb, ts, Pb);
			const int Sa = static_cast<int>(pa.radii.size()), Sb = static_cast<int>(pb.radii.size());
			for (size_t s = 0; s + 1 < ts.size(); ++s) {
				for (int k = 0; k < Sa; ++k) {
					for (int kk = 0; kk < Sb; ++kk) {
						double d0[3], e[3], de = 0.0, ee = 0.0;
						for (int i = 0; i < wd; ++i) {
							d0[i] = Pa(s * Sa + k, i) - Pb(s * Sb + kk, i);
							e[i] = (Pa((s + 1) * Sa + k, i) - Pb((s + 1) * Sb + kk, i)) - d0[i];
							de += d0[i] * e[i];
							ee += e[i] * e[i];
						}
						const double f = ee > 0.0 ? std::clamp(-de / ee, 0.0, 1.0) : 0.0;
						double d2 = 0.0;
						for (int i = 0; i < wd; ++i) d2 += (d0[i] + f * e[i]) * (d0[i] + f * e[i]);
						out.inter_agent = std::min(out.inter_agent, std::sqrt(d2) - pa.radii(k) - pb.radii(kk));
					}
				}
			}
		}
	}
	if (obstacles != nullptr) {
		for (const AgentSamples& a : agents) {
			const int S = static_cast<int>(a.radii.size());
			for (int row = 0; row < a.centers.rows(); ++row) {
				double x[3];
				for (int i = 0; i < wd; ++i) x[i] = a.centers(row, i);
				for (const Obstacle& o : obstacles->obstacles())
					out.obstacle = std::min(out.obstacle, ObstacleSdf(x, wd, o).value - a.radii(row % S));
			}
		}
	}
	return out;
}

}  // namespace spacetime_repulsion
