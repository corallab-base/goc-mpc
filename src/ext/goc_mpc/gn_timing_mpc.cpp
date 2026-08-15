#include "gn_timing_mpc.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <Eigen/Sparse>

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpTNLP.hpp>

#include "timing_gn_layout.hpp"

namespace {

using Ipopt::Index;
using Ipopt::Number;
using gn_timing::kTauMin;
using gn_timing::kTauMax;
using gn_timing::kInf;
using gn_timing::ProblemLayout;
using gn_timing::AssembleObjective;
using gn_timing::BuildProblemLayout;
using gn_timing::CumsumWithZero;

// Hand-written Ipopt::TNLP -- see gn_timing_mpc.hpp's top-of-file comment
// for why this bypasses Drake's MathematicalProgram/IpoptSolver entirely.
// Runs with hessian_approximation=exact (set by the caller in solve()) and
// supplies the objective's Gauss-Newton Hessian via eval_h; every
// constraint here is exactly linear, so the lambda-weighted constraint-
// Hessian term IPOPT's eval_h contract also asks for is exactly zero and
// simply omitted.
class GnTimingNLP : public Ipopt::TNLP {
public:
	GnTimingNLP(ProblemLayout layout, double time_cost, double time_cost2,
		    double acceleration_cost)
		: layout_(std::move(layout)),
		  time_cost_(time_cost),
		  time_cost2_(time_cost2),
		  acceleration_cost_(acceleration_cost) {
		BuildJacobianTriplets();
		BuildHessianPattern();
	}

	Eigen::VectorXd solution_x;
	Ipopt::SolverReturn solve_status = Ipopt::SolverReturn::UNASSIGNED;

	bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag,
			   IndexStyleEnum& index_style) override {
		n = layout_.n;
		m = static_cast<Index>(layout_.interactions.size());
		nnz_jac_g = static_cast<Index>(jac_triplets_.size());
		nnz_h_lag = hess_nnz_;
		index_style = C_STYLE;
		return true;
	}

	bool get_bounds_info(Index n, Number* x_l, Number* x_u,
			      Index m, Number* g_l, Number* g_u) override {
		for (int k = 0; k < n; ++k) {
			x_l[k] = -kInf;
			x_u[k] = kInf;
		}
		for (const auto& seg : layout_.segments) {
			x_l[seg.tau_idx] = kTauMin;
			x_u[seg.tau_idx] = kTauMax;
		}
		int row = 0;
		for (const auto& ir : layout_.interactions) {
			if (ir.type == AgentInteraction::Type::LESS_THAN) {
				g_l[row] = -kInf;
				g_u[row] = -ir.min_tau;
			} else {
				g_l[row] = 0.0;
				g_u[row] = 0.0;
			}
			++row;
		}
		return true;
	}

	bool get_starting_point(Index n, bool init_x, Number* x,
				 bool /*init_z*/, Number* /*z_L*/, Number* /*z_U*/,
				 Index /*m*/, bool /*init_lambda*/, Number* /*lambda*/) override {
		// z_L/z_U/lambda warm-starting is never requested (we don't set
		// warm_start_init_point), so init_z/init_lambda are always false.
		if (init_x) {
			for (int k = 0; k < n; ++k) x[k] = layout_.x_init(k);
		}
		return true;
	}

	bool eval_f(Index n, const Number* x, bool /*new_x*/, Number& obj_value) override {
		const Eigen::VectorXd xv = Eigen::Map<const Eigen::VectorXd>(x, n);
		double f = 0.0;
		AssembleObjective(layout_, acceleration_cost_, time_cost_, time_cost2_,
				   xv, &f, nullptr, nullptr);
		obj_value = f;
		return true;
	}

	bool eval_grad_f(Index n, const Number* x, bool /*new_x*/, Number* grad_f) override {
		const Eigen::VectorXd xv = Eigen::Map<const Eigen::VectorXd>(x, n);
		Eigen::VectorXd grad;
		AssembleObjective(layout_, acceleration_cost_, time_cost_, time_cost2_,
				   xv, nullptr, &grad, nullptr);
		for (int k = 0; k < n; ++k) grad_f[k] = grad(k);
		return true;
	}

	bool eval_g(Index /*n*/, const Number* x, bool /*new_x*/, Index /*m*/, Number* g) override {
		int row = 0;
		for (const auto& ir : layout_.interactions) {
			double si = 0.0, sj = 0.0;
			for (int k = 0; k < ir.count_i; ++k) si += x[ir.tau_offset_i + k];
			for (int k = 0; k < ir.count_j; ++k) sj += x[ir.tau_offset_j + k];
			g[row++] = si - sj;
		}
		return true;
	}

	bool eval_jac_g(Index /*n*/, const Number* /*x*/, bool /*new_x*/, Index /*m*/,
			 Index /*nele_jac*/, Index* iRow, Index* jCol, Number* values) override {
		if (values == nullptr) {
			for (size_t k = 0; k < jac_triplets_.size(); ++k) {
				iRow[k] = jac_triplets_[k].row();
				jCol[k] = jac_triplets_[k].col();
			}
		} else {
			for (size_t k = 0; k < jac_triplets_.size(); ++k) {
				values[k] = jac_triplets_[k].value();
			}
		}
		return true;
	}

	bool eval_h(Index n, const Number* x, bool /*new_x*/, Number obj_factor,
		    Index /*m*/, const Number* /*lambda*/, bool /*new_lambda*/,
		    Index /*nele_hess*/, Index* iRow, Index* jCol, Number* values) override {
		if (values == nullptr) {
			for (int k = 0; k < hess_nnz_; ++k) {
				iRow[k] = hess_rows_[k];
				jCol[k] = hess_cols_[k];
			}
			return true;
		}

		const Eigen::VectorXd xv = Eigen::Map<const Eigen::VectorXd>(x, n);
		std::vector<Eigen::Triplet<double>> triplets;
		AssembleObjective(layout_, acceleration_cost_, time_cost_, time_cost2_,
				   xv, nullptr, nullptr, &triplets);

		Eigen::SparseMatrix<double> H(layout_.n, layout_.n);
		H.setFromTriplets(triplets.begin(), triplets.end());
		H.makeCompressed();
		if (H.nonZeros() != hess_nnz_) {
			// The nonzero PATTERN is fully determined by problem structure
			// (which segments/blocks/interactions exist), never by x's
			// value -- so this can only mean a real bug in AssembleObjective
			// (e.g. a triplet emitted conditionally on a runtime value),
			// not an expected runtime condition. Fail loudly rather than
			// silently handing IPOPT a mismatched values array.
			throw std::runtime_error(
				"GnTimingMPC: Hessian sparsity pattern changed between "
				"calls (internal bug in AssembleObjective)");
		}

		int k = 0;
		for (int col = 0; col < H.outerSize(); ++col) {
			for (Eigen::SparseMatrix<double>::InnerIterator it(H, col); it; ++it) {
				values[k++] = obj_factor * it.value();
			}
		}
		return true;
	}

	void finalize_solution(Ipopt::SolverReturn status, Index n, const Number* x,
				const Number* /*z_L*/, const Number* /*z_U*/,
				Index /*m*/, const Number* /*g*/, const Number* /*lambda*/,
				Number /*obj_value*/, const Ipopt::IpoptData* /*ip_data*/,
				Ipopt::IpoptCalculatedQuantities* /*ip_cq*/) override {
		solution_x = Eigen::Map<const Eigen::VectorXd>(x, n);
		solve_status = status;
	}

private:
	void BuildJacobianTriplets() {
		// Each row only ever touches its own two agents' tau ranges, which
		// are disjoint by construction (BuildProblemLayout assigns every
		// agent a non-overlapping tau range) -- so unlike the Hessian, no
		// (row, col) pair here can repeat and there's no need to round-trip
		// through Eigen::SparseMatrix to deduplicate.
		int row = 0;
		for (const auto& ir : layout_.interactions) {
			for (int k = 0; k < ir.count_i; ++k) {
				jac_triplets_.emplace_back(row, ir.tau_offset_i + k, 1.0);
			}
			for (int k = 0; k < ir.count_j; ++k) {
				jac_triplets_.emplace_back(row, ir.tau_offset_j + k, -1.0);
			}
			++row;
		}
	}

	void BuildHessianPattern() {
		std::vector<Eigen::Triplet<double>> triplets;
		AssembleObjective(layout_, acceleration_cost_, time_cost_, time_cost2_,
				   layout_.x_init, nullptr, nullptr, &triplets);
		Eigen::SparseMatrix<double> H(layout_.n, layout_.n);
		H.setFromTriplets(triplets.begin(), triplets.end());
		H.makeCompressed();
		hess_nnz_ = static_cast<int>(H.nonZeros());
		hess_rows_.resize(hess_nnz_);
		hess_cols_.resize(hess_nnz_);
		int k = 0;
		for (int col = 0; col < H.outerSize(); ++col) {
			for (Eigen::SparseMatrix<double>::InnerIterator it(H, col); it; ++it) {
				hess_rows_[k] = it.row();
				hess_cols_[k] = it.col();
				++k;
			}
		}
	}

	ProblemLayout layout_;
	double time_cost_, time_cost2_, acceleration_cost_;
	std::vector<Eigen::Triplet<double>> jac_triplets_;
	int hess_nnz_ = 0;
	std::vector<int> hess_rows_, hess_cols_;
};

}  // namespace

/*
 * GnTimingMPC
 */

GnTimingMPC::GnTimingMPC(const GraphOfConstraints& graph,
			  std::vector<CubicConfigurationSpline> splines,
			  double time_cost,
			  double time_cost2,
			  double acceleration_cost)
	: _graph(&graph),
	  _splines(std::make_shared<std::vector<CubicConfigurationSpline>>(std::move(splines))),
	  _time_cost(time_cost),
	  _time_cost2(time_cost2),
	  _acceleration_cost(acceleration_cost) {

	const int num_agents = _graph->num_agents;
	const int num_nodes = _graph->structure.num_nodes();
	// Assuming all the same -- same assumption GraphTimingMPC's own
	// constructor makes.
	const int ambient_dim = _splines->at(0).ambient_dim();
	const int tangent_dim = _splines->at(0).tangent_dim();

	_wps_list.resize(num_agents);
	for (int i = 0; i < num_agents; ++i) _wps_list[i] = Eigen::MatrixXd::Zero(num_nodes, ambient_dim);
	_vs_list.resize(num_agents);
	for (int i = 0; i < num_agents; ++i) _vs_list[i] = Eigen::MatrixXd::Zero(num_nodes, tangent_dim);
	_time_deltas_list.resize(num_agents);
	for (int i = 0; i < num_agents; ++i) _time_deltas_list[i] = Eigen::VectorXd::Zero(num_nodes);
}

bool GnTimingMPC::solve(
		const Eigen::VectorXd& x0,
		const Eigen::VectorXd& v0,
		const std::vector<int>& remaining_vertices,
		const Eigen::MatrixXd& waypoints,
		const Eigen::VectorXi& assignments,
		const Eigen::VectorXd& t_by_node) {

	_timer.Start();

	// Snapshotted before this cycle's own values overwrite them below --
	// same warm-start purpose as GraphTimingMPC::solve()'s own snapshot.
	const std::vector<Eigen::MatrixXd> prev_vs_list = _vs_list;
	const std::vector<Eigen::VectorXd> prev_time_deltas_list = _time_deltas_list;
	const std::vector<std::vector<int>> prev_agent_nodes_list = _agent_nodes_list;
	const std::map<int, int> prev_agent_spline_length_map = _agent_spline_length_map;

	ProblemLayout layout;
	std::vector<Eigen::MatrixXd> wps_list;
	std::vector<std::vector<int>> agent_nodes_list;
	try {
		layout = BuildProblemLayout(
			*_graph, *_splines, remaining_vertices, waypoints, assignments, x0, v0,
			t_by_node, &wps_list, &agent_nodes_list,
			prev_vs_list, prev_time_deltas_list, prev_agent_nodes_list,
			prev_agent_spline_length_map);
	} catch (const std::exception& e) {
		std::cout << "Caught exception in GnTimingMPC problem construction: "
			  << e.what() << std::endl;
		return false;
	}

	_wps_list = wps_list;
	_agent_nodes_list = agent_nodes_list;

	if (layout.n == 0) {
		// Nothing left to plan for any agent.
		_last_solve_time = _timer.Tick();
		return true;
	}

	// Passed BY VALUE (copied, not moved) -- `layout` is still needed below
	// to recover each agent's tau/v ranges when reading the solution back
	// out, and a copy is cheap at this problem's scale.
	Ipopt::SmartPtr<GnTimingNLP> nlp =
		new GnTimingNLP(layout, _time_cost, _time_cost2, _acceleration_cost);

	Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();
	app->Options()->SetStringValue("sb", "yes");
	app->Options()->SetIntegerValue("print_level", 0);
	// The whole point of this class -- see gn_timing_mpc.hpp's top-of-file
	// comment: Drake's own IpoptSolver binding can never do this (it never
	// implements eval_h), so it's always stuck on limited-memory (L-BFGS).
	app->Options()->SetStringValue("hessian_approximation", "exact");
	if (std::getenv("GN_TIMING_MPC_DERIVATIVE_TEST") != nullptr) {
		app->Options()->SetStringValue("derivative_test", "second-order");
		app->Options()->SetNumericValue("derivative_test_tol", 1e-4);
		app->Options()->SetIntegerValue("print_level", 6);
	}

	const Ipopt::ApplicationReturnStatus init_status = app->Initialize();
	if (init_status != Ipopt::Solve_Succeeded) {
		std::cout << "GnTimingMPC: IPOPT initialization failed (status="
			  << static_cast<int>(init_status) << ")" << std::endl;
		return false;
	}

	Ipopt::ApplicationReturnStatus status;
	try {
		status = app->OptimizeTNLP(nlp);
	} catch (const std::exception& e) {
		std::cout << "Caught exception in GnTimingMPC solve: " << e.what() << std::endl;
		return false;
	}

	const bool success = (status == Ipopt::Solve_Succeeded ||
			       status == Ipopt::Solved_To_Acceptable_Level);
	if (!success) {
		std::cout << "GnTimingMPC: IPOPT did not succeed (status="
			  << static_cast<int>(status) << ")" << std::endl;
		return false;
	}

	_last_solve_time = _timer.Tick();

	const int tangent_dim = _splines->at(0).tangent_dim();
	for (int i = 0; i < _graph->num_agents; ++i) {
		const int K = static_cast<int>(_agent_nodes_list[i].size());
		if (K == 0) continue;
		_agent_spline_length_map[i] = K + 1;

		const int tau_offset = layout.agents[i].tau_offset;
		const int v_offset = layout.agents[i].v_offset;
		for (int j = 0; j < K; ++j) {
			_time_deltas_list[i](j) = nlp->solution_x(tau_offset + j);
		}
		if (K > 1) {
			for (int row = 0; row < K - 1; ++row) {
				for (int c = 0; c < tangent_dim; ++c) {
					_vs_list[i](row, c) = nlp->solution_x(v_offset + row * tangent_dim + c);
				}
			}
		}
	}
	return true;
}

int GnTimingMPC::get_agent_spline_length(int agent) const {
	if (!_agent_spline_length_map.contains(agent)) {
		return 0;
	} else {
		return _agent_spline_length_map.at(agent);
	}
}

std::vector<int> GnTimingMPC::get_agent_spline_nodes(int agent) const {
	if (agent < 0 || agent > _agent_nodes_list.size()) {
		return std::vector<int>();
	} else {
		return _agent_nodes_list.at(agent);
	}
}

std::set<int> GnTimingMPC::set_progressed_time(double delta, double tau_cutoff) {
	// Verbatim copy of GraphTimingMPC::set_progressed_time's logic -- see
	// that function's own comment for the "walk forward accumulating tau"
	// rationale. Only ever reads _agent_spline_length_map/_agent_nodes_list/
	// _time_deltas_list, none of which are Drake-specific.
	std::set<int> passed_nodes;

	for (int i = 0; i < _graph->num_agents; ++i) {
		if (_agent_spline_length_map[i] > 0) {
			double cumulative_tau = 0.0;
			for (int j = 0; j < static_cast<int>(_agent_nodes_list[i].size()); ++j) {
				cumulative_tau += _time_deltas_list[i](j);
				if (delta < cumulative_tau - tau_cutoff) {
					break;
				}
				const int node_id = _agent_nodes_list[i][j];
				if (node_id >= 0) {
					passed_nodes.insert(node_id);
				}
			}
		}
	}

	return passed_nodes;
}

void GnTimingMPC::fill_cubic_splines(std::vector<CubicConfigurationSpline*>& splines,
				      const Eigen::VectorXd& x0,
				      const Eigen::VectorXd& v0) const {
	// Verbatim copy of GraphTimingMPC::fill_cubic_splines's logic.
	const int a_d = splines[0]->ambient_dim();
	const int t_d = splines[0]->tangent_dim();

	for (int i = 0; i < _graph->num_agents; ++i) {
		const int spline_length_i = _agent_spline_length_map.at(i);

		if (spline_length_i > 1) {
			Eigen::VectorXd x0_i = x0.segment(i * a_d, a_d);
			Eigen::MatrixXd wps_i(spline_length_i, a_d);
			wps_i.row(0) = x0_i;
			wps_i.bottomRows(spline_length_i - 1) = _wps_list[i];

			Eigen::VectorXd v0_i = v0.segment(i * t_d, t_d);
			Eigen::MatrixXd vs_i(spline_length_i, t_d);
			vs_i.row(0) = v0_i;
			vs_i.block(1, 0, spline_length_i - 2, t_d) = _vs_list[i];
			vs_i.row(spline_length_i - 1).setZero();

			Eigen::VectorXd times_i = CumsumWithZero(_time_deltas_list[i], spline_length_i - 1);

			splines[i]->set(wps_i, vs_i, times_i);
		} else {
			Eigen::VectorXd x0_i = x0.segment(i * a_d, a_d);
			Eigen::MatrixXd wps_i(2, a_d);
			wps_i.row(0) = x0_i;
			wps_i.row(1) = x0_i;

			Eigen::VectorXd v0_i = v0.segment(i * t_d, t_d);
			Eigen::MatrixXd vs_i(2, t_d);
			vs_i.row(0) = v0_i;
			vs_i.row(1).setZero();

			Eigen::VectorXd times_i(2);
			times_i << 0.0, 1.0;

			splines[i]->set(wps_i, vs_i, times_i);
		}
	}
}

const std::vector<double> GnTimingMPC::get_next_taus() const {
	std::vector<double> result;
	for (int i = 0; i < _graph->num_agents; ++i) {
		const int spline_length_i = _agent_spline_length_map.at(i);
		if (spline_length_i > 1) {
			result.push_back(_time_deltas_list[i](0));
		}
	}
	return result;
}

const std::vector<int> GnTimingMPC::get_next_nodes() const {
	std::vector<int> result;
	for (int i = 0; i < _graph->num_agents; ++i) {
		const int spline_length_i = _agent_spline_length_map.at(i);
		if (spline_length_i > 1 && !_agent_nodes_list[i].empty()) {
			const int node = _agent_nodes_list[i].at(0);
			if (node >= 0) {
				result.push_back(node);
			}
		}
	}
	return result;
}
