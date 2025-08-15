CubicSplineLeapCost::CubicSplineLeapCost(const Eigen::VectorXd& x0,
                                         const Eigen::VectorXd& v0,
					 const Eigen::MatrixXd& wps,
					 double ctrl_cost,
                                         size_t K,
					 bool opt_last_vel,
					 bool opt_time_deltas)
	: x0_(x0),
	  v0_(v0),
	  wps_(wps),
	  ctrl_cost_(ctrl_cost),
	  K_(K),
	  d_(x0.size()),
	  opt_last_vel_(opt_last_vel),
	  opt_time_deltas_(opt_time_deltas),
	  /* num variables is K/K-1 * d velocity variables plus K tau variables */
	  drake::solvers::Cost((opt_last_vel ? K : K-1) * x0.size() + (opt_time_deltas ? K : 0))
{
	/* TODO: checks */
}

CubicSplineLeapCost::~CubicSplineLeapCost() = default;

/* template parameterizing over type of v0, type of v1, and return type. */
template <typename T1, typename T2, typename RT>
RT CubicPieceLeapCost(const Eigen::VectorXd& x0,
                      const Eigen::VectorXd& x1,
                      const Eigen::Matrix<T1, Eigen::Dynamic, 1>& v0,
                      const Eigen::Matrix<T2, Eigen::Dynamic, 1>& v1,
                      const RT tau) {
  using VecRT = Eigen::Matrix<RT, Eigen::Dynamic, 1>;

  const int dim = x0.size();
  DRAKE_DEMAND(v0.size() == dim);
  DRAKE_DEMAND(v1.size() == dim);

  const VecRT Dx = (x1 - x0).template cast<RT>();
  const VecRT D  = Dx - RT(0.5) * tau * (v0 + v1);
  const VecRT V  = v1 - v0;

  const double s12 = std::sqrt(12.0);
  const RT cost = RT(s12) * pow(tau, RT(-1.5)) * D.norm()
                + pow(tau, RT(-0.5)) * V.norm();
  return cost * cost;
}

// Evaluates cost with doubles
void CubicSplineLeapCost::DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
                                 Eigen::VectorXd* y) const {
  using T = double;
  y->resize(1);
  T total_cost = 0.0;

  const ssize_t vN = this->opt_last_vel_ ? this->K_ : this->K_ - 1;

  const auto v_flat = x.head(vN * this->d_);
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      v(v_flat.data(), vN, this->d_);

  const Eigen::VectorXd time_deltas = x.segment(vN * this->d_, this->K_);

  for (ssize_t k = 0; k < this->K_; ++k) {
    const Eigen::VectorXd x0 = (k == 0) ? this->x0_ : this->wps_.row(k - 1).transpose();
    const Eigen::VectorXd x1 = this->wps_.row(k).transpose();
    const double tau = time_deltas(k);

    if (k == 0) {
      const Eigen::VectorXd v0 = this->v0_;
      const Eigen::VectorXd v1 = v.row(k).transpose();
      total_cost += CubicPieceLeapCost<double, double, double>(x0, x1, v0, v1, tau);
    } else if (k == this->K_ - 1 && !this->opt_last_vel_) {
      const Eigen::VectorXd v0 = v.row(k - 1).transpose();
      const Eigen::VectorXd v1 = Eigen::VectorXd::Zero(this->d_);
      total_cost += CubicPieceLeapCost<double, double, double>(x0, x1, v0, v1, tau);
    } else {
      const Eigen::VectorXd v0 = v.row(k - 1).transpose();
      const Eigen::VectorXd v1 = v.row(k).transpose();
      total_cost += CubicPieceLeapCost<double, double, double>(x0, x1, v0, v1, tau);
    }
  }

  (*y)(0) = total_cost;
}

// Evaluates cost with AutoDiffXd
void CubicSplineLeapCost::DoEval(const Eigen::Ref<const drake::AutoDiffVecXd>& x,
                                 drake::AutoDiffVecXd* y) const {
  using AD = drake::AutoDiffXd;
  y->resize(1);
  AD total_cost = AD(0.0);

  const ssize_t vN = this->opt_last_vel_ ? this->K_ : this->K_ - 1;

  const auto v_flat = x.head(vN * this->d_);
  Eigen::Map<const Eigen::Matrix<AD, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      v(v_flat.data(), vN, this->d_);

  const Eigen::Matrix<AD, Eigen::Dynamic, 1> time_deltas = x.segment(vN * this->d_, this->K_);

  for (ssize_t k = 0; k < this->K_; ++k) {
    const Eigen::VectorXd x0 = (k == 0) ? this->x0_ : this->wps_.row(k - 1).transpose();
    const Eigen::VectorXd x1 = this->wps_.row(k).transpose();
    const AD tau = time_deltas(k);

    if (k == 0) {
      const Eigen::Matrix<AD, Eigen::Dynamic, 1> v0 = this->v0_.cast<AD>();
      const Eigen::Matrix<AD, Eigen::Dynamic, 1> v1 = v.row(k).transpose();
      total_cost += CubicPieceLeapCost<AD, AD, AD>(x0, x1, v0, v1, tau);
    } else if (k == this->K_ - 1 && !this->opt_last_vel_) {
      const Eigen::Matrix<AD, Eigen::Dynamic, 1> v0 = v.row(k - 1).transpose();
      const Eigen::Matrix<AD, Eigen::Dynamic, 1> v1 = Eigen::Matrix<AD, Eigen::Dynamic, 1>::Zero(this->d_);
      total_cost += CubicPieceLeapCost<AD, AD, AD>(x0, x1, v0, v1, tau);
    } else {
      const Eigen::Matrix<AD, Eigen::Dynamic, 1> v0 = v.row(k - 1).transpose();
      const Eigen::Matrix<AD, Eigen::Dynamic, 1> v1 = v.row(k).transpose();
      total_cost += CubicPieceLeapCost<AD, AD, AD>(x0, x1, v0, v1, tau);
    }
  }

  (*y)(0) = total_cost;
}

// Evaluates cost with symbolic variables
void CubicSplineLeapCost::DoEval(
    const Eigen::Ref<const Eigen::Matrix<drake::symbolic::Variable, -1, 1>>& x,
    Eigen::Matrix<drake::symbolic::Expression, -1, 1>* y) const {
  using drake::symbolic::Variable;
  using drake::symbolic::Expression;
  y->resize(1);
  Expression total_cost(0.0);

  const ssize_t vN = this->opt_last_vel_ ? this->K_ : this->K_ - 1;

  const auto v_flat = x.head(vN * this->d_);
  Eigen::Map<const Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      v(v_flat.data(), vN, this->d_);

  const Eigen::Matrix<Variable, Eigen::Dynamic, 1> time_deltas = x.segment(vN * this->d_, this->K_);

  // helpers to cast Variable -> Expression
  auto to_expr_vec = [](const auto& var_vec) {
    Eigen::Matrix<Expression, Eigen::Dynamic, 1> out(var_vec.size());
    for (int i = 0; i < var_vec.size(); ++i) out(i) = Expression(var_vec(i));
    return out;
  };

  for (ssize_t k = 0; k < this->K_; ++k) {
    const Eigen::VectorXd x0 = (k == 0) ? this->x0_ : this->wps_.row(k - 1).transpose();
    const Eigen::VectorXd x1 = this->wps_.row(k).transpose();
    const Expression tau = Expression(time_deltas(k));

    if (k == 0) {
      const auto v0 = this->v0_.unaryExpr([](double s) { return Expression(s); });
      const auto v1 = to_expr_vec(v.row(k).transpose());
      total_cost += CubicPieceLeapCost<Expression, Expression, Expression>(x0, x1, v0, v1, tau);
    } else if (k == this->K_ - 1 && !this->opt_last_vel_) {
      const auto v0 = to_expr_vec(v.row(k - 1).transpose());
      Eigen::Matrix<Expression, Eigen::Dynamic, 1> v1(this->d_);
      for (int i = 0; i < this->d_; ++i) v1(i) = Expression(0.0);
      total_cost += CubicPieceLeapCost<Expression, Expression, Expression>(x0, x1, v0, v1, tau);
    } else {
      const auto v0 = to_expr_vec(v.row(k - 1).transpose());
      const auto v1 = to_expr_vec(v.row(k).transpose());
      total_cost += CubicPieceLeapCost<Expression, Expression, Expression>(x0, x1, v0, v1, tau);
    }
  }

  (*y)(0) = total_cost;
}
