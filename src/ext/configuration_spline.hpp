#pragma once

#include <drake/solvers/mathematical_program.h>
#include <drake/math/rotation_matrix.h>
#include <drake/math/quaternion.h>
#include <drake/common/symbolic/expression.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <Eigen/Dense>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

using drake::symbolic::Expression;
template <typename T> using VecX = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T> using Vec3 = Eigen::Matrix<T, 3, 1>;
template <typename T, int N> using Vec = Eigen::Matrix<T, N, 1>;


static inline Expression sqr(const Expression& e) { return e * e; }

// Wrap a scalar angle difference into (-pi, pi].
template <typename T>
T wrap_to_pi(const T& delta) {
    using std::floor;
    const T two_pi = T(2.0 * M_PI);
    return delta - two_pi * floor((delta + T(M_PI)) / two_pi);
}

// Quaternion (w,x,y,z) -> RotationMatrix<Expression>
template <typename T>
inline drake::math::RotationMatrix<T>
RotFromQuatWxyz(const Eigen::Matrix<T,4,1>& qwxyz) {
    Eigen::Quaternion<T> q(
        qwxyz(0), qwxyz(1), qwxyz(2), qwxyz(3));
    return drake::math::RotationMatrix<T>(q);
}

// hat(·) operator: φ -> φ^ (skew)
static inline Eigen::Matrix<Expression,3,3> hat(const Vec<Expression,3>& a) {
	Eigen::Matrix<Expression,3,3> A;
	A << Expression(0), -a(2),        a(1),
		a(2),         Expression(0), -a(0),
		-a(1),         a(0),          Expression(0);
	return A;
}

namespace so3 {

	template <typename T>
	inline Eigen::Matrix<T,3,1> vee(const Eigen::Matrix<T,3,3>& S) {
		return Eigen::Matrix<T,3,1>(S(2,1)-S(1,2), S(0,2)-S(2,0), S(1,0)-S(0,1)) * T(0.5);
	}

	template <typename T>
	inline Eigen::Matrix<T,3,3> hat(const Eigen::Matrix<T,3,1>& w) {
		Eigen::Matrix<T,3,3> W;
		W << T(0), -w(2),  w(1),
			w(2),   T(0), -w(0),
			-w(1),  w(0),  T(0);
		return W;
	}

	namespace quat {
		template <typename T>
		inline Eigen::Quaternion<T> Exp(const Eigen::Matrix<T,3,1>& phi) {
			const T theta = phi.norm();
			if (theta < T(1e-12)) {
				// q ≈ [1, 0.5*phi]
				const Eigen::Matrix<T,3,1> half = T(0.5) * phi;
				return Eigen::Quaternion<T>(T(1), half(0), half(1), half(2)).normalized();
			}
			const Eigen::Matrix<T,3,1> a = phi / theta;
			const T half = T(0.5) * theta;
			const T s = std::sin(half);
			const T c = std::cos(half);
			return Eigen::Quaternion<T>(c, a(0)*s, a(1)*s, a(2)*s);
		}

		template <typename T>
		inline Eigen::Matrix<T,3,1> Log(const Eigen::Quaternion<T>& q_in) {
			// Canonicalize sign so w >= 0 to pick shortest rotation
			Eigen::Quaternion<T> q = (q_in.w() < T(0)) ? Eigen::Quaternion<T>(-q_in.w(), -q_in.x(), -q_in.y(), -q_in.z()) : q_in;
			T w = q.w();
			Eigen::Matrix<T,3,1> v(q.x(), q.y(), q.z());
			const T n = v.norm();
			const T eps = T(1e-12);
			if (n < eps) {
				// Log ~ 2*v (since angle ~ 2*|v| and axis ~ v/|v|)
				return T(2) * v;
			}

			if constexpr (std::is_same_v<T, Expression>) {
				// angle θ = 2 atan2(|v|, w), axis = v/|v|
				const T theta = T(2) * drake::symbolic::atan2(n, w);
				return (theta / n) * v;
			} else {
				const T theta = T(2) * std::atan2(n, w);
				return (theta / n) * v;
			}
		}
	}

	namespace mat {

		template <typename T>
		inline drake::math::RotationMatrix<T> Exp(const Eigen::Matrix<T,3,1>& phi) {
			const T theta_sq = phi.squaredNorm();
			const T theta = sqrt(theta_sq);

			Eigen::Matrix<T, 3, 3> phi_hat = hat(phi);

			const Eigen::Matrix<T, 3, 3> phi_hat_sq = phi_hat * phi_hat;
			Eigen::Matrix<T, 3, 3> R_mat;

			if (theta < T(1e-6)) {
				// Taylor expansion:
				// sin(theta)/theta approx 1 - theta^2 / 6
				// (1-cos(theta))/theta^2 approx 0.5 - theta^2 / 24
				R_mat = Eigen::Matrix<T, 3, 3>::Identity() +
					(T(1.0) - theta_sq / T(6.0)) * phi_hat +
					(T(0.5) - theta_sq / T(24.0)) * phi_hat_sq;
			} else {
				R_mat = Eigen::Matrix<T, 3, 3>::Identity() +
					(std::sin(theta) / theta) * phi_hat +
					((T(1.0) - std::cos(theta)) / theta_sq) * phi_hat_sq;
			}

			return drake::math::RotationMatrix<T>(R_mat);
		}

		template <typename T>
		inline Eigen::Matrix<T,3,1> Log(const drake::math::RotationMatrix<T>& Rrel) {
			Eigen::AngleAxis<T> aa = Rrel.ToAngleAxis();
			const T theta = aa.angle();
			const Eigen::Matrix<T,3,1> axis = aa.axis();
			return theta * axis;
		}

	}

	template <typename T>
	inline Eigen::Matrix<T,3,3> left_jacobian(const Eigen::Matrix<T,3,1>& phi) {
		const T theta = phi.norm();
		const T eps = T(1e-12);
		const Eigen::Matrix<T,3,3> I = Eigen::Matrix<T,3,3>::Identity();
		const auto S = hat(phi);
		if (theta < eps) {
			// Series: J ≈ I + 0.5 S + (1/6) S^2
			return I + T(0.5)*S + (T(1)/T(6))*S*S;
		}
		const T theta2 = theta*theta;
		const T A = (T(1) - std::cos(theta)) / theta2;
		const T B = (theta - std::sin(theta)) / (theta*theta2);
		return I + A*S + B*S*S;
	}

	template <typename T>
	inline Eigen::Matrix<T,3,3> left_jacobian_inv(const Eigen::Matrix<T,3,1>& phi) {
		const T theta = phi.norm();
		const T eps = T(1e-10);
		const Eigen::Matrix<T,3,3> I = Eigen::Matrix<T,3,3>::Identity();
		const auto S = hat(phi);
		if (theta < eps) {
			// J^{-1} ≈ I - 0.5 S + (1/12) S^2
			return I - T(0.5)*S + (T(1)/T(12))*S*S;
		}
		const T half = T(0.5)*theta;
		const T cot_half = std::cos(half)/std::sin(half);
		// J^{-1} = I - 0.5 S + (1 - theta*cot(theta/2)) / theta^2 * S^2
		const T C = (T(1) - theta * cot_half) / (theta*theta);
		return I - T(0.5)*S + C * (S*S);
	}

	template <typename T>
	inline Eigen::Matrix<T,3,1> d_left_jacobian_times_phidot(
		const Eigen::Matrix<T,3,1>& phi,
		const Eigen::Matrix<T,3,1>& phidot) {
		const T theta = phi.norm();
		const T eps = T(1e-10);
		if (theta < eps) {
			// Series around 0: J = I + 0.5 S + (1/6) S^2 + ...
			// dJ/dt * phidot ≈ 0.5 * hat(phidot) * phidot + (1/6)(hat(phidot) hat(phi) + hat(phi) hat(phidot)) phidot
			// The first term is zero since hat(phidot) phidot = 0.
			// Keep the leading nonzero term compactly:
			const auto S = hat(phi);
			const auto Sd = hat(phidot);
			return (T(1)/T(6)) * (Sd * (S * phidot) + S * (Sd * phidot));
		}
		// Use J = I + A S + B S^2, with A=(1-cosθ)/θ^2, B=(θ-sinθ)/θ^3
		const auto S  = hat(phi);
		const auto Sd = hat(phidot);
		const T theta2 = theta*theta;
		const T theta3 = theta2*theta;
		const T A  = (T(1) - std::cos(theta)) / theta2;
		const T B  = (theta - std::sin(theta)) / theta3;
		const T Ad = (theta*std::sin(theta) + T(2)*std::cos(theta) - T(2)) / (theta*theta2);   // dA/dθ / chain later
		const T Bd = (-theta*std::cos(theta) - T(2)*theta + T(3)*std::sin(theta)) / (theta*theta*theta*theta); // dB/dθ
		const T thetadot = (phi.dot(phidot)) / theta;

		const T dA_dt = Ad * thetadot;
		const T dB_dt = Bd * thetadot;

		// Precompute terms
		const Eigen::Matrix<T,3,1> Sphidot = S * phidot;                  // φ × φ̇
		const Eigen::Matrix<T,3,1> S2phidot = S * Sphidot;                // φ × (φ × φ̇)
		const Eigen::Matrix<T,3,1> Sd_S_phidot = Sd * Sphidot;            // φ̇ × (φ × φ̇)

		// dotJ * phidot = dA*S*phidot + A*Sd*phidot + dB*S^2*phidot + B*(Sd*S + S*Sd)*phidot
		// but Sd*phidot=0 and S*Sd*phidot = 0, so:
		return dA_dt * Sphidot + dB_dt * S2phidot + B * Sd_S_phidot;
	}

}

// --------- Torus helpers ------------
namespace torus {
	inline double wrap_pi(double a) {
		// Wrap to (-pi, pi]
		double r = std::fmod(a + M_PI, 2.0*M_PI);
		if (r <= 0) r += 2.0*M_PI;
		return r - M_PI;
	}
	inline double shortest_delta(double a1, double a0) {
		return wrap_pi(a1 - a0);
	}
} // namespace torus

// ==========================================================================
// CubicConfigurationSpline
//   Block-structured Hermite cubics on R^n, Torus(k), and SO(3) (quaternions).
//   - Ambient layout: concat blocks: R(k): k scalars; Torus(k): k angles (rad);
//     SO(3): unit quaternion [w, x, y, z]
//   - Tangent layout: concat blocks: R(k): k; Torus(k): k; SO(3): 3 (body ω)
// ==========================================================================

class CubicConfigurationSpline {
public:
	struct Block {
		enum class Type { R, Torus, SO3Quat, SO3Mat /*, SE3 (TODO)*/ };
		Type type;
		int  size;   // ambient size; for SO3Quat size == 4; for R/Torus size == k

		static Block R(int k)      { return Block{Type::R, k}; }
		static Block Torus(int k)  { return Block{Type::Torus, k}; }
		static Block SO3Quat()     { return Block{Type::SO3Quat, 4}; } // ambient 4 (quat)
		static Block SO3Mat()      { return Block{Type::SO3Mat, 9}; }
	};

	struct BlockOffset {
		int ambient_offset;
		int tangent_offset;
		int ambient_size;
		int tangent_size;
		Block::Type type;
	};

	struct EuclPiece {
		double tau{};
		Eigen::VectorXd x0, v0, x1, v1;
		Eigen::VectorXd a, b, c, d; // x(t) = d + t c + t^2 b + t^3 a
		int dim() const { return static_cast<int>(d.size()); }
	};

	struct TorusPiece {
		double tau{};
		Eigen::VectorXd a0, v0, delta, v1;
		Eigen::VectorXd a, b, c, d; // φ(t) cubic; q(t)=wrap(a0 + φ(t))
		int dim() const { return static_cast<int>(c.size()); }
	};

	struct SO3QuatPiece {
		double tau{};
		Eigen::Quaterniond q0;
		Eigen::Vector3d omega0, omega1;
		Eigen::Vector3d phi1; // log(q0^{-1} q1)
		Eigen::Vector3d a, b, c, d; // φ(t) cubic
	};

	struct SO3MatPiece {
		double tau{};
		drake::math::RotationMatrix<double> R0;
		Eigen::Vector3d omega0, omega1;
		Eigen::Vector3d phi1; // log(R0^{-1} R1)
		Eigen::Vector3d a, b, c, d; // φ(t) cubic
	};

	struct SegmentPiece {
		double t0{}, t1{};
		std::vector<std::variant<EuclPiece, TorusPiece, SO3QuatPiece, SO3MatPiece>> blocks;
	};

	using Spec = std::vector<Block>;

	Spec spec_;
	int ambient_dim_ = 0;
	int tan_dim_ = 0;
	std::vector<Block> blocks_;
	std::vector<BlockOffset> block_offsets_;

	Eigen::VectorXd times_;
	std::vector<SegmentPiece> pieces_;

	CubicConfigurationSpline()
		: block_offsets_(0),
		  blocks_(0) {}

	explicit CubicConfigurationSpline(Spec spec)
		: block_offsets_(0),
		  blocks_(0) { set_spec(std::move(spec)); }

	void set_spec(Spec spec) {
		spec_ = std::move(spec);
		ambient_dim_ = 0;
		tan_dim_ = 0;
		blocks_.clear();
		block_offsets_.clear();
		for (const auto& b : spec_) {
			int a_off = ambient_dim_;
			int t_off = tan_dim_;
			int a_sz, t_sz;
			if (b.type == Block::Type::SO3Quat) {
				a_sz  = 4;
				t_sz  = 3;
			} else if (b.type == Block::Type::SO3Mat) {
				a_sz  = 9;
				t_sz  = 3;
			} else {
				a_sz  = b.size;
				t_sz  = b.size;
			}

			ambient_dim_ += a_sz;
			tan_dim_ += t_sz;
			blocks_.push_back(b);
			block_offsets_.emplace_back(a_off, t_off, a_sz, t_sz, b.type);
		}
		clear();
	}

	int ambient_dim() const { return ambient_dim_; }
	int tangent_dim() const { return tan_dim_; }

	void clear() {
		times_.resize(0);
		pieces_.clear();
	}

	// ----- API: set from waypoints/vels/times -----
	// pts: (N x ambient_dim), vels: (N x tangent_dim), times: (N)
	template <typename DerivedP, typename DerivedV, typename DerivedT>
	void set(const Eigen::MatrixBase<DerivedP>& pts,
		 const Eigen::MatrixBase<DerivedV>& vels,
		 const Eigen::MatrixBase<DerivedT>& times) {
		using std::size_t;
		const int N = static_cast<int>(times.size());
		if (N < 2) throw std::runtime_error("Need at least 2 waypoints");
		if (pts.rows() != N || pts.cols() != ambient_dim_)
			throw std::runtime_error("pts has wrong shape");
		if (vels.rows() != N || vels.cols() != tan_dim_)
			throw std::runtime_error("vels has wrong shape");

		times_ = times;
		if ((times_.array().segment(1, N-1) - times_.array().segment(0, N-1)).minCoeff() <= 0.0)
			throw std::runtime_error("times must be strictly increasing");

		pieces_.clear();
		pieces_.reserve(N-1);

		for (int i = 0; i < N-1; ++i) {
			double t0 = times_(i);
			double t1 = times_(i+1);
			const double tau = t1 - t0;

			SegmentPiece seg;
			seg.t0 = t0; seg.t1 = t1;

			// Build each block's piece
			int a_off = 0, t_off = 0;
			for (const auto& bo : block_offsets_) {
				switch (bo.type) {
				case Block::Type::R: {
					EuclPiece p;
					p.tau = tau;
					const int k = bo.ambient_size;
					p.x0 = pts.row(i).segment(a_off, k).transpose();
					p.v0 = vels.row(i).segment(t_off, k).transpose();
					p.x1 = pts.row(i+1).segment(a_off, k).transpose();
					p.v1 = vels.row(i+1).segment(t_off, k).transpose();
					// Hermite coeffs in R^k for x(t) = d + t c + t^2 b + t^3 a
					p.d = p.x0;
					p.c = p.v0;
					p.b = (3.0*(p.x1 - p.x0) - tau*(2.0*p.v0 + p.v1)) / (tau*tau);
					p.a = (-2.0*(p.x1 - p.x0) + tau*(p.v0 + p.v1)) / (tau*tau*tau);
					seg.blocks.emplace_back(std::move(p));
					a_off += k; t_off += k;
					break;
				}
				case Block::Type::Torus: {
					TorusPiece p;
					p.tau = tau;
					const int k = bo.ambient_size;
					p.a0 = pts.row(i).segment(a_off, k).transpose();
					p.v0 = vels.row(i).segment(t_off, k).transpose();
					p.delta = Eigen::VectorXd(k);
					p.v1 = vels.row(i+1).segment(t_off, k).transpose();
					for (int j = 0; j < k; ++j) {
						const double a0 = p.a0(j);
						const double a1 = pts(i+1, a_off + j);
						const double d  = torus::shortest_delta(a1, a0);
						p.delta(j) = d;
					}
					// φ(t) cubic with φ(0)=0, φ'(0)=v0, φ(τ)=delta, φ'(τ)=v1
					p.d = Eigen::VectorXd::Zero(k);
					p.c = p.v0;
					p.b = (3.0*p.delta - tau*(2.0*p.v0 + p.v1)) / (tau*tau);
					p.a = (-2.0*p.delta + tau*(p.v0 + p.v1)) / (tau*tau*tau);
					seg.blocks.emplace_back(std::move(p));
					a_off += k; t_off += k;
					break;
				}
				case Block::Type::SO3Quat: {
					SO3QuatPiece p;
					p.tau = tau;
					// Read and canonicalize quats (w,x,y,z)
					Eigen::Quaterniond q0(pts(i, a_off+0), pts(i, a_off+1), pts(i, a_off+2), pts(i, a_off+3));
					Eigen::Quaterniond q1(pts(i+1, a_off+0), pts(i+1, a_off+1), pts(i+1, a_off+2), pts(i+1, a_off+3));
					q0.normalize(); if (q0.w() < 0) q0.coeffs() *= -1.0;
					q1.normalize(); if (q1.w() < 0) q1.coeffs() *= -1.0;
					// Body angular velocities
					p.omega0 = vels.row(i).segment(t_off, 3).transpose();
					p.omega1 = vels.row(i+1).segment(t_off, 3).transpose();

					// Relative motion φ = Log(q0^{-1} * q1), in so(3)
					Eigen::Quaterniond qrel = q0.conjugate() * q1;
					p.phi1 = so3::quat::Log(qrel);

					// Cubic in φ(t): φ(0)=0, φ'(0)=v0φ, φ(τ)=φ1, φ'(τ)=v1φ
					// v0φ = ω0 (since J(0)=I), v1φ = J(φ1)^{-1} ω1
					const Eigen::Matrix3d J1_inv = so3::left_jacobian_inv(p.phi1);
					const Eigen::Vector3d v0phi = p.omega0;
					const Eigen::Vector3d v1phi = J1_inv * p.omega1;
					p.d = Eigen::Vector3d::Zero();
					p.c = v0phi;
					p.b = (3.0*p.phi1 - tau*(2.0*v0phi + v1phi)) / (tau*tau);
					p.a = (-2.0*p.phi1 + tau*(v0phi + v1phi)) / (tau*tau*tau);

					p.q0 = q0;
					seg.blocks.emplace_back(std::move(p));
					a_off += 4; t_off += 3;
					break;
				}
				case Block::Type::SO3Mat: {
					SO3MatPiece p;
					p.tau = tau;

					// Read rotation matricies
					// The validity of these should be
					// enforced by the constraints of the
					// waypoint problem or their initialization.
					Eigen::Matrix3d R0_mat;
					R0_mat << pts(i, a_off+0), pts(i, a_off+1), pts(i, a_off+2),
						pts(i, a_off+3), pts(i, a_off+4), pts(i, a_off+5),
						pts(i, a_off+6), pts(i, a_off+7), pts(i, a_off+8);
					drake::math::RotationMatrix<double> R0(R0_mat);
					Eigen::Matrix3d R1_mat;
					R1_mat << pts(i+1, a_off+0), pts(i+1, a_off+1), pts(i+1, a_off+2),
						pts(i+1, a_off+3), pts(i+1, a_off+4), pts(i+1, a_off+5),
						pts(i+1, a_off+6), pts(i+1, a_off+7), pts(i+1, a_off+8);
					drake::math::RotationMatrix<double> R1(R1_mat);

					// Body angular velocities
					p.omega0 = vels.row(i).segment(t_off, 3).transpose();
					p.omega1 = vels.row(i+1).segment(t_off, 3).transpose();

					// Relative motion φ = Log(R0^{-1} * R1), in so(3)
					drake::math::RotationMatrix<double> Rrel = R0.inverse() * R1;
					p.phi1 = so3::mat::Log(Rrel);

					// Cubic in φ(t): φ(0)=0, φ'(0)=v0φ, φ(τ)=φ1, φ'(τ)=v1φ
					// v0φ = ω0 (since J(0)=I), v1φ = J(φ1)^{-1} ω1
					const Eigen::Matrix3d J1_inv = so3::left_jacobian_inv(p.phi1);
					const Eigen::Vector3d v0phi = p.omega0;
					const Eigen::Vector3d v1phi = J1_inv * p.omega1;
					p.d = Eigen::Vector3d::Zero();
					p.c = v0phi;
					p.b = (3.0*p.phi1 - tau*(2.0*v0phi + v1phi)) / (tau*tau);
					p.a = (-2.0*p.phi1 + tau*(v0phi + v1phi)) / (tau*tau*tau);

					p.R0 = R0;
					seg.blocks.emplace_back(std::move(p));
					a_off += 9; t_off += 3;
					break;
				}
				}
			}

			pieces_.push_back(std::move(seg));
		}
	}

	bool initialized() const { return times_.size() >= 2; }
	int num_pieces() const { return static_cast<int>(pieces_.size()); }
	double begin() const {
		if (!initialized()) throw std::runtime_error("empty");
		return times_(0);
	}
	double end() const {
		if (!initialized()) throw std::runtime_error("empty");
		return times_(times_.size()-1);
	}

	// Evaluate at global time t (clamped to [begin, end]).
	// Returns ambient q(t), tangent v(t), tangent a(t).
	struct Eval {
		Eigen::VectorXd q_ambient;
		Eigen::VectorXd v_tangent;
		Eigen::VectorXd a_tangent;
	};

	Eval eval(double t) const {
		if (!initialized()) throw std::runtime_error("CubicConfigurationSpline not initialized");
		if      (t <= begin()) t = begin();
		else if (t >= end())   t = end();

		const int k = find_piece(t);
		const auto& seg = pieces_[k];
		const double tau = seg.t1 - seg.t0;
		const double local_t = t - seg.t0;

		Eigen::VectorXd q(ambient_dim_);
		Eigen::VectorXd v(tan_dim_);
		Eigen::VectorXd a(tan_dim_);

		int a_off = 0, t_off = 0;
		for (size_t bi = 0; bi < seg.blocks.size(); ++bi) {
			const auto& bo = block_offsets_[bi];
			const auto& blk = seg.blocks[bi];
			if (std::holds_alternative<EuclPiece>(blk)) {
				const auto& p = std::get<EuclPiece>(blk);
				const double tt = local_t;
				Eigen::VectorXd x  = p.d + tt * (p.c + tt*(p.b + tt*p.a));
				Eigen::VectorXd xd = p.c + tt * (2.0*p.b + tt*3.0*p.a);
				Eigen::VectorXd xdd= 2.0*p.b + tt*6.0*p.a;

				q.segment(a_off, p.dim()) = x;
				v.segment(t_off, p.dim()) = xd;
				a.segment(t_off, p.dim()) = xdd;

				a_off += p.dim(); t_off += p.dim();
			} else if (std::holds_alternative<TorusPiece>(blk)) {
				const auto& p = std::get<TorusPiece>(blk);
				const double tt = local_t;
				Eigen::VectorXd phi  = p.d + tt * (p.c + tt*(p.b + tt*p.a));
				Eigen::VectorXd phid = p.c + tt * (2.0*p.b + tt*3.0*p.a);
				Eigen::VectorXd phidd= 2.0*p.b + tt*6.0*p.a;

				const int kdim = p.dim();
				for (int j = 0; j < kdim; ++j) {
					q(a_off + j) = torus::wrap_pi(p.a0(j) + phi(j));
				}
				v.segment(t_off, kdim) = phid;
				a.segment(t_off, kdim) = phidd;

				a_off += kdim; t_off += kdim;
			} else if (std::holds_alternative<SO3QuatPiece>(blk)) {
				const auto& p = std::get<SO3QuatPiece>(blk);
				const double tt = local_t;
				Eigen::Matrix<double,3,1> phi  = p.d + tt * (p.c + tt*(p.b + tt*p.a));
				Eigen::Matrix<double,3,1> phid = p.c + tt * (2.0*p.b + tt*3.0*p.a);
				Eigen::Matrix<double,3,1> phidd= 2.0*p.b + tt*6.0*p.a;

				// q(t) = q0 * Exp(phi(t))
				Eigen::Quaterniond qe = so3::quat::Exp(phi);
				Eigen::Quaterniond qt = (p.q0 * qe).normalized();
				if (qt.w() < 0) qt.coeffs() *= -1.0; // canonicalize

				// ω = J(phi) φdot,  ωdot = J φ¨ + (dJ/dt) φdot
				Eigen::Matrix3d J   = so3::left_jacobian(phi);
				Eigen::Vector3d omg = J * phid;
				Eigen::Vector3d dJphid = so3::d_left_jacobian_times_phidot(phi, phid);
				Eigen::Vector3d omgd = J * phidd + dJphid;

				q(a_off+0) = qt.w(); q(a_off+1) = qt.x(); q(a_off+2) = qt.y(); q(a_off+3) = qt.z();
				v.segment(t_off, 3) = omg;
				a.segment(t_off, 3) = omgd;

				a_off += 4; t_off += 3;
			} else if (std::holds_alternative<SO3MatPiece>(blk)) {
				const auto& p = std::get<SO3MatPiece>(blk);
				const double tt = local_t;
				Eigen::Matrix<double,3,1> phi  = p.d + tt * (p.c + tt*(p.b + tt*p.a));
				Eigen::Matrix<double,3,1> phid = p.c + tt * (2.0*p.b + tt*3.0*p.a);
				Eigen::Matrix<double,3,1> phidd= 2.0*p.b + tt*6.0*p.a;

				// q(t) = q0 * Exp(phi(t))
				drake::math::RotationMatrix<double> Re = so3::mat::Exp(phi);
				drake::math::RotationMatrix<double> Rt = p.R0 * Re;

				// ω = J(phi) φdot,  ωdot = J φ¨ + (dJ/dt) φdot
				Eigen::Matrix3d J   = so3::left_jacobian(phi);
				Eigen::Vector3d omg = J * phid;
				Eigen::Vector3d dJphid = so3::d_left_jacobian_times_phidot(phi, phid);
				Eigen::Vector3d omgd = J * phidd + dJphid;

				q.segment(a_off+0, 3) = Rt.row(0);
				q.segment(a_off+3, 3) = Rt.row(1);
				q.segment(a_off+6, 3) = Rt.row(2);
				v.segment(t_off, 3) = omg;
				a.segment(t_off, 3) = omgd;

				a_off += 9; t_off += 3;
			} else {
				DRAKE_UNREACHABLE();
			}
		}
		return Eval{std::move(q), std::move(v), std::move(a)};
	}

	std::pair<Eigen::MatrixXd, Eigen::MatrixXd> eval_multiple(const Eigen::VectorXd& times) const {
		const int N = static_cast<int>(times.size());
		Eigen::MatrixXd Q  = Eigen::MatrixXd::Zero(N, ambient_dim_);
		Eigen::MatrixXd V  = Eigen::MatrixXd::Zero(N, tan_dim_);

		for (int i = 0; i < N; ++i) {
			const Eval e = eval(times(i));     // clamps to [begin(), end()] internally
			Q.row(i) = e.q_ambient.transpose();
			V.row(i) = e.v_tangent.transpose();
		}
		return {std::move(Q), std::move(V)};
	}

	Expression squared_distance(
		const VecX<Expression>& q1,
		const VecX<Expression>& q2) const {

		Expression total(0.0);
		for (const struct BlockOffset off : block_offsets_) {
			const int a0 = off.ambient_offset;
			const int aN = off.ambient_size;

			switch (off.type) {
			case Block::Type::R: {
				const auto d = q1.segment(a0, aN) - q2.segment(a0, aN);
				total += d.squaredNorm();
				break;
			}
			case Block::Type::Torus: {
				VecX<Expression> dw(aN);
				for (int k = 0; k < aN; ++k) {
					dw[k] = wrap_to_pi(q1[a0 + k] - q2[a0 + k]);
				}
				total += dw.squaredNorm();
				break;
			}
			case Block::Type::SO3Quat: {
				// ambient 4: (w,x,y,z)
				DRAKE_DEMAND(aN == 4);
				const auto q1wxyz = q1.segment(a0, 4);
				const auto q2wxyz = q2.segment(a0, 4);

				// total += (q2wxyz - q1wxyz).squaredNorm();
				// total += (1 - drake::symbolic::abs(q1wxyz.dot(q2wxyz)));
				// total += drake::symbolic::min((q1wxyz - q2wxyz).squaredNorm(), (q1wxyz + q2wxyz).squaredNorm());
				total += (1 - (q1wxyz.dot(q2wxyz))*(q1wxyz.dot(q2wxyz)));
				break;
			}
			case Block::Type::SO3Mat: {
				DRAKE_DEMAND(aN == 9);

				Eigen::Matrix<Expression, 3, 3> R1;
				R1 << q1(a0+0), q1(a0+1), q1(a0+2),
					q1(a0+3), q1(a0+4), q1(a0+5),
					q1(a0+6), q1(a0+7), q1(a0+8);
				Eigen::Matrix<Expression, 3, 3> R2;
				R2 << q2(a0+0), q2(a0+1), q2(a0+2),
					q2(a0+3), q2(a0+4), q2(a0+5),
					q2(a0+6), q2(a0+7), q2(a0+8);

				// squared frobenius norm (chordal distance)
				total += (R2 - R1).squaredNorm();
				break;
			}

			default:
				DRAKE_UNREACHABLE();
			}
		}

		return total;
	}

	template <typename T>
	T compute_ctrl_cost(
		const VecX<T>& xJ,
		const VecX<T>& xJm1,
		const VecX<T>& vJ,
		const VecX<T>& vJm1,
		const T& tau) const {
		/* Minimum acceleration cost */

		const T inv_tau  = T(1.0) / tau;
		const T inv_tau2 = inv_tau * inv_tau;
		const T inv_tau3 = inv_tau2 * inv_tau;

		T total = T(0.0);

		for (const BlockOffset& off : block_offsets_) {
			const int a0 = off.ambient_offset, aN = off.ambient_size;
			const int t0 = off.tangent_offset, tN = off.tangent_size;

			switch (off.type) {

			case Block::Type::R: {
				const auto xj   = xJ.segment(a0, aN);
				const auto xjm1 = xJm1.segment(a0, aN);
				const auto vj   = vJ.segment(t0, tN);
				const auto vjm1 = vJm1.segment(t0, tN);

				const VecX<T> D = (xj - xjm1) - T(0.5) * tau * (vjm1 + vj);
				const VecX<T> V = (vj - vjm1);

				total += T(12.0) * inv_tau3 * D.squaredNorm()
				       + inv_tau * V.squaredNorm();
				break;
			}
			case Block::Type::Torus: {
				// Torus block: ambient == tangent == aN == tN
				// Use wrapped angle difference for the position residual (componentwise).
				const auto xj   = xJ.segment(a0, aN);
				const auto xjm1 = xJm1.segment(a0, aN);
				const auto vj   = vJ.segment(t0, tN);
				const auto vjm1 = vJm1.segment(t0, tN);

				VecX<T> Dw(aN);
				for (int k = 0; k < aN; ++k) {
					Dw[k] = wrap_to_pi(xj[k] - xjm1[k]);
				}
				const VecX<T> D = Dw - T(0.5) * tau * (vjm1 + vj);
				const VecX<T> V = (vj - vjm1);

				total += T(12.0) * inv_tau3 * D.squaredNorm()
					+ inv_tau * V.squaredNorm();
				break;
			}
			case Block::Type::SO3Quat: {
				DRAKE_DEMAND(aN == 4 && tN == 3);

				const Eigen::Matrix<T,4,1> qjm1 = xJm1.segment(a0, 4);
				const Eigen::Matrix<T,4,1> qj   = xJ.segment(a0, 4);
				const Vec<T,3> wjm1 = vJm1.segment(t0, 3);
				const Vec<T,3> wj   = vJ.segment(t0, 3);

				const auto Rjm1 = RotFromQuatWxyz(qjm1);
				const auto Rj   = RotFromQuatWxyz(qj);
				const auto Rrel = Rjm1.transpose() * Rj;

				const Vec<T,3> dphi = so3::mat::Log(Rrel);
				const Vec<T,3> D = dphi - T(0.5) * tau * (wjm1 + wj);
				const Vec<T,3> V = (wj - wjm1);

				total += T(12.0) * inv_tau3 * D.squaredNorm()
					+ inv_tau * V.squaredNorm();
				break;
			}
			case Block::Type::SO3Mat: {
				DRAKE_DEMAND(aN == 9 && tN == 3);
				break;
			}
			default:
				DRAKE_UNREACHABLE();
			}
		}

		return total;
	}

	template <typename T>
	T compute_energy_cost(
		const VecX<T>& xJ,
		const VecX<T>& xJm1,
		const VecX<T>& vJ,
		const VecX<T>& vJm1,
		const T& tau) const {

		const T inv_tau = T(1.0) / tau;
		T total = T(0.0);

		for (const BlockOffset& off : block_offsets_) {
			const int a0 = off.ambient_offset, aN = off.ambient_size;
			const int t0 = off.tangent_offset, tN = off.tangent_size;

			switch (off.type) {

			case Block::Type::R: {
				const auto xj   = xJ.segment(a0, aN);
				const auto xjm1 = xJm1.segment(a0, aN);
				const auto vj   = vJ.segment(t0, tN);
				const auto vjm1 = vJm1.segment(t0, tN);

				// Displacement (Delta x)
				const VecX<T> disp = (xj - xjm1); // Use wrap_to_pi for Torus, so3_log for SO3

				// Deviation from constant velocity (your existing D)
				const VecX<T> D = disp - T(0.5) * tau * (vjm1 + vj);

				// Velocity change (your existing V)
				const VecX<T> V = (vj - vjm1);

				// Analytical integral of velocity squared:
				// Term 1: The 'Straight Line' energy (Distance^2 / Time)
				// Term 2: The 'Wiggle' energy (Deviation from straight)
				// Term 3: The 'Acceleration' energy (Velocity change)
				total += inv_tau * disp.squaredNorm()
					+ T(2.4) * inv_tau * D.squaredNorm()
					+ T(1.0 / 60.0) * tau * V.squaredNorm();
				break;
			}
			case Block::Type::Torus: {
				// Use wrapped angle difference for the position residual (componentwise).
				const auto xj   = xJ.segment(a0, aN);
				const auto xjm1 = xJm1.segment(a0, aN);
				const auto vj   = vJ.segment(t0, tN);
				const auto vjm1 = vJm1.segment(t0, tN);

				// Displacement (Delta x)
				VecX<T> disp(aN);
				for (int k = 0; k < aN; ++k) {
					disp[k] = wrap_to_pi(xj[k] - xjm1[k]);
				}

				// Deviation from constant velocity (your existing D)
				const VecX<T> D = disp - T(0.5) * tau * (vjm1 + vj);

				// Velocity change (your existing V)
				const VecX<T> V = (vj - vjm1);

				// Analytical integral of velocity squared:
				// Term 1: The 'Straight Line' energy (Distance^2 / Time)
				// Term 2: The 'Wiggle' energy (Deviation from straight)
				// Term 3: The 'Acceleration' energy (Velocity change)
				total += inv_tau * disp.squaredNorm()
					+ T(2.4) * inv_tau * D.squaredNorm()
					+ T(1.0 / 60.0) * tau * V.squaredNorm();
				break;
			}
			case Block::Type::SO3Quat: {
				DRAKE_DEMAND(aN == 4 && tN == 3);

				const Eigen::Matrix<T,4,1> qjm1 = xJm1.segment(a0, 4);
				const Eigen::Matrix<T,4,1> qj   = xJ.segment(a0, 4);
				const Vec<T,3> wjm1 = vJm1.segment(t0, 3);
				const Vec<T,3> wj   = vJ.segment(t0, 3);

				const auto Rjm1 = RotFromQuatWxyz(qjm1);
				const auto Rj   = RotFromQuatWxyz(qj);
				const auto Rrel = Rjm1.transpose() * Rj;

				const Vec<T,3> disp = so3::mat::Log(Rrel);

				// Deviation from constant velocity (your existing D)
				const VecX<T> D = disp - T(0.5) * tau * (wjm1 + wj);

				// Velocity change (your existing V)
				const Vec<T,3> V = (wj - wjm1);

				// Analytical integral of velocity squared:
				// Term 1: The 'Straight Line' energy (Distance^2 / Time)
				// Term 2: The 'Wiggle' energy (Deviation from straight)
				// Term 3: The 'Acceleration' energy (Velocity change)
				total += inv_tau * disp.squaredNorm()
					+ T(2.4) * inv_tau * D.squaredNorm()
					+ T(1.0 / 60.0) * tau * V.squaredNorm();
				break;
			}
			case Block::Type::SO3Mat: {
				DRAKE_DEMAND(aN == 9 && tN == 3);
				break;
			}
			default:
				DRAKE_UNREACHABLE();
			}
		}
		return total;
	}

	template <typename T>
	T compute_arclength_cost(
		const VecX<T>& xJ,
		const VecX<T>& xJm1,
		const VecX<T>& vJ,
		const VecX<T>& vJm1,
		const T& tau) const {

		T total = T(0.0);

		// Gauss-Legendre 3-point weights and nodes for interval [0, 1]
		const std::vector<T> w = { T(5.0/18.0), T(8.0/18.0), T(5.0/18.0) };
		const std::vector<T> nodes = {
			T(0.5 - 0.5 * 0.7745966692), // 0.5 - 0.5 * sqrt(3/5)
			T(0.5),
			T(0.5 + 0.5 * 0.7745966692)  // 0.5 + 0.5 * sqrt(3/5)
		};

		for (const BlockOffset& off : block_offsets_) {
			const int a0 = off.ambient_offset, aN = off.ambient_size;
			const int t0 = off.tangent_offset, tN = off.tangent_size;

			switch (off.type) {

			case Block::Type::R: {
				const auto xj   = xJ.segment(a0, aN);
				const auto xjm1 = xJm1.segment(a0, aN);
				const auto vj   = vJ.segment(t0, tN);
				const auto vjm1 = vJm1.segment(t0, tN);

				/* For each quadrature point */
				for (int i = 0; i < 3; ++i) {
					T u = nodes[i];

					// Cubic Hermite derivative basis functions at u in [0, 1]
					// These calculate velocity p'(u) normalized by tau
					T h0_dot = T(6.0*u*u - 6.0*u);
					T h1_dot = T(3.0*u*u - 4.0*u + 1.0);
					T h2_dot = T(-6.0*u*u + 6.0*u);
					T h3_dot = T(3.0*u*u - 2.0*u);

					// Velocity at this point (p_dot = (1/tau) * dp/du)
					// Note: Since we integrate over dt, tau factors out:
					// integral ||p_dot(t)|| dt = integral ||dp/du|| du
					VecX<T> vel_u = h0_dot * (xj - xjm1) / tau // simplistic for brevity
						+ h1_dot * vjm1
						+ h3_dot * vj;

					// Add weighted norm to total
					total += w[i] * vel_u.norm() * tau;
				}

				break;
			}
			case Block::Type::Torus: {
				const auto vj   = vJ.segment(t0, tN);
				const auto vjm1 = vJm1.segment(t0, tN);

				T v0_norm = sqrt(vjm1.squaredNorm() + T(1e-8));
				T v1_norm = sqrt(vj.squaredNorm() + T(1e-8));
				total += tau * (v0_norm + v1_norm) / T(2.0);

				break;
			}
			case Block::Type::SO3Quat: {
				DRAKE_DEMAND(aN == 4 && tN == 3);

				const Vec<T,3> wjm1 = vJm1.segment(t0, 3);
				const Vec<T,3> wj   = vJ.segment(t0, 3);

				// Simple trapezoidal approximation for angular arc length
				// Arc length ≈ τ * (‖ω₀‖ + ‖ω₁‖) / 2
				T w0_norm = sqrt(wjm1.squaredNorm() + T(1e-8));
				T w1_norm = sqrt(wj.squaredNorm() + T(1e-8));
				total += tau * (w0_norm + w1_norm) / T(2.0);

				break;
			}
			case Block::Type::SO3Mat: {
				DRAKE_DEMAND(aN == 9 && tN == 3);
				break;
			}
			default:
				DRAKE_UNREACHABLE();
			}
		}
		return total;
	}

	template <typename T>
	std::pair<VecX<T>, VecX<T>> select_linear_blocks(
		const VecX<T>& x,
		const VecX<T>& v) const {

		int lin_size = 0;
		for (const BlockOffset& off : block_offsets_) {
			switch (off.type) {
			case Block::Type::R: {
				// ambient size and tangent size are equal here.
				lin_size += off.ambient_size;
				break;
			}
			default:
				;
			}
		}

		int i = 0;
		VecX<T> x_lin(lin_size);
		VecX<T> v_lin(lin_size);
		for (const BlockOffset& off : block_offsets_) {
			const int a0 = off.ambient_offset, aN = off.ambient_size;
			const int t0 = off.tangent_offset, tN = off.tangent_size;

			switch (off.type) {
			case Block::Type::R: {
				x_lin.segment(i, aN) = x.segment(a0, aN);
				v_lin.segment(i, tN) = v.segment(t0, tN);
				i += aN;
				break;
			}
			default:
				;
			}
		}
		return std::make_pair(x_lin, v_lin);
	}


private:
	// -------- Internal per-block piece types ----------
	int find_piece(double t) const {
		// Binary search; _times is strictly increasing
		int lo = 0, hi = static_cast<int>(times_.size()) - 2;
		while (lo <= hi) {
			int mid = (lo + hi) / 2;
			if (t < pieces_[mid].t1) hi = mid - 1;
			else lo = mid + 1;
		}
		int idx = std::max(0, hi + 1);
		if (idx >= static_cast<int>(pieces_.size())) idx = static_cast<int>(pieces_.size()) - 1;
		return idx;
	}
};
