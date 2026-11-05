#pragma once

#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <drake/common/symbolic/expression.h>

// Rows of symbolic expressions over a few variables, compiled once into a
// flat instruction list -- every distinct subexpression one instruction, so
// common subexpressions (a kinematic chain's shared rotations) are computed
// once -- and evaluated with their Jacobian by forward mode.
//
// Supported: constants, variables, +, *, /, constant powers, sqrt, sin,
// cos, exp, log, abs; anything else throws at construction.
class ExpressionTape {
public:
	ExpressionTape(const drake::VectorX<drake::symbolic::Expression>& rows,
		       const drake::VectorX<drake::symbolic::Variable>& vars);

	int num_rows() const { return static_cast<int>(_outputs.size()); }
	int num_vars() const { return _num_vars; }

	// The rows' values and their Jacobian (num_rows x num_vars) at x.
	void Eval(const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd* value, Eigen::MatrixXd* jacobian) const;

private:
	enum class Op { kConst, kVar, kAffine, kMul, kDiv, kPow, kSin, kCos, kExp, kLog, kAbs };
	struct Instr {
		Op op;
		double c = 0.0;     // constant / affine constant / power exponent
		int a = -1, b = -1;  // operands (instruction indices) / variable index
		int terms_begin = 0, terms_end = 0;  // kAffine: c + sum coef * value(operand)
	};

	int Compile(const drake::symbolic::Expression& e);
	int Push(Instr instr);

	int _num_vars;
	std::vector<Instr> _code;
	std::vector<std::pair<int, double>> _terms;
	std::vector<int> _outputs;
	std::unordered_map<drake::symbolic::Expression, int> _seen;
	std::unordered_map<drake::symbolic::Variable::Id, int> _var_index;
	// Scratch for Eval: each instruction's value and gradient.
	mutable Eigen::VectorXd _v;
	mutable Eigen::MatrixXd _d;
};
