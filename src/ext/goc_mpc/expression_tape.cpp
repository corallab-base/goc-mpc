#include "expression_tape.hpp"

#include <cmath>
#include <stdexcept>

#include <fmt/format.h>

namespace sym = drake::symbolic;
using sym::Expression;
using sym::ExpressionKind;

ExpressionTape::ExpressionTape(const drake::VectorX<Expression>& rows, const drake::VectorX<sym::Variable>& vars)
	: _num_vars(static_cast<int>(vars.size())) {
	for (int i = 0; i < vars.size(); ++i) _var_index.emplace(vars(i).get_id(), i);
	for (int r = 0; r < rows.size(); ++r) _outputs.push_back(Compile(rows(r)));
	_seen.clear();
	_v.resize(_code.size());
	_d.resize(_num_vars, _code.size());
}

int ExpressionTape::Push(Instr instr) {
	_code.push_back(instr);
	return static_cast<int>(_code.size()) - 1;
}

int ExpressionTape::Compile(const Expression& e) {
	if (const auto it = _seen.find(e); it != _seen.end()) return it->second;
	auto constant_exponent = [&](const Expression& exponent) {
		if (!sym::is_constant(exponent))
			throw std::runtime_error(fmt::format("ExpressionTape: only constant exponents, got {}", e.to_string()));
		return sym::get_constant_value(exponent);
	};
	auto power = [&](int base, double exponent) {
		if (exponent == 1.0) return base;
		Instr p{Op::kPow, exponent, base};
		return Push(p);
	};
	auto unary = [&](Op op) {
		Instr u{op, 0.0, Compile(sym::get_argument(e))};
		return Push(u);
	};
	int out;
	switch (e.get_kind()) {
	case ExpressionKind::Constant:
		out = Push({Op::kConst, sym::get_constant_value(e)});
		break;
	case ExpressionKind::Var: {
		const auto it = _var_index.find(sym::get_variable(e).get_id());
		if (it == _var_index.end())
			throw std::runtime_error(fmt::format("ExpressionTape: {} isn't one of the variables",
							     sym::get_variable(e).get_name()));
		out = Push({Op::kVar, 0.0, it->second});
		break;
	}
	case ExpressionKind::Add: {
		std::vector<std::pair<int, double>> terms;
		for (const auto& [term, coef] : sym::get_expr_to_coeff_map_in_addition(e))
			terms.emplace_back(Compile(term), coef);
		Instr sum{Op::kAffine, sym::get_constant_in_addition(e)};
		sum.terms_begin = static_cast<int>(_terms.size());
		_terms.insert(_terms.end(), terms.begin(), terms.end());
		sum.terms_end = static_cast<int>(_terms.size());
		out = Push(sum);
		break;
	}
	case ExpressionKind::Mul: {
		std::vector<int> factors;
		for (const auto& [base, exponent] : sym::get_base_to_exponent_map_in_multiplication(e))
			factors.push_back(power(Compile(base), constant_exponent(exponent)));
		int acc = factors[0];
		for (size_t k = 1; k < factors.size(); ++k) acc = Push({Op::kMul, 0.0, acc, factors[k]});
		const double c = sym::get_constant_in_multiplication(e);
		if (c != 1.0) {
			Instr scale{Op::kAffine, 0.0};
			scale.terms_begin = static_cast<int>(_terms.size());
			_terms.emplace_back(acc, c);
			scale.terms_end = static_cast<int>(_terms.size());
			acc = Push(scale);
		}
		out = acc;
		break;
	}
	case ExpressionKind::Div: {
		const int a = Compile(sym::get_first_argument(e));
		const int b = Compile(sym::get_second_argument(e));
		out = Push({Op::kDiv, 0.0, a, b});
		break;
	}
	case ExpressionKind::Pow:
		out = power(Compile(sym::get_first_argument(e)), constant_exponent(sym::get_second_argument(e)));
		break;
	case ExpressionKind::Sqrt:
		out = power(Compile(sym::get_argument(e)), 0.5);
		break;
	case ExpressionKind::Sin: out = unary(Op::kSin); break;
	case ExpressionKind::Cos: out = unary(Op::kCos); break;
	case ExpressionKind::Exp: out = unary(Op::kExp); break;
	case ExpressionKind::Log: out = unary(Op::kLog); break;
	case ExpressionKind::Abs: out = unary(Op::kAbs); break;
	default:
		throw std::runtime_error(fmt::format(
			"ExpressionTape: unsupported operation in {} (constants, variables, +, *, /, constant powers, "
			"sqrt, sin, cos, exp, log, abs only)", e.to_string()));
	}
	_seen.emplace(e, out);
	return out;
}

void ExpressionTape::Eval(const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd* value,
			  Eigen::MatrixXd* jacobian) const {
	for (size_t i = 0; i < _code.size(); ++i) {
		const Instr& in = _code[i];
		auto d = _d.col(i);
		switch (in.op) {
		case Op::kConst:
			_v(i) = in.c;
			d.setZero();
			break;
		case Op::kVar:
			_v(i) = x(in.a);
			d.setZero();
			d(in.a) = 1.0;
			break;
		case Op::kAffine:
			_v(i) = in.c;
			d.setZero();
			for (int t = in.terms_begin; t < in.terms_end; ++t) {
				const auto& [operand, coef] = _terms[t];
				_v(i) += coef * _v(operand);
				d += coef * _d.col(operand);
			}
			break;
		case Op::kMul:
			_v(i) = _v(in.a) * _v(in.b);
			d = _v(in.b) * _d.col(in.a) + _v(in.a) * _d.col(in.b);
			break;
		case Op::kDiv:
			_v(i) = _v(in.a) / _v(in.b);
			d = (_d.col(in.a) - _v(i) * _d.col(in.b)) / _v(in.b);
			break;
		case Op::kPow:
			_v(i) = std::pow(_v(in.a), in.c);
			d = (in.c * std::pow(_v(in.a), in.c - 1.0)) * _d.col(in.a);
			break;
		case Op::kSin:
			_v(i) = std::sin(_v(in.a));
			d = std::cos(_v(in.a)) * _d.col(in.a);
			break;
		case Op::kCos:
			_v(i) = std::cos(_v(in.a));
			d = -std::sin(_v(in.a)) * _d.col(in.a);
			break;
		case Op::kExp:
			_v(i) = std::exp(_v(in.a));
			d = _v(i) * _d.col(in.a);
			break;
		case Op::kLog:
			_v(i) = std::log(_v(in.a));
			d = _d.col(in.a) / _v(in.a);
			break;
		case Op::kAbs:
			_v(i) = std::abs(_v(in.a));
			d = (_v(in.a) >= 0.0 ? 1.0 : -1.0) * _d.col(in.a);
			break;
		}
	}
	value->resize(_outputs.size());
	jacobian->resize(_outputs.size(), _num_vars);
	for (size_t r = 0; r < _outputs.size(); ++r) {
		(*value)(r) = _v(_outputs[r]);
		jacobian->row(r) = _d.col(_outputs[r]).transpose();
	}
}
