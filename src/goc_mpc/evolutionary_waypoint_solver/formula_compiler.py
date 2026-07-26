"""Compiles drake::symbolic Formula/Expression trees (as produced by
GraphOfConstraints' unified symbolic constraint API -- add_constraint,
add_assignable_constraint, add_edge_constraint, add_conditional_edge_ordering)
into jax-traceable functions, via pydrake.symbolic's Unapply() recursive
pattern-matching. Two independent compilers live here, for two different jobs:

  - compile_condition: Formula -> gate(owner_vagent, cond_binary) -> jnp bool.
    Python analog of milp_waypoint_mpc.cpp's CompileVarEq/CompileConditionToBinary
    (cpp:741-886), retargeted to emit boolean gates instead of MILP big-M
    binaries. Narrow vocabulary (VarEq/BinVarEq atoms, And/Or/Not) matching
    conditional-ordering formulas' actual grammar exactly.

  - compile_expression / compile_relational_formula: Expression/Formula ->
    jax-traceable value/residual functions, for general algebraic node/edge
    constraints (add_constraint et al). Wider arithmetic vocabulary (Add, Mul,
    Div, Pow, transcendental functions) since these formulas encode real
    waypoint-value constraints, not just boolean conditions over discrete
    assignment variables.

    add_edge_constraint formulas compile through this same
    compile_relational_formula path regardless of which of its two forms
    they take -- spec.py just varies how many times and against which
    row(s) the resulting fn is registered: once over both endpoint rows for
    a *relational* formula (u_agent_q/v_agent_q etc.), or once against a
    single-row resolver but registered twice (once per endpoint) for an
    "along the edge" formula (the plain agent_q/object_q/var_agent_q
    placeholders) -- see GraphOrderingSpec._resolve_symbolic_constraints.
"""

import jax.numpy as jnp
import pydrake.symbolic as sym


def as_variable(expr):
    """Returns the underlying Variable if `expr` is a bare symbolic variable
    (ExpressionKind.Var), else None."""
    if expr.get_kind() != sym.ExpressionKind.Var:
        return None
    _, operands = expr.Unapply()
    return operands[0]


def compile_condition(formula, var_sym_ids: dict, cond_sym_ids: dict):
    """Compiles a Drake symbolic::Formula (as produced by
    GraphOfConstraints.add_conditional_edge_ordering / conditional_ordering_map)
    into gate(owner_vagent, cond_binary) -> jnp bool scalar, traceable under
    jax.jit/jax.vmap.

    var_sym_ids: Variable.get_id() -> index into owner_vagent (built from
        graph.assignment_sym(v) for v in range(graph.num_variables)).
    cond_sym_ids: Variable.get_id() -> index into cond_binary (built from
        graph.binary_cond_sym_vars).
    """
    kind = formula.get_kind()

    if kind == sym.FormulaKind.Eq:
        _, (lhs, rhs) = formula.Unapply()

        lv, rv = as_variable(lhs), as_variable(rhs)
        if lv is not None and rv is not None and \
           lv.get_id() in var_sym_ids and rv.get_id() in var_sym_ids:
            i0, i1 = var_sym_ids[lv.get_id()], var_sym_ids[rv.get_id()]

            def gate(owner_vagent, cond_binary, i0=i0, i1=i1):
                return owner_vagent[i0] == owner_vagent[i1]
            return gate

        def try_binary_eq(var_expr, val_expr):
            v = as_variable(var_expr)
            if v is None or v.get_id() not in cond_sym_ids:
                return None
            idx = cond_sym_ids[v.get_id()]
            val = val_expr.Evaluate()

            def gate(owner_vagent, cond_binary, idx=idx, val=val):
                is_one = cond_binary[idx] >= 0.5
                return is_one if val >= 0.5 else jnp.logical_not(is_one)
            return gate

        gate = try_binary_eq(lhs, rhs) or try_binary_eq(rhs, lhs)
        if gate is not None:
            return gate
        raise ValueError("Unsupported Eq atom in conditional edge formula")

    if kind == sym.FormulaKind.And:
        _, subs = formula.Unapply()
        gates = [compile_condition(s, var_sym_ids, cond_sym_ids) for s in subs]

        def gate(owner_vagent, cond_binary, gates=gates):
            result = jnp.asarray(True)
            for g in gates:
                result = jnp.logical_and(result, g(owner_vagent, cond_binary))
            return result
        return gate

    if kind == sym.FormulaKind.Or:
        _, subs = formula.Unapply()
        gates = [compile_condition(s, var_sym_ids, cond_sym_ids) for s in subs]

        def gate(owner_vagent, cond_binary, gates=gates):
            result = jnp.asarray(False)
            for g in gates:
                result = jnp.logical_or(result, g(owner_vagent, cond_binary))
            return result
        return gate

    if kind == sym.FormulaKind.Not:
        _, (sub,) = formula.Unapply()
        sub_gate = compile_condition(sub, var_sym_ids, cond_sym_ids)

        def gate(owner_vagent, cond_binary, sub_gate=sub_gate):
            return jnp.logical_not(sub_gate(owner_vagent, cond_binary))
        return gate

    raise ValueError(f"Unsupported FormulaKind {kind} in conditional edge ordering")


# ---------------------------------------------------------------------------
# General algebraic Expression/Formula compilation, for add_constraint /
# add_assignable_constraint / add_edge_constraint value constraints.
# ---------------------------------------------------------------------------

# ExpressionKind.Pow's exponent, ExpressionKind.Div's numerator/denominator etc.
# are always Expression operands (never bare Python floats) -- only Add/Mul's
# leading "constant" operand is a plain float, everything else recurses.
_UNARY_JNP_FNS = {
    sym.ExpressionKind.Sin: jnp.sin,
    sym.ExpressionKind.Cos: jnp.cos,
    sym.ExpressionKind.Tan: jnp.tan,
    sym.ExpressionKind.Asin: jnp.arcsin,
    sym.ExpressionKind.Acos: jnp.arccos,
    sym.ExpressionKind.Atan: jnp.arctan,
    sym.ExpressionKind.Sinh: jnp.sinh,
    sym.ExpressionKind.Cosh: jnp.cosh,
    sym.ExpressionKind.Tanh: jnp.tanh,
    sym.ExpressionKind.Sqrt: jnp.sqrt,
    sym.ExpressionKind.Abs: jnp.abs,
    sym.ExpressionKind.Exp: jnp.exp,
    sym.ExpressionKind.Log: jnp.log,
    sym.ExpressionKind.Ceil: jnp.ceil,
    sym.ExpressionKind.Floor: jnp.floor,
}


def compile_expression(expr, resolve_placeholder):
    """Compiles a pydrake.symbolic.Expression into a jax-traceable function
    fn(*rows) -> jnp value, via Unapply()-based recursive descent (same
    pattern as compile_condition, widened to arithmetic node kinds).

    `resolve_placeholder(variable) -> Callable[..., jnp value]` resolves a
    leaf Variable (e.g. one component of agent_q(k)/var_agent_q(var)) to a
    function of the same *rows signature as the returned compiled function
    (e.g. fn(wp_row) for a node constraint, fn(wp_u_row, wp_v_row) for an
    edge constraint) -- it should raise if the placeholder can't be resolved
    (e.g. object_q, or agent_q(k) for an agent this solver's per-node scope
    doesn't track).

    Note: there is no separate UnaryMinus ExpressionKind -- Drake represents
    negation as Mul with a -1.0 leading constant, handled by the Mul branch.
    """
    kind = expr.get_kind()

    if kind == sym.ExpressionKind.Var:
        _, (var,) = expr.Unapply()
        return resolve_placeholder(var)

    if kind == sym.ExpressionKind.Constant:
        _, (val,) = expr.Unapply()
        val = float(val)
        return lambda *rows, val=val: jnp.asarray(val)

    if kind == sym.ExpressionKind.Add:
        _, operands = expr.Unapply()
        const = float(operands[0])
        terms = [compile_expression(o, resolve_placeholder) for o in operands[1:]]

        def fn(*rows, const=const, terms=terms):
            out = jnp.asarray(const)
            for t in terms:
                out = out + t(*rows)
            return out
        return fn

    if kind == sym.ExpressionKind.Mul:
        _, operands = expr.Unapply()
        const = float(operands[0])
        terms = [compile_expression(o, resolve_placeholder) for o in operands[1:]]

        def fn(*rows, const=const, terms=terms):
            out = jnp.asarray(const)
            for t in terms:
                out = out * t(*rows)
            return out
        return fn

    if kind == sym.ExpressionKind.Div:
        _, (num, den) = expr.Unapply()
        num_fn = compile_expression(num, resolve_placeholder)
        den_fn = compile_expression(den, resolve_placeholder)

        def fn(*rows, num_fn=num_fn, den_fn=den_fn):
            return num_fn(*rows) / den_fn(*rows)
        return fn

    if kind == sym.ExpressionKind.Pow:
        _, (base, exponent) = expr.Unapply()
        base_fn = compile_expression(base, resolve_placeholder)
        exponent_fn = compile_expression(exponent, resolve_placeholder)

        def fn(*rows, base_fn=base_fn, exponent_fn=exponent_fn):
            return base_fn(*rows) ** exponent_fn(*rows)
        return fn

    if kind in _UNARY_JNP_FNS:
        _, (arg,) = expr.Unapply()
        arg_fn = compile_expression(arg, resolve_placeholder)
        jnp_fn = _UNARY_JNP_FNS[kind]

        def fn(*rows, arg_fn=arg_fn, jnp_fn=jnp_fn):
            return jnp_fn(arg_fn(*rows))
        return fn

    raise ValueError(f"Unsupported ExpressionKind {kind} in symbolic constraint compilation")


def _flatten_conjunction(formula):
    """Flattens a top-level And of relational atoms into a flat list (what
    pydrake.math.eq(vector, ...) produces once conjoined by the add_constraint
    pybind wrapper); a single relational atom returns as a 1-element list."""
    if formula.get_kind() == sym.FormulaKind.And:
        _, subs = formula.Unapply()
        out = []
        for s in subs:
            out.extend(_flatten_conjunction(s))
        return out
    return [formula]


_EQ_KINDS = {sym.FormulaKind.Eq}
_INEQ_KINDS = {sym.FormulaKind.Leq, sym.FormulaKind.Lt, sym.FormulaKind.Geq, sym.FormulaKind.Gt}
_GEQ_LIKE_KINDS = {sym.FormulaKind.Geq, sym.FormulaKind.Gt}


def compile_relational_formula(formula, resolve_placeholder):
    """Compiles a Formula (a single relational atom, or a top-level And of
    relational atoms) into (residual_fn, kind), matching the
    add_python_constraint(node, fn, kind) contract exactly: residual_fn(*rows)
    returns a (k,) jnp array, feasible at ==0 for kind="eq" or <=0 for
    kind="ineq". All atoms in `formula` must share the same eq/ineq kind."""
    atoms = _flatten_conjunction(formula)
    kinds = {a.get_kind() for a in atoms}

    if kinds <= _EQ_KINDS:
        out_kind = "eq"
    elif kinds <= _INEQ_KINDS:
        out_kind = "ineq"
    else:
        raise ValueError(
            f"Unsupported or mixed FormulaKinds {kinds} in symbolic constraint "
            "compilation -- expected a single eq/ineq kind (Eq, or one of "
            "Leq/Lt/Geq/Gt) across every conjoined atom")

    residual_fns = []
    for atom in atoms:
        _, (lhs, rhs) = atom.Unapply()
        lhs_fn = compile_expression(lhs, resolve_placeholder)
        rhs_fn = compile_expression(rhs, resolve_placeholder)
        # lhs>=rhs / lhs>rhs is feasible at <=0 under -(lhs-rhs); every other
        # kind (Eq, Leq, Lt) is already feasible at <=0 (or ==0) under +(lhs-rhs).
        sign = -1.0 if atom.get_kind() in _GEQ_LIKE_KINDS else 1.0

        def residual(*rows, lhs_fn=lhs_fn, rhs_fn=rhs_fn, sign=sign):
            return sign * (lhs_fn(*rows) - rhs_fn(*rows))
        residual_fns.append(residual)

    def residual_fn(*rows, residual_fns=residual_fns):
        return jnp.stack([r(*rows) for r in residual_fns])

    return residual_fn, out_kind
