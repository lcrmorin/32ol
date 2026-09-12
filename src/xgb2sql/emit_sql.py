"""Emit a SQL CASE-expression from the canonical IR (see ir.py).

Missing-value behavior for a Split with `missing=None` (i.e. the source
model has no native missing-value handling - this is every sklearn CART
tree; xgboost splits always carry a missing branch so this path never
triggers for xgboost): the emitted SQL still checks `{col} IS NULL` and
routes it to the "no" branch. This matches ordinary SQL semantics (a NULL
comparison like `x < 5` evaluates to UNKNOWN, which a bare `CASE WHEN`
treats as not-matched, i.e. falls through to ELSE = "no"), so a NULL input
behaves the same whether or not this explicit check is present - it's
written out for readability, not because it changes behavior. It does NOT
mean nulls are "supported" the way xgboost's missing handling is: the
source model was never trained to route missing values on purpose, so
whatever the "no" branch computes is an unvalidated default, not a modeled
prediction. See README.md.
"""

from typing import Optional

import numpy as np

from xgb2sql.ir import Ensemble, Leaf, Node, Split

_SQL_RESERVED = frozenset(
    "group order select from where table index key column values set check default "
    "desc asc limit offset join left right inner outer cross on having between like "
    "in not null true false case when then else end and or is as union all distinct "
    "insert update delete create drop alter into primary foreign references grant "
    "revoke with recursive exists any some cast rank row rows range partition over "
    "window user role schema database trigger view procedure function type level zone "
    "year month day hour minute second date time timestamp interval".split()
)


def quote_col(name: str) -> str:
    if name.lower() in _SQL_RESERVED or not name.isidentifier():
        return f'"{name}"'
    return name


def quote_str_literal(value: str) -> str:
    """Escape a single-quoted SQL string literal (doubling embedded quotes)."""
    return "'" + value.replace("'", "''") + "'"


def _node_to_sql(node: Node, float_type: str, double_type: str, precision: str) -> str:
    if isinstance(node, Leaf):
        return f"{node.value:.15f}"

    assert isinstance(node, Split)
    feat = quote_col(node.feature)
    yes_sql = _node_to_sql(node.yes, float_type, double_type, precision)
    no_sql = _node_to_sql(node.no, float_type, double_type, precision)
    missing_sql = (
        _node_to_sql(node.missing, float_type, double_type, precision)
        if node.missing is not None else no_sql
    )

    if node.categories is not None:
        if not node.categories:
            return f"CASE WHEN {feat} IS NULL THEN {missing_sql} ELSE {no_sql} END"
        parts = ", ".join(
            quote_str_literal(v) if isinstance(v, str) else str(v) for v in node.categories
        )
        return (
            f"CASE WHEN {feat} IS NULL THEN {missing_sql} "
            f"WHEN {feat} IN ({parts}) THEN {yes_sql} ELSE {no_sql} END"
        )

    op = "<=" if node.le else "<"
    if precision == "float32":
        # Force the comparison to happen in float32, matching the source
        # model's internal split precision (see ir.py / TESTING_PLAN.md).
        thresh = f"{float(np.float32(node.threshold)):.20e}"
        cast_col = f"CAST({feat} AS {float_type})"
    else:
        # Full float64 precision, no truncating cast.
        thresh = repr(float(node.threshold))
        cast_col = f"CAST({feat} AS {double_type})"
    return (
        f"CASE WHEN {feat} IS NULL THEN {missing_sql} "
        f"WHEN {cast_col} {op} {thresh} "
        f"THEN {yes_sql} ELSE {no_sql} END"
    )


def ensemble_to_sql(ensemble: Ensemble, float_type: str = "FLOAT", double_type: str = "DOUBLE") -> str:
    """Render one Ensemble (one output/class) as a SQL scalar expression."""
    precision = ensemble.threshold_precision
    tree_sqls = []
    for t in ensemble.trees:
        expr = f"({_node_to_sql(t.root, float_type, double_type, precision)})"
        if t.weight != 1.0:
            expr = f"({t.weight:.15f} * {expr})"
        tree_sqls.append(expr)
    raw = f"{ensemble.base_score:.15f} + {' + '.join(tree_sqls)}" if tree_sqls else f"{ensemble.base_score:.15f}"
    if ensemble.link == "logistic":
        return f"1.0 / (1.0 + EXP(-({raw})))"
    if ensemble.link == "exp":
        return f"EXP({raw})"
    return raw


def multiclass_to_sql(ensembles, float_type: str = "FLOAT", double_type: str = "DOUBLE") -> dict:
    """Render a list of per-class Ensembles (raw margins, no per-class link
    applied - softmax is applied here across all classes) as {class_idx: sql}.
    """
    def tree_expr(t, precision):
        base = f"({_node_to_sql(t.root, float_type, double_type, precision)})"
        return f"({t.weight:.15f} * {base})" if t.weight != 1.0 else base

    raws = [
        (
            f"({ensembles[c].base_score:.15f} + "
            + " + ".join(tree_expr(t, ensembles[c].threshold_precision) for t in ensembles[c].trees)
            + ")"
        )
        if ensembles[c].trees
        else f"({ensembles[c].base_score:.15f})"
        for c in range(len(ensembles))
    ]
    denom = f"({' + '.join(f'EXP({r})' for r in raws)})"
    return {c: f"(EXP({raws[c]}) / {denom})" for c in range(len(ensembles))}
