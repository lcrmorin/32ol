"""Emit a SAS DATA-step expression from the canonical IR (see ir.py).

IMPORTANT, read before trusting this for anything xgboost-derived: this
emitter is only implemented for `Ensemble.threshold_precision == "float64"`
(i.e. sklearn-sourced ensembles). It deliberately RAISES for "float32"
(xgboost-sourced) ensembles rather than guessing.

Why: the SQL emitter's float32 correctness trick is `CAST(col AS FLOAT)`,
which relies on the target database having a real 4-byte float type so the
*incoming value* gets truncated to float32 before comparing against the
float32 threshold - matching what XGBoost did internally. A SAS DATA step
has no equivalent native 4-byte truncation (SAS numeric variables are IEEE
double precision; declaring `LENGTH x 4` changes on-disk storage width but
its rounding behavior has not been checked against IEEE-754 binary32
round-to-nearest). Skipping the truncation isn't a safe default either -
the SQLite case in TESTING_PLAN.md showed skipping it silently misroutes
~7% of rows with up to 0.43 absolute error. So: not supported here until
verified against a real SAS instance, exactly the same posture as the
still-open Postgres/MySQL item.

sklearn-sourced ensembles ("float64") need no such trick - SAS numerics are
natively IEEE double precision, matching sklearn's own comparison exactly.

Missing-value handling: every branch explicitly checks `MISSING(col)` first,
rather than relying on SAS's comparison-operator behavior for a missing
operand. This is deliberate, not just style: SAS treats a numeric missing
value as smaller than any real number for ordinary comparisons (`. < 5` is
TRUE), which is NOT the same as SQL's three-valued NULL logic this codebase
was built around - so an implicit-missing-propagation design that mirrors
the SQL emitter would silently misroute missing values into the "yes"
branch instead of the dedicated missing branch. Explicit MISSING() avoids
relying on that comparison quirk at all.

Column names that aren't valid plain SAS identifiers are emitted as SAS
name literals (`'my col'n`), which requires `OPTIONS VALIDVARNAME=ANY;` in
the SAS session - noted here, not silently assumed.
"""

import re

from xgb2sql.ir import Ensemble, Leaf, Node, Split

_VALID_SAS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")


def quote_col(name: str) -> str:
    if _VALID_SAS_NAME.match(name):
        return name
    return "'" + name.replace("'", "''") + "'n"


def quote_str_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _emit_node(node: Node) -> str:
    if isinstance(node, Leaf):
        return f"{node.value:.15f}"

    assert isinstance(node, Split)
    feat = quote_col(node.feature)
    yes_sas = _emit_node(node.yes)
    no_sas = _emit_node(node.no)
    missing_sas = _emit_node(node.missing) if node.missing is not None else no_sas

    if node.categories is not None:
        if not node.categories:
            cond = "0"
        else:
            parts = ", ".join(
                quote_str_literal(v) if isinstance(v, str) else str(v) for v in node.categories
            )
            cond = f"({feat} IN ({parts}))"
    else:
        op = "<=" if node.le else "<"
        cond = f"({feat} {op} {node.threshold!r})"

    return f"IFN(MISSING({feat}), {missing_sas}, IFN({cond}, {yes_sas}, {no_sas}))"


def ensemble_to_sas(ensemble: Ensemble) -> str:
    """Render one Ensemble (one output/class) as a SAS scalar expression,
    usable as `prediction = {expr};` in a DATA step.
    """
    if ensemble.threshold_precision != "float64":
        raise NotImplementedError(
            "SAS emission is only implemented for threshold_precision='float64' "
            "(sklearn-sourced ensembles). xgboost-sourced ('float32') ensembles "
            "are not supported yet - see this module's docstring for why "
            "(no verified float32-truncation equivalent in a SAS DATA step)."
        )
    tree_sass = []
    for t in ensemble.trees:
        expr = f"({_emit_node(t.root)})"
        if t.weight != 1.0:
            expr = f"({t.weight!r} * {expr})"
        tree_sass.append(expr)
    raw = f"{ensemble.base_score!r} + {' + '.join(tree_sass)}" if tree_sass else f"{ensemble.base_score!r}"
    if ensemble.link == "logistic":
        return f"(1 / (1 + exp(-({raw}))))"
    if ensemble.link == "exp":
        return f"(exp({raw}))"
    return f"({raw})"
