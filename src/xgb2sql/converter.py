"""
XGBoost model → SQL converter with correct float32 handling and categorical support.

Key details:
- XGBoost uses float32 for split thresholds. We CAST the column to FLOAT in SQL
  and write the threshold as its exact float32-as-float64 value using %.20e format.
  (DuckDB's decimal parser loses precision with repr(); don't CAST the literal.)
- Categorical splits are detected by split_condition being a list of int codes.
  These are mapped back to original labels via cat_label_maps.
- Tested with DuckDB. Other engines: use FLOAT (MySQL) or REAL (PostgreSQL, SQL Server).

Known, verified-by-test limitations (see TESTING_PLAN.md for the full matrix):
- Only booster="gbtree" is supported. "dart" is silently WRONG if you bypass the
  guard below - at inference time DART's dropout-adjusted tree weights are not
  recoverable from get_dump() alone, so a plain sum of leaf values does not match
  Booster.predict(). "gblinear" is a different model family entirely.
- num_parallel_tree > 1 (random forest mode) has been verified to need NO special
  handling: XGBoost's own leaf values already fold in whatever averaging is
  needed, so summing every dumped tree as usual is correct.
- The link function (how the summed raw margin turns into a prediction) is
  auto-detected from the model's objective. Objectives that aren't in the
  verified allow-list raise an error instead of silently returning a raw
  margin that looks like a prediction but isn't one.
"""

import json
import numpy as np
import xgboost as xgb
import pandas as pd
from typing import Optional

# Objective -> link function, verified empirically against Booster.predict()
# (see TESTING_PLAN.md). "identity": raw margin == prediction. "logistic":
# prediction = sigmoid(margin), and base_score is itself a probability.
# "exp": prediction = exp(margin), and base_score is itself on the output
# (mean/count) scale, not the margin scale.
_OBJECTIVE_LINKS = {
    "binary:logistic": "logistic",
    "reg:logistic": "logistic",
    "reg:squarederror": "identity",
    "reg:squaredlogerror": "identity",
    "reg:pseudohubererror": "identity",
    "reg:absoluteerror": "identity",
    "binary:logitraw": "identity",
    "count:poisson": "exp",
    "reg:gamma": "exp",
    "reg:tweedie": "exp",
}

_UNSUPPORTED_BOOSTERS = {
    "dart": (
        "booster='dart' is not supported: at inference time DART applies "
        "per-tree dropout weighting that is not recoverable from get_dump() "
        "alone, so summing leaf values gives a different number than "
        "Booster.predict() (verified empirically - see TESTING_PLAN.md). "
        "Retrain with booster='gbtree', or open an issue if you need this."
    ),
    "gblinear": (
        "booster='gblinear' is not a tree model and is not supported by "
        "this tree-to-SQL converter."
    ),
}


def _check_booster_supported(cfg: dict) -> None:
    name = cfg["learner"]["gradient_booster"]["name"]
    if name in _UNSUPPORTED_BOOSTERS:
        raise NotImplementedError(_UNSUPPORTED_BOOSTERS[name])
    if name != "gbtree":
        raise NotImplementedError(
            f"booster='{name}' has not been verified against a live prediction "
            "comparison and is not supported. Open an issue if you need it."
        )


def _resolve_link(cfg: dict, sigmoid: bool, link: Optional[str]) -> str:
    """Decide the link function, preferring an explicit override, falling
    back to the deprecated `sigmoid` flag, then to objective auto-detection.
    """
    if link is not None:
        if link not in ("identity", "logistic", "exp"):
            raise ValueError(f"link must be one of identity/logistic/exp, got {link!r}")
        return link
    if sigmoid:
        return "logistic"

    objective = cfg["learner"]["objective"]["name"]
    resolved = _OBJECTIVE_LINKS.get(objective)
    if resolved is None:
        raise NotImplementedError(
            f"objective='{objective}' is not in the verified allow-list "
            f"({sorted(_OBJECTIVE_LINKS)}). Pass link='identity'/'logistic'/'exp' "
            "explicitly once you've confirmed the correct link function for this "
            "objective against a real prediction comparison (see TESTING_PLAN.md) "
            "- do not guess, an unverified link silently produces wrong numbers."
        )
    return resolved

# SQL reserved words to quote in column names
_SQL_RESERVED = frozenset(
    "group order select from where table index key column values set check default "
    "desc asc limit offset join left right inner outer cross on having between like "
    "in not null true false case when then else end and or is as union all distinct "
    "insert update delete create drop alter into primary foreign references grant "
    "revoke with recursive exists any some cast rank row rows range partition over "
    "window user role schema database trigger view procedure function type level zone "
    "year month day hour minute second date time timestamp interval".split()
)


def _quote_col(name: str) -> str:
    if name.lower() in _SQL_RESERVED or not name.isidentifier():
        return f'"{name}"'
    return name


def _quote_str_literal(value: str) -> str:
    """Quote a string as a SQL literal, escaping embedded single quotes.

    The unescaped version (f"'{v}'") breaks - or, worse, silently produces a
    syntactically different WHERE clause - for any category label containing
    a quote (e.g. "O'Brien", "5' ft", a free-text category). Standard SQL
    escapes a single quote by doubling it, which DuckDB/Postgres/MySQL/SQL
    Server all accept.
    """
    return "'" + value.replace("'", "''") + "'"


def _tree_node_to_sql(node, cat_label_maps=None, float_type="FLOAT"):
    """Recursively convert a tree node (JSON dump) to a SQL CASE expression."""
    if "leaf" in node:
        return f"{node['leaf']:.15f}"

    feat = _quote_col(node["split"])
    children = {c["nodeid"]: c for c in node["children"]}
    sub = {k: _tree_node_to_sql(children[node[k]], cat_label_maps, float_type)
           for k in ("yes", "no", "missing")}

    sc = node["split_condition"]

    # Categorical split: split_condition is a list of int codes → yes branch
    if isinstance(sc, list):
        if cat_label_maps and node["split"] in cat_label_maps:
            labels = [cat_label_maps[node["split"]].get(c, c) for c in sc]
        else:
            labels = sc
        if not labels:
            return f"CASE WHEN {feat} IS NULL THEN {sub['missing']} ELSE {sub['no']} END"
        parts = ", ".join(_quote_str_literal(v) if isinstance(v, str) else str(v) for v in labels)
        return (f"CASE WHEN {feat} IS NULL THEN {sub['missing']} "
                f"WHEN {feat} IN ({parts}) THEN {sub['yes']} ELSE {sub['no']} END")

    # Numeric split: cast column to float32, use exact f32 threshold literal
    if isinstance(sc, str):
        sc = float(sc.strip("[]"))
    thresh = f"{float(np.float32(float(sc))):.20e}"
    return (f"CASE WHEN {feat} IS NULL THEN {sub['missing']} "
            f"WHEN CAST({feat} AS {float_type}) < {thresh} "
            f"THEN {sub['yes']} ELSE {sub['no']} END")


def _build_cat_label_maps(df, cat_feature_names):
    """Build {feature: {int_code: label}} from pd.Categorical columns."""
    maps = {}
    for col in cat_feature_names:
        if col not in df.columns:
            continue
        s = df[col] if hasattr(df[col], "cat") else df[col].astype("category")
        maps[col] = dict(enumerate(s.cat.categories))
    return maps


def _parse_base_score(cfg):
    """Parse base_score from config. Returns list[float] (per-class for multiclass)."""
    raw = cfg["learner"]["learner_model_param"]["base_score"]
    if isinstance(raw, (int, float)):
        return [float(raw)]
    raw = raw.strip("[]")
    return [float(x) for x in raw.split(",")] if "," in raw else [float(raw)]


def _base_score_to_margin(base_score: float, link: str) -> float:
    """Invert the link function to turn base_score into an additive margin.

    base_score is stored on the OUTPUT scale (a probability for "logistic",
    a mean/count for "exp"), not the margin scale that tree leaf values add
    into - so it has to be inverse-transformed before summing with the trees.
    Verified empirically against Booster.predict() - see TESTING_PLAN.md.
    """
    if link == "identity":
        return base_score
    if link == "logistic":
        if not (0.0 < base_score < 1.0):
            raise ValueError(
                f"base_score={base_score} is exactly 0 or 1, so logit(base_score) "
                "is +/-infinity. This should not happen from real training (XGBoost "
                "clips it internally); if you set base_score manually, use a value "
                "strictly between 0 and 1."
            )
        return float(np.log(base_score / (1 - base_score)))
    if link == "exp":
        if base_score <= 0.0:
            raise ValueError(
                f"base_score={base_score} is <= 0, so log(base_score) is undefined "
                "for an exp-link objective (count:poisson/reg:gamma/reg:tweedie)."
            )
        return float(np.log(base_score))
    raise AssertionError(f"unreachable: unknown link {link!r}")  # _resolve_link validates this


def _resolve_label_maps(cat_label_maps, cat_feature_names, training_df):
    if cat_label_maps is not None:
        return cat_label_maps
    if cat_feature_names and training_df is not None:
        return _build_cat_label_maps(training_df, cat_feature_names)
    return None


def _trees_to_sql(model, label_maps, float_type):
    return [f"({_tree_node_to_sql(json.loads(t), label_maps, float_type)})"
            for t in model.get_dump(dump_format="json")]


def xgboost_to_sql(
    model: xgb.Booster,
    sigmoid: bool = False,
    cat_feature_names: Optional[list[str]] = None,
    cat_label_maps: Optional[dict] = None,
    training_df: Optional[pd.DataFrame] = None,
    float_type: str = "FLOAT",
    link: Optional[str] = None,
) -> str:
    """Convert XGBoost model to a SQL expression.

    Args:
        model: Trained xgb.Booster. Only booster="gbtree" is supported (raises
            NotImplementedError for "dart"/"gblinear" - see module docstring).
        sigmoid: Deprecated alias for link="logistic". Kept for backward
            compatibility; if both are omitted, the link is auto-detected from
            the model's objective, and an unrecognized objective raises rather
            than silently returning a raw margin as if it were a prediction.
        cat_feature_names: Categorical feature names (to build label maps from training_df).
        cat_label_maps: Pre-built {feature: {code: label}} maps (overrides training_df).
        training_df: DataFrame with .cat columns to build label maps from.
        float_type: SQL float32 type name ("FLOAT" for DuckDB/MySQL, "REAL" for PostgreSQL).
        link: Explicit link function override: "identity", "logistic", or
            "exp" (for count:poisson/reg:gamma/reg:tweedie). Takes precedence
            over `sigmoid` and over auto-detection.

    Returns:
        SQL expression string for SELECT {expr} AS prediction FROM table.
    """
    cfg = json.loads(model.save_config())
    _check_booster_supported(cfg)
    resolved_link = _resolve_link(cfg, sigmoid, link)

    label_maps = _resolve_label_maps(cat_label_maps, cat_feature_names, training_df)
    base_score = _parse_base_score(cfg)[0]
    base_margin = _base_score_to_margin(base_score, resolved_link)

    tree_sqls = _trees_to_sql(model, label_maps, float_type)
    raw = f"{base_margin:.15f} + {' + '.join(tree_sqls)}"
    if resolved_link == "logistic":
        return f"1.0 / (1.0 + EXP(-({raw})))"
    if resolved_link == "exp":
        return f"EXP({raw})"
    return raw


def xgboost_to_sql_multiclass(
    model: xgb.Booster,
    num_class: int,
    cat_feature_names: Optional[list[str]] = None,
    cat_label_maps: Optional[dict] = None,
    training_df: Optional[pd.DataFrame] = None,
    float_type: str = "FLOAT",
) -> dict[int, str]:
    """Convert multiclass XGBoost model to {class_idx: sql_expression} with softmax.

    Trees are stored round-robin: tree i belongs to class (i % num_class).
    Works for both multi:softprob and multi:softmax models - they share the
    same tree structure, this always returns per-class probabilities
    regardless of which one trained the model (verified in tests).
    """
    cfg = json.loads(model.save_config())
    _check_booster_supported(cfg)

    actual_num_class = int(cfg["learner"]["learner_model_param"].get("num_class", 0))
    if actual_num_class and actual_num_class != num_class:
        raise ValueError(
            f"num_class={num_class} was passed but the model was trained with "
            f"num_class={actual_num_class}. Using the wrong value silently "
            "misassigns trees to classes."
        )

    label_maps = _resolve_label_maps(cat_label_maps, cat_feature_names, training_df)
    base_scores = _parse_base_score(cfg)
    class_base = base_scores if len(base_scores) == num_class else [base_scores[0]] * num_class

    tree_sqls = _trees_to_sql(model, label_maps, float_type)
    if len(tree_sqls) % num_class != 0:
        raise ValueError(
            f"model has {len(tree_sqls)} trees, which is not a multiple of "
            f"num_class={num_class}; trees can't be assigned round-robin to classes."
        )
    class_trees = {c: [] for c in range(num_class)}
    for i, sql in enumerate(tree_sqls):
        class_trees[i % num_class].append(sql)

    raw = {c: f"({class_base[c]:.15f} + {' + '.join(class_trees[c])})" for c in range(num_class)}
    denom = f"({' + '.join(f'EXP({raw[c]})' for c in range(num_class))})"
    return {c: f"(EXP({raw[c]}) / {denom})" for c in range(num_class)}


def prepare_df_for_duckdb(df, cat_feature_names=None):
    """Convert DataFrame for DuckDB: categoricals → strings, numerics → float."""
    result = df.copy()
    cat_cols = set(cat_feature_names or [])
    for col in result.columns:
        if col in cat_cols:
            if hasattr(result[col], "cat"):
                codes, cats = result[col].cat.codes, result[col].cat.categories
                result[col] = pd.array(
                    [str(cats[c]) if c >= 0 else None for c in codes], dtype="object")
        elif result[col].dtype.kind in ("i", "u", "f"):
            result[col] = result[col].astype(float)
    return result
