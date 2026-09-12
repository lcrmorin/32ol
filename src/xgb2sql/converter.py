"""XGBoost model -> SQL converter with correct float32 handling and categorical support.

Thin wrapper: parse_xgb.py turns the Booster into the canonical IR (ir.py),
emit_sql.py renders that IR as SQL. See those modules for the real logic;
this module is just the public xgboost-specific entry points, kept for
backward compatibility with the pre-refactor API.

- XGBoost uses float32 for split thresholds. We CAST the column to FLOAT in SQL
  and write the threshold as its exact float32-as-float64 value using %.20e format.
  (DuckDB's decimal parser loses precision with repr(); don't CAST the literal.)
- Tested with DuckDB. Other engines: use FLOAT (MySQL) or REAL (PostgreSQL, SQL Server) -
  MySQL/PostgreSQL support is UNVERIFIED against a real instance, see TESTING_PLAN.md.
"""

from typing import Optional

import pandas as pd
import xgboost as xgb

from xgb2sql.emit_sas import ensemble_to_sas
from xgb2sql.emit_sql import ensemble_to_sql, multiclass_to_sql
from xgb2sql.ir import Ensemble
from xgb2sql.parse_xgb import xgb_to_ensemble, xgb_to_multiclass_ensembles
from xgb2sql.quantize import check_xgb_sas_safety


def xgboost_to_sql(
    model: xgb.Booster,
    sigmoid: bool = False,
    cat_feature_names: Optional[list] = None,
    cat_label_maps: Optional[dict] = None,
    training_df: Optional[pd.DataFrame] = None,
    float_type: str = "FLOAT",
    link: Optional[str] = None,
) -> str:
    """Convert XGBoost model to a SQL expression.

    Args:
        model: Trained xgb.Booster. Only booster="gbtree" is supported (raises
            NotImplementedError for "dart"/"gblinear" - see parse_xgb.py).
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
    ensemble = xgb_to_ensemble(
        model,
        sigmoid=sigmoid,
        cat_feature_names=cat_feature_names,
        cat_label_maps=cat_label_maps,
        training_df=training_df,
        link=link,
    )
    return ensemble_to_sql(ensemble, float_type=float_type)


def xgboost_to_sql_multiclass(
    model: xgb.Booster,
    num_class: int,
    cat_feature_names: Optional[list] = None,
    cat_label_maps: Optional[dict] = None,
    training_df: Optional[pd.DataFrame] = None,
    float_type: str = "FLOAT",
) -> dict:
    """Convert multiclass XGBoost model to {class_idx: sql_expression} with softmax.

    Works for both multi:softprob and multi:softmax models - they share the
    same tree structure, this always returns per-class probabilities
    regardless of which one trained the model (verified in tests).
    """
    ensembles = xgb_to_multiclass_ensembles(
        model,
        num_class=num_class,
        cat_feature_names=cat_feature_names,
        cat_label_maps=cat_label_maps,
        training_df=training_df,
    )
    return multiclass_to_sql(ensembles, float_type=float_type)


def xgboost_to_sas(
    model: xgb.Booster,
    bins_per_feature: dict,
    sigmoid: bool = False,
    cat_feature_names: Optional[list] = None,
    cat_label_maps: Optional[dict] = None,
    training_df: Optional[pd.DataFrame] = None,
    link: Optional[str] = None,
) -> str:
    """Convert an XGBoost model to SAS - ONLY if every feature was quantized
    to a small number of integer levels via xgb2sql.quantize.IntegerBinner
    (or an equivalent scheme) AND the resulting model's thresholds are
    verified safe against those achievable values. Raises ValueError with
    the specific violations if not - see quantize.py for why xgboost needs
    this and DecisionTree/RandomForest/GradientBoosting/HistGradientBoosting
    don't (sklearn_to_sas has no such restriction).

    Args:
        model: Trained xgb.Booster, fit on data that was quantized with the
            SAME IntegerBinner you pass bins_per_feature from.
        bins_per_feature: {feature_name: n_bins} - e.g. IntegerBinner().levels().
            Every numeric-split feature in the model must have an entry.
        (other args: see xgboost_to_sql)

    Returns:
        SAS expression string for `prediction = {expr};`.

    Raises:
        ValueError: if any threshold isn't safely separated from an
            achievable quantized value - the model (or its bin count /
            max_depth) needs to change, not this function's logic.
    """
    ensemble = xgb_to_ensemble(
        model, sigmoid=sigmoid, cat_feature_names=cat_feature_names,
        cat_label_maps=cat_label_maps, training_df=training_df, link=link,
    )
    violations = check_xgb_sas_safety(ensemble, bins_per_feature)
    if violations:
        shown = violations[:5]
        more = f" (+{len(violations) - 5} more)" if len(violations) > 5 else ""
        raise ValueError(
            "xgboost_to_sas refused: this model's thresholds are not safely separated "
            f"from the declared achievable quantized values. {len(violations)} violation(s), "
            f"first {len(shown)}: " + "; ".join(shown) + more +
            ". Try fewer bins (a coarser IntegerBinner), a shallower model (lower "
            "max_depth), or fewer boosting rounds - see quantize.py for why this isn't "
            "guaranteed by quantization alone."
        )
    # Safety verified for this specific declared grid: emit as float64-precision
    # (no cast needed) rather than the blocked float32 path.
    safe_ensemble = Ensemble(
        trees=ensemble.trees, base_score=ensemble.base_score,
        link=ensemble.link, threshold_precision="float64",
    )
    return ensemble_to_sas(safe_ensemble)


def prepare_df_for_duckdb(df, cat_feature_names=None):
    """Convert DataFrame for DuckDB: categoricals -> strings, numerics -> float."""
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
