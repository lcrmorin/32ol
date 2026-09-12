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

from xgb2sql.emit_sql import ensemble_to_sql, multiclass_to_sql
from xgb2sql.parse_xgb import xgb_to_ensemble, xgb_to_multiclass_ensembles


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
