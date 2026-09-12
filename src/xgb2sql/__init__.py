"""xgb2sql: convert a trained XGBoost model into a SQL expression for in-database scoring.

Converts a trained XGBoost Booster into a SQL CASE WHEN expression, with
float32-precision-correct thresholds, categorical-split support, and
multiclass support. See README.md for usage and TESTING_PLAN.md for test
coverage and open verification items.
"""

from xgb2sql.converter import (
    xgboost_to_sas,
    xgboost_to_sql,
    xgboost_to_sql_multiclass,
    prepare_df_for_duckdb,
)
from xgb2sql.quantize import IntegerBinner, check_xgb_sas_safety
from xgb2sql.sklearn_api import (
    sklearn_to_sas,
    sklearn_to_sas_multiclass,
    sklearn_to_sql,
    sklearn_to_sql_multiclass,
)

__version__ = "0.1.0"
__all__ = [
    "xgboost_to_sql",
    "xgboost_to_sql_multiclass",
    "xgboost_to_sas",
    "prepare_df_for_duckdb",
    "sklearn_to_sql",
    "sklearn_to_sql_multiclass",
    "sklearn_to_sas",
    "sklearn_to_sas_multiclass",
    "IntegerBinner",
    "check_xgb_sas_safety",
]
