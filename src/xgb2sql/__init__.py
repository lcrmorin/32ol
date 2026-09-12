"""xgb2sql: convert a trained XGBoost model into a SQL expression for in-database scoring.

Converts a trained XGBoost Booster into a SQL CASE WHEN expression, with
float32-precision-correct thresholds, categorical-split support, and
multiclass support. See README.md for usage and TESTING_PLAN.md for test
coverage and open verification items.
"""

from xgb2sql.converter import (
    xgboost_to_sql,
    xgboost_to_sql_multiclass,
    prepare_df_for_duckdb,
)

__version__ = "0.1.0"
__all__ = ["xgboost_to_sql", "xgboost_to_sql_multiclass", "prepare_df_for_duckdb"]
