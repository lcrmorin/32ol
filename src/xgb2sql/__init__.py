"""xgb2sql: convert a trained tree model (XGBoost, scikit-learn, LightGBM,
CatBoost) into a SQL or SAS expression for in-database/in-process scoring.

float32-precision-correct thresholds, categorical-split support, and
multiclass support, per source library - see README.md for the full support
matrix and usage, and TESTING_PLAN.md for test coverage and open
verification items.

Each source library is an OPTIONAL dependency (`pip install xgb2sql[xgboost]`,
`xgb2sql[sklearn]`, `xgb2sql[lightgbm]`, `xgb2sql[catboost]`, or `xgb2sql[all]`
for every one). `import xgb2sql` always succeeds regardless of which of
those are installed - each block below is its own try/except, so a missing
library only breaks the functions that actually need it. Calling one of
those functions without its library installed raises a plain ImportError
naming the extra to install, rather than xgb2sql itself failing to import.
"""


def _stub(fn_name: str, extra: str):
    def _raise(*args, **kwargs):
        raise ImportError(
            f"{fn_name}() needs the optional '{extra}' dependency, which is not "
            f"installed in this environment. Install it with: pip install xgb2sql[{extra}]"
        )

    _raise.__name__ = fn_name
    return _raise


try:
    from xgb2sql.converter import (
        xgboost_to_sas,
        xgboost_to_sql,
        xgboost_to_sql_multiclass,
        prepare_df_for_duckdb,
    )
except ImportError:
    xgboost_to_sas = _stub("xgboost_to_sas", "xgboost")
    xgboost_to_sql = _stub("xgboost_to_sql", "xgboost")
    xgboost_to_sql_multiclass = _stub("xgboost_to_sql_multiclass", "xgboost")
    prepare_df_for_duckdb = _stub("prepare_df_for_duckdb", "xgboost")

try:
    from xgb2sql.sklearn_api import (
        sklearn_to_sas,
        sklearn_to_sas_multiclass,
        sklearn_to_sql,
        sklearn_to_sql_multiclass,
    )
except ImportError:
    sklearn_to_sas = _stub("sklearn_to_sas", "sklearn")
    sklearn_to_sas_multiclass = _stub("sklearn_to_sas_multiclass", "sklearn")
    sklearn_to_sql = _stub("sklearn_to_sql", "sklearn")
    sklearn_to_sql_multiclass = _stub("sklearn_to_sql_multiclass", "sklearn")

try:
    from xgb2sql.lgbm_api import (
        lgbm_to_sas,
        lgbm_to_sas_multiclass,
        lgbm_to_sql,
        lgbm_to_sql_multiclass,
    )
except ImportError:
    lgbm_to_sas = _stub("lgbm_to_sas", "lightgbm")
    lgbm_to_sas_multiclass = _stub("lgbm_to_sas_multiclass", "lightgbm")
    lgbm_to_sql = _stub("lgbm_to_sql", "lightgbm")
    lgbm_to_sql_multiclass = _stub("lgbm_to_sql_multiclass", "lightgbm")

try:
    from xgb2sql.catboost_api import (
        catboost_to_sas,
        catboost_to_sas_multiclass,
        catboost_to_sql,
        catboost_to_sql_multiclass,
    )
except ImportError:
    catboost_to_sas = _stub("catboost_to_sas", "catboost")
    catboost_to_sas_multiclass = _stub("catboost_to_sas_multiclass", "catboost")
    catboost_to_sql = _stub("catboost_to_sql", "catboost")
    catboost_to_sql_multiclass = _stub("catboost_to_sql_multiclass", "catboost")

# numpy/pandas only - always available regardless of which ML libraries are installed.
from xgb2sql.quantize import IntegerBinner, check_xgb_sas_safety

__version__ = "0.2.0"
__all__ = [
    "xgboost_to_sql",
    "xgboost_to_sql_multiclass",
    "xgboost_to_sas",
    "prepare_df_for_duckdb",
    "sklearn_to_sql",
    "sklearn_to_sql_multiclass",
    "sklearn_to_sas",
    "sklearn_to_sas_multiclass",
    "lgbm_to_sql",
    "lgbm_to_sas",
    "lgbm_to_sql_multiclass",
    "lgbm_to_sas_multiclass",
    "catboost_to_sql",
    "catboost_to_sas",
    "catboost_to_sql_multiclass",
    "catboost_to_sas_multiclass",
    "IntegerBinner",
    "check_xgb_sas_safety",
]
