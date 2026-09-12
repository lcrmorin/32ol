"""Public entry points for CatBoost models -> SQL and SAS.

See parse_catboost.py for exactly what's supported and why. Unlike
LightGBM, CatBoost needs the SAME treatment as xgboost for SAS: its
thresholds are float32 and the incoming feature value is truncated to
float32 before comparison too (verified - see parse_catboost.py), so
catboost_to_sas is gated behind the same IntegerBinner-quantize +
check_xgb_sas_safety path as xgboost_to_sas, not a free pass like
sklearn/LightGBM. (check_xgb_sas_safety is named for its original xgboost
use case but is generic over any float32-precision Ensemble - it just walks
Split nodes - so it's reused here as-is rather than duplicated.)
"""

from typing import Optional

from xgb2sql.emit_sas import ensemble_to_sas
from xgb2sql.emit_sql import ensemble_to_sql
from xgb2sql.ir import Ensemble
from xgb2sql.parse_catboost import catboost_to_ensemble
from xgb2sql.quantize import check_xgb_sas_safety


def catboost_to_sql(
    model,
    float_type: str = "FLOAT",
    double_type: str = "DOUBLE",
    link: Optional[str] = None,
) -> str:
    """Convert a fitted CatBoostRegressor or binary CatBoostClassifier to a
    single SQL expression (the regression value, or P(class=1) for a
    classifier).

    Args:
        model: Fitted CatBoostRegressor or CatBoostClassifier. Only
            grow_policy="SymmetricTree" (the default), numeric-only features,
            and single-target regression/binary classification are
            supported - see parse_catboost.py.
        float_type / double_type: SQL type names for the float32-cast trick
            (needed here, unlike LightGBM/sklearn - see parse_catboost.py)
            and the float64 cast respectively.
        link: Explicit link override ("identity"/"logistic"). Auto-detected
            from the model's loss_function if omitted.

    Returns:
        SQL expression string for SELECT {expr} AS prediction FROM table.
    """
    ensemble = catboost_to_ensemble(model, link=link)
    return ensemble_to_sql(ensemble, float_type=float_type, double_type=double_type)


def catboost_to_sas(model, bins_per_feature: dict, link: Optional[str] = None) -> str:
    """Convert a CatBoost model to SAS - ONLY if every feature was quantized
    to a small number of integer levels via xgb2sql.quantize.IntegerBinner
    (or an equivalent scheme) AND the resulting model's thresholds are
    verified safe against those achievable values. Raises ValueError with
    the specific violations if not - see quantize.py and parse_catboost.py
    for why CatBoost needs this (its comparison is float32-truncated,
    same as xgboost - unlike LightGBM/sklearn, which don't).

    Args:
        model: Fitted CatBoostRegressor or CatBoostClassifier, trained on
            data that was quantized with the SAME IntegerBinner you pass
            bins_per_feature from.
        bins_per_feature: {feature_name: n_bins} - e.g. IntegerBinner().levels().
            Every numeric-split feature in the model must have an entry.
        link: see catboost_to_sql.

    Returns:
        SAS expression string for `prediction = {expr};`.

    Raises:
        ValueError: if any threshold isn't safely separated from an
            achievable quantized value.
    """
    ensemble = catboost_to_ensemble(model, link=link)
    violations = check_xgb_sas_safety(ensemble, bins_per_feature)
    if violations:
        shown = violations[:5]
        more = f" (+{len(violations) - 5} more)" if len(violations) > 5 else ""
        raise ValueError(
            "catboost_to_sas refused: this model's thresholds are not safely separated "
            f"from the declared achievable quantized values. {len(violations)} violation(s), "
            f"first {len(shown)}: " + "; ".join(shown) + more +
            ". Try fewer bins (a coarser IntegerBinner), a shallower model (lower "
            "depth), or fewer iterations - see quantize.py for why this isn't "
            "guaranteed by quantization alone."
        )
    safe_ensemble = Ensemble(
        trees=ensemble.trees, base_score=ensemble.base_score,
        link=ensemble.link, threshold_precision="float64",
    )
    return ensemble_to_sas(safe_ensemble)
