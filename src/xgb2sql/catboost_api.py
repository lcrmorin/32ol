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

from xgb2sql.emit_sas import ensemble_to_sas, multiclass_to_sas
from xgb2sql.emit_sql import ensemble_to_sql, multiclass_to_sql
from xgb2sql.ir import Ensemble
from xgb2sql.parse_catboost import catboost_to_ensemble, catboost_to_multiclass_ensembles
from xgb2sql.quantize import check_xgb_sas_safety


def catboost_to_sql(
    model,
    float_type: str = "FLOAT",
    double_type: str = "DOUBLE",
    link: Optional[str] = None,
    cat_feature_values: Optional[dict] = None,
) -> str:
    """Convert a fitted CatBoostRegressor or binary CatBoostClassifier to a
    single SQL expression (the regression value, or P(class=1) for a
    classifier).

    Args:
        model: Fitted CatBoostRegressor or CatBoostClassifier. Only
            grow_policy="SymmetricTree" (the default) and single-target
            regression/binary classification are supported - see
            parse_catboost.py. Categorical features are supported only when
            every categorical split is "OneHotFeature" (see below).
        float_type / double_type: SQL type names for the float32-cast trick
            (needed here, unlike LightGBM/sklearn - see parse_catboost.py)
            and the float64 cast respectively.
        link: Explicit link override ("identity"/"logistic"). Auto-detected
            from the model's loss_function if omitted.
        cat_feature_values: Required if the model has categorical features:
            {feature_name: [every distinct raw category value that column
            had at training time]} - see parse_catboost.py for why.

    Returns:
        SQL expression string for SELECT {expr} AS prediction FROM table.
    """
    ensemble = catboost_to_ensemble(model, link=link, cat_feature_values=cat_feature_values)
    return ensemble_to_sql(ensemble, float_type=float_type, double_type=double_type)


def catboost_to_sas(
    model, bins_per_feature: dict, link: Optional[str] = None, cat_feature_values: Optional[dict] = None,
) -> str:
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
    ensemble = catboost_to_ensemble(model, link=link, cat_feature_values=cat_feature_values)
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


def catboost_to_sql_multiclass(
    model, float_type: str = "FLOAT", double_type: str = "DOUBLE",
    cat_feature_values: Optional[dict] = None,
) -> dict:
    """Convert a fitted multiclass CatBoostClassifier (loss_function=
    "MultiClass") to {class_index: sql_expression}, with softmax applied
    across classes - same convention as xgboost/LightGBM multiclass.

    cat_feature_values: see catboost_to_sql - required if the model has
    categorical features.
    """
    ensembles = catboost_to_multiclass_ensembles(model, cat_feature_values=cat_feature_values)
    return multiclass_to_sql(ensembles, float_type=float_type, double_type=double_type)


def catboost_to_sas_multiclass(
    model, bins_per_feature: dict, cat_feature_values: Optional[dict] = None,
) -> dict:
    """Multiclass equivalent of catboost_to_sas - same quantizer requirement
    (CatBoost is float32-truncated regardless of class count), checked
    independently for EVERY class's thresholds since they can differ even
    though the split structure is shared (different leaf values per class
    don't change threshold safety, but this checks each ensemble rather
    than assuming one class's safety implies another's - the splits/borders
    are actually identical across classes within a tree, so in practice
    checking class 0 would suffice, but checking all of them costs little
    and doesn't rely on that being true forever).
    """
    ensembles = catboost_to_multiclass_ensembles(model, cat_feature_values=cat_feature_values)
    all_violations = {}
    for c, ensemble in enumerate(ensembles):
        v = check_xgb_sas_safety(ensemble, bins_per_feature)
        if v:
            all_violations[c] = v
    if all_violations:
        parts = [f"class {c}: {'; '.join(v[:3])}" for c, v in all_violations.items()]
        raise ValueError(
            "catboost_to_sas_multiclass refused: thresholds are not safely separated "
            "from the declared achievable quantized values. " + " | ".join(parts) +
            ". Try fewer bins, a shallower model, or fewer iterations."
        )
    safe_ensembles = [
        Ensemble(trees=e.trees, base_score=e.base_score, link=e.link, threshold_precision="float64")
        for e in ensembles
    ]
    return multiclass_to_sas(safe_ensembles)
