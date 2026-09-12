"""Public entry points for LightGBM models -> SQL and SAS.

See parse_lgbm.py for exactly what's supported and why (verified allow-list,
same philosophy as the xgboost/sklearn sides). Unlike xgboost, LightGBM's
thresholds are native float64 (verified - see parse_lgbm.py), so SAS
emission needs no quantizer and no safety check: it just works, the same
way it does for sklearn.
"""

from typing import Optional

import lightgbm as lgb

from xgb2sql.emit_sas import ensemble_to_sas, multiclass_to_sas
from xgb2sql.emit_sql import ensemble_to_sql, multiclass_to_sql
from xgb2sql.parse_lgbm import lgbm_to_ensemble, lgbm_to_multiclass_ensembles


def _to_booster(model) -> lgb.Booster:
    """Accept either a raw lgb.Booster or the sklearn-wrapper's fitted
    LGBMRegressor/LGBMClassifier (which carries the Booster as .booster_).
    """
    if isinstance(model, lgb.Booster):
        return model
    booster = getattr(model, "booster_", None)
    if booster is not None:
        return booster
    raise TypeError(
        f"{type(model).__name__} is not a lgb.Booster and has no .booster_ "
        "attribute. Pass the result of lgb.train(), or a fitted "
        "LGBMRegressor/LGBMClassifier."
    )


def lgbm_to_sql(
    model,
    feature_names: Optional[list] = None,
    float_type: str = "FLOAT",
    double_type: str = "DOUBLE",
    link: Optional[str] = None,
    categorical_feature_values: Optional[dict] = None,
) -> str:
    """Convert a fitted LightGBM regressor or binary classifier to a single
    SQL expression (the regression value, or P(class=1) for a classifier).

    Args:
        model: lgb.Booster (from lgb.train), or a fitted LGBMRegressor/
            LGBMClassifier. Only boosting_type="gbdt" (the default),
            objective in {"regression", "binary"}, single-class (binary,
            not multiclass) is supported - see parse_lgbm.py. Categorical
            features are supported - see categorical_feature_values below.
        feature_names: Feature names in training column order. If omitted,
            uses the names stored on the Booster itself.
        float_type / double_type: unused here (LightGBM always compares in
            float64 - no float32-cast trick needed, unlike xgboost); kept
            for a consistent call signature across xgboost/sklearn/lgbm.
        link: Explicit link override ("identity"/"logistic"). Auto-detected
            from the model's objective if omitted.
        categorical_feature_values: Required only if the model has a
            pandas-Categorical-dtype categorical feature: {feature_name:
            [its raw category values, any order]} - see parse_lgbm.py for
            why (a plain int/numeric-dtype categorical column needs no
            entry here - its codes are already the raw values).

    Returns:
        SQL expression string for SELECT {expr} AS prediction FROM table.
    """
    ensemble = lgbm_to_ensemble(_to_booster(model), feature_names=feature_names, link=link,
                                 categorical_feature_values=categorical_feature_values)
    return ensemble_to_sql(ensemble, float_type=float_type, double_type=double_type)


def lgbm_to_sas(
    model, feature_names: Optional[list] = None, link: Optional[str] = None,
    categorical_feature_values: Optional[dict] = None,
) -> str:
    """Convert a fitted LightGBM regressor or binary classifier to a SAS
    DATA-step expression, usable as `prediction = {expr};`.

    No quantization or safety check needed (unlike xgboost_to_sas) -
    LightGBM's thresholds are natively float64, verified empirically (see
    parse_lgbm.py's module docstring). categorical_feature_values: see
    lgbm_to_sql.
    """
    ensemble = lgbm_to_ensemble(_to_booster(model), feature_names=feature_names, link=link,
                                 categorical_feature_values=categorical_feature_values)
    return ensemble_to_sas(ensemble)


def lgbm_to_sql_multiclass(
    model, feature_names: Optional[list] = None, float_type: str = "FLOAT", double_type: str = "DOUBLE",
    categorical_feature_values: Optional[dict] = None,
) -> dict:
    """Convert a fitted multiclass LightGBM model to {class_index: sql_expression},
    with softmax applied across classes - same convention as xgboost_to_sql_multiclass.
    categorical_feature_values: see lgbm_to_sql.
    """
    ensembles = lgbm_to_multiclass_ensembles(_to_booster(model), feature_names=feature_names,
                                              categorical_feature_values=categorical_feature_values)
    return multiclass_to_sql(ensembles, float_type=float_type, double_type=double_type)


def lgbm_to_sas_multiclass(
    model, feature_names: Optional[list] = None, categorical_feature_values: Optional[dict] = None,
) -> dict:
    """SAS equivalent of lgbm_to_sql_multiclass. No quantizer needed here
    either - see lgbm_to_sas.
    """
    ensembles = lgbm_to_multiclass_ensembles(_to_booster(model), feature_names=feature_names,
                                              categorical_feature_values=categorical_feature_values)
    return multiclass_to_sas(ensembles)
