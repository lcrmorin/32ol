"""Public entry points for scikit-learn tree models -> SQL and SAS.

See parse_sklearn.py for exactly which estimators/losses are supported and
why (verified allow-list, same philosophy as the xgboost side), and
emit_sas.py for why SAS support is currently sklearn-only.
"""

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from xgb2sql.emit_sas import ensemble_to_sas
from xgb2sql.emit_sql import ensemble_to_sql
from xgb2sql.parse_sklearn import (
    decision_tree_to_ensemble,
    gradient_boosting_classifier_to_ensemble,
    gradient_boosting_regressor_to_ensemble,
    random_forest_to_ensemble,
)

_SUPPORTED = (
    "DecisionTreeRegressor, DecisionTreeClassifier, RandomForestRegressor, "
    "RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier"
)


def _model_to_ensemble(model, feature_names: list):
    """Regressor -> its Ensemble. Binary classifier -> the positive-class
    (index 1) probability Ensemble. Raises for a multiclass classifier -
    use the _multiclass variant instead.
    """
    if isinstance(model, DecisionTreeRegressor):
        return decision_tree_to_ensemble(model, feature_names, class_index=None)
    if isinstance(model, DecisionTreeClassifier):
        _require_binary(model)
        return decision_tree_to_ensemble(model, feature_names, class_index=1)
    if isinstance(model, RandomForestRegressor):
        return random_forest_to_ensemble(model, feature_names, class_index=None)
    if isinstance(model, RandomForestClassifier):
        _require_binary(model)
        return random_forest_to_ensemble(model, feature_names, class_index=1)
    if isinstance(model, GradientBoostingRegressor):
        return gradient_boosting_regressor_to_ensemble(model, feature_names)
    if isinstance(model, GradientBoostingClassifier):
        return gradient_boosting_classifier_to_ensemble(model, feature_names)
    raise NotImplementedError(f"{type(model).__name__} is not supported. Supported: {_SUPPORTED}.")


def _multiclass_ensemble_builder(model, feature_names: list):
    if isinstance(model, DecisionTreeClassifier):
        return lambda c: decision_tree_to_ensemble(model, feature_names, class_index=c)
    if isinstance(model, RandomForestClassifier):
        return lambda c: random_forest_to_ensemble(model, feature_names, class_index=c)
    raise NotImplementedError(
        f"{type(model).__name__} multiclass is not supported here. "
        "DecisionTreeClassifier and RandomForestClassifier are."
    )


def sklearn_to_sql(model, feature_names: list, float_type: str = "FLOAT", double_type: str = "DOUBLE") -> str:
    """Convert a fitted regressor or BINARY classifier to a single SQL
    expression (the regression value, or P(class=1) for a classifier).

    For a multiclass classifier, use sklearn_to_sql_multiclass instead.

    Args:
        model: one of DecisionTreeRegressor, DecisionTreeClassifier (binary),
            RandomForestRegressor, RandomForestClassifier (binary),
            GradientBoostingRegressor (loss="squared_error"),
            GradientBoostingClassifier (binary, loss="log_loss").
        feature_names: feature names in the same column order the model was
            fit with (model.feature_names_in_ if fit on a DataFrame).
        float_type / double_type: SQL type names for the float32-cast trick
            (unused here - sklearn always compares in float64) and the
            float64 cast respectively. double_type: "DOUBLE" (DuckDB/MySQL 8+),
            "DOUBLE PRECISION" (PostgreSQL).
    """
    ensemble = _model_to_ensemble(model, feature_names)
    return ensemble_to_sql(ensemble, float_type=float_type, double_type=double_type)


def sklearn_to_sql_multiclass(
    model, feature_names: list, float_type: str = "FLOAT", double_type: str = "DOUBLE"
) -> dict:
    """Convert a fitted multiclass DecisionTreeClassifier or
    RandomForestClassifier to {class_index: sql_expression_for_that_class_proba}.

    Each expression is already a valid probability on its own (they sum to 1
    across classes by construction - both are averages of per-leaf class
    proportions) - no softmax step needed, unlike xgboost/GBM multiclass.
    GradientBoostingClassifier with >2 classes is NOT supported (see
    parse_sklearn.py).
    """
    builder = _multiclass_ensemble_builder(model, feature_names)
    return {
        c: ensemble_to_sql(builder(c), float_type=float_type, double_type=double_type)
        for c in range(len(model.classes_))
    }


def sklearn_to_sas(model, feature_names: list) -> str:
    """Convert a fitted regressor or BINARY classifier to a SAS DATA-step
    expression, usable as `prediction = {expr};`.

    Same model support as sklearn_to_sql. Raises NotImplementedError if the
    resulting ensemble isn't float64-precision (see emit_sas.py) - in
    practice this only happens if you build an Ensemble by hand with a
    non-default threshold_precision; every model this function accepts
    produces a float64 ensemble.
    """
    ensemble = _model_to_ensemble(model, feature_names)
    return ensemble_to_sas(ensemble)


def sklearn_to_sas_multiclass(model, feature_names: list) -> dict:
    """SAS equivalent of sklearn_to_sql_multiclass - see its docstring."""
    builder = _multiclass_ensemble_builder(model, feature_names)
    return {c: ensemble_to_sas(builder(c)) for c in range(len(model.classes_))}


def _require_binary(model) -> None:
    if len(model.classes_) != 2:
        raise NotImplementedError(
            f"{type(model).__name__} has {len(model.classes_)} classes; "
            "sklearn_to_sql/sklearn_to_sas only handle binary classification. Use "
            "the _multiclass variant for a multiclass DecisionTreeClassifier "
            "or RandomForestClassifier."
        )
