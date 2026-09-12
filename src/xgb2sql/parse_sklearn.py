"""Parse fitted scikit-learn tree models into the canonical IR (see ir.py).

Supported: DecisionTreeRegressor, DecisionTreeClassifier, RandomForestRegressor,
RandomForestClassifier, GradientBoostingRegressor (loss="squared_error" only),
GradientBoostingClassifier (binary only, loss="log_loss" only),
HistGradientBoostingRegressor, HistGradientBoostingClassifier (binary only,
non-categorical splits only - see below).

Not supported (raises NotImplementedError rather than guessing): multiclass
GradientBoostingClassifier or HistGradientBoostingClassifier (both need a
per-round softmax across n_classes trees, not yet implemented/verified), any
GradientBoosting loss other than the defaults checked below, HistGradientBoosting
models trained with `categorical_features` set (the bitset-based categorical
routing hasn't been implemented/verified yet - raises rather than silently
ignoring the categorical split), ExtraTrees* (same tree_ shape as RandomForest
but the extra randomization hasn't been separately verified end-to-end).

HistGradientBoosting note: unlike XGBoost, its stored split thresholds
(`num_threshold`) are genuine float64 - confirmed empirically by walking the
raw node arrays and matching .predict()/.predict_proba() exactly (zero
floating-point difference) on continuous, non-quantized data, missing values
included. So no quantization or truncation trick is needed for it at all -
it's float64-safe "for free", unlike XGBoost (see the SAS-quantization
discussion in TESTING_PLAN.md for why XGBoost needs a different approach).

Missing-value handling: sklearn's CART trees (>=1.3) DO support NaN routing via
a per-node `missing_go_to_left` flag on the fitted tree - this has been verified
empirically against a model's own .predict() with NaN inputs (see TESTING_PLAN.md),
so it is treated as real support, not a fallback default, unlike the plain
sklearn-has-no-missing-branch case emit_sql.py documents for other sources.

Split precision: sklearn compares thresholds in float64 (unlike xgboost's
float32), and its condition is "value <= threshold" for the left/yes branch
(xgboost: "value < threshold"). Both are captured explicitly in the IR
(Split.le, Ensemble.threshold_precision="float64") rather than assumed.
"""

from typing import Optional

import numpy as np

from xgb2sql.ir import Ensemble, Leaf, Node, Split, Tree

_GB_REGRESSOR_LOSSES = {"squared_error"}
_GB_CLASSIFIER_LOSSES = {"log_loss"}


def _sk_tree_to_ir(tree_, feature_names, class_index: Optional[int]) -> Node:
    has_missing = hasattr(tree_, "missing_go_to_left")

    def build(node_id: int) -> Node:
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        if left == -1 and right == -1:
            if class_index is None:
                value = float(tree_.value[node_id, 0, 0])
            else:
                counts = tree_.value[node_id, 0, :]
                total = counts.sum()
                value = float(counts[class_index] / total) if total > 0 else 0.0
            return Leaf(value)

        feature = feature_names[tree_.feature[node_id]]
        threshold = float(tree_.threshold[node_id])
        yes_node = build(left)
        no_node = build(right)
        if has_missing:
            missing_node = yes_node if tree_.missing_go_to_left[node_id] else no_node
        else:
            missing_node = None
        return Split(feature=feature, threshold=threshold, le=True,
                     yes=yes_node, no=no_node, missing=missing_node)

    return build(0)


def decision_tree_to_ensemble(model, feature_names: list, class_index: Optional[int] = None) -> Ensemble:
    """DecisionTreeRegressor (class_index=None) or DecisionTreeClassifier
    (class_index=which class's probability to compute) -> single-tree Ensemble.
    """
    root = _sk_tree_to_ir(model.tree_, feature_names, class_index)
    return Ensemble(trees=(Tree(root=root, weight=1.0),), base_score=0.0,
                     link="identity", threshold_precision="float64")


def random_forest_to_ensemble(model, feature_names: list, class_index: Optional[int] = None) -> Ensemble:
    """RandomForestRegressor/Classifier -> Ensemble that averages every tree
    (weight = 1/n_estimators each). For a classifier, each tree's leaf is
    already that tree's own predicted probability for `class_index`, so the
    weighted sum is exactly sklearn's predict_proba mean-of-trees (verified
    empirically - see TESTING_PLAN.md); no extra link/normalization needed.
    """
    n = len(model.estimators_)
    trees = tuple(
        Tree(root=_sk_tree_to_ir(est.tree_, feature_names, class_index), weight=1.0 / n)
        for est in model.estimators_
    )
    return Ensemble(trees=trees, base_score=0.0, link="identity", threshold_precision="float64")


def gradient_boosting_regressor_to_ensemble(model, feature_names: list) -> Ensemble:
    loss = getattr(model, "loss", "squared_error")
    if loss not in _GB_REGRESSOR_LOSSES:
        raise NotImplementedError(
            f"GradientBoostingRegressor(loss={loss!r}) is not supported - only "
            f"{sorted(_GB_REGRESSOR_LOSSES)} has been verified against a real "
            "prediction comparison (see TESTING_PLAN.md). Other losses (huber, "
            "quantile, absolute_error) likely need a different base-score/link "
            "formula that hasn't been checked."
        )
    base_score = float(model.init_.constant_[0][0])
    trees = tuple(
        Tree(root=_sk_tree_to_ir(est.tree_, feature_names, None), weight=model.learning_rate)
        for est in model.estimators_[:, 0]
    )
    return Ensemble(trees=trees, base_score=base_score, link="identity", threshold_precision="float64")


def gradient_boosting_classifier_to_ensemble(model, feature_names: list) -> Ensemble:
    """Binary GradientBoostingClassifier only. Multiclass raises - it needs a
    per-round softmax across n_classes trees, not implemented here yet.
    """
    if len(model.classes_) != 2:
        raise NotImplementedError(
            f"GradientBoostingClassifier with {len(model.classes_)} classes is not "
            "supported - only binary classification is implemented. Multiclass "
            "needs a per-round softmax across n_classes trees per round that "
            "hasn't been built/verified yet."
        )
    loss = getattr(model, "loss", "log_loss")
    if loss not in _GB_CLASSIFIER_LOSSES:
        raise NotImplementedError(
            f"GradientBoostingClassifier(loss={loss!r}) is not supported - only "
            f"{sorted(_GB_CLASSIFIER_LOSSES)} has been verified (see TESTING_PLAN.md)."
        )
    prior = float(model.init_.class_prior_[1])
    if not (0.0 < prior < 1.0):
        raise ValueError(f"class prior={prior} is exactly 0 or 1; logit is undefined.")
    base_score = float(np.log(prior / (1 - prior)))
    trees = tuple(
        Tree(root=_sk_tree_to_ir(est.tree_, feature_names, None), weight=model.learning_rate)
        for est in model.estimators_[:, 0]
    )
    return Ensemble(trees=trees, base_score=base_score, link="logistic", threshold_precision="float64")


def _hgb_tree_to_ir(nodes, feature_names) -> Node:
    def build(node_id: int) -> Node:
        n = nodes[node_id]
        if n["is_leaf"]:
            return Leaf(float(n["value"]))
        if n["is_categorical"]:
            raise NotImplementedError(
                "This HistGradientBoosting model has a categorical split (trained "
                "with categorical_features set). Categorical routing there uses a "
                "bitset representation that hasn't been implemented/verified here "
                "yet - retrain without categorical_features, or open an issue."
            )
        feature = feature_names[n["feature_idx"]]
        threshold = float(n["num_threshold"])
        yes_node = build(n["left"])
        no_node = build(n["right"])
        missing_node = yes_node if n["missing_go_to_left"] else no_node
        return Split(feature=feature, threshold=threshold, le=True,
                     yes=yes_node, no=no_node, missing=missing_node)

    return build(0)


def _hgb_baseline(model) -> float:
    return float(np.asarray(model._baseline_prediction).ravel()[0])


def hist_gradient_boosting_regressor_to_ensemble(model, feature_names: list) -> Ensemble:
    """HistGradientBoostingRegressor -> Ensemble. Each stored leaf `value`
    already has the learning rate folded in (confirmed empirically - no extra
    per-tree weight needed, unlike GradientBoostingRegressor), so every tree
    here uses weight=1.0.
    """
    trees = tuple(
        Tree(root=_hgb_tree_to_ir(pred.nodes, feature_names), weight=1.0)
        for round_predictors in model._predictors
        for pred in round_predictors
    )
    return Ensemble(trees=trees, base_score=_hgb_baseline(model), link="identity", threshold_precision="float64")


def hist_gradient_boosting_classifier_to_ensemble(model, feature_names: list) -> Ensemble:
    """Binary HistGradientBoostingClassifier only - see module docstring for why
    multiclass isn't supported.
    """
    if len(model.classes_) != 2:
        raise NotImplementedError(
            f"HistGradientBoostingClassifier with {len(model.classes_)} classes is "
            "not supported - only binary classification is implemented. Multiclass "
            "needs a per-round softmax across n_classes trees per round that "
            "hasn't been built/verified yet."
        )
    trees = tuple(
        Tree(root=_hgb_tree_to_ir(pred.nodes, feature_names), weight=1.0)
        for round_predictors in model._predictors
        for pred in round_predictors
    )
    return Ensemble(trees=trees, base_score=_hgb_baseline(model), link="logistic", threshold_precision="float64")
