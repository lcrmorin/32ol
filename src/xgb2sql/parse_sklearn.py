"""Parse fitted scikit-learn tree models into the canonical IR (see ir.py).

Supported: DecisionTreeRegressor, DecisionTreeClassifier, RandomForestRegressor,
RandomForestClassifier, GradientBoostingRegressor (loss="squared_error" only),
GradientBoostingClassifier (binary only, loss="log_loss" only),
HistGradientBoostingRegressor, HistGradientBoostingClassifier (binary and
multiclass; categorical splits included - see below).

Not supported (raises NotImplementedError rather than guessing): any
GradientBoosting loss other than the defaults checked below, ExtraTrees*
(same tree_ shape as RandomForest but the extra randomization hasn't been
separately verified end-to-end).

HistGradientBoosting categorical splits (`categorical_features=[...]`):
supported. Decoded from the fitted predictor's `raw_left_cat_bitsets`
(one 256-bit bitset per categorical split in the tree, indexed by each
node's own `bitset_idx` - NOT by feature identity; a value's bit is set iff
that raw category routes to the "yes"/left branch) plus the bin mapper's
`known_categories` (every category value seen at train time for that
feature). Reverse-engineered from sklearn's own Cython predictor source
(`_predictor.pyx`) rather than guessed from the public docs.

Two real bugs were caught this way, not assumed away:

1. An initial black-box experiment (varying one feature at a time and
   watching `.predict()`) showed a value outside the feature's known
   categories being silently routed as a "missing" case (see the next
   paragraph) - this is real sklearn behavior, not a bug, but it looked at
   first like the split was reading the wrong column, which led to bug #2.
2. `node.feature_idx` does NOT index the user's original column order once
   a model has ANY categorical feature - confirmed both empirically (a
   categorical split's `feature_idx` pointed at a different, wrong column
   when tested with mixed categorical+numeric features - caught by an
   end-to-end DuckDB comparison test, not by inspection) and then in source:
   `BaseHistGradientBoosting._preprocess_X` runs a `ColumnTransformer` that
   reorders every categorical column FIRST (original relative order), then
   every numerical column (original relative order), before the array ever
   reaches the binner/grower - so `node.feature_idx`, `_bin_mapper.known_categories`,
   and `raw_left_cat_bitsets` are all in THIS reordered space, not the raw
   input's. This affects both categorical AND numeric splits whenever the
   model has at least one categorical feature (a pure-numeric or
   pure-categorical model happens to leave the order unchanged, which is why
   earlier single-categorical-feature testing didn't surface it). Fixed by
   `_hgb_feature_order()`, which reproduces the same reorder on
   `feature_names` (using `model.is_categorical_`, the exact boolean mask
   `ColumnTransformer` split on) before any `feature_idx` lookup - verified
   against a mixed a/b/c fixture (c categorical) via DuckDB.

The one HistGradientBoosting-specific behavior that has no equivalent
elsewhere in this project: a category value that was NEVER SEEN during
training is treated as MISSING at predict time (confirmed directly in
`_predictor.pyx`: `in_bitset` against the "known categories" bitset decides
this, falling through to `missing_go_to_left` when the value matches
neither the left-bitset nor the known-categories bitset) - not silently
routed as if it were a normal "not in the yes-list" value. The IR's
`Split.known_categories` field exists specifically to carry this (see
ir.py) - `emit_sql.py`/`emit_sas.py` treat "not NULL, not in `categories`,
also not in `known_categories`" as an extra way to hit the missing branch.

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


def gradient_boosting_classifier_multiclass_to_ensembles(model, feature_names: list) -> list:
    """Multiclass GradientBoostingClassifier -> num_class raw-margin Ensembles
    (link="identity" - softmax across classes is the emitter's job, via
    emit_sql.multiclass_to_sql, not encoded per-ensemble - same shape as
    xgboost multiclass).

    Per-class base_score is log(model.init_.class_prior_[c]) - verified
    empirically against predict_proba (see TESTING_PLAN.md). Note this is
    the UNCENTERED log-prior (sklearn's actual internal init uses a
    class-mean-centered version) - confirmed they produce identical softmax
    output, since softmax is shift-invariant to adding the same constant to
    every class's raw score, so the simpler uncentered form is used here.
    estimators_[:, c] gives class c's tree for every round in order,
    weighted by learning_rate - same per-round-per-class tree layout HGB
    multiclass uses below.
    """
    loss = getattr(model, "loss", "log_loss")
    if loss not in _GB_CLASSIFIER_LOSSES:
        raise NotImplementedError(
            f"GradientBoostingClassifier(loss={loss!r}) is not supported - only "
            f"{sorted(_GB_CLASSIFIER_LOSSES)} has been verified (see TESTING_PLAN.md)."
        )
    n_classes = len(model.classes_)
    priors = model.init_.class_prior_
    if np.any(priors <= 0.0):
        raise ValueError("a class prior is <= 0; log(prior) is undefined.")
    ensembles = []
    for c in range(n_classes):
        trees = tuple(
            Tree(root=_sk_tree_to_ir(est.tree_, feature_names, None), weight=model.learning_rate)
            for est in model.estimators_[:, c]
        )
        ensembles.append(Ensemble(trees=trees, base_score=float(np.log(priors[c])),
                                   link="identity", threshold_precision="float64"))
    return ensembles


def gradient_boosting_classifier_to_ensemble(model, feature_names: list) -> Ensemble:
    """Binary GradientBoostingClassifier only - use
    gradient_boosting_classifier_multiclass_to_ensembles for >2 classes.
    """
    if len(model.classes_) != 2:
        raise NotImplementedError(
            f"GradientBoostingClassifier with {len(model.classes_)} classes is not "
            "binary - use gradient_boosting_classifier_multiclass_to_ensembles "
            "(or sklearn_to_sql_multiclass/sklearn_to_sas_multiclass) instead."
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


def _bitset_contains(bitset_row, value: int) -> bool:
    """bitset_row is 8x uint32 (a 256-bit set). Bit for integer `value` lives
    at word (value // 32), bit (value % 32) - verified against sklearn's own
    `in_bitset_memoryview` (`_predictor.pyx` / `_bitset.pyx`).
    """
    if value < 0 or value >= 256:
        return False
    word = int(bitset_row[value // 32])
    return bool((word >> (value % 32)) & 1)


def _hgb_feature_order(model, feature_names: list) -> list:
    """`node.feature_idx` does NOT index the user's original column order
    when the model has any categorical features - confirmed empirically
    (a categorical split's feature_idx pointed at the wrong column when
    tested with mixed categorical+numeric features) and then confirmed in
    source: `BaseHistGradientBoosting._preprocess_X` runs a `ColumnTransformer`
    that puts every categorical column FIRST (in original relative order),
    then every numerical column (in original relative order), before handing
    the array to the binner/grower - so `feature_idx`, `_bin_mapper.known_categories`,
    and `raw_left_cat_bitsets` are all in THIS reordered space, not the raw
    input's. `model.is_categorical_` (per raw column, None if the model has
    no categorical features at all) is exactly the boolean mask that
    ColumnTransformer split on, so it's used here to reproduce the same
    reorder on `feature_names` before indexing into it.
    """
    is_cat = getattr(model, "is_categorical_", None)
    if is_cat is None:
        return feature_names
    cat_idx = [i for i, c in enumerate(is_cat) if c]
    num_idx = [i for i, c in enumerate(is_cat) if not c]
    return [feature_names[i] for i in cat_idx + num_idx]


def _hgb_tree_to_ir(nodes, feature_names, raw_left_cat_bitsets, known_categories_per_feature) -> Node:
    """`feature_names` here must already be in the REORDERED (categorical-
    first) column order `_hgb_feature_order` produces - see its docstring.
    `raw_left_cat_bitsets` is the fitted TreePredictor's own array, shape
    (n_categorical_splits_in_this_tree, 8) - indexed by each node's
    `bitset_idx`, NOT by feature. `known_categories_per_feature` is
    `model._bin_mapper.known_categories` - a list indexed by the SAME
    reordered feature_idx space, each entry either None (non-categorical
    feature) or an array of every category value seen at train time for
    that feature. All of this, and the routing logic below, come directly
    from reading sklearn's compiled `_predict_one_from_raw_data` (see
    module docstring).
    """
    def build(node_id: int) -> Node:
        n = nodes[node_id]
        if n["is_leaf"]:
            return Leaf(float(n["value"]))
        feature_idx = int(n["feature_idx"])
        feature = feature_names[feature_idx]
        yes_node = build(n["left"])
        no_node = build(n["right"])
        missing_node = yes_node if n["missing_go_to_left"] else no_node
        if n["is_categorical"]:
            known = known_categories_per_feature[feature_idx]
            known_cats = tuple(sorted(int(v) for v in known))
            bitset_row = raw_left_cat_bitsets[int(n["bitset_idx"])]
            left_cats = tuple(v for v in known_cats if _bitset_contains(bitset_row, v))
            return Split(feature=feature, categories=left_cats, known_categories=known_cats,
                         yes=yes_node, no=no_node, missing=missing_node)
        threshold = float(n["num_threshold"])
        return Split(feature=feature, threshold=threshold, le=True,
                     yes=yes_node, no=no_node, missing=missing_node)

    return build(0)


def _hgb_baseline(model, class_index: Optional[int] = None) -> float:
    baseline = np.asarray(model._baseline_prediction).ravel()
    return float(baseline[0] if class_index is None else baseline[class_index])


def hist_gradient_boosting_regressor_to_ensemble(model, feature_names: list) -> Ensemble:
    """HistGradientBoostingRegressor -> Ensemble. Each stored leaf `value`
    already has the learning rate folded in (confirmed empirically - no extra
    per-tree weight needed, unlike GradientBoostingRegressor), so every tree
    here uses weight=1.0.
    """
    known_categories = model._bin_mapper.known_categories
    ordered_names = _hgb_feature_order(model, feature_names)
    trees = tuple(
        Tree(root=_hgb_tree_to_ir(pred.nodes, ordered_names, pred.raw_left_cat_bitsets, known_categories), weight=1.0)
        for round_predictors in model._predictors
        for pred in round_predictors
    )
    return Ensemble(trees=trees, base_score=_hgb_baseline(model), link="identity", threshold_precision="float64")


def hist_gradient_boosting_classifier_to_ensemble(model, feature_names: list) -> Ensemble:
    """Binary HistGradientBoostingClassifier only - use
    hist_gradient_boosting_classifier_multiclass_to_ensembles for >2 classes.
    """
    if len(model.classes_) != 2:
        raise NotImplementedError(
            f"HistGradientBoostingClassifier with {len(model.classes_)} classes is "
            "not binary - use hist_gradient_boosting_classifier_multiclass_to_ensembles "
            "(or sklearn_to_sql_multiclass/sklearn_to_sas_multiclass) instead."
        )
    known_categories = model._bin_mapper.known_categories
    ordered_names = _hgb_feature_order(model, feature_names)
    trees = tuple(
        Tree(root=_hgb_tree_to_ir(pred.nodes, ordered_names, pred.raw_left_cat_bitsets, known_categories), weight=1.0)
        for round_predictors in model._predictors
        for pred in round_predictors
    )
    return Ensemble(trees=trees, base_score=_hgb_baseline(model), link="logistic", threshold_precision="float64")


def hist_gradient_boosting_classifier_multiclass_to_ensembles(model, feature_names: list) -> list:
    """Multiclass HistGradientBoostingClassifier -> num_class raw-margin
    Ensembles (link="identity" - softmax applied by the emitter across
    classes, same shape as xgboost/GradientBoostingClassifier multiclass).

    model._predictors is [round][class] - one tree per class per round, in
    that order - and model._baseline_prediction is a (1, n_classes) array,
    one bias per class. Verified exactly (0.0 diff) against predict_proba -
    see TESTING_PLAN.md.
    """
    n_classes = len(model.classes_)
    known_categories = model._bin_mapper.known_categories
    ordered_names = _hgb_feature_order(model, feature_names)
    ensembles = []
    for c in range(n_classes):
        trees = tuple(
            Tree(root=_hgb_tree_to_ir(round_predictors[c].nodes, ordered_names,
                                       round_predictors[c].raw_left_cat_bitsets, known_categories), weight=1.0)
            for round_predictors in model._predictors
        )
        ensembles.append(Ensemble(trees=trees, base_score=_hgb_baseline(model, class_index=c),
                                   link="identity", threshold_precision="float64"))
    return ensembles
