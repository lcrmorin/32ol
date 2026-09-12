"""Parse a trained LightGBM Booster into the canonical IR (see ir.py).

Key details (verified empirically before writing this module - see
TESTING_PLAN.md for the actual numbers):

- Unlike XGBoost, LightGBM compares split thresholds in **native float64**,
  the same as sklearn. Verified by walking a real dumped tree two ways (a
  plain float64 comparison, and xgboost's float32-cast-truncated comparison)
  against Booster.predict(): the float64 walk matched exactly (0.0 diff)
  including on adversarial rows deliberately placed in the float32
  rounding danger band, where the float32-cast walk was off by up to 0.49.
  So LightGBM needs none of XGBoost's quantization/SAS-blocking machinery -
  `threshold_precision="float64"` unconditionally, and SAS emission "just
  works" the same way it does for sklearn.
- The numeric-split condition is "value <= threshold" for the left/yes
  branch (decision_type "<=" in the dump) - same direction as sklearn
  (Split.le=True), not xgboost's "<".
- Missing values route via each split's own `default_left` flag - verified
  against real predict() with NaN inputs, including for splits that never
  saw a NaN at train time (missing_type "None" in the dump): default_left
  still governs predict-time NaN routing correctly in that case too, not
  just when missing_type is "NaN".
- The learning rate and any boost_from_average bias are already folded into
  the dumped leaf_value (verified: summing raw leaf values across all trees,
  with no extra scaling, matches Booster.predict(raw_score=True) exactly,
  for both a "regression" model with a large constant offset and a "binary"
  model). So every Tree here uses weight=1.0 and Ensemble.base_score=0.0
  always - there is no separate base-score term to invert, unlike xgboost.
- Only boosting_type="gbdt" (the default) is supported. "dart" and "rf" are
  NOT verified here and are blocked - LightGBM's own docs describe dropout
  and bagging mechanics for those modes that could plausibly affect what a
  plain leaf-value sum means, and this hasn't been checked as carefully as
  the gbdt case above (a quick spot-check suggested dart might sum cleanly
  too, but "suggested by one quick test" is not this project's bar for
  shipping a path as safe - see TESTING_PLAN.md).
- Categorical splits (decision_type "==", where `threshold` is a
  "||"-joined string of category codes rather than a number) are detected
  and raise NotImplementedError, the same posture as
  HistGradientBoosting's categorical splits in parse_sklearn.py - the
  bin-id-to-raw-category mapping (`pandas_categorical` in the model dump)
  hasn't been implemented/verified here.
- Multiclass (num_class > 1) is not supported yet - same reason as
  multiclass GradientBoostingClassifier: needs a per-round softmax across
  n_classes trees that hasn't been built/verified for LightGBM specifically.
"""

from typing import Optional

import lightgbm as lgb

from xgb2sql.ir import Ensemble, Leaf, Node, Split, Tree

# objective -> link function, verified empirically the same way as
# _OBJECTIVE_LINKS in parse_xgb.py (see module docstring point above).
_OBJECTIVE_LINKS = {
    "regression": "identity",
    "binary": "logistic",
}

_UNSUPPORTED_BOOSTING = {
    "dart": (
        "boosting_type='dart' is not supported: LightGBM's DART mode applies "
        "dropout during training, and while a quick check suggested a plain "
        "leaf-value sum might still match Booster.predict() for dart models, "
        "that has not been verified as rigorously as the gbdt case (see "
        "parse_lgbm.py's module docstring and TESTING_PLAN.md). Retrain with "
        "the default boosting_type='gbdt', or open an issue if you need dart "
        "support and can help verify it."
    ),
    "rf": (
        "boosting_type='rf' (random forest mode) is not supported here - not "
        "verified against a real prediction comparison. Retrain with the "
        "default boosting_type='gbdt', or open an issue if you need this."
    ),
    "goss": (
        "boosting_type='goss' is not supported here - not verified against a "
        "real prediction comparison. Retrain with the default "
        "boosting_type='gbdt', or open an issue if you need this."
    ),
}


def _get_boosting_type(model: lgb.Booster) -> str:
    params = getattr(model, "params", None) or {}
    boosting = params.get("boosting") or params.get("boosting_type")
    if boosting:
        return boosting
    # Fall back to parsing the serialized model text (works even if the
    # Booster was loaded from a file rather than freshly trained, where
    # .params may be empty) - LightGBM always writes a "[boosting: ...]" line.
    for line in model.model_to_string().splitlines():
        line = line.strip()
        if line.startswith("[boosting:"):
            return line[len("[boosting:"):].strip(" ]")
    return "gbdt"  # LightGBM's own default when nothing says otherwise.


def _check_boosting_supported(model: lgb.Booster) -> None:
    boosting = _get_boosting_type(model)
    if boosting in _UNSUPPORTED_BOOSTING:
        raise NotImplementedError(_UNSUPPORTED_BOOSTING[boosting])
    if boosting != "gbdt":
        raise NotImplementedError(
            f"boosting_type='{boosting}' has not been verified against a live "
            "prediction comparison and is not supported. Open an issue if you need it."
        )


def _resolve_link(objective: str, link: Optional[str]) -> str:
    if link is not None:
        if link not in ("identity", "logistic", "exp"):
            raise ValueError(f"link must be one of identity/logistic/exp, got {link!r}")
        return link
    base_objective = objective.split()[0] if objective else ""
    resolved = _OBJECTIVE_LINKS.get(base_objective)
    if resolved is None:
        raise NotImplementedError(
            f"objective='{base_objective}' is not in the verified allow-list "
            f"({sorted(_OBJECTIVE_LINKS)}). Pass link='identity'/'logistic' "
            "explicitly once you've confirmed the correct link function for this "
            "objective against a real prediction comparison (see TESTING_PLAN.md) "
            "- do not guess, an unverified link silently produces wrong numbers."
        )
    return resolved


def _lgbm_node_to_ir(node: dict) -> Node:
    if "leaf_value" in node:
        return Leaf(float(node["leaf_value"]))

    if node.get("decision_type") == "==":
        raise NotImplementedError(
            "This LightGBM model has a categorical split (trained with "
            "categorical_feature set). LightGBM encodes the routed category "
            "set as internal bin ids ('threshold' is a '||'-joined list of "
            "codes, not raw category labels) and mapping those back via "
            "pandas_categorical hasn't been implemented/verified here yet - "
            "retrain without categorical_feature, or open an issue."
        )
    if node.get("decision_type") != "<=":
        raise NotImplementedError(
            f"Unrecognized LightGBM decision_type={node.get('decision_type')!r} "
            "- only '<=' (numeric) splits are supported."
        )

    feature_idx = node["split_feature"]
    threshold = float(node["threshold"])
    yes_node = _lgbm_node_to_ir(node["left_child"])
    no_node = _lgbm_node_to_ir(node["right_child"])
    missing_node = yes_node if node["default_left"] else no_node
    return Split(feature=feature_idx, threshold=threshold, le=True,
                 yes=yes_node, no=no_node, missing=missing_node)


def _rename_feature(node: Node, feature_names: list) -> Node:
    """The raw dump uses integer feature indices; rewrite them to names,
    matching every other parser in this package (feature is looked up by
    name, not position, at emit time).
    """
    if isinstance(node, Leaf):
        return node
    name = feature_names[node.feature] if isinstance(node.feature, int) else node.feature
    return Split(
        feature=name, threshold=node.threshold, le=node.le, categories=node.categories,
        yes=_rename_feature(node.yes, feature_names),
        no=_rename_feature(node.no, feature_names),
        missing=_rename_feature(node.missing, feature_names) if node.missing is not None else None,
    )


def lgbm_to_ensemble(
    model: lgb.Booster,
    feature_names: Optional[list] = None,
    link: Optional[str] = None,
) -> Ensemble:
    """Parse a binary/regression LightGBM Booster into a single Ensemble.

    Args:
        model: Trained lgb.Booster (lgb.train's return value, or
            LGBMRegressor/LGBMClassifier's .booster_ attribute). Only
            boosting_type="gbdt" (the default) is supported.
        feature_names: Feature names in the same column order the model was
            trained with. If omitted, uses model.feature_name() (LightGBM
            stores these on the Booster itself, unlike xgboost's Booster).
        link: Explicit link function override ("identity" or "logistic").
            Auto-detected from the model's objective if omitted.
    """
    _check_boosting_supported(model)
    dump = model.dump_model()

    num_class = dump.get("num_class", 1)
    if num_class != 1:
        raise NotImplementedError(
            f"Multiclass LightGBM models (num_class={num_class}) are not supported "
            "yet - needs a per-round softmax across n_classes trees that hasn't "
            "been built/verified for LightGBM specifically."
        )

    resolved_link = _resolve_link(dump.get("objective", ""), link)
    names = feature_names if feature_names is not None else model.feature_name()

    trees = tuple(
        Tree(root=_rename_feature(_lgbm_node_to_ir(t["tree_structure"]), names), weight=1.0)
        for t in dump["tree_info"]
    )
    return Ensemble(trees=trees, base_score=0.0, link=resolved_link, threshold_precision="float64")
