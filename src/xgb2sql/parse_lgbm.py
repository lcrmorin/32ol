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
  model). So every Tree here uses weight=1.0 and Ensemble.base_score=0.0 -
  no separate base-score term to invert, unlike xgboost - EXCEPT for
  boosting_type="rf" (see next point).
- boosting_type "gbdt", "dart", "rf", and "goss" are ALL supported, but NOT
  all with the same plain leaf-value sum - rf is a real exception, caught by
  a mistake worth documenting. The first verification pass compared a plain
  leaf-sum against Booster.predict(raw_score=True) across 9 configurations
  (dart with varying drop_rate/max_drop, rf with varying bagging/feature
  fraction, goss with varying top_rate/other_rate, with and without missing
  values) and got 0.0 diff every time, including dart - xgboost's DART needs
  blocking because its per-tree dropout weighting isn't recoverable from a
  dumped tree, but LightGBM bakes any dart/goss rescaling into the stored
  leaf values, so a plain sum is genuinely correct for dart/goss/gbdt. BUT
  that first pass was checking the wrong thing for rf: raw_score=True always
  returns the plain sum by definition, regardless of boosting_type - it does
  NOT mean that sum is the real prediction. Checking against the ACTUAL
  Booster.predict()/predict_proba() (which is what every downstream test and
  every other parser in this project actually compares against) revealed rf
  mode's predict() divides the sum by the number of trees before applying
  the link function - confirmed exactly (0.0 diff) once every Tree.weight is
  set to 1/n_trees for rf specifically, for both regression and binary
  classification. Every other verification claim in this file was checked
  against real predict()/predict_proba() from the start; this one wasn't,
  initially, and it would have shipped a bug (predictions too large by a
  factor of n_trees) if the downstream SQL/SAS test comparison against
  predict() hadn't caught it - see TESTING_PLAN.md.
- Categorical splits (decision_type "==", where `threshold` is a
  "||"-joined string of category CODES, e.g. "2||4") are supported. Two
  genuinely different cases, both handled:

  * A plain integer/numeric-dtype column passed via categorical_feature=[...]
    (not pandas Categorical dtype): the codes in `threshold` ARE the raw
    values (verified: LightGBM's `feature_infos[name]["values"]` for such a
    column is just its own literal value set) - no decoding needed, and the
    generated SQL/SAS compares the raw column directly against those
    integers.
  * A pandas Categorical dtype column: the codes are pandas' own `.cat.codes`
    (not raw labels), and the model dump's `pandas_categorical` field holds,
    for EVERY such column in the training DataFrame (regardless of whether
    any tree actually split on it), its `.cat.categories` array - i.e. the
    code -> raw-label lookup, in code order. But `pandas_categorical` is a
    plain ORDERED LIST with no feature-name tag, so this module can't itself
    tell "entry 0 belongs to which column" from the fitted model alone
    (mirrors why CatBoost's cat_feature_values is needed in parse_catboost.py
    - a different root cause, same shape of problem: the model doesn't
    self-describe this mapping). So `lgbm_to_ensemble(...,
    categorical_feature_values={'col': [...]})` takes the caller-supplied
    raw category list for each pandas-dtype column that needs decoding, and
    resolves which `pandas_categorical` entry is "col" by matching value
    SETS (`set(caller's list) == set(pandas_categorical[i])`) - not by
    position, so the caller doesn't need to get the order right, only the
    membership. Every entry in `pandas_categorical` must end up matched to
    exactly one feature or this raises rather than silently leaving one
    column's codes undecoded (which would emit SQL comparing an internal
    code against a column that actually holds string labels - wrong, and
    the kind of mistake that wouldn't be obvious from the generated SQL
    alone). If `pandas_categorical` is empty, every categorical feature in
    the model is the first (undecoded) kind and no cat_feature_values are
    needed at all.

  Bit convention (left=match/right=no-match, i.e. the OPPOSITE of a numeric
  split's `yes`=`<=`/`no`=`>` framing needing no swap here since "match" IS
  the IR's `categories` yes-branch) and missing-value routing were both
  confirmed against LightGBM's own C++ predictor source
  (`include/LightGBM/tree.h`, `CategoricalDecision`), not guessed from
  black-box testing alone (black-box testing agreed, but this is exactly
  the kind of claim - see the CatBoost/HistGradientBoosting precedent in
  this project - worth pinning to source): a categorical split ALWAYS
  routes NaN and any unseen/negative category code to the "no-match" (right)
  child, UNCONDITIONALLY - `default_left` is a numeric-split-only field
  that categorical splits don't consult at all (confirmed: dumped
  categorical splits were never observed with default_left=True across many
  configurations including ones deliberately constructed so a learned
  default_left=True would clearly be better, and the source function never
  reads it for the categorical branch). This is a real, if easy to miss,
  LightGBM behavior worth flagging since it's NOT the same missing-handling
  shape as this module's own numeric splits two paragraphs up.

  No float32/float64 precision question applies here at all (unlike this
  module's numeric-split float64-vs-xgboost's-float32 finding above):
  category codes are always small exact integers on both sides of the
  comparison, so there is no rounding-boundary case to get wrong regardless
  of which SQL numeric type stores them.
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

_SUPPORTED_BOOSTING = {"gbdt", "dart", "rf", "goss"}


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
    if boosting not in _SUPPORTED_BOOSTING:
        raise NotImplementedError(
            f"boosting_type='{boosting}' has not been verified against a live "
            f"prediction comparison and is not supported (verified: {sorted(_SUPPORTED_BOOSTING)}). "
            "Open an issue if you need it."
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


def _lgbm_categorical_decode_map(dump: dict, categorical_feature_values: Optional[dict]) -> dict:
    """{feature_name: tuple(raw labels, indexed by pandas .cat code)} for
    every pandas-Categorical-dtype feature that needs decoding. See this
    module's docstring for why this can't be derived from the model alone
    and why matching is done by value-SET identity rather than position.
    """
    pandas_cat = dump.get("pandas_categorical") or []
    if not pandas_cat:
        return {}
    categorical_feature_values = categorical_feature_values or {}
    feature_infos = dump.get("feature_infos", {})
    cat_features = {name for name, info in feature_infos.items() if info.get("values")}

    decode_map: dict = {}
    used_pc_indices: dict = {}
    for name, values in categorical_feature_values.items():
        if name not in cat_features:
            raise ValueError(
                f"{name!r} is not a categorical feature in this model "
                f"(per feature_infos) - categorical_feature_values should only "
                f"list categorical_feature=[...] columns."
            )
        target_set = set(values)
        matches = [i for i, pc in enumerate(pandas_cat) if set(pc) == target_set]
        if not matches:
            raise ValueError(
                f"categorical_feature_values[{name!r}] (values {sorted(map(str, target_set))}) "
                "doesn't match any pandas-categorical column recorded in this model's "
                "pandas_categorical (matched by value-set equality) - check it's the exact "
                "set of categories that column had at training time (df[col].cat.categories)."
            )
        if len(matches) > 1:
            raise ValueError(
                f"categorical_feature_values[{name!r}]'s value set is ambiguous - {len(matches)} "
                "different pandas-categorical columns in this model share that exact category "
                "set, so which one is {name!r} can't be told apart by value alone."
            )
        idx = matches[0]
        if idx in used_pc_indices:
            raise ValueError(
                f"categorical_feature_values[{name!r}] and "
                f"categorical_feature_values[{used_pc_indices[idx]!r}] both matched the same "
                "underlying pandas-categorical column - can't both be right."
            )
        used_pc_indices[idx] = name
        decode_map[name] = tuple(pandas_cat[idx])

    if len(used_pc_indices) < len(pandas_cat):
        unresolved_names = sorted(cat_features - set(decode_map))
        raise ValueError(
            f"This model has {len(pandas_cat)} pandas-categorical-dtype column(s) recorded "
            f"(pandas_categorical) but categorical_feature_values only resolved "
            f"{len(used_pc_indices)} of them. At least one of these categorical feature(s) "
            f"needs an entry: {unresolved_names} - pass "
            "categorical_feature_values={'<col>': [<its raw category values, any order>]} "
            "for it (see this module's docstring for why every pandas-dtype column must be "
            "accounted for, not just the ones you think matter)."
        )
    return decode_map


def _lgbm_node_to_ir(node: dict, names: list, decode_map: dict) -> Node:
    if "leaf_value" in node:
        return Leaf(float(node["leaf_value"]))

    decision_type = node.get("decision_type")
    if decision_type == "==":
        feature_name = names[node["split_feature"]]
        codes = [int(c) for c in node["threshold"].split("||")]
        if feature_name in decode_map:
            labels = decode_map[feature_name]
            categories = tuple(labels[c] for c in codes)
        else:
            categories = tuple(codes)
        match_node = _lgbm_node_to_ir(node["left_child"], names, decode_map)
        nomatch_node = _lgbm_node_to_ir(node["right_child"], names, decode_map)
        # NaN AND any unseen/negative category code ALWAYS route to
        # no-match, unconditionally - confirmed in LightGBM's own C++
        # predictor source (CategoricalDecision in tree.h), which never
        # consults default_left for a categorical split - see module
        # docstring. known_categories is intentionally left unset: an
        # unseen-but-valid-looking code (e.g. a genuinely new category
        # string) behaves exactly like "not in categories", i.e. plain
        # no-match, NOT a separate missing case the way HistGradientBoosting
        # treats it - there is nothing extra to model here.
        return Split(feature=feature_name, categories=categories,
                     yes=match_node, no=nomatch_node, missing=nomatch_node)
    if decision_type != "<=":
        raise NotImplementedError(
            f"Unrecognized LightGBM decision_type={decision_type!r} "
            "- only '<=' (numeric) and '==' (categorical) splits are supported."
        )

    feature_idx = node["split_feature"]
    threshold = float(node["threshold"])
    yes_node = _lgbm_node_to_ir(node["left_child"], names, decode_map)
    no_node = _lgbm_node_to_ir(node["right_child"], names, decode_map)
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
    categorical_feature_values: Optional[dict] = None,
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
        categorical_feature_values: Required only if the model has a
            pandas-Categorical-dtype categorical feature: {feature_name:
            [its raw category values, any order]} - see this module's
            docstring for why and how this is matched.
    """
    _check_boosting_supported(model)
    dump = model.dump_model()

    num_class = dump.get("num_class", 1)
    if num_class != 1:
        raise NotImplementedError(
            f"Multiclass LightGBM models (num_class={num_class}) are not supported "
            "by lgbm_to_ensemble - use lgbm_to_multiclass_ensembles (or "
            "lgbm_to_sql_multiclass/lgbm_to_sas_multiclass) instead."
        )

    resolved_link = _resolve_link(dump.get("objective", ""), link)
    names = feature_names if feature_names is not None else model.feature_name()
    decode_map = _lgbm_categorical_decode_map(dump, categorical_feature_values)

    boosting = _get_boosting_type(model)
    # rf mode is the one real exception to "leaf values already have
    # everything baked in, weight=1.0 always": LightGBM's predict() AVERAGES
    # the bagged trees (divides by tree count) rather than summing them -
    # raw_score=True still returns the plain sum, which is why this needed
    # its own verification rather than being covered by the gbdt/dart/goss
    # check above. Verified: predict(raw_score=True) == sum(leaves), but the
    # actual predict()/predict_proba() == sigmoid(sum(leaves) / n_trees) for
    # rf specifically - see TESTING_PLAN.md.
    n = len(dump["tree_info"])
    weight = (1.0 / n) if (boosting == "rf" and n > 0) else 1.0

    trees = tuple(
        Tree(root=_rename_feature(_lgbm_node_to_ir(t["tree_structure"], names, decode_map), names), weight=weight)
        for t in dump["tree_info"]
    )
    return Ensemble(trees=trees, base_score=0.0, link=resolved_link, threshold_precision="float64")


def lgbm_to_multiclass_ensembles(
    model: lgb.Booster,
    feature_names: Optional[list] = None,
    categorical_feature_values: Optional[dict] = None,
) -> list:
    """Parse a multiclass LightGBM Booster into num_class raw-margin
    Ensembles (link="identity" - softmax across classes is the emitter's
    job, same convention as xgboost/GradientBoostingClassifier multiclass).

    Trees are stored round-robin, same as xgboost: tree i belongs to class
    (i % num_class) - verified against a real model's predict() (softmax of
    the round-robin-assigned raw sums matched exactly; a "class-major"
    ordering hypothesis was also checked and did NOT match, ruling it out
    rather than assuming round-robin from the xgboost precedent alone - see
    TESTING_PLAN.md). base_score=0.0 for every class, same reasoning as the
    binary/regression case above.
    """
    _check_boosting_supported(model)
    dump = model.dump_model()
    num_class = dump.get("num_class", 1)
    if num_class < 2:
        raise ValueError(f"model has num_class={num_class}; not a multiclass model.")

    names = feature_names if feature_names is not None else model.feature_name()
    decode_map = _lgbm_categorical_decode_map(dump, categorical_feature_values)
    boosting = _get_boosting_type(model)
    all_trees = dump["tree_info"]
    if len(all_trees) % num_class != 0:
        raise ValueError(
            f"model has {len(all_trees)} trees, which is not a multiple of "
            f"num_class={num_class}; trees can't be assigned round-robin to classes."
        )
    n_rounds = len(all_trees) // num_class
    weight = (1.0 / n_rounds) if (boosting == "rf" and n_rounds > 0) else 1.0

    class_trees = {c: [] for c in range(num_class)}
    for i, t in enumerate(all_trees):
        class_trees[i % num_class].append(t)

    return [
        Ensemble(
            trees=tuple(
                Tree(root=_rename_feature(_lgbm_node_to_ir(t["tree_structure"], names, decode_map), names), weight=weight)
                for t in class_trees[c]
            ),
            base_score=0.0, link="identity", threshold_precision="float64",
        )
        for c in range(num_class)
    ]
