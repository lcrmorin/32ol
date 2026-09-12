"""Parse a trained CatBoost model into the canonical IR (see ir.py).

Key details (verified empirically before writing this module - see
TESTING_PLAN.md for the actual numbers):

- CatBoost's default tree structure (grow_policy="SymmetricTree") is an
  OBLIVIOUS/symmetric tree: every node at a given depth level shares the
  SAME split condition, so a depth-d tree is stored as just `d` splits plus
  2**d leaf values, not a general binary tree. This module reconstructs an
  equivalent conventional binary tree (2**d - 1 internal nodes) from that
  compact form, because the shared IR (ir.py) only knows how to represent
  conventional per-node splits.
- The leaf a row lands in is `sum(bit_j * 2**j for j, split_j in
  enumerate(tree["splits"]))`, where `bit_j = 1 if value > split_j["border"]
  else 0` - i.e. splits[0] is the LEAST significant bit of the leaf index,
  and ">" (not ">=") is the direction that sets that bit. Verified by
  brute-force checking all 4 combinations of (bit order, comparison
  direction) against a real model's own .predict() - only "splits[0] = LSB,
  '>' sets the bit" reproduced predict() exactly (0.0 diff); the other three
  were off by tens of units.
- Unlike LightGBM (float64-native) but LIKE xgboost, CatBoost's stored split
  borders ARE float32 values (every border in a real model's JSON dump was
  bit-identical to its own float32 rounding), AND - this is the part that
  actually matters for correctness - the INCOMING feature value is also
  truncated to float32 before the comparison at prediction time. Verified
  the same way as xgboost/LightGBM: adversarial rows placed in the exact
  float32-rounding danger band matched a float32-cast-truncated manual walk
  exactly (0.0 diff) and disagreed with a plain float64 walk (diff ~48 on a
  toy model). So CatBoost needs the SAME treatment as xgboost:
  `threshold_precision="float32"` - the SQL CAST(col AS FLOAT) trick, and
  the SAME quantizer-gated path (not "just works") to reach SAS. This was
  not obvious going in (LightGBM turned out float64-native, so it would
  have been easy to assume CatBoost - also a "modern" gradient booster -
  behaves the same way; it doesn't).
- Missing values: each float feature has a single, GLOBAL
  `nan_value_treatment` ("AsTrue", "AsFalse", or "AsIs") in the model's own
  JSON dump, not a per-split flag like xgboost/LightGBM/sklearn. "AsTrue"
  means NaN always satisfies "value > border" (routes like bit=1) on every
  split for that feature; "AsFalse" and "AsIs" both route NaN like bit=0 -
  verified empirically, including the specific case of a feature that never
  saw a NaN at train time ("AsIs") still being fed one at predict time
  (CatBoost's predict() silently accepts this and routes it as bit=0, 0.0
  diff against a manual walk assuming that).
- base_score: the model's `scale_and_bias` field ([scale, [bias]]) applies
  ONCE across the whole ensemble - prediction = bias + scale * sum(every
  tree's selected leaf value) - not per-tree like GradientBoostingRegressor's
  learning_rate. Folded into the IR by giving every Tree weight=scale and
  the Ensemble base_score=bias (the IR's own base_score + sum(weight_i *
  tree_i) formula reproduces bias + scale*sum(trees) exactly when every
  weight_i is the same `scale`).
- Categorical features (CatBoost's own headline feature - target-statistic
  encoding, order-dependent permutation counters) are explicitly NOT
  supported: detected via the model dump's features_info["categorical_features"]
  and raises NotImplementedError rather than silently ignoring them or
  guessing at the ctr formula.
- Only grow_policy="SymmetricTree" (the default) is supported - "Lossguide"
  and "Depthwise" produce a structurally different (non-oblivious) tree
  dump that this parser doesn't read.
- Only single-target regression (loss_function="RMSE") and binary
  classification (loss_function="Logloss") are in the verified allow-list,
  the same restricted-allow-list philosophy as parse_xgb.py/parse_sklearn.py.
  Multiclass is not supported yet (needs a per-round softmax across classes,
  not built/verified here).
"""

import json
import tempfile
from typing import Optional

from xgb2sql.ir import Ensemble, Leaf, Node, Split, Tree

_LOSS_LINKS = {
    "RMSE": "identity",
    "Logloss": "logistic",
}


def _dump_json(model) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        model.save_model(f.name, format="json")
        with open(f.name) as fh:
            return json.load(fh)


def _check_supported(model, dump: dict) -> None:
    if dump.get("features_info", {}).get("categorical_features"):
        raise NotImplementedError(
            "This CatBoost model has categorical features. CatBoost's native "
            "categorical handling (order-dependent target-statistic counters, "
            "computed over the training permutation) is not implemented/verified "
            "here - retrain on numeric-only features (encode categoricals "
            "yourself beforehand), or open an issue."
        )
    grow_policy = model.get_all_params().get("grow_policy", "SymmetricTree")
    if grow_policy != "SymmetricTree":
        raise NotImplementedError(
            f"grow_policy={grow_policy!r} is not supported - only the default "
            "'SymmetricTree' (oblivious trees) has a parser here; 'Lossguide' and "
            "'Depthwise' produce a structurally different, non-oblivious tree "
            "representation this module doesn't read."
        )
    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) > 2:
        raise NotImplementedError(
            f"CatBoost model has {len(classes)} classes - multiclass is not "
            "supported yet (needs a per-round softmax across classes, not "
            "built/verified for CatBoost specifically)."
        )


def _resolve_link(dump: dict, link: Optional[str]) -> str:
    if link is not None:
        if link not in ("identity", "logistic", "exp"):
            raise ValueError(f"link must be one of identity/logistic/exp, got {link!r}")
        return link
    loss = dump["model_info"].get("params", {}).get("loss_function", {}).get("type", "")
    resolved = _LOSS_LINKS.get(loss)
    if resolved is None:
        raise NotImplementedError(
            f"loss_function={loss!r} is not in the verified allow-list "
            f"({sorted(_LOSS_LINKS)}). Pass link='identity'/'logistic' explicitly "
            "once you've confirmed the correct link function for this loss against "
            "a real prediction comparison (see TESTING_PLAN.md) - do not guess."
        )
    return resolved


def _feature_id_by_index(dump: dict) -> dict:
    return {ff["feature_index"]: ff["feature_id"] for ff in dump["features_info"]["float_features"]}


def _nan_bit(dump: dict) -> dict:
    """{float_feature_index: True/False} - True means a NaN on that feature
    should be treated as satisfying "value > border" (routes like bit=1).
    """
    return {
        ff["feature_index"]: (ff.get("nan_value_treatment") == "AsTrue")
        for ff in dump["features_info"]["float_features"]
    }


def _symmetric_tree_to_ir(tree: dict, feature_id_by_index: dict, nan_bit: dict) -> Node:
    splits = tree["splits"]
    leaves = tree["leaf_values"]
    depth = len(splits)

    def build(level: int, idx: int) -> Node:
        if level == depth:
            return Leaf(float(leaves[idx]))
        s = splits[level]
        fidx = s["float_feature_index"]
        feature = feature_id_by_index[fidx]
        border = float(s["border"])
        # bit=1 ("value > border") contributes 2**level to the leaf index.
        yes_node = build(level + 1, idx)                  # bit=0: value <= border
        no_node = build(level + 1, idx + (1 << level))     # bit=1: value > border
        missing_node = no_node if nan_bit.get(fidx, False) else yes_node
        return Split(feature=feature, threshold=border, le=True,
                     yes=yes_node, no=no_node, missing=missing_node)

    return build(0, 0)


def catboost_to_ensemble(model, link: Optional[str] = None) -> Ensemble:
    """Parse a fitted CatBoostRegressor or binary CatBoostClassifier into a
    single Ensemble.

    Args:
        model: Fitted CatBoostRegressor or CatBoostClassifier. Only
            grow_policy="SymmetricTree" (the default), numeric-only features
            (no cat_features), and single-target regression/binary
            classification are supported - see this module's docstring.
        link: Explicit link override ("identity"/"logistic"). Auto-detected
            from the model's loss_function if omitted.
    """
    dump = _dump_json(model)
    _check_supported(model, dump)
    resolved_link = _resolve_link(dump, link)

    feature_id_by_index = _feature_id_by_index(dump)
    nan_bit = _nan_bit(dump)
    scale, bias_list = dump.get("scale_and_bias", [1.0, [0.0]])
    scale = float(scale)
    bias = float(bias_list[0]) if bias_list else 0.0

    trees = tuple(
        Tree(root=_symmetric_tree_to_ir(t, feature_id_by_index, nan_bit), weight=scale)
        for t in dump.get("oblivious_trees", [])
    )
    return Ensemble(trees=trees, base_score=bias, link=resolved_link, threshold_precision="float32")
