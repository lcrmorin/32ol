"""Parse a trained XGBoost Booster into the canonical IR (see ir.py).

Key details (see TESTING_PLAN.md for how each of these was verified):
- XGBoost uses float32 for split thresholds internally; emitters that care
  about exact threshold matching should treat `Split.threshold` as already
  being the correct float32 value (this module rounds it there).
- Only booster="gbtree" is supported. "dart" is silently WRONG if bypassed -
  DART's dropout-adjusted tree weights are not recoverable from get_dump()
  alone, so a plain sum of leaf values does not match Booster.predict().
  "gblinear" is a different model family entirely (not tree-based).
- num_parallel_tree > 1 (random forest mode) needs NO special handling:
  XGBoost's own leaf values already fold in whatever averaging is needed.
- The link function (how the summed raw margin becomes a prediction) is
  auto-detected from the model's objective via a verified allow-list.
"""

import json
from typing import Optional

import pandas as pd
import xgboost as xgb

from xgb2sql.ir import Ensemble, Leaf, Split, Tree

# Objective -> link function, verified empirically against Booster.predict().
# "identity": raw margin == prediction. "logistic": prediction =
# sigmoid(margin), and base_score is itself a probability. "exp": prediction
# = exp(margin), and base_score is itself on the output (mean/count) scale.
_OBJECTIVE_LINKS = {
    "binary:logistic": "logistic",
    "reg:logistic": "logistic",
    "reg:squarederror": "identity",
    "reg:squaredlogerror": "identity",
    "reg:pseudohubererror": "identity",
    "reg:absoluteerror": "identity",
    "binary:logitraw": "identity",
    "count:poisson": "exp",
    "reg:gamma": "exp",
    "reg:tweedie": "exp",
}

_UNSUPPORTED_BOOSTERS = {
    "dart": (
        "booster='dart' is not supported: at inference time DART applies "
        "per-tree dropout weighting that is not recoverable from get_dump() "
        "alone, so summing leaf values gives a different number than "
        "Booster.predict() (verified empirically - see TESTING_PLAN.md). "
        "Retrain with booster='gbtree', or open an issue if you need this."
    ),
    "gblinear": (
        "booster='gblinear' is not a tree model and is not supported by "
        "this tree-to-code converter."
    ),
}


def _check_booster_supported(cfg: dict) -> None:
    gb = cfg["learner"]["gradient_booster"]
    name = gb["name"]
    if name in _UNSUPPORTED_BOOSTERS:
        raise NotImplementedError(_UNSUPPORTED_BOOSTERS[name])
    if name != "gbtree":
        raise NotImplementedError(
            f"booster='{name}' has not been verified against a live prediction "
            "comparison and is not supported. Open an issue if you need it."
        )
    # XGBoost (>=3.3-ish, exact version not pinned down) deprecated
    # booster="dart" as a separate gradient_booster and folded it into
    # "gbtree" instead - a dart-trained model now reports
    # gradient_booster.name == "gbtree" too, with its dropout parameters
    # under a dart_train_param block, rather than name == "dart" (verified
    # empirically: a model trained with booster="dart" and rate_drop=0.3 on
    # xgboost 3.4.1 reports name="gbtree", while the same call on xgboost
    # 3.2.0 still reports name="dart" - see TESTING_PLAN.md). This means the
    # `name in _UNSUPPORTED_BOOSTERS` check above silently stopped catching
    # DART on newer xgboost - a real, caught-in-CI gap, not hypothetical:
    # summing the dumped trees for such a model still disagreed with
    # Booster.predict() by ~3x on a real example, exactly the failure this
    # guard exists to prevent.
    #
    # A plain gbtree config ALSO carries a dart_train_param block (always
    # present, verified on xgboost 3.4.1 for a model trained with
    # booster="gbtree" and no dart parameters at all) - but with
    # rate_drop=one_drop=skip_drop=0, which has no effect. So `name` alone no
    # longer distinguishes dart from gbtree; check whether dropout is
    # actually configured to do anything instead.
    dart_cfg = gb.get("dart_train_param")
    if dart_cfg is not None:
        rate_drop = float(dart_cfg.get("rate_drop", 0))
        one_drop = float(dart_cfg.get("one_drop", 0))
        skip_drop = float(dart_cfg.get("skip_drop", 0))
        if rate_drop > 0 or one_drop != 0 or skip_drop > 0:
            raise NotImplementedError(_UNSUPPORTED_BOOSTERS["dart"])


def _resolve_link(cfg: dict, sigmoid: bool, link: Optional[str]) -> str:
    if link is not None:
        if link not in ("identity", "logistic", "exp"):
            raise ValueError(f"link must be one of identity/logistic/exp, got {link!r}")
        return link
    if sigmoid:
        return "logistic"

    objective = cfg["learner"]["objective"]["name"]
    resolved = _OBJECTIVE_LINKS.get(objective)
    if resolved is None:
        raise NotImplementedError(
            f"objective='{objective}' is not in the verified allow-list "
            f"({sorted(_OBJECTIVE_LINKS)}). Pass link='identity'/'logistic'/'exp' "
            "explicitly once you've confirmed the correct link function for this "
            "objective against a real prediction comparison (see TESTING_PLAN.md) "
            "- do not guess, an unverified link silently produces wrong numbers."
        )
    return resolved


def _base_score_to_margin(base_score: float, link: str) -> float:
    """Invert the link function: base_score is stored on the OUTPUT scale
    (a probability for "logistic", a mean/count for "exp"), not the margin
    scale trees add into. Verified empirically - see TESTING_PLAN.md.
    """
    if link == "identity":
        return base_score
    if link == "logistic":
        if not (0.0 < base_score < 1.0):
            raise ValueError(
                f"base_score={base_score} is exactly 0 or 1, so logit(base_score) "
                "is +/-infinity. This should not happen from real training (XGBoost "
                "clips it internally); if you set base_score manually, use a value "
                "strictly between 0 and 1."
            )
        import numpy as np
        return float(np.log(base_score / (1 - base_score)))
    if link == "exp":
        if base_score <= 0.0:
            raise ValueError(
                f"base_score={base_score} is <= 0, so log(base_score) is undefined "
                "for an exp-link objective (count:poisson/reg:gamma/reg:tweedie)."
            )
        import numpy as np
        return float(np.log(base_score))
    raise AssertionError(f"unreachable: unknown link {link!r}")


def _parse_base_score(cfg):
    raw = cfg["learner"]["learner_model_param"]["base_score"]
    if isinstance(raw, (int, float)):
        return [float(raw)]
    raw = raw.strip("[]")
    return [float(x) for x in raw.split(",")] if "," in raw else [float(raw)]


def _build_cat_label_maps(df, cat_feature_names):
    maps = {}
    for col in cat_feature_names:
        if col not in df.columns:
            continue
        s = df[col] if hasattr(df[col], "cat") else df[col].astype("category")
        maps[col] = dict(enumerate(s.cat.categories))
    return maps


def _resolve_label_maps(cat_label_maps, cat_feature_names, training_df):
    if cat_label_maps is not None:
        return cat_label_maps
    if cat_feature_names and training_df is not None:
        return _build_cat_label_maps(training_df, cat_feature_names)
    return None


def _json_node_to_ir(node: dict, cat_label_maps) -> "Node":
    if "leaf" in node:
        return Leaf(float(node["leaf"]))

    children = {c["nodeid"]: c for c in node["children"]}
    yes = _json_node_to_ir(children[node["yes"]], cat_label_maps)
    no = _json_node_to_ir(children[node["no"]], cat_label_maps)
    missing = _json_node_to_ir(children[node["missing"]], cat_label_maps)

    sc = node["split_condition"]
    if isinstance(sc, list):
        if cat_label_maps and node["split"] in cat_label_maps:
            labels = tuple(cat_label_maps[node["split"]].get(c, c) for c in sc)
        else:
            labels = tuple(sc)
        return Split(feature=node["split"], categories=labels, yes=yes, no=no, missing=missing)

    if isinstance(sc, str):
        sc = float(sc.strip("[]"))
    import numpy as np
    threshold = float(np.float32(float(sc)))
    return Split(feature=node["split"], threshold=threshold, yes=yes, no=no, missing=missing)


def _trees_to_ir(model, label_maps):
    return [Tree(root=_json_node_to_ir(json.loads(t), label_maps)) for t in model.get_dump(dump_format="json")]


def xgb_to_ensemble(
    model: xgb.Booster,
    sigmoid: bool = False,
    cat_feature_names=None,
    cat_label_maps=None,
    training_df: Optional[pd.DataFrame] = None,
    link: Optional[str] = None,
) -> Ensemble:
    """Parse a binary/regression XGBoost Booster into a single Ensemble."""
    cfg = json.loads(model.save_config())
    _check_booster_supported(cfg)
    resolved_link = _resolve_link(cfg, sigmoid, link)

    label_maps = _resolve_label_maps(cat_label_maps, cat_feature_names, training_df)
    base_score = _parse_base_score(cfg)[0]
    base_margin = _base_score_to_margin(base_score, resolved_link)

    trees = _trees_to_ir(model, label_maps)
    return Ensemble(trees=tuple(trees), base_score=base_margin, link=resolved_link)


def xgb_to_multiclass_ensembles(
    model: xgb.Booster,
    num_class: int,
    cat_feature_names=None,
    cat_label_maps=None,
    training_df: Optional[pd.DataFrame] = None,
) -> list:
    """Parse a multiclass XGBoost Booster into num_class raw-margin Ensembles
    (link="identity" - softmax across classes is the emitter's job, not
    encoded per-ensemble). Trees are stored round-robin: tree i belongs to
    class (i % num_class). Works for both multi:softprob and multi:softmax -
    they share the same tree structure (verified in tests).
    """
    cfg = json.loads(model.save_config())
    _check_booster_supported(cfg)

    actual_num_class = int(cfg["learner"]["learner_model_param"].get("num_class", 0))
    if actual_num_class and actual_num_class != num_class:
        raise ValueError(
            f"num_class={num_class} was passed but the model was trained with "
            f"num_class={actual_num_class}. Using the wrong value silently "
            "misassigns trees to classes."
        )

    label_maps = _resolve_label_maps(cat_label_maps, cat_feature_names, training_df)
    base_scores = _parse_base_score(cfg)
    class_base = base_scores if len(base_scores) == num_class else [base_scores[0]] * num_class

    all_trees = _trees_to_ir(model, label_maps)
    if len(all_trees) % num_class != 0:
        raise ValueError(
            f"model has {len(all_trees)} trees, which is not a multiple of "
            f"num_class={num_class}; trees can't be assigned round-robin to classes."
        )
    class_trees = {c: [] for c in range(num_class)}
    for i, t in enumerate(all_trees):
        class_trees[i % num_class].append(t)

    return [
        Ensemble(trees=tuple(class_trees[c]), base_score=class_base[c], link="identity")
        for c in range(num_class)
    ]
