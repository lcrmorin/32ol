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
- Categorical features (`cat_features=[...]`): PARTIALLY supported, split by
  which split type CatBoost actually used for that feature - detected per
  split via the JSON dump's `split_type`, not assumed from `one_hot_max_size`
  (a feature can end up using either, or a mix across trees):

  * "OneHotFeature" splits (used when the feature's cardinality stays at or
    under `one_hot_max_size`): SUPPORTED. A OneHotFeature split just tests
    "does this row's raw value equal one specific training-time category",
    no target-statistic anything - the only wrinkle is that the split stores
    a HASH of that category (an int, e.g. -1284790409), not the original
    string, and the model dump has no reverse hash->string table. CatBoost's
    own official "export to C++" feature has the exact same problem and
    solves it the same way this module does: it requires a training-data
    `Pool` to bake a `{raw_string: hash}` lookup table into the exported
    code (confirmed directly - calling `model.save_model(..., format="CPP")`
    without a `pool=` argument raises `"Need to use training dataset Pool to
    save mapping {categorical feature value -> hash value} due to the
    absence of a hash function in the model"`). So this module does the same
    thing: `catboost_to_ensemble(model, cat_feature_values={...})` takes a
    caller-supplied `{feature_name: [raw category values]}`, builds a
    throwaway `Pool` from just those values (verified empirically that the
    resulting hash table depends only on the categorical column's own string
    content, not on the other columns/target/row count - a 4-row synthetic
    pool and the real 500-row training pool produced byte-identical hashes),
    exports it via `format="CPP"`, and regexes the `CatFeatureHashes` table
    out of the generated source. No hash FORMULA is guessed or reimplemented
    anywhere - CatBoost computes every hash itself, this just reads the
    answer back out. Bit-order/branch convention (which child is "matches"
    vs "doesn't") was verified empirically the same way as the numeric-split
    convention below: an isolated single-tree model's leaf_values, read out
    for a matching vs. non-matching input, pinned "bit=1 (the child at
    idx+2**level) means MATCH" - the opposite sense from a numeric split's
    bit, where bit=1 means "value > border". Missing/NaN categorical values
    are NOT modeled: CatBoost's own `.predict()` actively REJECTS a NaN/None
    categorical value at predict time (confirmed: raises "Invalid type for
    cat_feature...values should be converted to string"), so there is no
    real prediction to match against - a NULL input in the generated SQL
    just falls through to the "no match" branch, an unvalidated default,
    same posture as sklearn's no-native-missing CART case in emit_sql.py.

  * "OnlineCtr" splits (used once cardinality exceeds `one_hot_max_size` -
    CatBoost's real target-statistic encoding; called "CtrFeature" in the
    CPP-export enum but "OnlineCtr" in the JSON dump's own `split_type` -
    both names refer to the same thing, note here because it cost a wasted
    round-trip to notice the JSON dump doesn't use the CPP exporter's name):
    NOT YET supported, but no longer an "unknown black box" - raises
    NotImplementedError with a specific reason, not a blanket refusal.
    Investigated by generating a real `format="CPP"` export for a CTR-using
    model: it turns out CatBoost's own exporter embeds a COMPLETE, concrete
    `CalcHash(a, b)` combination function (`result = (a + b * MAGIC_MULT)
    ...`-shaped, `MAGIC_MULT = 0x4906ba494954cb65ull`, in 64-bit unsigned
    arithmetic) plus, per CTR, a static `IndexHashViewer`/`CtrMeanHistory`/
    `CtrTotal` lookup table - so unlike the OneHotFeature case, this
    genuinely IS a documented, exact formula now (found in the generated
    source, not guessed), not an undocumented reverse-engineering dead end.
    What's still missing before it could be supported: porting a 64-bit-
    wraparound multiplicative hash into SQL in a way that's portable across
    dialects (signed/unsigned 64-bit overflow behavior differs by engine -
    the same class of risk the float32 threshold-cast trick exists to avoid
    elsewhere in this project), plus emitting the per-CTR static bucket
    table and the `(count+prior_num)/(total+prior_denom)*scale+shift`
    formula, and end-to-end verification against DuckDB the same way
    everything else here is checked. That's a real, scoped follow-up, not
    attempted in this pass - raising honestly here rather than shipping an
    unverified guess. The workaround in the meantime: raise
    `one_hot_max_size` so a feature's cardinality stays under it (forces
    OneHotFeature, which IS supported), or encode the categorical feature
    yourself before training.
- Only grow_policy="SymmetricTree" (the default) is supported - "Lossguide"
  and "Depthwise" produce a structurally different (non-oblivious) tree
  dump that this parser doesn't read.
- Single-target regression (loss_function="RMSE"), binary classification
  (loss_function="Logloss"), and multiclass (loss_function="MultiClass")
  are in the verified allow-list, the same restricted-allow-list philosophy
  as parse_xgb.py/parse_sklearn.py. For multiclass, every class shares the
  SAME oblivious tree structure within a round (CatBoost builds one tree
  per round, not one per class) - `leaf_values` is `2**depth * num_class`
  long, CLASS-MINOR (leaf i's class-c value is at `leaf_values[i*num_class+c]`,
  not a per-class contiguous block) - verified against
  predict(prediction_type="RawFormulaVal"); the other plausible layout was
  checked and ruled out, not assumed. `scale_and_bias`'s bias list has one
  entry per class for a multiclass model.
"""

import json
import re
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


def _split_types_used(dump: dict) -> set:
    return {
        s.get("split_type", "FloatFeature")
        for t in dump.get("oblivious_trees", [])
        for s in t["splits"]
    }


def _cat_feature_hash_map(model, cat_info: list, cat_feature_values: dict) -> dict:
    """Resolve every OneHotFeature split's stored integer hash back to the
    original raw category string, using CatBoost's OWN hashing (never
    reimplemented here) - see this module's docstring for why this needs a
    caller-supplied Pool at all (the model dump has no reverse mapping, and
    neither does CatBoost's own "export to CPP" feature without one).
    """
    import pandas as pd
    from catboost import Pool

    cat_names = {cf["feature_id"] for cf in cat_info}
    missing = cat_names - set(cat_feature_values or {})
    if missing:
        raise ValueError(
            f"cat_feature_values is missing entries for categorical feature(s) "
            f"{sorted(missing)} - supply every raw category value this model's "
            "one-hot splits might need to resolve (e.g. every distinct value the "
            "column had at training time): cat_feature_values={'<col>': [...], ...}."
        )
    for name in cat_names:
        if not cat_feature_values[name]:
            raise ValueError(f"cat_feature_values[{name!r}] is empty.")

    feature_names = list(model.feature_names_)
    n_rows = max(len(cat_feature_values[name]) for name in cat_names)
    cols = {}
    for name in feature_names:
        if name in cat_names:
            vals = list(cat_feature_values[name])
            cols[name] = [str(vals[i % len(vals)]) for i in range(n_rows)]
        else:
            cols[name] = [0.0] * n_rows
    X = pd.DataFrame(cols)[feature_names]
    pool = Pool(X, [0.0] * n_rows, cat_features=sorted(cat_names))

    with tempfile.NamedTemporaryFile(suffix=".cpp") as f:
        model.save_model(f.name, format="CPP", pool=pool)
        with open(f.name) as fh:
            text = fh.read()

    m = re.search(r"CatFeatureHashes\s*=\s*\{(.*?)\};", text, re.S)
    if not m:
        raise RuntimeError(
            "Could not find the CatFeatureHashes table in CatBoost's own CPP "
            "export - this likely means CatBoost changed its export format; "
            "this module's categorical support relies on that table's exact "
            "shape (see parse_catboost.py docstring)."
        )
    entries = re.findall(r'\{"((?:[^"\\]|\\.)*)",\s*(-?\d+)\}', m.group(1))
    hash_to_str: dict = {}
    for raw, h in entries:
        h = int(h)
        s = raw.replace('\\"', '"')
        if h in hash_to_str and hash_to_str[h] != s:
            raise RuntimeError(
                f"Hash collision resolving CatBoost categorical values: both "
                f"{hash_to_str[h]!r} and {s!r} hash to {h} - cannot disambiguate "
                "which one a OneHotFeature split targeting this hash actually means."
            )
        hash_to_str[h] = s
    return hash_to_str


def _check_supported(
    model, dump: dict, allow_multiclass: bool = False, cat_feature_values: Optional[dict] = None,
) -> None:
    # `features_info.categorical_features` lists every feature DECLARED
    # cat_features=[...] at train time, whether or not the fitted model
    # actually ended up splitting on it (a categorical feature the model
    # found useless is simply never referenced by any split and needs no
    # hash resolution at all) - so gate on the split TYPES actually present
    # in the trees, not on that declaration list.
    split_types = _split_types_used(dump)
    if split_types & {"OneHotFeature", "OnlineCtr"}:
        if "OnlineCtr" in split_types:
            raise NotImplementedError(
                "This CatBoost model uses CTR (target-statistic) categorical splits "
                "(split_type='OnlineCtr' in the model dump) for at least one categorical "
                "feature - not supported yet (see this module's docstring for what IS now "
                "known about the hash/CTR formula and what's still missing to port it "
                "safely to SQL/SAS). Workarounds: raise one_hot_max_size so this feature's "
                "cardinality stays under it (CatBoost then uses plain OneHotFeature splits, "
                "which ARE supported - pass cat_feature_values), or encode the categorical "
                "feature yourself before training."
            )
        if cat_feature_values is None:
            raise NotImplementedError(
                "This CatBoost model has one-hot categorical splits (OneHotFeature) - "
                "supported, but resolving them to the original raw category strings "
                "requires cat_feature_values={'<col>': [<every distinct raw value that "
                "column had at training time>], ...} (CatBoost's own model dump has no "
                "reverse hash->string table - see this module's docstring)."
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
    if not allow_multiclass and classes is not None and len(classes) > 2:
        raise NotImplementedError(
            f"CatBoost model has {len(classes)} classes - use "
            "catboost_to_multiclass_ensembles (or catboost_to_sql_multiclass/"
            "catboost_to_sas_multiclass) instead."
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


def _symmetric_tree_to_ir(
    tree: dict, feature_id_by_index: dict, nan_bit: dict,
    cat_feature_id_by_index: Optional[dict] = None, hash_to_str: Optional[dict] = None,
    num_class: int = 1, class_index: int = 0,
) -> Node:
    """Build one conventional binary tree from CatBoost's compact oblivious
    encoding. For a single-target model (num_class=1) `leaf_values` has one
    entry per leaf. For a multiclass model it has `2**depth * num_class`
    entries, CLASS-MINOR: leaf i's value for class c is at
    `leaf_values[i * num_class + class_index]` - verified against
    Booster.predict(prediction_type="RawFormulaVal") (the other plausible
    layout, "class-major" - leaf_values[class_index * 2**depth + i] - was
    also checked and did NOT match, ruling it out rather than assuming
    class-minor from convention alone; see TESTING_PLAN.md). All classes
    for one tree share the exact same split structure - only which leaf
    value gets read out differs.

    A level's split can also be "OneHotFeature" (a categorical equality
    test) rather than "FloatFeature" - its bit convention is the OPPOSITE
    of a numeric split's: bit=1 (the child at idx+2**level) means the raw
    value MATCHES the split's target category, bit=0 means it doesn't -
    verified empirically (see module docstring) by isolating a single-tree
    model and reading which leaf_values entry a matching vs. non-matching
    input actually landed on.
    """
    splits = tree["splits"]
    leaves = tree["leaf_values"]
    depth = len(splits)

    def build(level: int, idx: int) -> Node:
        if level == depth:
            return Leaf(float(leaves[idx * num_class + class_index]))
        s = splits[level]
        if s.get("split_type") == "OneHotFeature":
            feature = cat_feature_id_by_index[s["cat_feature_index"]]
            target_str = hash_to_str[s["value"]]
            match_node = build(level + 1, idx + (1 << level))   # bit=1: raw value == target
            nomatch_node = build(level + 1, idx)                # bit=0: raw value != target
            # No native missing handling (CatBoost's own .predict() rejects a
            # NaN/None categorical value outright - see module docstring), so
            # missing=None: the emitter's documented "route NULL to no" default.
            return Split(feature=feature, categories=(target_str,),
                         yes=match_node, no=nomatch_node, missing=None)
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


def catboost_to_ensemble(model, link: Optional[str] = None, cat_feature_values: Optional[dict] = None) -> Ensemble:
    """Parse a fitted CatBoostRegressor or binary CatBoostClassifier into a
    single Ensemble.

    Args:
        model: Fitted CatBoostRegressor or CatBoostClassifier. Only
            grow_policy="SymmetricTree" (the default) and single-target
            regression/binary classification are supported - see this
            module's docstring. Categorical features are supported only if
            every categorical split CatBoost used is a plain "OneHotFeature"
            (not the CTR-based "OnlineCtr") - see this module's docstring.
        link: Explicit link override ("identity"/"logistic"). Auto-detected
            from the model's loss_function if omitted.
        cat_feature_values: Required if the model has categorical features:
            {feature_name: [every distinct raw category value that column
            had at training time]} - used to resolve CatBoost's internal
            per-category hashes back to the original strings (see this
            module's docstring for why this can't be read off the model
            alone).
    """
    dump = _dump_json(model)
    _check_supported(model, dump, cat_feature_values=cat_feature_values)
    resolved_link = _resolve_link(dump, link)

    feature_id_by_index = _feature_id_by_index(dump)
    nan_bit = _nan_bit(dump)
    cat_info = dump.get("features_info", {}).get("categorical_features") or []
    cat_feature_id_by_index = {cf["feature_index"]: cf["feature_id"] for cf in cat_info}
    needs_hash_map = "OneHotFeature" in _split_types_used(dump)
    hash_to_str = _cat_feature_hash_map(model, cat_info, cat_feature_values) if needs_hash_map else None
    scale, bias_list = dump.get("scale_and_bias", [1.0, [0.0]])
    scale = float(scale)
    bias = float(bias_list[0]) if bias_list else 0.0

    trees = tuple(
        Tree(root=_symmetric_tree_to_ir(t, feature_id_by_index, nan_bit,
                                         cat_feature_id_by_index, hash_to_str), weight=scale)
        for t in dump.get("oblivious_trees", [])
    )
    return Ensemble(trees=trees, base_score=bias, link=resolved_link, threshold_precision="float32")


_MULTICLASS_LOSSES = {"MultiClass"}


def catboost_to_multiclass_ensembles(model, cat_feature_values: Optional[dict] = None) -> list:
    """Parse a multiclass CatBoostClassifier (loss_function="MultiClass")
    into num_class raw-margin Ensembles (link="identity" - softmax across
    classes is the emitter's job, same convention as xgboost/LightGBM/
    GradientBoostingClassifier multiclass).

    Every class shares the SAME split structure within a tree (CatBoost's
    oblivious trees are built once per round, not once per class) - only
    the leaf value read out differs per class (class-minor layout, see
    _symmetric_tree_to_ir). scale_and_bias's bias list has one entry per
    class here (verified: 3 classes -> [scale, [bias0, bias1, bias2]]).
    "MultiClassOneVsAll" and other multiclass losses are not in the
    verified allow-list and raise.

    cat_feature_values: see catboost_to_ensemble - required if the model has
    categorical features.
    """
    dump = _dump_json(model)
    _check_supported(model, dump, allow_multiclass=True, cat_feature_values=cat_feature_values)

    loss = dump["model_info"].get("params", {}).get("loss_function", {}).get("type", "")
    if loss not in _MULTICLASS_LOSSES:
        raise NotImplementedError(
            f"loss_function={loss!r} is not in the verified multiclass allow-list "
            f"({sorted(_MULTICLASS_LOSSES)}). Only 'MultiClass' has been checked "
            "against a real prediction comparison - see TESTING_PLAN.md."
        )

    classes = getattr(model, "classes_", None)
    num_class = len(classes) if classes is not None else None
    if not num_class or num_class < 2:
        raise ValueError(f"model has {num_class} classes; not a multiclass model.")

    feature_id_by_index = _feature_id_by_index(dump)
    nan_bit = _nan_bit(dump)
    cat_info = dump.get("features_info", {}).get("categorical_features") or []
    cat_feature_id_by_index = {cf["feature_index"]: cf["feature_id"] for cf in cat_info}
    needs_hash_map = "OneHotFeature" in _split_types_used(dump)
    hash_to_str = _cat_feature_hash_map(model, cat_info, cat_feature_values) if needs_hash_map else None
    scale, bias_list = dump.get("scale_and_bias", [1.0, [0.0] * num_class])
    scale = float(scale)
    biases = [float(b) for b in bias_list] if bias_list else [0.0] * num_class
    if len(biases) != num_class:
        raise ValueError(
            f"scale_and_bias has {len(biases)} bias entries but the model has "
            f"{num_class} classes - can't assign biases to classes unambiguously."
        )

    trees_raw = dump.get("oblivious_trees", [])
    return [
        Ensemble(
            trees=tuple(
                Tree(root=_symmetric_tree_to_ir(t, feature_id_by_index, nan_bit,
                                                 cat_feature_id_by_index, hash_to_str,
                                                 num_class=num_class, class_index=c), weight=scale)
                for t in trees_raw
            ),
            base_score=biases[c], link="identity", threshold_precision="float32",
        )
        for c in range(num_class)
    ]
