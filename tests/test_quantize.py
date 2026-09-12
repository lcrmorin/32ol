"""Tests for xgb2sql.quantize: the IntegerBinner + check_xgb_sas_safety that
together unlock a verified xgboost -> SAS path.

Two things are being tested here, and it matters to keep them separate:
1. That check_xgb_sas_safety correctly ACCEPTS models whose thresholds really
   are safe (verified end-to-end against the SAS interpreter AND against
   Booster.predict on the same quantized grid).
2. That it does NOT false-positive-reject a model just because a threshold
   happens to sit exactly ON an achievable integer (that's the safe case,
   not the risky one - see quantize.py's docstring for why a naive
   distance-to-nearest-integer heuristic gets this backwards, which an
   earlier version of this checker did).

There is no hand-constructed "genuinely unsafe" fixture here: every real
model tried during development came back safe once the checker was fixed
to simulate the actual comparison instead of using a distance heuristic (see
TESTING_PLAN.md). test_refuses_when_feature_not_declared covers the one
failure mode that's easy to construct on purpose (a missing bins_per_feature
entry) and test_synthetic_unsafe_threshold_is_caught constructs an ensemble
by hand with a threshold deliberately placed to disagree, to prove the
checker's simulation logic actually catches a real disagreement when one
exists, not just that it happens to accept everything.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from sas_interp import eval_sas_row
from xgb2sql import IntegerBinner, xgboost_to_sas
from xgb2sql.ir import Ensemble, Leaf, Split, Tree
from xgb2sql.quantize import check_xgb_sas_safety


@pytest.fixture(scope="module")
def quantized_model():
    rng = np.random.default_rng(3)
    n = 4000
    Xc = pd.DataFrame(rng.uniform(0, 1000, size=(n, 3)), columns=["a", "b", "c"])
    y = (Xc["a"] + 2 * Xc["b"] - Xc["c"] > 500).astype(int)
    binner = IntegerBinner(n_bins=16).fit(Xc)
    Xb = binner.transform(Xc)
    dtrain = xgb.DMatrix(Xb, label=y, feature_names=["a", "b", "c"])
    bst = xgb.train({"objective": "binary:logistic", "max_depth": 5}, dtrain, num_boost_round=40)
    return bst, binner, Xb


def test_integer_binner_output_range(quantized_model):
    _, binner, Xb = quantized_model
    for col in ["a", "b", "c"]:
        assert Xb[col].min() >= 0
        assert Xb[col].max() < binner.n_levels(col)
        assert (Xb[col] == Xb[col].round()).all()


def test_safety_check_accepts_verified_safe_model(quantized_model):
    bst, binner, _ = quantized_model
    from xgb2sql.parse_xgb import xgb_to_ensemble
    ensemble = xgb_to_ensemble(bst)
    violations = check_xgb_sas_safety(ensemble, binner.levels())
    assert violations == []


def test_xgboost_to_sas_matches_booster_predict(quantized_model):
    bst, binner, Xb = quantized_model
    sas_expr = xgboost_to_sas(bst, bins_per_feature=binner.levels())
    Xtest = Xb.iloc[:100]
    proba = bst.predict(xgb.DMatrix(Xtest, feature_names=["a", "b", "c"]))
    got = np.array([
        eval_sas_row(sas_expr, ["a", "b", "c"], {k: float(v) for k, v in row.items()})
        for _, row in Xtest.iterrows()
    ])
    # tolerance set by float32 threshold rounding, not a bug budget
    np.testing.assert_allclose(got, proba, atol=1e-6)


def test_xgboost_to_sas_refuses_when_feature_not_declared(quantized_model):
    bst, binner, _ = quantized_model
    incomplete = {k: v for k, v in binner.levels().items() if k != "a"}
    with pytest.raises(ValueError, match="not safely separated|no bin count was declared"):
        xgboost_to_sas(bst, bins_per_feature=incomplete)


def test_synthetic_unsafe_threshold_is_caught():
    """Hand-build an Ensemble with a threshold placed exactly where float64
    and float32-truncated comparison of an achievable value disagree, to
    prove the checker's simulation catches a real disagreement.
    """
    # A float64 threshold just above 5.0 (so plain float64 "5.0 < th" is True)
    # that rounds DOWN to exactly float32(5.0) (so the float32-truncated
    # comparison "float32(5.0) < float32(th)" is False) - a real disagreement.
    raw = 5.0 + 1e-7
    assert (5.0 < raw) != (np.float32(5.0) < np.float32(raw)), "fixture didn't land in the danger band"

    tree = Tree(root=Split(feature="a", threshold=raw, le=False, yes=Leaf(1.0), no=Leaf(0.0)))
    ensemble = Ensemble(trees=(tree,), base_score=0.0, link="identity", threshold_precision="float32")
    violations = check_xgb_sas_safety(ensemble, {"a": 16})
    assert violations, "checker should have flagged the deliberately-unsafe threshold"


def test_integer_binner_missing_values_pass_through():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.uniform(0, 100, 200)})
    df.loc[::10, "x"] = np.nan
    binner = IntegerBinner(n_bins=8).fit(df)
    out = binner.transform(df)
    assert out["x"].isna().sum() == df["x"].isna().sum()
    assert (out["x"].dropna() >= 0).all()
