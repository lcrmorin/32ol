"""Test suite for xgb2sql's CatBoost -> SQL and CatBoost -> SAS support.

SQL correctness is checked against real DuckDB execution. SAS is checked
against tests/sas_interp.py (self-consistency only, same caveat as
everywhere else). Unlike LightGBM, CatBoost's thresholds are float32-cast
at prediction time (verified in parse_catboost.py) - same as xgboost - so
catboost_to_sas is only reachable through the IntegerBinner quantize +
check_xgb_sas_safety gate, exactly like xgboost_to_sas.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import duckdb
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor

from sas_interp import SAS_MISSING, eval_sas_row
from xgb2sql import (
    IntegerBinner,
    catboost_to_sas,
    catboost_to_sas_multiclass,
    catboost_to_sql,
    catboost_to_sql_multiclass,
)
from xgb2sql.parse_catboost import catboost_to_ensemble

FEATURES = ["a", "b", "c"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.uniform(0, 100, size=(500, 3)), columns=FEATURES)
    y_reg = X["a"] * 2 + X["b"] - X["c"] + 3000  # constant offset exercises boost_from_average
    y_bin = (X["a"] + X["b"] > 100).astype(int)
    return X, y_reg, y_bin


@pytest.fixture(scope="module")
def duck(data):
    X, _, _ = data
    con = duckdb.connect()
    con.register("t", X)
    return con


def _check_sql(con, sql, expected, tol=1e-6):
    got = con.execute(f"SELECT {sql} AS p FROM t").df()["p"].values
    np.testing.assert_allclose(got, expected, atol=tol)


def _check_sas(expr, X, expected, tol=1e-6):
    got = np.array([
        eval_sas_row(expr, FEATURES, {k: (SAS_MISSING if pd.isna(v) else float(v)) for k, v in row.items()})
        for _, row in X.iterrows()
    ])
    np.testing.assert_allclose(got, expected, atol=tol)


def _train_regressor(X, y, **params):
    p = dict(iterations=15, depth=4, verbose=False, random_seed=0)
    p.update(params)
    m = CatBoostRegressor(**p)
    m.fit(X, y)
    return m


def _train_classifier(X, y, **params):
    p = dict(iterations=15, depth=4, verbose=False, random_seed=0)
    p.update(params)
    m = CatBoostClassifier(**p)
    m.fit(X, y)
    return m


def test_regressor_sql(data, duck):
    X, y, _ = data
    m = _train_regressor(X, y)
    _check_sql(duck, catboost_to_sql(m), m.predict(X))


def test_classifier_binary_sql(data, duck):
    X, _, yb = data
    m = _train_classifier(X, yb)
    _check_sql(duck, catboost_to_sql(m), m.predict_proba(X)[:, 1])


def test_missing_values_sql(data, duck):
    X, y, _ = data
    Xm = X.copy()
    Xm.loc[::5, "a"] = np.nan
    Xm.loc[::7, "b"] = np.nan
    m = _train_regressor(Xm, y)
    con = duckdb.connect()
    con.register("t", Xm)
    _check_sql(con, catboost_to_sql(m), m.predict(Xm))


def test_missing_values_unseen_at_train_still_route_correctly(data, duck):
    """A feature with nan_value_treatment='AsIs' (no NaN seen at train) must
    still route an unexpected predict-time NaN the same way CatBoost itself
    does (verified empirically in parse_catboost.py: like bit=0).
    """
    X, y, _ = data  # no NaNs at train time
    m = _train_regressor(X, y)
    Xm = X.copy()
    Xm.loc[::4, "a"] = np.nan
    con = duckdb.connect()
    con.register("t", Xm)
    _check_sql(con, catboost_to_sql(m), m.predict(Xm))


def test_nan_mode_max_asTrue_routing(data, duck):
    X, y, _ = data
    Xm = X.copy()
    Xm.loc[::5, "a"] = np.nan
    m = _train_regressor(Xm, y, nan_mode="Max")
    con = duckdb.connect()
    con.register("t", Xm)
    _check_sql(con, catboost_to_sql(m), m.predict(Xm))


def test_adversarial_float32_rounding_band_requires_cast(data, duck):
    """The whole reason CatBoost needs the same float32-cast SQL trick as
    xgboost: directly construct rows in the exact danger band where a plain
    float64 comparison would disagree with the float32-truncated one CatBoost
    actually performs, and confirm catboost_to_sql (which casts) still
    matches Booster.predict() there.
    """
    X, y, _ = data
    m = _train_regressor(X, y)
    ensemble = catboost_to_ensemble(m)

    thresholds = []
    def collect(node):
        if hasattr(node, "threshold") and node.threshold is not None:
            thresholds.append((node.feature, node.threshold))
            collect(node.yes)
            collect(node.no)
    for t in ensemble.trees:
        collect(t.root)

    adv_rows = []
    for feat, th in thresholds[:30]:
        t32 = np.float32(th)
        vv = th
        for _ in range(2000):
            vv = np.nextafter(vv, vv + 1000.0)
            if np.float32(vv) == t32 and vv > th:
                row = X.iloc[0].copy()
                row[feat] = vv
                adv_rows.append(row)
                break
            if np.float32(vv) != t32:
                break

    assert len(adv_rows) > 5, "expected several adversarial rows to make this test meaningful"
    advX = pd.DataFrame(adv_rows)
    con = duckdb.connect()
    con.register("t", advX)
    _check_sql(con, catboost_to_sql(m), m.predict(advX))


def test_multiclass_raises():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 100, size=(150, 3)), columns=FEATURES)
    y3 = pd.cut(X["a"], 3, labels=False)
    m = CatBoostClassifier(iterations=5, depth=3, verbose=False, random_seed=0, loss_function="MultiClass")
    m.fit(X, y3)
    with pytest.raises(NotImplementedError):
        catboost_to_sql(m)


def test_multiclass_sql():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 100, size=(400, 3)), columns=FEATURES)
    y3 = pd.cut(X["a"], 3, labels=False)
    m = CatBoostClassifier(iterations=10, depth=4, verbose=False, random_seed=0, loss_function="MultiClass")
    m.fit(X, y3)
    sqls = catboost_to_sql_multiclass(m)
    proba = m.predict_proba(X)
    con = duckdb.connect()
    con.register("t", X)
    for c, sql in sqls.items():
        _check_sql(con, sql, proba[:, c])


def test_multiclass_to_sas_via_quantizer():
    """Same quantizer requirement as binary/regression catboost_to_sas -
    CatBoost's float32 truncation doesn't go away for multiclass.
    """
    rng = np.random.default_rng(17)
    n = 3000
    Xc = pd.DataFrame(rng.uniform(0, 1000, size=(n, 3)), columns=FEATURES)
    y3 = pd.cut(Xc["a"] + Xc["b"] - Xc["c"], 3, labels=False)
    binner = IntegerBinner(n_bins=16).fit(Xc)
    Xb = binner.transform(Xc)
    m = CatBoostClassifier(iterations=15, depth=4, verbose=False, random_seed=0, loss_function="MultiClass")
    m.fit(Xb, y3)

    sass = catboost_to_sas_multiclass(m, bins_per_feature=binner.levels())
    proba = m.predict_proba(Xb)
    Xs = Xb.iloc[:40]
    for c, expr in sass.items():
        got = np.array([
            eval_sas_row(expr, FEATURES, {k: float(v) for k, v in row.items()})
            for _, row in Xs.iterrows()
        ])
        np.testing.assert_allclose(got, proba[:40, c], atol=1e-6)


def test_categorical_feature_ctr_raises():
    """Default one_hot_max_size (2) with a 4-category, target-dependent
    feature makes CatBoost fall back to CTR ('OnlineCtr') splits - not yet
    supported (see parse_catboost.py for exactly what's known and missing).
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 100, size=(300, 2)), columns=["a", "b"])
    X["cat"] = rng.integers(0, 4, size=300).astype(str)
    y = X["a"] + X["b"] + (X["cat"] == "2").astype(float) * 50
    m = CatBoostRegressor(iterations=5, depth=3, verbose=False, random_seed=0, cat_features=["cat"])
    m.fit(X, y)
    with pytest.raises(NotImplementedError, match="OnlineCtr"):
        catboost_to_sql(m, cat_feature_values={"cat": ["0", "1", "2", "3"]})


def test_categorical_feature_onehot_missing_cat_feature_values_raises():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 100, size=(300, 2)), columns=["a", "b"])
    X["cat"] = rng.integers(0, 4, size=300).astype(str)
    y = X["a"] + X["b"] + (X["cat"] == "2").astype(float) * 50
    m = CatBoostRegressor(iterations=8, depth=4, verbose=False, random_seed=0,
                           cat_features=["cat"], one_hot_max_size=10)
    m.fit(X, y)
    with pytest.raises(NotImplementedError, match="cat_feature_values"):
        catboost_to_sql(m)


def test_categorical_feature_onehot_sql(data, duck):
    """A categorical feature that stays under one_hot_max_size uses plain
    OneHotFeature splits - fully supported. cat_feature_values is resolved
    via CatBoost's own hashing (see parse_catboost.py), never reimplemented.
    """
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(rng.uniform(0, 100, size=(n, 2)), columns=["a", "b"])
    X["cat"] = rng.integers(0, 4, size=n).astype(str)
    y = X["a"] + X["b"] + (X["cat"] == "2").astype(float) * 50
    m = CatBoostRegressor(iterations=12, depth=4, verbose=False, random_seed=0,
                           cat_features=["cat"], one_hot_max_size=10)
    m.fit(X, y)
    sql = catboost_to_sql(m, cat_feature_values={"cat": ["0", "1", "2", "3"]})
    con = duckdb.connect()
    con.register("t", X)
    _check_sql(con, sql, m.predict(X))


def test_categorical_feature_onehot_classifier_and_multiclass_sql():
    rng = np.random.default_rng(1)
    n = 400
    X = pd.DataFrame(rng.uniform(0, 100, size=(n, 2)), columns=["a", "b"])
    X["cat"] = rng.integers(0, 4, size=n).astype(str)
    cat_vals = {"cat": ["0", "1", "2", "3"]}
    con = duckdb.connect()
    con.register("t", X)

    yb = ((X["a"] + X["b"] > 100) | (X["cat"] == "2")).astype(int)
    mb = CatBoostClassifier(iterations=10, depth=4, verbose=False, random_seed=0,
                             cat_features=["cat"], one_hot_max_size=10)
    mb.fit(X, yb)
    _check_sql(con, catboost_to_sql(mb, cat_feature_values=cat_vals), mb.predict_proba(X)[:, 1])

    y3 = pd.cut(X["a"] + (X["cat"] == "2").astype(float) * 30, 3, labels=False)
    m3 = CatBoostClassifier(iterations=10, depth=4, verbose=False, random_seed=0,
                             cat_features=["cat"], one_hot_max_size=10, loss_function="MultiClass")
    m3.fit(X, y3)
    sqls = catboost_to_sql_multiclass(m3, cat_feature_values=cat_vals)
    proba = m3.predict_proba(X)
    for c, sql in sqls.items():
        _check_sql(con, sql, proba[:, c])


def test_categorical_feature_declared_but_unused_needs_no_cat_feature_values():
    """A feature declared cat_features=[...] that the fitted model never
    actually split on (irrelevant to the target) needs no cat_feature_values
    at all - the split-type gate looks at what's actually in the trees, not
    at what was merely declared at train time.
    """
    import json
    import tempfile

    rng = np.random.default_rng(1)
    n = 300
    X = pd.DataFrame(rng.uniform(0, 100, size=(n, 2)), columns=["a", "b"])
    X["cat"] = rng.integers(0, 4, size=n).astype(str)
    y = X["a"] + X["b"]  # genuinely independent of "cat"
    m = CatBoostRegressor(iterations=8, depth=3, verbose=False, random_seed=0, cat_features=["cat"])
    m.fit(X, y)
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        m.save_model(f.name, format="json")
        dump = json.load(open(f.name))
    split_types = {s.get("split_type") for t in dump["oblivious_trees"] for s in t["splits"]}
    assert split_types <= {"FloatFeature"}, (
        f"fixture didn't stay cat-independent - model used {split_types}, "
        "adjust the fixture so this test still exercises the no-cat_feature_values path"
    )
    con = duckdb.connect()
    con.register("t", X)
    _check_sql(con, catboost_to_sql(m), m.predict(X))


def test_lossguide_grow_policy_raises():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 100, size=(300, 2)), columns=["a", "b"])
    y = X["a"] + X["b"]
    m = CatBoostRegressor(iterations=5, depth=4, verbose=False, random_seed=0, grow_policy="Lossguide")
    m.fit(X, y)
    with pytest.raises(NotImplementedError):
        catboost_to_sql(m)


# ---- quantized -> SAS path (same shape as xgboost_to_sas) ----

@pytest.fixture(scope="module")
def quantized_model():
    rng = np.random.default_rng(13)
    n = 3000
    Xc = pd.DataFrame(rng.uniform(0, 1000, size=(n, 3)), columns=FEATURES)
    y = (Xc["a"] + 2 * Xc["b"] - Xc["c"] > 500).astype(int)
    binner = IntegerBinner(n_bins=16).fit(Xc)
    Xb = binner.transform(Xc)
    m = CatBoostClassifier(iterations=25, depth=5, verbose=False, random_seed=0)
    m.fit(Xb, y)
    return m, binner, Xb


def test_catboost_to_sas_matches_predict_when_safe(quantized_model):
    m, binner, Xb = quantized_model
    sas_expr = catboost_to_sas(m, bins_per_feature=binner.levels())
    Xtest = Xb.iloc[:60]
    proba = m.predict_proba(Xtest)[:, 1]
    got = np.array([
        eval_sas_row(sas_expr, FEATURES, {k: float(v) for k, v in row.items()})
        for _, row in Xtest.iterrows()
    ])
    np.testing.assert_allclose(got, proba, atol=1e-6)


def test_catboost_to_sas_refuses_when_feature_not_declared(quantized_model):
    m, binner, _ = quantized_model
    incomplete = {k: v for k, v in binner.levels().items() if k != "a"}
    with pytest.raises(ValueError, match="not safely separated|no bin count was declared"):
        catboost_to_sas(m, bins_per_feature=incomplete)


def test_catboost_to_sql_plain_xgboost_style_model_still_needs_no_quantizer():
    """catboost_to_sql (unlike catboost_to_sas) works on an ordinary,
    non-quantized model - the float32 caution only applies to the SAS path.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 100, size=(300, 2)), columns=["a", "b"])
    y = X["a"] + X["b"]
    m = CatBoostRegressor(iterations=10, depth=4, verbose=False, random_seed=0)
    m.fit(X, y)
    con = duckdb.connect()
    con.register("t", X)
    _check_sql(con, catboost_to_sql(m), m.predict(X))
