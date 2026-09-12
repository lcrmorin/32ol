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
from xgb2sql import IntegerBinner, catboost_to_sas, catboost_to_sql
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


def test_categorical_feature_raises():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 100, size=(300, 2)), columns=["a", "b"])
    X["cat"] = rng.integers(0, 4, size=300).astype(str)
    y = X["a"] + X["b"] + (X["cat"] == "2").astype(float) * 50
    m = CatBoostRegressor(iterations=5, depth=3, verbose=False, random_seed=0, cat_features=["cat"])
    m.fit(X, y)
    with pytest.raises(NotImplementedError):
        catboost_to_sql(m)


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
