"""Test suite for xgb2sql's LightGBM -> SQL and LightGBM -> SAS support.

SQL correctness is checked against real DuckDB execution. SAS correctness is
checked with tests/sas_interp.py (self-consistency only - no real SAS has
been available, same caveat as everywhere else in this project). Unlike
xgboost, LightGBM needs no quantizer/safety-check gate for SAS - it's
float64-native (verified in parse_lgbm.py) - so lgbm_to_sas is called
directly here, the same way sklearn_to_sas is.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from sas_interp import SAS_MISSING, eval_sas_row
from xgb2sql import lgbm_to_sas, lgbm_to_sas_multiclass, lgbm_to_sql, lgbm_to_sql_multiclass
from xgb2sql.parse_lgbm import lgbm_to_ensemble

FEATURES = ["a", "b", "c"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.uniform(0, 100, size=(400, 3)), columns=FEATURES)
    y_reg = X["a"] * 2 + X["b"] - X["c"] + 5000  # constant offset exercises boost_from_average
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


def _check_sas_mixed(expr, X, expected, tol=1e-6):
    """Like _check_sas, but for a DataFrame with a categorical (string)
    column - only numeric-dtype columns get the float() conversion, string
    values pass through as-is (matching what a real categorical column's
    membership check compares against) and missing values become SAS_MISSING
    regardless of dtype.
    """
    got = np.array([
        eval_sas_row(expr, list(X.columns), {
            k: (SAS_MISSING if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.floating)) else v))
            for k, v in row.items()
        })
        for _, row in X.iterrows()
    ])
    np.testing.assert_allclose(got, expected, atol=tol)


def _train_regressor(X, y, **params):
    dtrain = lgb.Dataset(X, label=y, feature_name=FEATURES)
    p = {"objective": "regression", "verbose": -1, "max_depth": 4, "num_leaves": 12, "min_data_in_leaf": 5}
    p.update(params)
    return lgb.train(p, dtrain, num_boost_round=12)


def _train_classifier(X, y, **params):
    dtrain = lgb.Dataset(X, label=y, feature_name=FEATURES)
    p = {"objective": "binary", "verbose": -1, "max_depth": 4, "num_leaves": 12, "min_data_in_leaf": 5}
    p.update(params)
    return lgb.train(p, dtrain, num_boost_round=12)


# SAS correctness is checked with tests/sas_interp.py, a pure-Python
# interpreter that deliberately shares SAS's documented non-short-circuiting
# IFN evaluation (every branch runs on every call) - this is slow enough
# that, matching the precedent already set for the sklearn SAS tests on
# larger ensembles, SAS checks here run on a row subset rather than the
# full fixture.
SAS_CHECK_ROWS = 60


def test_regressor_sql(data, duck):
    X, y, _ = data
    m = _train_regressor(X, y)
    _check_sql(duck, lgbm_to_sql(m), m.predict(X))


def test_regressor_sas(data):
    X, y, _ = data
    m = _train_regressor(X, y)
    Xs = X.iloc[:SAS_CHECK_ROWS]
    _check_sas(lgbm_to_sas(m), Xs, m.predict(Xs))


def test_classifier_binary_sql(data, duck):
    X, _, yb = data
    m = _train_classifier(X, yb)
    _check_sql(duck, lgbm_to_sql(m), m.predict(X))


def test_classifier_binary_sas(data):
    X, _, yb = data
    m = _train_classifier(X, yb)
    Xs = X.iloc[:SAS_CHECK_ROWS]
    _check_sas(lgbm_to_sas(m), Xs, m.predict(Xs))


def test_missing_values_sql_and_sas(data, duck):
    X, y, _ = data
    Xm = X.copy()
    Xm.loc[::5, "a"] = np.nan
    Xm.loc[::7, "b"] = np.nan
    m = _train_regressor(Xm, y)
    con = duckdb.connect()
    con.register("t", Xm)
    _check_sql(con, lgbm_to_sql(m), m.predict(Xm))
    Xms = Xm.iloc[:SAS_CHECK_ROWS]
    _check_sas(lgbm_to_sas(m), Xms, m.predict(Xms))


def test_missing_values_unseen_at_train_still_route_correctly(data, duck):
    """default_left must govern NaN routing at predict time even for splits
    whose training data never included a NaN (missing_type='None' in the
    dump) - verified in the exploratory testing behind this module; this
    locks that behavior in as a regression test.
    """
    X, y, _ = data  # no NaNs at train time
    m = _train_regressor(X, y)
    Xm = X.copy()
    Xm.loc[::4, "a"] = np.nan
    con = duckdb.connect()
    con.register("t", Xm)
    _check_sql(con, lgbm_to_sql(m), m.predict(Xm))
    Xms = Xm.iloc[:SAS_CHECK_ROWS]
    _check_sas(lgbm_to_sas(m), Xms, m.predict(Xms))


def test_adversarial_float32_rounding_band_matches_float64(data):
    """The whole reason LightGBM doesn't need xgboost's quantizer: thresholds
    compare in native float64. Directly construct rows placed in the exact
    danger band where a float32-cast comparison would disagree with float64,
    and confirm the emitted (float64, no-cast) SQL/SAS still matches
    Booster.predict() there - i.e. lgbm_to_sql/_to_sas never needed the cast.
    """
    X, y, _ = data
    m = _train_regressor(X, y)
    ensemble = lgbm_to_ensemble(m)

    thresholds = []
    def collect(node):
        if hasattr(node, "threshold") and node.threshold is not None:
            thresholds.append((node.feature, node.threshold))
            collect(node.yes)
            collect(node.no)
    for t in ensemble.trees:
        collect(t.root)

    adv_rows = []
    for feat, th in thresholds[:25]:
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

    assert len(adv_rows) > 5, "expected to find several adversarial rows to make this test meaningful"
    advX = pd.DataFrame(adv_rows)
    con = duckdb.connect()
    con.register("t", advX)
    _check_sql(con, lgbm_to_sql(m), m.predict(advX))
    _check_sas(lgbm_to_sas(m), advX, m.predict(advX))


def test_multiclass_raises():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 100, size=(150, 3)), columns=FEATURES)
    y3 = pd.cut(X["a"], 3, labels=False)
    dtrain = lgb.Dataset(X, label=y3, feature_name=FEATURES)
    m = lgb.train(
        {"objective": "multiclass", "num_class": 3, "verbose": -1, "max_depth": 3},
        dtrain, num_boost_round=5,
    )
    with pytest.raises(NotImplementedError):
        lgbm_to_sql(m)


def test_multiclass_sql_and_sas():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 100, size=(400, 3)), columns=FEATURES)
    y3 = pd.cut(X["a"], 3, labels=False)
    dtrain = lgb.Dataset(X, label=y3, feature_name=FEATURES)
    m = lgb.train(
        {"objective": "multiclass", "num_class": 3, "verbose": -1, "max_depth": 4, "num_leaves": 12},
        dtrain, num_boost_round=10,
    )
    proba = m.predict(X)

    sqls = lgbm_to_sql_multiclass(m)
    con = duckdb.connect()
    con.register("t", X)
    for c, sql in sqls.items():
        _check_sql(con, sql, proba[:, c])

    sass = lgbm_to_sas_multiclass(m)
    Xs = X.iloc[:SAS_CHECK_ROWS]
    for c, expr in sass.items():
        _check_sas(expr, Xs, proba[:SAS_CHECK_ROWS, c])


def _has_cat_split(node):
    if "decision_type" not in node:
        return False
    if node["decision_type"] == "==":
        return True
    return _has_cat_split(node["left_child"]) or _has_cat_split(node["right_child"])


def test_categorical_plain_int_sql_and_sas(duck):
    """A plain int/numeric-dtype column passed via categorical_feature=[...]
    (not pandas Categorical dtype) needs no categorical_feature_values at
    all - its codes ARE the raw values.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 100, size=(400, 2)), columns=["a", "b"])
    X["c"] = rng.integers(0, 5, size=400).astype(int)
    y = X["a"] * 0.1 + (X["c"] >= 3).astype(int) * 40
    dtrain = lgb.Dataset(X, label=y, categorical_feature=["c"], feature_name=["a", "b", "c"])
    m = lgb.train({"objective": "regression", "verbose": -1, "max_depth": 4, "num_leaves": 16},
                   dtrain, num_boost_round=12)
    dump = m.dump_model()
    assert any(_has_cat_split(t["tree_structure"]) for t in dump["tree_info"]), \
        "fixture didn't actually produce a categorical split - test is vacuous"
    con = duckdb.connect()
    con.register("t", X)
    _check_sql(con, lgbm_to_sql(m), m.predict(X))
    Xs = X.iloc[:SAS_CHECK_ROWS]
    _check_sas(lgbm_to_sas(m), Xs, m.predict(Xs))


def test_categorical_pandas_dtype_missing_values_raises(duck):
    rng = np.random.default_rng(1)
    n = 400
    a = rng.uniform(0, 100, size=n)
    labels = np.array(["red", "green", "blue", "yellow", "purple"])
    cat = labels[rng.integers(0, 5, size=n)]
    y = a * 0.05 + (cat == "blue").astype(float) * 100
    X = pd.DataFrame({"a": a, "cat": pd.Categorical(cat)})
    m = lgb.LGBMRegressor(n_estimators=10, max_depth=4, verbose=-1).fit(X, y)
    with pytest.raises(ValueError, match="categorical_feature_values"):
        lgbm_to_sql(m)


@pytest.fixture(scope="module")
def pandas_categorical_data():
    rng = np.random.default_rng(2)
    n = 600
    a = rng.uniform(0, 100, size=n)
    labels = np.array(["red", "green", "blue", "yellow", "purple"])
    cat = labels[rng.integers(0, 5, size=n)].astype(object)
    nan_mask = rng.random(n) < 0.15
    cat[nan_mask] = None
    y = a * 0.05 + (pd.Series(cat) == "blue").astype(float) * 100 + (pd.Series(cat) == "purple").astype(float) * 40
    X = pd.DataFrame({"a": a, "cat": pd.Categorical(cat, categories=labels)})
    return X, y, labels


def test_categorical_pandas_dtype_sql_and_sas(pandas_categorical_data):
    """The common LightGBM categorical workflow: pandas Categorical dtype.
    Codes are decoded to raw labels via categorical_feature_values, matched
    against the model's own pandas_categorical by value-set identity (see
    parse_lgbm.py). Includes rows with real missing values (NaN routes to
    no-match unconditionally - verified against LightGBM's own C++ source).
    """
    X, y, labels = pandas_categorical_data
    m = lgb.LGBMRegressor(n_estimators=15, max_depth=4, verbose=-1).fit(X, y)
    sql = lgbm_to_sql(m, categorical_feature_values={"cat": list(labels)})
    con = duckdb.connect()
    con.register("t", X)
    _check_sql(con, sql, m.predict(X))
    Xs = X.iloc[:SAS_CHECK_ROWS]
    sas = lgbm_to_sas(m, categorical_feature_values={"cat": list(labels)})
    _check_sas_mixed(sas, Xs, m.predict(Xs))


def test_categorical_pandas_dtype_category_order_does_not_matter(pandas_categorical_data):
    """categorical_feature_values is matched by value SET, not position -
    passing the categories in a different order than pandas assigned must
    still decode correctly.
    """
    X, y, labels = pandas_categorical_data
    m = lgb.LGBMRegressor(n_estimators=8, max_depth=3, verbose=-1).fit(X, y)
    shuffled = list(labels[::-1])
    sql = lgbm_to_sql(m, categorical_feature_values={"cat": shuffled})
    con = duckdb.connect()
    con.register("t", X)
    _check_sql(con, sql, m.predict(X))


def test_categorical_pandas_dtype_unseen_category_at_predict_time(pandas_categorical_data):
    """A category value never seen at train time routes exactly the way
    LightGBM's own predictor does (no-match, same as NaN - see
    parse_lgbm.py) - checked against the model's own .predict(), not an
    assumption.
    """
    X, y, labels = pandas_categorical_data
    m = lgb.LGBMRegressor(n_estimators=10, max_depth=4, verbose=-1).fit(X, y)
    sql = lgbm_to_sql(m, categorical_feature_values={"cat": list(labels)})
    Xu = X.copy()
    Xu["cat"] = Xu["cat"].astype(object)
    Xu.loc[::9, "cat"] = "NEVER_SEEN_AT_TRAIN"
    Xu["cat"] = pd.Categorical(Xu["cat"], categories=list(labels) + ["NEVER_SEEN_AT_TRAIN"])
    con = duckdb.connect()
    con.register("t", Xu)
    _check_sql(con, sql, m.predict(Xu))


def test_categorical_multiclass_sql():
    rng = np.random.default_rng(3)
    n = 500
    a = rng.uniform(0, 100, size=n)
    labels = np.array(["red", "green", "blue"])
    cat = labels[rng.integers(0, 3, size=n)]
    X = pd.DataFrame({"a": a, "cat": pd.Categorical(cat)})
    y3 = pd.cut(a + (pd.Series(cat) == "blue").astype(float) * 50, 3, labels=False)
    m = lgb.LGBMClassifier(n_estimators=10, max_depth=3, verbose=-1,
                            objective="multiclass", num_class=3).fit(X, y3)
    sqls = lgbm_to_sql_multiclass(m, categorical_feature_values={"cat": list(labels)})
    proba = m.predict_proba(X)
    con = duckdb.connect()
    con.register("t", X)
    for c, sql in sqls.items():
        _check_sql(con, sql, proba[:, c])


def test_dart_boosting_sql(data, duck):
    """dart is supported - LightGBM bakes any dropout-related rescaling into
    the stored leaf values, unlike xgboost's DART (see parse_lgbm.py).
    """
    X, y, _ = data
    m = _train_regressor(X, y, boosting="dart", drop_rate=0.3)
    _check_sql(duck, lgbm_to_sql(m), m.predict(X))


def test_rf_boosting_sql(data, duck):
    X, y, _ = data
    m = _train_regressor(
        X, y, boosting="rf", bagging_freq=1, bagging_fraction=0.7, feature_fraction=0.7
    )
    _check_sql(duck, lgbm_to_sql(m), m.predict(X))


def test_goss_boosting_sql(data, duck):
    X, y, _ = data
    m = _train_regressor(X, y, boosting="goss")
    _check_sql(duck, lgbm_to_sql(m), m.predict(X))


def test_unsupported_boosting_raises():
    """Guardrail: an unrecognized boosting_type string still raises rather
    than silently being treated as gbdt.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 100, size=(200, 2)), columns=["a", "b"])
    y = X["a"] + X["b"]
    dtrain = lgb.Dataset(X, label=y, feature_name=["a", "b"])
    m = lgb.train({"objective": "regression", "verbose": -1}, dtrain, num_boost_round=5)
    m.params["boosting"] = "made_up_mode"  # simulate an unrecognized value
    with pytest.raises(NotImplementedError):
        lgbm_to_sql(m)


def test_sklearn_wrapper_accepted(data, duck):
    """lgbm_to_sql/_to_sas also accept a fitted LGBMRegressor/LGBMClassifier
    directly (via its .booster_), not just a raw lgb.Booster.
    """
    from lightgbm import LGBMRegressor
    X, y, _ = data
    m = LGBMRegressor(
        n_estimators=15, max_depth=5, num_leaves=31, min_child_samples=5, verbose=-1
    ).fit(X, y)
    _check_sql(duck, lgbm_to_sql(m, feature_names=FEATURES), m.predict(X))
