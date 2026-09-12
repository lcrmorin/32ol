"""Test suite for xgb2sql's scikit-learn -> SQL and scikit-learn -> SAS support.

SQL correctness is checked against real DuckDB execution, the same way
test_xgb2sql.py checks xgboost - a real, independent database engine
running the generated SQL and comparing to model.predict()/predict_proba().

SAS correctness is checked with tests/sas_interp.py, a hand-written
interpreter for the exact narrow SAS subset emit_sas.py produces. This is a
self-consistency check, NOT verification against a real SAS instance - no
SAS has been available anywhere this project has been built. Treat it the
same way as the still-open Postgres/MySQL item in TESTING_PLAN.md: the
logic has been checked as carefully as this sandbox allows, but the "does
it actually run in real SAS" question is still open.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sas_interp import SAS_MISSING, eval_sas_row
from xgb2sql import (
    sklearn_to_sas,
    sklearn_to_sas_multiclass,
    sklearn_to_sql,
    sklearn_to_sql_multiclass,
)

FEATURES = ["a", "b", "c"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((300, 3)), columns=FEATURES)
    y_reg = X["a"] * 2 + X["b"] - X["c"]
    y_bin = (X["a"] + X["b"] > 1).astype(int)
    y_multi = pd.cut(X["a"], 3, labels=False)
    return X, y_reg, y_bin, y_multi


@pytest.fixture(scope="module")
def duck(data):
    X, _, _, _ = data
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


# ---- regressors ----

def test_decision_tree_regressor_sql(data, duck):
    X, y, _, _ = data
    m = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
    _check_sql(duck, sklearn_to_sql(m, FEATURES), m.predict(X))


def test_decision_tree_regressor_sas(data):
    X, y, _, _ = data
    m = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
    _check_sas(sklearn_to_sas(m, FEATURES), X, m.predict(X))


def test_random_forest_regressor_sql(data, duck):
    X, y, _, _ = data
    m = RandomForestRegressor(n_estimators=9, max_depth=4, random_state=0).fit(X, y)
    _check_sql(duck, sklearn_to_sql(m, FEATURES), m.predict(X))


def test_gradient_boosting_regressor_sql(data, duck):
    X, y, _, _ = data
    m = GradientBoostingRegressor(n_estimators=12, max_depth=3, random_state=0).fit(X, y)
    _check_sql(duck, sklearn_to_sql(m, FEATURES), m.predict(X))


def test_gradient_boosting_regressor_sas(data):
    X, y, _, _ = data
    m = GradientBoostingRegressor(n_estimators=12, max_depth=3, random_state=0).fit(X, y)
    _check_sas(sklearn_to_sas(m, FEATURES), X, m.predict(X))


def test_hist_gradient_boosting_regressor_sql(data, duck):
    X, y, _, _ = data
    m = HistGradientBoostingRegressor(max_iter=15, max_depth=5, random_state=0).fit(X, y)
    _check_sql(duck, sklearn_to_sql(m, FEATURES), m.predict(X))


def test_hist_gradient_boosting_regressor_sas(data):
    X, y, _, _ = data
    m = HistGradientBoostingRegressor(max_iter=15, max_depth=5, random_state=0).fit(X, y)
    _check_sas(sklearn_to_sas(m, FEATURES), X, m.predict(X))


def test_hist_gradient_boosting_regressor_missing_values(data, duck):
    X, y, _, _ = data
    Xm = X.copy()
    Xm.loc[::5, "a"] = np.nan
    m = HistGradientBoostingRegressor(max_iter=15, max_depth=5, random_state=0).fit(X, y)
    con = duckdb.connect()
    con.register("t", Xm)
    _check_sql(con, sklearn_to_sql(m, FEATURES), m.predict(Xm))
    _check_sas(sklearn_to_sas(m, FEATURES), Xm, m.predict(Xm))


def test_hist_gradient_boosting_classifier_binary_sql(data, duck):
    X, _, yb, _ = data
    m = HistGradientBoostingClassifier(max_iter=15, max_depth=5, random_state=0).fit(X, yb)
    _check_sql(duck, sklearn_to_sql(m, FEATURES), m.predict_proba(X)[:, 1])


def test_hist_gradient_boosting_classifier_binary_sas(data):
    X, _, yb, _ = data
    m = HistGradientBoostingClassifier(max_iter=15, max_depth=5, random_state=0).fit(X, yb)
    _check_sas(sklearn_to_sas(m, FEATURES), X, m.predict_proba(X)[:, 1])


def test_hist_gradient_boosting_classifier_multiclass_raises():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((90, 3)), columns=FEATURES)
    y3 = pd.cut(X["a"], 3, labels=False)
    m = HistGradientBoostingClassifier(max_iter=5, max_depth=2).fit(X, y3)
    with pytest.raises(NotImplementedError):
        sklearn_to_sql(m, FEATURES)


def test_hist_gradient_boosting_categorical_split_raises():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((300, 2)), columns=["a", "b"])
    cat = rng.integers(0, 4, size=300).astype(float)
    X["c"] = cat
    y = (cat >= 2).astype(int)  # purely category-driven, so a categorical split is used
    m = HistGradientBoostingClassifier(max_iter=5, max_depth=3, categorical_features=[2]).fit(X, y)
    assert any(
        pred.nodes["is_categorical"].any() for round_ in m._predictors for pred in round_
    ), "fixture didn't actually produce a categorical split"
    with pytest.raises(NotImplementedError):
        sklearn_to_sql(m, ["a", "b", "c"])


def test_gradient_boosting_regressor_unsupported_loss_raises():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((50, 3)), columns=FEATURES)
    y = X["a"]
    m = GradientBoostingRegressor(n_estimators=3, loss="huber").fit(X, y)
    with pytest.raises(NotImplementedError):
        sklearn_to_sql(m, FEATURES)


# ---- binary classifiers ----

def test_decision_tree_classifier_binary_sql(data, duck):
    X, _, yb, _ = data
    m = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, yb)
    _check_sql(duck, sklearn_to_sql(m, FEATURES), m.predict_proba(X)[:, 1])


def test_random_forest_classifier_binary_sql(data, duck):
    X, _, yb, _ = data
    m = RandomForestClassifier(n_estimators=9, max_depth=4, random_state=0).fit(X, yb)
    _check_sql(duck, sklearn_to_sql(m, FEATURES), m.predict_proba(X)[:, 1])


def test_random_forest_classifier_binary_sas(data):
    X, _, yb, _ = data
    m = RandomForestClassifier(n_estimators=9, max_depth=4, random_state=0).fit(X, yb)
    _check_sas(sklearn_to_sas(m, FEATURES), X, m.predict_proba(X)[:, 1])


def test_gradient_boosting_classifier_binary_sql(data, duck):
    X, _, yb, _ = data
    m = GradientBoostingClassifier(n_estimators=12, max_depth=3, random_state=0).fit(X, yb)
    _check_sql(duck, sklearn_to_sql(m, FEATURES), m.predict_proba(X)[:, 1])


def test_gradient_boosting_classifier_binary_sas(data):
    X, _, yb, _ = data
    m = GradientBoostingClassifier(n_estimators=12, max_depth=3, random_state=0).fit(X, yb)
    _check_sas(sklearn_to_sas(m, FEATURES), X, m.predict_proba(X)[:, 1])


def test_gradient_boosting_classifier_multiclass_raises():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((90, 3)), columns=FEATURES)
    y3 = pd.cut(X["a"], 3, labels=False)
    m = GradientBoostingClassifier(n_estimators=3, max_depth=2).fit(X, y3)
    with pytest.raises(NotImplementedError):
        sklearn_to_sql(m, FEATURES)


# ---- multiclass (DecisionTree / RandomForest only) ----

def test_decision_tree_classifier_multiclass_sql(data, duck):
    X, _, _, y3 = data
    m = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y3)
    sqls = sklearn_to_sql_multiclass(m, FEATURES)
    proba = m.predict_proba(X)
    for c, sql in sqls.items():
        _check_sql(duck, sql, proba[:, c])


def test_random_forest_classifier_multiclass_sql(data, duck):
    X, _, _, y3 = data
    m = RandomForestClassifier(n_estimators=9, max_depth=4, random_state=0).fit(X, y3)
    sqls = sklearn_to_sql_multiclass(m, FEATURES)
    proba = m.predict_proba(X)
    for c, sql in sqls.items():
        _check_sql(duck, sql, proba[:, c])


def test_decision_tree_classifier_multiclass_sas(data):
    X, _, _, y3 = data
    m = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y3)
    sass = sklearn_to_sas_multiclass(m, FEATURES)
    proba = m.predict_proba(X)
    for c, expr in sass.items():
        _check_sas(expr, X, proba[:, c])


# ---- missing values (sklearn's own native NaN support, verified) ----

def test_missing_values_sql(data, duck):
    X, y, _, _ = data
    Xm = X.copy()
    Xm.loc[::5, "a"] = np.nan
    Xm.loc[::7, "b"] = np.nan
    m = DecisionTreeRegressor(max_depth=5, random_state=0).fit(X, y)  # trained without NaNs
    con = duckdb.connect()
    con.register("t", Xm)
    _check_sql(con, sklearn_to_sql(m, FEATURES), m.predict(Xm))


def test_missing_values_sas(data):
    X, y, _, _ = data
    Xm = X.copy()
    Xm.loc[::5, "a"] = np.nan
    Xm.loc[::7, "b"] = np.nan
    m = DecisionTreeRegressor(max_depth=5, random_state=0).fit(X, y)
    _check_sas(sklearn_to_sas(m, FEATURES), Xm, m.predict(Xm))


# ---- guardrails ----

def test_multiclass_classifier_rejected_by_binary_entrypoint(data):
    X, _, _, y3 = data
    m = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y3)
    with pytest.raises(NotImplementedError):
        sklearn_to_sql(m, FEATURES)
    with pytest.raises(NotImplementedError):
        sklearn_to_sas(m, FEATURES)


def test_unsupported_model_type_raises():
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit(np.random.rand(10, 3), np.random.rand(10))
    with pytest.raises(NotImplementedError):
        sklearn_to_sql(m, FEATURES)


def test_xgboost_ensemble_rejected_by_sas_emitter(data):
    import xgboost as xgb
    from xgb2sql.emit_sas import ensemble_to_sas
    from xgb2sql.parse_xgb import xgb_to_ensemble

    X, y, _, _ = data
    bst = xgb.train({"objective": "reg:squarederror"}, xgb.DMatrix(X, label=y), num_boost_round=3)
    ensemble = xgb_to_ensemble(bst)
    with pytest.raises(NotImplementedError):
        ensemble_to_sas(ensemble)


def test_column_name_needing_sas_name_literal():
    from xgb2sql.emit_sas import quote_col
    assert quote_col("normal_name") == "normal_name"
    assert quote_col("weird name") == "'weird name'n"
    assert quote_col("3startswithdigit") == "'3startswithdigit'n"


def test_sql_string_literal_and_sas_string_literal_escape_embedded_quote():
    from xgb2sql.emit_sas import quote_str_literal as sas_quote
    from xgb2sql.emit_sql import quote_str_literal as sql_quote
    assert sas_quote("O'Brien") == "'O''Brien'"
    assert sql_quote("O'Brien") == "'O''Brien'"
