"""Test suite for xgb2sql.

Two kinds of tests here:
1. Prediction-equivalence tests (the original suite): train a real XGBoost
   model, convert it to SQL, run the SQL in DuckDB, and assert the numbers
   match Booster.predict() to a tight tolerance.
2. Guard-rail tests (new, added when hardening this for real-life use): each
   one locks in a specific finding from TESTING_PLAN.md - either "this
   dangerous case is now rejected with a clear error" or "this case was
   worried about but is actually fine, verified empirically".

Run with: pytest tests/test_xgb2sql.py -v
"""

import duckdb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from xgb2sql import (
    prepare_df_for_duckdb,
    xgboost_to_sql,
    xgboost_to_sql_multiclass,
)


# ─────────────────────────── shared helpers ────────────────────────────────

def assert_close(xgb_p, sql_p, tol=1e-6, rtol=None):
    diff = np.abs(xgb_p - sql_p)
    if rtol is not None:
        rel = diff / (np.abs(xgb_p) + 1)
        assert rel.max() < rtol, f"Max relative diff {rel.max():.2e} exceeds {rtol:.2e}"
    else:
        assert diff.max() < tol, f"Max diff {diff.max():.2e} exceeds {tol:.2e}"


def train_and_compare(df, feats, target, params, num_rounds,
                       sigmoid=False, link=None, cat_feats=None,
                       tol=1e-6, rtol=None):
    """Train a model, convert to SQL, run it in DuckDB, compare to predict()."""
    df = df.copy()
    df["target"] = target
    enable_cat = bool(cat_feats)
    dtrain = xgb.DMatrix(df[feats], label=df["target"], enable_categorical=enable_cat)
    model = xgb.train(params, dtrain, num_rounds, verbose_eval=False)
    xgb_preds = model.predict(dtrain)

    sql_expr = xgboost_to_sql(
        model, sigmoid=sigmoid, link=link,
        cat_feature_names=cat_feats, training_df=df[feats] if cat_feats else None,
    )
    con = duckdb.connect()
    con.register("data", prepare_df_for_duckdb(df[feats], cat_feature_names=cat_feats))
    sql_preds = con.execute(f"SELECT {sql_expr} AS p FROM data").df()["p"].values
    assert_close(xgb_preds, sql_preds, tol=tol, rtol=rtol)
    return model, xgb_preds, sql_expr


# ───────────────────── 1. prediction-equivalence suite ─────────────────────

def test_numeric_binary():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "x1": rng.uniform(800, 4000, n), "x2": rng.uniform(1, 4.5, n),
        "x3": rng.uniform(0.5, 50, n),
    })
    target = (df["x1"] / 1000 * 0.3 + df["x2"] * 0.15 + df["x3"] / -10 * 0.15
              + rng.normal(0, 0.3, n))
    train_and_compare(df, list(df.columns), (target > np.median(target)).astype(int),
                       {"objective": "binary:logistic", "max_depth": 5, "eta": 0.1, "seed": 42},
                       20, sigmoid=True)


def test_numeric_regression():
    rng = np.random.default_rng(123)
    n = 150
    df = pd.DataFrame({"x1": rng.uniform(-10, 10, n), "x2": rng.uniform(0, 100, n),
                        "x3": rng.normal(0, 5, n)})
    target = 3 * df["x1"] + 0.5 * df["x2"] - 2 * df["x3"] + rng.normal(0, 5, n)
    train_and_compare(df, list(df.columns), target,
                       {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.3, "seed": 99},
                       30, tol=1e-4)


def test_categorical_binary():
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({
        "color": pd.Categorical(rng.choice(["red", "green", "blue", "yellow"], n)),
        "city": pd.Categorical(rng.choice(["NYC", "LA", "CHI", "HOU", "PHX"], n)),
        "area": rng.uniform(50, 300, n), "price": rng.uniform(100, 1000, n),
    })
    target = np.zeros(n)
    target[df["color"] == "red"] += 1
    target[df["color"] == "blue"] += 0.5
    target[df["city"].isin(["NYC", "LA"])] += 1
    target += df["area"] / 300 + rng.normal(0, 0.5, n)
    train_and_compare(
        df, list(df.columns), (target > np.median(target)).astype(int),
        {"objective": "binary:logistic", "max_depth": 4, "eta": 0.1,
         "seed": 42, "tree_method": "hist"},
        20, sigmoid=True, cat_feats=["color", "city"])


def test_categorical_regression():
    rng = np.random.default_rng(55)
    n = 200
    df = pd.DataFrame({
        "material": pd.Categorical(rng.choice(["wood", "steel", "concrete", "glass"], n)),
        "zone_id": pd.Categorical(rng.choice(["A", "B", "C"], n)),
        "sqft": rng.uniform(500, 5000, n), "age": rng.uniform(0, 50, n),
    })
    target = df["sqft"] * 100 + df["age"] * -500 + rng.normal(0, 10000, n)
    target[df["material"] == "steel"] += 50000
    target[df["zone_id"] == "A"] += 30000
    train_and_compare(
        df, list(df.columns), target,
        {"objective": "reg:squarederror", "max_depth": 5, "eta": 0.2,
         "seed": 42, "tree_method": "hist"},
        30, cat_feats=["material", "zone_id"], rtol=1e-5)


def test_missing_values():
    rng = np.random.default_rng(88)
    n = 200
    df = pd.DataFrame({"x1": rng.uniform(0, 100, n), "x2": rng.uniform(-50, 50, n),
                        "x3": rng.uniform(0, 10, n)})
    target = (df["x1"] + df["x2"] > 50).astype(int)
    for col in df.columns:
        df.loc[rng.random(n) < 0.15, col] = np.nan
    train_and_compare(df, list(df.columns), target,
                       {"objective": "binary:logistic", "max_depth": 3, "eta": 0.1, "seed": 42},
                       15, sigmoid=True)


def test_categorical_missing():
    rng = np.random.default_rng(333)
    n = 250
    df = pd.DataFrame({
        "color": pd.Categorical(
            [c if c else np.nan for c in rng.choice(["red", "green", "blue", None], n)],
            categories=["red", "green", "blue"]),
        "val": rng.uniform(0, 100, n),
    })
    target = df["val"] / 50 + rng.normal(0, 0.5, n)
    target[df["color"] == "red"] += 1
    train_and_compare(
        df, ["color", "val"], (target > np.median(target)).astype(int),
        {"objective": "binary:logistic", "max_depth": 4, "eta": 0.15,
         "seed": 42, "tree_method": "hist"},
        15, sigmoid=True, cat_feats=["color"])


def test_many_categories_partition_splits():
    rng = np.random.default_rng(404)
    n = 400
    cats = [f"cat_{i:02d}" for i in range(20)]
    df = pd.DataFrame({
        "grp": pd.Categorical(rng.choice(cats, n), categories=cats),
        "x1": rng.uniform(0, 100, n),
    })
    target = df["x1"] / 100 + rng.normal(0, 0.3, n)
    for i, c in enumerate(cats):
        target[df["grp"] == c] += np.sin(i) * 0.5
    train_and_compare(
        df, ["grp", "x1"], (target > np.median(target)).astype(int),
        {"objective": "binary:logistic", "max_depth": 6, "eta": 0.15,
         "seed": 42, "tree_method": "hist", "max_cat_to_onehot": 5},
        25, sigmoid=True, cat_feats=["grp"])


def test_multiclass_softmax():
    rng = np.random.default_rng(777)
    n = 200
    df = pd.DataFrame({f"x{i}": rng.uniform(0, 10, n) for i in range(3)})
    feats = list(df.columns)
    df["target"] = np.argmax(df[feats].values, axis=1)
    num_class = 3

    dtrain = xgb.DMatrix(df[feats], label=df["target"])
    model = xgb.train(
        {"objective": "multi:softprob", "num_class": num_class,
         "max_depth": 4, "eta": 0.2, "seed": 42},
        dtrain, 20, verbose_eval=False)
    xgb_probs = model.predict(dtrain)

    exprs = xgboost_to_sql_multiclass(model, num_class=num_class)
    select = ", ".join(f"{e} AS p{c}" for c, e in exprs.items())
    con = duckdb.connect()
    con.register("data", prepare_df_for_duckdb(df[feats]))
    result = con.execute(f"SELECT {select} FROM data").df()
    sql_probs = np.column_stack([result[f"p{c}"].values for c in range(num_class)])
    assert_close(xgb_probs, sql_probs, tol=1e-5)


# ───────────────────── 2. new: real-life-use guard rails ───────────────────

def test_multiclass_with_categorical_features():
    """Gap found during hardening: multiclass + categorical was never tested
    combined (only separately). Real credit/marketing models are often both."""
    rng = np.random.default_rng(909)
    n = 300
    df = pd.DataFrame({
        "region": pd.Categorical(rng.choice(["north", "south", "east", "west"], n)),
        "score": rng.uniform(0, 100, n),
    })
    target = np.zeros(n, dtype=int)
    target[df["region"] == "north"] = 1
    target[(df["region"] == "south") & (df["score"] > 50)] = 2
    num_class = 3

    dtrain = xgb.DMatrix(df, label=target, enable_categorical=True)
    model = xgb.train(
        {"objective": "multi:softprob", "num_class": num_class,
         "max_depth": 4, "eta": 0.2, "seed": 42, "tree_method": "hist"},
        dtrain, 25, verbose_eval=False)
    xgb_probs = model.predict(dtrain)

    exprs = xgboost_to_sql_multiclass(model, num_class=num_class,
                                       cat_feature_names=["region"], training_df=df)
    select = ", ".join(f"{e} AS p{c}" for c, e in exprs.items())
    con = duckdb.connect()
    con.register("data", prepare_df_for_duckdb(df, cat_feature_names=["region"]))
    result = con.execute(f"SELECT {select} FROM data").df()
    sql_probs = np.column_stack([result[f"p{c}"].values for c in range(num_class)])
    assert_close(xgb_probs, sql_probs, tol=1e-5)


def test_category_label_with_embedded_quote_does_not_corrupt_sql():
    """Real bug found in review: category labels were interpolated into SQL
    string literals with no escaping (f"'{v}'"). Any label containing a
    single quote - "O'Brien", a free-text tag, "5' 2\"" - produced a
    syntactically broken (or, worse, silently different) WHERE clause."""
    rng = np.random.default_rng(11)
    n = 200
    labels = ["O'Brien", "Smith", "D'Angelo", "Lee"]
    df = pd.DataFrame({
        "name": pd.Categorical(rng.choice(labels, n)),
        "amount": rng.uniform(0, 100, n),
    })
    target = (df["name"] == "O'Brien").astype(int)
    train_and_compare(
        df, ["name", "amount"], target,
        {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3,
         "seed": 1, "tree_method": "hist"},
        10, sigmoid=True, cat_feats=["name"],
    )


def test_dart_booster_is_rejected_not_silently_wrong():
    """Empirically verified: DART's dropout-adjusted weights are not
    recoverable from get_dump(), so plain-sum SQL gives a materially
    different number than Booster.predict() (checked: ~0.007 vs ~0.25 on a
    real example). Must raise loudly, not return a wrong prediction."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x1": rng.uniform(0, 10, n), "x2": rng.uniform(0, 10, n)})
    y = (df["x1"] + df["x2"] > 10).astype(int)
    dtrain = xgb.DMatrix(df, label=y)
    model = xgb.train(
        {"objective": "binary:logistic", "booster": "dart", "max_depth": 3,
         "eta": 0.3, "rate_drop": 0.3, "seed": 1},
        dtrain, 20, verbose_eval=False,
    )
    with pytest.raises(NotImplementedError, match="dart"):
        xgboost_to_sql(model, sigmoid=True)


def test_gblinear_booster_is_rejected():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"x1": rng.uniform(0, 10, n)})
    y = (df["x1"] > 5).astype(int)
    dtrain = xgb.DMatrix(df, label=y)
    model = xgb.train({"objective": "binary:logistic", "booster": "gblinear"},
                       dtrain, 10, verbose_eval=False)
    with pytest.raises(NotImplementedError, match="gblinear"):
        xgboost_to_sql(model, sigmoid=True)


def test_random_forest_mode_num_parallel_tree_needs_no_special_casing():
    """Worried this needed dividing by num_parallel_tree; verified empirically
    it does not - XGBoost's own leaf values already account for it. This test
    locks that finding in so a future change can't silently break it."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x1": rng.uniform(0, 10, n), "x2": rng.uniform(0, 10, n)})
    y = (df["x1"] + df["x2"] > 10).astype(int)
    train_and_compare(
        df, ["x1", "x2"], y,
        {"objective": "binary:logistic", "booster": "gbtree", "max_depth": 3,
         "eta": 0.3, "num_parallel_tree": 4, "seed": 1},
        5, sigmoid=True,
    )


def test_count_poisson_exp_link_auto_detected():
    """New capability: previously only 'sigmoid or raw' existed, so a
    poisson/gamma model called with the default args silently returned a raw
    margin as if it were a prediction. Verified empirically that base_score
    for count:poisson is on the *output* (mean) scale, needing log() before
    it can be summed with the margin-scale tree leaves."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x1": rng.uniform(0, 10, n)})
    y = rng.poisson(3, n)
    train_and_compare(
        df, ["x1"], y,
        {"objective": "count:poisson", "max_depth": 3, "eta": 0.3, "seed": 1},
        15, rtol=1e-4,
    )


def test_unrecognized_objective_raises_instead_of_guessing():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"x1": rng.uniform(0, 10, n), "x2": rng.uniform(0, 10, n)})
    y = rng.integers(0, 3, n)
    dtrain = xgb.DMatrix(df, label=y)
    model = xgb.train({"objective": "rank:pairwise", "max_depth": 3, "eta": 0.3},
                       dtrain, 5, verbose_eval=False)
    with pytest.raises(NotImplementedError, match="rank:pairwise"):
        xgboost_to_sql(model)


def test_explicit_link_override_bypasses_auto_detection():
    """An explicit link= should work even for an objective not in the
    allow-list, since the caller has presumably verified it themselves."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"x1": rng.uniform(0, 10, n)})
    y = (df["x1"] > 5).astype(int)
    dtrain = xgb.DMatrix(df, label=y)
    model = xgb.train({"objective": "binary:logitraw", "max_depth": 3, "eta": 0.3},
                       dtrain, 5, verbose_eval=False)
    sql_expr = xgboost_to_sql(model, link="identity")
    assert "EXP" not in sql_expr


def test_multiclass_num_class_mismatch_raises():
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({f"x{i}": rng.uniform(0, 10, n) for i in range(3)})
    y = rng.integers(0, 3, n)
    dtrain = xgb.DMatrix(df, label=y)
    model = xgb.train({"objective": "multi:softprob", "num_class": 3,
                        "max_depth": 3, "eta": 0.2}, dtrain, 10, verbose_eval=False)
    with pytest.raises(ValueError, match="num_class"):
        xgboost_to_sql_multiclass(model, num_class=4)


def test_reserved_word_and_quote_needing_column_name_end_to_end():
    """Column named after a SQL reserved word, exercised end-to-end against
    DuckDB (not just unit-tested against _quote_col in isolation)."""
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({"select": rng.uniform(0, 10, n), "order": rng.uniform(0, 10, n)})
    y = (df["select"] + df["order"] > 10).astype(int)
    train_and_compare(
        df, ["select", "order"], y,
        {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3, "seed": 1},
        10, sigmoid=True,
    )


def test_sql_is_deterministic_across_repeated_conversion():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"x1": rng.uniform(0, 10, n)})
    y = (df["x1"] > 5).astype(int)
    dtrain = xgb.DMatrix(df, label=y)
    model = xgb.train({"objective": "binary:logistic", "max_depth": 3, "eta": 0.3},
                       dtrain, 5, verbose_eval=False)
    assert xgboost_to_sql(model, sigmoid=True) == xgboost_to_sql(model, sigmoid=True)


def test_sqlite_cast_to_float_does_not_truncate_to_float32():
    """Real, confirmed limitation (not a bug in this module): the docstring's
    claim "use FLOAT for MySQL, REAL for PostgreSQL" was never checked against
    a second live engine. Running the exact same SQL this module generates
    against SQLite shows CAST(x AS FLOAT) is a no-op there - SQLite has no
    true 4-byte float type, so the cast stays full double precision. Since
    the threshold literal is deliberately the float32-rounded value (to match
    XGBoost's internal float32 comparison), a handful of rows whose true value
    sits between the float32 and float64 versions of the threshold get routed
    down the wrong branch: this run showed up to 0.43 absolute error on ~7% of
    rows, not a rounding-level difference. Any target engine needs to be
    checked for genuine float32 CAST support before trusting this converter's
    output there - see TESTING_PLAN.md."""
    import sqlite3

    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"x1": rng.uniform(0, 10, n), "x2": rng.uniform(0, 10, n)})
    y = (df["x1"] + df["x2"] > 10).astype(int)
    dtrain = xgb.DMatrix(df, label=y)
    model = xgb.train({"objective": "binary:logistic", "max_depth": 3, "eta": 0.3, "seed": 1},
                       dtrain, 10, verbose_eval=False)
    xgb_preds = model.predict(dtrain)

    sql_expr = xgboost_to_sql(model, sigmoid=True)
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE data (x1 REAL, x2 REAL)")
    con.executemany("INSERT INTO data VALUES (?, ?)", df[["x1", "x2"]].values.tolist())
    sql_preds = np.array([r[0] for r in con.execute(f"SELECT {sql_expr} FROM data")])

    diff = np.abs(xgb_preds - sql_preds)
    # Documents the failure mode rather than hiding it: most rows match
    # tightly, but SQLite's lack of float32 CAST support does cause real,
    # non-trivial routing mismatches on some rows. This is expected to fail
    # here and is the reason SQLite is not a supported target engine.
    assert diff.max() > 1e-3, (
        "expected to reproduce SQLite's known float32-CAST limitation, but "
        "predictions matched closely - either SQLite's CAST behavior changed, "
        "or this test's assumption needs re-checking"
    )
