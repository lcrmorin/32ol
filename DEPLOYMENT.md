# Deploying xgb2sql output in production

Everything in this repo verifies that the SQL/SAS text this package generates encodes the *same logic* as the Python model it came from. That is necessary and it is checked hard (DuckDB, a hand-written SAS interpreter) - but it is not the same problem as running that text safely, forever, in someone else's production system. This doc is about the second problem: what changes once the generated expression leaves this package and is pasted into a warehouse view, a scoring job, or a SAS program someone else owns.

## 1. Treat the generated expression as a build artifact, not a snippet

The temptation is to run `xgboost_to_sql(...)`, copy the string into a `CREATE VIEW`, and move on. That throws away the one thing you actually have: a reproducible mapping from *this exact model file* to *this exact SQL text*. Once that link is broken, nobody six months from now can answer "does the SQL in production still match the model in the registry?" without re-deriving it by hand.

Concretely:

- Generate the SQL/SAS as part of the same pipeline step that saves the model, not as a separate manual action later.
- Name and store the output next to the model, keyed by the same version identifier (a model registry version, a content hash of the serialized model file, a run ID - whatever your team already uses to identify "which model is this"). `scores_v2024_11_03_a1b2c3d.sql`, not `scores.sql`.
- Commit the generated file to version control (or an equivalent artifact store) even though it's machine-generated. It is the thing actually running in production; the training code is one step removed from that.
- Never hand-edit the generated SQL/SAS to "fix" something. If it's wrong, the bug is in the model, the parser, or the emitter - fix one of those and regenerate. A hand patch silently breaks the model-to-SQL link this whole package exists to preserve.

## 2. Gate every promotion on regeneration + re-verification, not just regeneration

Regenerating the SQL from a new model file is cheap and can silently produce something wrong (a new sklearn/xgboost/lightgbm/catboost version changing internal behavior, a training change that trips one of this package's NotImplementedError guards in a code path nobody tested manually, a feature list drifting out of order). The pattern this repo already uses for its own test suite is the one to run in CI on *your* model too, every time, not just when someone remembers:

1. Train or load the model.
2. Generate the SQL/SAS expression.
3. Run the expression against a real instance of your actual target engine (or the closest verified proxy - DuckDB for SQL, per the caveats in section 4) on a held-out batch of real or synthetic rows, including rows that exercise every branch you care about (missing values, values sitting exactly on a threshold, category values not seen at fit time if relevant).
4. Compare against the model's own `.predict()`/`.predict_proba()` on the same rows, with an explicit numeric tolerance - not eyeballed.
5. Only promote the new SQL/SAS artifact if step 4 passes. A regenerate-and-ship pipeline with no comparison step is strictly worse than a manual process, because it looks automated and safe while providing neither.

This is exactly what `tests/` in this repo does for every supported model type - the only thing that changes for your deployment is that step 1 uses your real model and step 3 uses your real target engine, not a toy fixture.

## 3. Monitor the deployed expression against the live model, not just at deploy time

A SQL expression that matched at generation time can drift from "correct" in production for reasons that have nothing to do with this package:

- The upstream table's column types change (an INT column silently becomes a source of different rounding behavior than the FLOAT/DOUBLE this package assumed).
- A new category value or a rare NaN-only-in-production value hits a code path the offline verification batch happened not to cover.
- Someone updates the view/procedure without going through the regeneration pipeline (see section 1 - this is the failure mode versioning is meant to prevent, but the boundary is a process, not a database constraint).
- The SQL engine itself changes version and quietly changes float-casting or `NULL`-comparison behavior.

The concrete mitigation: periodically (daily/weekly, matched to how often your model or data actually changes) score the same batch of rows through both the live Python model and the deployed SQL/SAS, and alert on divergence above your tolerance - the same comparison as the CI gate in section 2, run continuously against production rather than once at build time. This catches drift from *outside* this package's control, which no amount of testing inside this package can.

## 4. Know exactly which parts of "correct" are actually verified

This package is explicit in its own docs about what has and hasn't been checked against a real target engine (see `README.md` / `TESTING_PLAN.md`), and that distinction should carry through to how much you trust each deployment target:

- **DuckDB**: verified directly, real execution, in this repo's own test suite.
- **PostgreSQL / MySQL**: NOT verified against a real instance anywhere this package has been built (no Docker daemon has been available in any sandbox used). The SQL is written to be standard and engine-agnostic, but "should work" is not the same claim as "was run and checked." Before relying on this in production: run the same generate -> execute -> compare-to-predict() loop from section 2 against your actual Postgres/MySQL instance, once, as a one-time qualification, and ideally fold it into CI if that instance is reachable from your pipeline.
- **SAS**: NOT verified against a real SAS installation anywhere this package has been built. `tests/sas_interp.py` confirms the generated SAS is internally self-consistent (it encodes the logic it claims to), which catches real logic bugs, but it is a hand-written Python model of SAS semantics, not SAS itself. If you have real SAS access, running the qualification loop there before any production use is not optional - it's the one target in this table that has never touched the real thing.

Treat "this package's test suite is green" as necessary, not sufficient, for the two unverified targets. Someone with real access to that engine needs to run the comparison at least once before this goes anywhere near production - this package can get you a verified-correct expression to hand them, but it can't verify the engine it's never been run against.

## 5. The XGBoost/CatBoost-to-SAS quantizer is a real accuracy tradeoff, not a free unlock

`xgboost_to_sas` and `catboost_to_sas` only work at all if the model was trained on features quantized to a small number of integer levels (`IntegerBinner`), and even then only after `check_xgb_sas_safety` confirms every threshold is actually safe. This is a genuine, verified path - but it is not free:

- Quantizing a continuous feature to (say) 16 ordinal levels throws away information the unquantized model would have used. Expect some accuracy loss relative to the same model trained on full-precision features - how much depends entirely on your data and how much genuine signal lives in the fine-grained differences within a bin.
- More bins reduces that loss but works against the safety checker (more achievable values means more chances for a threshold to land unsafely close to one, requiring a shallower model or fewer bins to pass `check_xgb_sas_safety` again). This is a real three-way tradeoff between bin count, tree depth/complexity, and whether the safety check passes at all - not a dial you can crank up freely.
- **If XGBoost's or CatBoost's specific regularization/behavior isn't a hard requirement, and SAS is the actual target, training a `HistGradientBoostingRegressor`/`Classifier` (via `sklearn_to_sas`) or a LightGBM model with `boosting_type="gbdt"` (via `lgbm_to_sas`) instead sidesteps this whole tradeoff.** Both compare in native float64 (verified in this repo) and need no quantization, no safety check, and no accuracy sacrifice to reach SAS. Reach for the XGBoost/CatBoost quantizer path specifically when you've already established that XGBoost or CatBoost outperforms the alternatives enough to be worth the integer-quantization cost - not as the default route to SAS.

## Summary checklist before first production use

1. Generated SQL/SAS is produced by a pipeline step, versioned against the exact model it came from, and never hand-edited.
2. CI regenerates and re-verifies (execute + compare to `predict()`, with an explicit tolerance) on every model update, and blocks promotion on failure.
3. A recurring job compares live-model vs. deployed-expression scores on production data and alerts on divergence.
4. If the target is Postgres, MySQL, or SAS: someone with real access has run the generate -> execute -> compare loop against the real engine at least once - this package's own test suite does not cover those targets.
5. If you're using the XGBoost/CatBoost-to-SAS quantizer, you've confirmed you actually need XGBoost/CatBoost specifically - not just reaching for it because it was the first thing you tried.
