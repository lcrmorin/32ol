# Why `xgb2sql` — prior art check

Closest prior art found, checked against GitHub/PyPI metadata (as of 2026-09-12):

| Project | Status |
|---|---|
| [gbm2sql](https://github.com/crst/gbm2sql) | 4 stars, 1 commit — a demo snapshot, not a maintained tool. |
| [sqlgbm](https://github.com/mattismegevand/sqlgbm) | 2 stars, README literally says "not ready for production use yet." |
| [xgb2sql](https://github.com/chengjunhou/xgb2sql) (R package of a similar name) | R only, not usable from Python. |
| [m2cgen](https://github.com/BayesWitnesses/m2cgen) | General-purpose model-export tool, 3,000 stars, but no release since April 2022 and doesn't target SQL as an output language at all. |

None of these is a real substitute — hence a dedicated package rather than depending on an existing one. See `TESTING_PLAN.md` for what's been hardened and verified so far.

## Prior art check: model -> SAS

`m2cgen` (BayesWitnesses/m2cgen, 3,000 stars) is the obvious general-purpose "model to native code" tool to check first — it targets Java, C, Python, Go, JavaScript, Visual Basic, C#, R, PowerShell, PHP, Dart, Haskell, Ruby, F#, Rust. **SAS is not in that list**, confirmed against the project's own README (checked 2026-09-12).

The only SAS-adjacent prior art found: [cydalytics/xgboost_to_sas](https://github.com/cydalytics/xgboost_to_sas) (7 stars, 2 forks) — a small demo/tutorial repo, not a maintained tool. It doesn't convert to SAS directly either: it runs `m2cgen` to get Visual Basic output, then manually hand-translates that VBA into SAS. Two Jupyter notebooks on the Iris dataset, explicitly framed as a proof-of-concept write-up (there's a companion "Towards Data Science" article), not something built for production use or to handle the edge cases (missing values, categorical splits, verified link functions) this package's XGBoost side already covers.

Bottom line: there is no real existing tool for model -> SAS, direct or automated. That's a genuine gap, not just an underused one.

## Prior art re-check: LightGBM and CatBoost as source models (2026-09-12)

`m2cgen` does support LightGBM and CatBoost as source models (it's a general "model to native code" tool, not tree-specific) - but this doesn't change the bottom line above, since SQL isn't in its target language list at all and SAS isn't either. Breadth of source-model support doesn't substitute for the actual gap (SQL/SAS as targets), and `m2cgen`'s own no-release-since-2022 status stands regardless of which models it can read. No other prior-art candidate found for LightGBM/CatBoost -> SQL or -> SAS specifically beyond what's already listed above.
