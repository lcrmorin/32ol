# Why `xgb2sql` — prior art check

Closest prior art found, checked against GitHub/PyPI metadata (as of 2026-09-12):

| Project | Status |
|---|---|
| [gbm2sql](https://github.com/crst/gbm2sql) | 4 stars, 1 commit — a demo snapshot, not a maintained tool. |
| [sqlgbm](https://github.com/mattismegevand/sqlgbm) | 2 stars, README literally says "not ready for production use yet." |
| [xgb2sql](https://github.com/chengjunhou/xgb2sql) (R package of a similar name) | R only, not usable from Python. |
| [m2cgen](https://github.com/BayesWitnesses/m2cgen) | General-purpose model-export tool, 3,000 stars, but no release since April 2022 and doesn't target SQL as an output language at all. |

None of these is a real substitute — hence a dedicated package rather than depending on an existing one. See `TESTING_PLAN.md` for what's been hardened and verified so far.
