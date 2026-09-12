"""An ordinal integer quantizer, plus a safety checker that together unlock a
verified xgboost -> SAS path (see xgboost_to_sas in sklearn_api-adjacent module
`xgb_sas.py`).

Why this exists: xgboost's split thresholds are float32, and a SAS DATA step
has no verified float32-truncation equivalent to SQL's `CAST(col AS FLOAT)`
trick (see emit_sas.py). Feeding xgboost features that only ever take a small
number of distinct INTEGER values changes the picture: if every threshold in
the trained model turns out to be safely far (in float32-epsilon terms) from
every achievable integer, then plain float64 comparison in SAS - no cast
needed - can't disagree with xgboost's float32-truncated comparison, because
there's no achievable value close enough to the threshold for truncation to
matter.

IMPORTANT - this is checked per-model, not assumed from the quantization
alone. Empirically, "quantize to integers" does NOT reliably produce safe
thresholds on its own: tested against a real trained model with 1000 integer
levels and max_depth=8, hundreds of thresholds landed close to (in a few
cases apparently near-coincident with) an achievable integer - not the clean
"always exactly X.5" pattern seen with fewer levels and shallower trees (see
TESTING_PLAN.md for the actual numbers). The mechanism is not fully
characterized (it may depend on max_bin, tree_method, tie-breaking, or
something else), so quantizing your features is necessary but not sufficient
- `check_xgb_sas_safety` verifies the SPECIFIC trained model actually landed
in the safe regime, and `xgboost_to_sas` refuses to emit anything if it
didn't. Trust the check, not the intuition about why it should work.
"""

from typing import Optional

import numpy as np
import pandas as pd

from xgb2sql.ir import Ensemble, Split


class IntegerBinner:
    """A minimal ordinal binner: fit on training data, transform maps each
    column to an integer code in [0, n_bins) (quantile-based bin edges), NaN
    passes through as NaN. Use the SAME fitted instance to transform both
    training data (before fitting the xgboost model) and any data you'll
    later score - consistency between the two is what the safety argument
    depends on.

    This is deliberately simple rather than a sklearn Pipeline-integrated
    transformer, so the actual bin edges are inspectable (`.edges_`) rather
    than hidden inside a fitted object.
    """

    def __init__(self, n_bins: int = 32):
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.n_bins = n_bins
        self.edges_: dict = {}

    def fit(self, df: pd.DataFrame, columns: Optional[list] = None) -> "IntegerBinner":
        columns = columns or list(df.columns)
        for col in columns:
            s = df[col].dropna().astype(float)
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            edges = np.unique(np.quantile(s, quantiles))
            if len(edges) < 2:
                raise ValueError(f"column {col!r} has too few distinct values to bin")
            self.edges_[col] = edges
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, edges in self.edges_.items():
            values = df[col].astype(float)
            # np.digitize with the interior edges gives a code in [0, len(edges)-2]
            codes = np.digitize(values.values, edges[1:-1], right=False).astype(float)
            codes[values.isna().values] = np.nan
            out[col] = codes
        return out

    def n_levels(self, col: str) -> int:
        """Number of distinct integer codes column `col` can take after
        transform - i.e. codes are in [0, n_levels(col)).
        """
        return len(self.edges_[col]) - 1

    def levels(self) -> dict:
        """{column: n_levels} for every fitted column - the `bins_per_feature`
        argument xgboost_to_sas needs.
        """
        return {col: self.n_levels(col) for col in self.edges_}


def check_xgb_sas_safety(ensemble: Ensemble, bins_per_feature: dict) -> list:
    """Check every numeric-split threshold in `ensemble` against the declared
    achievable integer range [0, bins_per_feature[feature]) for that feature.
    Returns a list of human-readable violation strings (empty = safe).
    Categorical splits (exact integer-code membership, not an inequality) and
    the missing branch (an exact-equality MISSING() check, not an inequality)
    are not subject to this float32-boundary risk and are not checked.

    This directly SIMULATES the comparison for every achievable value near
    each threshold - both the naive float64 way (what SAS, with no cast,
    would compute) and xgboost's own float32-truncated way - and flags a
    disagreement, rather than using a distance heuristic. A threshold that
    happens to sit exactly ON an achievable integer is safe (both a query
    equal to it and its float32 image compare identically); a distance-based
    heuristic would have wrongly flagged that as the worst case - this
    function does not make that mistake.
    """
    violations = []

    def check_split(feature, th):
        n_bins = bins_per_feature.get(feature)
        if n_bins is None:
            return [
                f"feature {feature!r} has a numeric split but no bin count was "
                "declared in bins_per_feature - cannot verify safety for it"
            ]
        lo = max(0, int(np.floor(th)) - 2)
        hi = min(n_bins - 1, int(np.ceil(th)) + 2)
        bad = []
        for v in range(lo, hi + 1):
            no_cast = float(v) < th
            with_cast = np.float32(v) < np.float32(th)
            if no_cast != with_cast:
                bad.append(v)
        if bad:
            return [
                f"feature {feature!r}: threshold {th!r} disagrees (float64-vs-float32-truncated "
                f"routing) for achievable value(s) {bad}"
            ]
        return []

    def walk(node):
        if not isinstance(node, Split):
            return
        if node.categories is None:
            violations.extend(check_split(node.feature, node.threshold))
        walk(node.yes)
        walk(node.no)
        if node.missing is not None:
            walk(node.missing)

    for t in ensemble.trees:
        walk(t.root)
    return violations
