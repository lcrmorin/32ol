"""Canonical intermediate representation (IR) for a tree ensemble.

The point of this module: every model-library parser (xgboost, sklearn, ...)
converts its own native tree format into this shape ONCE, and every
target-language emitter (sql, sas, ...) consumes only this shape. Adding a
new model library means writing one parser; adding a new target language
means writing one emitter. Neither has to know about the other.

A single decision tree is a `Node` tree: either a `Leaf` (a value) or a
`Split` (a feature test with up to three children - yes/no/missing). Missing
is optional: a source model that has no native missing-value handling
(plain sklearn CART trees) produces splits with `missing=None`, and it is
the emitter's job to decide what a NULL input does in that case (see each
emitter's docstring - this is a real, disclosed behavioral choice, not
free-form nonsense).

An `Ensemble` is one or more weighted trees (weight handles RandomForest
averaging and GradientBoosting's learning rate) plus how their summed raw
output turns into a final prediction (base_score + link function). A model
with multiple outputs (multiclass) is represented as a list of Ensembles,
one per class, combined with softmax by the caller.
"""

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass(frozen=True)
class Leaf:
    value: float


@dataclass(frozen=True)
class Split:
    feature: str
    # Exactly one of (threshold) or (categories) is set.
    threshold: Optional[float] = None   # numeric split: yes-branch if value < threshold (or <=, see `le`)
    le: bool = False                    # False: yes-branch is "value < threshold" (xgboost).
                                         # True:  yes-branch is "value <= threshold" (sklearn CART).
    categories: Optional[tuple] = None  # categorical split: yes-branch if value in categories
    yes: "Node" = None
    no: "Node" = None
    # None = source model has no concept of missing for this split; the
    # emitter decides the fallback behavior (documented per-emitter).
    missing: "Node" = None


Node = Union[Leaf, Split]


@dataclass(frozen=True)
class Tree:
    root: Node
    weight: float = 1.0  # multiplies this tree's raw contribution (RF averaging, GBM learning_rate)


LINKS = ("identity", "logistic", "exp")


_THRESHOLD_PRECISIONS = ("float32", "float64")


@dataclass(frozen=True)
class Ensemble:
    trees: tuple  # tuple[Tree, ...]
    base_score: float = 0.0  # additive margin-scale offset, already link-inverted by the parser
    link: str = "identity"
    # "float32": thresholds are float32 values (xgboost splits internally in
    #   float32) - emitters must force a float32-precision comparison (e.g.
    #   CAST col AS FLOAT) or predictions silently disagree near a threshold.
    # "float64": thresholds are full float64 precision (sklearn CART splits
    #   compare in float64) - emitters must NOT truncate to float32.
    threshold_precision: str = "float32"

    def __post_init__(self):
        if self.link not in LINKS:
            raise ValueError(f"link must be one of {LINKS}, got {self.link!r}")
        if self.threshold_precision not in _THRESHOLD_PRECISIONS:
            raise ValueError(
                f"threshold_precision must be one of {_THRESHOLD_PRECISIONS}, "
                f"got {self.threshold_precision!r}"
            )
