"""A tiny interpreter for the exact, narrow SAS expression subset emit_sas.py
produces - NOT a general SAS parser.

There is no real SAS available anywhere this project has been built, so this
is a self-consistency check, not proof the code runs correctly in real SAS:
it evaluates the emitted expression text using a hand-written evaluator that
shares the same understanding of IFN/MISSING/name-literal semantics as the
emitter. It catches real bugs (wrong operator, wrong branch, bad quoting,
sign errors, precedence mistakes) but a shared blind spot between emitter
and interpreter would not be caught. Treat SAS output as unverified against
a real SAS instance - same posture as the still-open Postgres/MySQL item.
"""

import math
import re


class _SasMissing(float):
    """Stand-in for a SAS missing numeric value: compares as smaller than any
    real number (matching real SAS's documented behavior for ordinary
    numeric comparisons - "." sorts before any number), while still being
    distinguishable from a real value by MISSING(). This exists because
    SAS's IFN function evaluates ALL of its arguments (it is not a
    short-circuiting IF-THEN/ELSE) - so even when the missing-branch is the
    one that will be used, the not-taken branch's comparison still gets
    evaluated and must not raise.
    """

    def __new__(cls):
        return super().__new__(cls, float("-inf"))


SAS_MISSING = _SasMissing()  # pass this as a row value in place of NaN/None

_NAME_LITERAL = re.compile(r"'((?:[^']|'')*)'n")
_STRING_LITERAL = re.compile(r"'((?:[^']|'')*)'")


def _sas_to_python(expr: str, feature_names) -> str:
    # 1. Name literals ('col name'n) -> __col__("col name")
    def repl_name_literal(m):
        name = m.group(1).replace("''", "'")
        return '__col__("' + name + '")'

    out = _NAME_LITERAL.sub(repl_name_literal, expr)

    # 2. Bare identifiers that are exactly a known feature name -> __col__("name")
    for fn in sorted(feature_names, key=len, reverse=True):
        out = re.sub(rf"(?<![\w'\"]){re.escape(fn)}(?![\w'\"])", f'__col__("{fn}")', out)

    # 3. Remaining plain string literals (category labels) -> Python string literals
    def repl_string_literal(m):
        return "'" + m.group(1).replace("''", "\\'") + "'"

    out = _STRING_LITERAL.sub(repl_string_literal, out)

    # 4. Function/keyword casing
    out = out.replace("IFN(", "ifn(").replace("MISSING(", "missing_(")
    out = re.sub(r"\bIN\s*\(", " in (", out)
    out = re.sub(r"\bOR\b", "or", out)
    out = re.sub(r"\bAND\b", "and", out)
    out = re.sub(r"\bNOT\b", "not", out)

    # 5. A single-category IN (...) becomes a Python parenthesized
    # expression, not a 1-tuple, without a trailing comma - "in ('blue')"
    # is "in" on the plain string 'blue' (a substring check, needing a
    # string on the left) rather than tuple membership, which crashes when
    # the left side is a non-string missing sentinel even though IFN's
    # non-short-circuit evaluation means that branch's value is discarded
    # either way. Add the trailing comma SAS's own IN doesn't need but
    # Python's tuple literal does.
    def _singleton_tuple(m):
        inner = m.group(1)
        return f"in ({inner},)" if inner.strip() and "," not in inner else m.group(0)
    out = re.sub(r"\bin\s*\(([^()]*)\)", _singleton_tuple, out)

    return out


def eval_sas_row(expr: str, feature_names, row: dict) -> float:
    """Evaluate one SAS-subset expression against one input row (a dict of
    feature_name -> value, with a real Python None/NaN for a missing value).
    """
    py_expr = _sas_to_python(expr, feature_names)

    def __col__(name):
        return row[name]

    def ifn(cond, a, b):
        return a if cond else b

    def missing_(v):
        return isinstance(v, _SasMissing)

    return eval(py_expr, {"__builtins__": {}}, {
        "__col__": __col__, "ifn": ifn, "missing_": missing_, "exp": math.exp,
    })
