"""Standalone verifier written by traceX for a Yukon-style benchmark: reads the submission JSON, writes
{"score": n, "metrics": {}} to the score path and exits 0, or exits 1 with no score file (invalid submission)."""
import json, math, os, sys

class Invalid(ValueError):
    """The submission fails the verifier's gate: no score."""


def _finite(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise Invalid(f"not a finite number: {x!r}")
    return x


def _rows(v, n, width, what):
    if not isinstance(v, list) or len(v) != n:
        raise Invalid(f"{what}: expected {n} entries")
    out = []
    for row in v:
        if width is None:
            out.append(_finite(row))
        else:
            if not isinstance(row, list) or len(row) != width:
                raise Invalid(f"{what}: each entry has {width} coordinates")
            out.append([_finite(x) for x in row])
    return out


def circle_packing_sum_radii(solution, instance):
    """Pack n disjoint circles in the unit square; maximize the sum of their radii. solution: {"centers": [[x, y]] * n,
    "radii": [r] * n}; instance: {"n": n}. The reference check's conditions in IEEE double arithmetic, which every
    platform rounds the same way (+, -, * and a correctly rounded square root): each circle has r >= 0 and
    r <= x <= 1 - r, r <= y <= 1 - r; two circles overlap when r_i + r_j > sqrt(dx*dx + dy*dy) (touching is allowed).
    Rounding can let two circles share at most about one unit in the last place, worth some 1e-16 of score, far below
    any challenge's minimum step. The score is the correctly rounded sum of the radii (math.fsum)."""
    try:
        n = int(instance["n"])
        if not 1 <= n <= 1000 or not isinstance(solution, dict):
            raise Invalid("bad instance or solution")
        c = [(float(x), float(y)) for x, y in _rows(solution.get("centers"), n, 2, "centers")]
        r = [float(x) for x in _rows(solution.get("radii"), n, None, "radii")]
        for (x, y), ri in zip(c, r):
            if ri < 0 or not (ri <= x <= 1 - ri and ri <= y <= 1 - ri):
                raise Invalid("a circle is not inside the unit square")
        for i in range(n):
            xi, yi = c[i]
            for j in range(i + 1, n):
                dx, dy = xi - c[j][0], yi - c[j][1]
                if r[i] + r[j] > math.sqrt(dx * dx + dy * dy):
                    raise Invalid("two circles overlap")
        return math.fsum(r)
    except Invalid:
        return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


if __name__ == "__main__":
    sub, out = sys.argv[1], sys.argv[2]
    if os.path.exists(out):
        os.remove(out)
    with open(sub, encoding="utf-8") as f:
        solution = json.load(f)
    score = circle_packing_sum_radii(solution, {"n": 26})
    if score is None or not math.isfinite(score):
        sys.exit(1)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"score": score, "metrics": {}}, f)
