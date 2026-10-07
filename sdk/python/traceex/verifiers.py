"""Verifiers for challenge bounties (SPEC 4k). Standard library only, deterministic, safe for a node to run: each takes
a submitted solution (JSON) and returns a score, or None when the solution is invalid (the deterministic gate).

    score = circle_packing_sum_radii({"centers": [[x, y], ...], "radii": [...]}, {"n": 26})

Contract (the same for every kind, and the same as a Yukon-style benchmark command): a verifier either returns one
finite number, the score, or says the submission is invalid (None here; a nonzero exit and no score file for a command).
It never returns a partial score for an invalid submission, and the same input always gives the same score: the node
runs every python verifier twice and refuses a submission whose two scores differ.

Kinds
  * python   functions in this module (and any the operator registers in Python, never over HTTP: they run code). The
             node runs them itself.
  * lean4    a Lean 4 proof of an exact statement (formal-conjectures). lean_gate() is the cheap, pure-Python part (the
             statement must be the challenge's, no sorry / admit / new axioms); LeanRunner compiles it in a checkout of
             the statement's repository and checks `#print axioms`. The node runs LeanRunner only where the operator
             set one up; otherwise drawn validators run it and reveal the result.
  * command  a Yukon-style benchmark: setupCommand, then benchmarkCommand, which writes {"score": n, "metrics": {...}} at
             scorePath and exits 0, or exits nonzero with no score. Never run by the node (it runs submitted code):
             validators run it in their own sandbox and reveal the score.
  * registry a failure's pass rate on validators' own hidden cases (an escalated failure, 4g): validators measure it.

The circle-packing and Tammes verifiers check the same conditions as the verification code in Google DeepMind's
AlphaEvolve repository of problems (https://github.com/google-deepmind/alphaevolve_repository_of_problems, Apache-2.0;
see NOTICE), re-implemented here with no dependencies, in IEEE double arithmetic that every platform rounds the same
way, so every node and validator gives the same score.
"""
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter


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


def tammes_min_distance(solution, instance):
    """n points on the unit sphere in R^3; maximize the smallest distance between two of them (the Tammes problem).
    solution: {"points": [[x, y, z]] * n} (each normalised to the sphere, as the reference evaluation does);
    instance: {"n": n}. A zero vector is invalid. Computed in a fixed order with correctly rounded square roots, so
    the score is the same everywhere."""
    try:
        n = int(instance["n"])
        if not 2 <= n <= 1000 or not isinstance(solution, dict):
            return None
        pts = _rows(solution.get("points"), n, 3, "points")
        unit = []
        for p in pts:
            norm = math.sqrt(math.fsum(x * x for x in p))
            if not norm > 0:
                return None
            unit.append([x / norm for x in p])
        best = math.inf
        for i in range(n):
            for j in range(i + 1, n):
                best = min(best, math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(unit[i], unit[j]))))
        return float(best) if math.isfinite(best) else None
    except Invalid:
        return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


BUILTIN = {
    "circle-packing-sum-radii@1": circle_packing_sum_radii,
    "tammes-min-distance@1": tammes_min_distance,
}


# --- lineage: is a submission derived from an earlier one? ------------------------------------------------------------
def _leaves(x):
    """A solution's values (numbers and text), never its keys: two solutions share structure, not content."""
    if isinstance(x, dict):
        return [v for k in sorted(x) for v in _leaves(x[k])]
    if isinstance(x, (list, tuple)):
        return [v for y in x for v in _leaves(y)]
    return [x]


def similarity(a, b):
    """How much of one solution another repeats: for numeric constructions, the share of numbers (rounded to 6
    places) they have in common; for text (a proof), the share of words. 1.0 means one is a copy of the other."""
    la, lb = _leaves(a), _leaves(b)
    num = lambda xs: [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    na, nb = num(la), num(lb)
    if len(na) >= 4 and len(nb) >= 4:
        ca = Counter(f"{float(x):.6f}" for x in na)
        cb = Counter(f"{float(x):.6f}" for x in nb)
        return sum((ca & cb).values()) / max(sum(ca.values()), sum(cb.values()))
    words = lambda xs: set(w for x in xs for w in re.findall(r"\w+", str(x)))
    wa, wb = words(la), words(lb)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


# --- Lean 4 (formal-conjectures) -------------------------------------------------------------------------------------
FORBIDDEN_LEAN = re.compile(
    r"\b(sorry|admit|axiom|axioms|unsafe|implemented_by|extern|native_decide|opaque|partial)\b"
    r"|#exit|\belab\b|\bmacro\b|\bsyntax\b|\binitialize\b|\brun_cmd\b|\brun_tac\b|set_option\s+debug|"
    r"\bdecide\s*:=\s*true|Lean\.ofReduceBool|\bimport\b")
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def _ws(s):
    return re.sub(r"\s+", " ", s).strip()


def lean_gate(statement, theorem, proof):
    """The pure-Python part of the Lean contract, run before anything compiles. `proof` is the whole replacement
    declaration: `theorem <name> <statement> := <proof>`. Its statement must be the challenge's, character for
    character up to whitespace (an `answer(sorry)` may be filled with an answer); it may not use sorry, admit, new
    axioms, unsafe code, imports or tactics that skip the kernel. Returns the answer it fills in (or "") or raises
    Invalid."""
    if not isinstance(proof, str) or len(proof) > 200_000:
        raise Invalid("a Lean proof is text, at most 200 KB")
    m = re.match(r"\s*theorem\s+([\w.'₀-₉]+)\s*(.*?):=(.*)\Z", proof, re.S)
    if not m or m[1] != theorem:
        raise Invalid(f"the proof must be the declaration `theorem {theorem} ... := ...`")
    want = _ws(statement)
    got = _ws(m[2])
    answer = ""
    if "answer(sorry)" in want:
        pat = re.escape(want).replace(re.escape("answer(sorry)"), r"answer\((.+?)\)")
        mm = re.fullmatch(pat, got)
        if not mm:
            raise Invalid("the statement differs from the challenge's")
        answer = mm[1]
        if FORBIDDEN_LEAN.search(answer):
            raise Invalid("the answer may not use sorry")
    elif got != want:
        raise Invalid("the statement differs from the challenge's")
    if FORBIDDEN_LEAN.search(m[3]):
        raise Invalid(f"forbidden in a proof: {FORBIDDEN_LEAN.search(m[3])[0]}")
    return answer


class LeanRunner:
    """Compile a submitted proof inside a checkout of the statement's repository (e.g. google-deepmind/
    formal-conjectures, built once with `lake exe cache get && lake build`) and accept it only if Lean reports no
    error, no `sorry`, and `#print axioms` shows nothing beyond propext, Classical.choice and Quot.sound. Returns 1.0
    (proved) or None. Not exercised by the test suite (it needs Lean, Mathlib and a built checkout); the gate above is.

        node.register_verifier("lean4@formal-conjectures", LeanRunner("/srv/formal-conjectures"), author=OPERATOR)
    """

    def __init__(self, checkout, lake="lake", timeout=900):
        self.checkout, self.lake, self.timeout = os.path.abspath(checkout), lake, int(timeout)

    def __call__(self, solution, instance):
        try:
            proof = (solution or {}).get("proof")
            lean_gate(instance["statement"], instance["theorem"], proof)
            src = open(os.path.join(self.checkout, instance["file"]), encoding="utf-8").read()
            decl = re.compile(r"theorem\s+" + re.escape(instance["theorem"]) + r"\b.*?:=\s*by\s+sorry", re.S)
            if not decl.search(src):
                return None
            body = decl.sub(lambda _: proof, src, count=1)
            full = ".".join(x for x in (instance.get("namespace"), instance["theorem"]) if x)
            body += f"\n#print axioms {full}\n"
            fd, path = tempfile.mkstemp(suffix=".lean", prefix="TracexCheck_",
                                        dir=os.path.dirname(os.path.join(self.checkout, instance["file"])))
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(body)
            try:
                r = subprocess.run([self.lake, "env", "lean", path], cwd=self.checkout, capture_output=True, text=True,
                                   timeout=self.timeout)
            finally:
                os.unlink(path)
            out = r.stdout + r.stderr
            if r.returncode != 0 or "sorry" in out or "error" in out.lower():
                return None
            used = set(re.findall(r"[\w.]+", out.split("depends on axioms:", 1)[1])) if "depends on axioms" in out else set()
            return 1.0 if used <= ALLOWED_AXIOMS else None
        except (Invalid, OSError, KeyError, subprocess.TimeoutExpired):
            return None


def digest(obj):
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
