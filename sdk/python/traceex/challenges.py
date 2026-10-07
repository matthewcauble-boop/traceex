"""Challenge bounties (SPEC 4k): big open problems with a deterministic verifier, a direction and a public
leaderboard, paid per verified improvement. This module is the client-side half: the challenge file format
(`challenge/0.1`), the payout curve both sides compute, importers from openly licensed sources, and a reader and
writer for Yukon-style benchmark.json manifests.

    python -m traceex.challenges import examples/challenges/sample/*.json --node http://localhost:8787 --address 0x…
    python -m traceex.challenges show examples/challenges/sample/formal_conjectures/Erdos28.lean
    python -m traceex.challenges export 3 --node http://localhost:8787 --out ./bench

The file format
  {"v": "challenge/0.1", "title", "statement", "path": "math/packing",
   "key": "packing:circles-in-square:sum-radii:n=26",   # what the problem IS: the same problem from two sources merges
   "aliases": ["erdos:28", …],
   "metric": {"name", "direction": "maximize" | "minimize", "baseline", "baseline_source", "target"?,
              "scale"?, "min_step"?, "min_step_rel"?, "final_share"?},
   "verifier": {"id": "circle-packing-sum-radii@1", "kind": "python" | "lean4" | "command" | "registry" | "none",
                "instance": {...}, "author"?: "0x…", "command"?: {Yukon fields}},
   "instances": {"hidden": {"digest": "sha256:…", "count": n}}?,     # held by validators; only the hash is public
   "source": {"name", "url", "licence", "ref"}, "days"?: 182, "window_days"?: 14,
   "reference"?: <a solution>}       # when posting: the node scores it and takes that as the baseline

How a challenge pays (the curve; node/challenges.py moves the money)
  progress u = (score - baseline) / scale, or (baseline - score) / scale when minimizing (never below 0).
  With a target: P(u) = (1 - final_share) * min(u, 1) + final_share * [u >= 1]  (scale = |target - baseline|).
  Without:       P(u) = 1 - 2^-u   (every `scale` of improvement releases half of what is left).
  A pledge made when the best was b (or at `from_score`) has released floor(amount * (P(best) - P(b)) / (1 - P(b)))
  once the best reaches `best`. P is a function of the best score alone, so the payouts telescope: k small steps pay
  exactly what one step to the same score pays, and splitting an improvement into many submissions earns nothing but
  extra fees. An improvement counts only when it beats the best by the minimum step (min_step, or min_step_rel x |best|,
  whichever is larger; with validators, by that much beyond twice their standard error).
"""
import argparse
import ast
import hashlib
import json
import math
import os
import re
import sys

from .canon import canonical
from .verifiers import BUILTIN

VERSION = "challenge/0.1"
DIRECTIONS = {"maximize": 1, "max": 1, "+": 1, "higher": 1, "minimize": -1, "min": -1, "-": -1, "lower": -1}
KINDS = ("python", "lean4", "command", "registry", "none")
PATH = re.compile(r"[a-z_]+(/[a-z_]+){0,5}")
FINAL_SHARE = 0.5
WINDOW_DAYS, MIN_WINDOW_DAYS = 14, 7       # paid tranches and the bond wait this long for prior art (wall-clock days)                 # with a target: half the pledges stream out along the way, half on reaching it


def direction(metric):
    d = DIRECTIONS.get(str((metric or {}).get("direction", "maximize")).lower())
    if d is None:
        raise ValueError("direction is maximize or minimize (or Yukon's + / -)")
    return d


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:60] or "challenge"


def normalize(c):
    """Check a challenge file and fill in its defaults (scale, minimum step, final share). Returns a new dict."""
    if not isinstance(c, dict) or c.get("v", VERSION) != VERSION:
        raise ValueError(f"a challenge file is {VERSION}")
    c = json.loads(json.dumps(c))
    c["v"] = VERSION
    for k, cap in (("title", 200), ("statement", 20_000)):
        if not isinstance(c.get(k), str) or not c[k].strip() or len(c[k]) > cap:
            raise ValueError(f"{k} is text, at most {cap} characters")
    c["path"] = str(c.get("path") or "math").strip("/")
    if not PATH.fullmatch(c["path"]):
        raise ValueError("path looks like a taxonomy branch, e.g. math/packing")
    m = c.setdefault("metric", {})
    d = direction(m)
    m["direction"] = "maximize" if d > 0 else "minimize"
    m.setdefault("name", "score")
    try:
        m["baseline"] = float(m.get("baseline", 0.0))
    except (TypeError, ValueError):
        raise ValueError("metric.baseline is a number: the best score known when the challenge is posted")
    if not math.isfinite(m["baseline"]):
        raise ValueError("metric.baseline is finite")
    if m.get("target") is not None:
        m["target"] = float(m["target"])
        if d * (m["target"] - m["baseline"]) <= 0:
            raise ValueError("the target must be better than the baseline")
        m["scale"] = abs(m["target"] - m["baseline"])
    for k in ("scale", "min_step", "min_step_rel"):
        if m.get(k) is not None:
            m[k] = float(m[k])
            if not (m[k] >= 0 and math.isfinite(m[k])):
                raise ValueError(f"metric.{k} is a number >= 0")
    if not m.get("scale"):
        m["scale"] = 100 * m["min_step"] if m.get("min_step") else (abs(m["baseline"]) * 0.01 or 1.0)
    if m.get("min_step") is None:
        m["min_step"] = m["scale"] / 100
    m.setdefault("min_step_rel", 0.0)
    m["final_share"] = float(m.get("final_share", FINAL_SHARE if m.get("target") is not None else 0.0))
    if not 0 <= m["final_share"] <= 1:
        raise ValueError("metric.final_share is between 0 and 1")
    v = c.setdefault("verifier", {"kind": "none"})
    if v.get("kind", "none") not in KINDS:
        raise ValueError(f"verifier.kind is one of {KINDS}")
    v.setdefault("kind", "none")
    v.setdefault("id", f"{v['kind']}:{slug(c['title'])}@1" if v["kind"] != "none" else "")
    v.setdefault("instance", {})
    c["aliases"] = sorted({str(a).strip().lower() for a in c.get("aliases") or [] if str(a).strip()})[:20]
    c["key"] = str(c.get("key") or problem_key(c)).strip().lower()[:200]
    w = c.get("window_days")
    c["window_days"] = max(MIN_WINDOW_DAYS, float(WINDOW_DAYS if w is None else w))   # the prior-art window, wall-clock
    c.setdefault("source", {})
    return c


def problem_key(c):
    """What a problem is, when its file names no key: the statement, metric, direction and verifier instance."""
    body = [re.sub(r"\s+", " ", c["statement"]).strip().lower(), c["metric"].get("name"), c["metric"]["direction"],
            (c.get("verifier") or {}).get("id"), (c.get("verifier") or {}).get("instance")]
    return "sha256:" + hashlib.sha256(canonical(body)).hexdigest()


# --- the payout curve (shared with the node) ------------------------------------------------------------------------
def progress(metric, score):
    if score is None:
        return 0.0
    return max(0.0, direction(metric) * (float(score) - metric["baseline"]) / (metric["scale"] or 1.0))


def released(metric, score):
    """P(u): the share of a pledge made at the baseline that has been released once the best reaches `score`."""
    u = progress(metric, score)
    if metric.get("target") is not None:
        fs = metric.get("final_share", FINAL_SHARE)
        return 1.0 if u >= 1 else (1 - fs) * u
    return 1.0 - 2.0 ** (-u)


def owed(amount, metric, from_score, score):
    """msats a pledge of `amount` made at `from_score` has released once the best is `score` (integer, floor)."""
    p0, p1 = released(metric, from_score), released(metric, score)
    if p1 <= p0 or p0 >= 1:
        return 0
    if p1 >= 1:
        return int(amount)
    return min(int(amount), int(math.floor(int(amount) * (p1 - p0) / (1 - p0))))


def min_step(metric, best):
    return max(float(metric.get("min_step") or 0), float(metric.get("min_step_rel") or 0) * abs(best or 0))


def improves(metric, best, score, se=0.0):
    """Does `score` beat `best` by the minimum step (beyond twice the validators' standard error)?"""
    if score is None:
        return False
    return direction(metric) * (float(score) - float(best)) - 2 * float(se or 0) >= max(min_step(metric, best), 1e-12)


def reached(metric, score):
    t = metric.get("target")
    return t is not None and score is not None and direction(metric) * (float(score) - t) >= 0


# --- importers ----------------------------------------------------------------------------------------------------------
AE_REPO = "https://github.com/google-deepmind/alphaevolve_repository_of_problems"
AE_SOURCE = {"name": "AlphaEvolve repository of problems (Google DeepMind)", "url": AE_REPO,
             "licence": "Apache-2.0 (software); CC-BY-4.0 (other materials)",
             "cite": "Mathematical exploration and discovery at scale, arXiv:2511.02864 (2025)"}
AE_FAMILIES = {
    "packing_circles_max_sum_of_radii": {
        "verifier": "circle-packing-sum-radii@1", "metric": "sum_of_radii", "direction": "maximize",
        "path": "math/packing", "scale": 0.001, "min_step": 1e-6,
        "title": "Pack {n} disjoint circles in a unit square, maximizing the sum of their radii",
        "statement": ("Place {n} disjoint circles (touching allowed) inside the unit square [0,1]^2 so that the sum of "
                      "their radii is as large as possible. Submit {{\"centers\": [[x, y]] * {n}, \"radii\": [r] * {n}}}."),
        "arrays": r"(centers|radii)_(\d+)\s*=\s*np\.array\(\s*(\[.*?\])\s*\)"},
    "tammes_problem": {
        "verifier": "tammes-min-distance@1", "metric": "min_distance", "direction": "maximize",
        "path": "math/packing", "scale": 0.001, "min_step": 1e-7,
        "title": "Tammes problem, n = {n}: spread points on a sphere",
        "statement": ("Place {n} points on the unit sphere in R^3 so that the smallest distance between two of them is "
                      "as large as possible. Submit {{\"points\": [[x, y, z]] * {n}}} (each point is normalised)."),
        "arrays": r"(\d+)\s*:\s*np\.array\(\s*(\[.*?\])\s*,?\s*\)"},
}


def from_alphaevolve_notebook(nb, family, *, ref=None, only=None):
    """Challenges from one notebook of the AlphaEvolve repository of problems (the .ipynb JSON, as dict or text): one
    per construction the notebook holds, for the families with a verifier here (AE_FAMILIES). The baseline is our
    verifier's score of the notebook's own best construction, which is also returned as the reference solution, so the
    record a challenge starts from is checked, not copied."""
    if family not in AE_FAMILIES:
        raise ValueError(f"no verifier for {family!r} yet; families: {sorted(AE_FAMILIES)}")
    fam = AE_FAMILIES[family]
    nb = json.loads(nb) if isinstance(nb, str) else nb
    cells = ["".join(c.get("source") or []) for c in nb.get("cells", [])]
    code = "\n".join(s for s, c in zip(cells, nb.get("cells", [])) if c.get("cell_type") == "code")
    cands, pending = {}, {}
    for m in re.finditer(fam["arrays"], code, re.S):
        try:                                   # plain number literals only: nothing in a notebook is ever executed
            if family == "packing_circles_max_sum_of_radii":
                n = int(m[2])
                pending.setdefault(n, {})[m[1]] = ast.literal_eval(m[3])
                if len(pending[n]) == 2:
                    cands.setdefault(n, []).append(pending.pop(n))
            else:
                cands.setdefault(int(m[1]), []).append({"points": ast.literal_eval(m[2])})
        except (ValueError, SyntaxError):
            continue
    out = []
    verify = BUILTIN[fam["verifier"]]
    sign = DIRECTIONS[fam["direction"]]
    for n, sols in sorted(cands.items()):
        if only and n not in only:
            continue
        scored = [(verify(x, {"n": n}), i) for i, x in enumerate(sols)]
        scored = [(sc, i) for sc, i in scored if sc is not None]
        if not scored:
            continue
        score, i = max(scored, key=lambda t: (sign * t[0], -t[1]))      # the notebook's best valid construction
        sol = sols[i]
        out.append({"challenge": normalize({
            "v": VERSION, "title": fam["title"].format(n=n), "statement": fam["statement"].format(n=n),
            "path": fam["path"], "key": f"alphaevolve:{family}:n={n}",
            "metric": {"name": fam["metric"], "direction": fam["direction"], "baseline": score,
                       "baseline_source": f"best construction in {AE_REPO}/tree/main/experiments/{family}",
                       "scale": fam["scale"], "min_step": fam["min_step"]},
            "verifier": {"id": fam["verifier"], "kind": "python", "instance": {"n": n}},
            "source": dict(AE_SOURCE, ref=ref or f"experiments/{family}/{family}.ipynb")}), "reference": sol})
    return out


def from_formal_conjectures(text, *, file, commit="main"):
    """Open statements from one formal-conjectures Lean file (google-deepmind/formal-conjectures, Apache-2.0): every
    theorem tagged `@[category research open, …]` whose proof is `sorry` becomes a challenge with a Lean verifier
    (proved: score 1, the target; final share 1). Erdős problems get the alias erdos:<n>, so the same problem from
    another source merges with it."""
    ns = re.search(r"^namespace\s+([\w.]+)", text, re.M)
    ns = ns[1] if ns else ""
    erdos = re.search(r"ErdosProblems/(\d+)\.lean$", file.replace("\\", "/"))
    pat = re.compile(r"(?:/--(?P<doc>(?:(?!-/).)*?)-/\s*)?@\[(?P<attrs>[^\]]*\bcategory\s+research\s+open\b[^\]]*)\]\s*"
                     r"theorem\s+(?P<name>[\w.'₀-₉]+)(?P<stmt>.*?):=\s*by\s+sorry", re.S)
    out = []
    for m in pat.finditer(text):
        name, stmt = m["name"], m["stmt"].strip()
        doc = re.sub(r"\s+", " ", (m["doc"] or "").strip())
        main = bool(erdos) and name == f"erdos_{erdos[1]}"
        aliases = [f"erdos:{erdos[1]}"] if main else []
        title = (f"Erdős Problem {erdos[1]}" + ("" if main else f" ({name.split('.', 1)[-1]})") if erdos
                 else name.replace("_", " "))
        out.append(normalize({
            "v": VERSION, "title": f"{title}: prove it in Lean"[:200],
            "statement": (doc + "\n\n" if doc else "") + f"theorem {name} {stmt}",
            "path": "math/proof", "key": f"lean:formal-conjectures:{file}:{ns + '.' if ns else ''}{name}",
            "aliases": aliases,
            "metric": {"name": "proved", "direction": "maximize", "baseline": 0.0, "target": 1.0, "final_share": 1.0,
                       "min_step": 1.0},
            "verifier": {"id": "lean4@formal-conjectures", "kind": "lean4",
                         "instance": {"file": file, "theorem": name, "namespace": ns, "statement": stmt}},
            "source": {"name": "formal-conjectures (Google DeepMind)",
                       "url": f"https://github.com/google-deepmind/formal-conjectures/blob/{commit}/{file}",
                       "licence": "Apache-2.0", "ref": f"{file}:{name}"}}))
    return out


def _yaml_value(v):
    v = v.strip()
    if not v:
        return None
    try:
        return json.loads(v)
    except ValueError:
        return v.strip("'\"")


def parse_erdosproblems_yaml(text):
    """The subset of YAML that teorth/erdosproblems' data/problems.yaml uses: a list of maps, one level of nested maps,
    JSON-style scalars and flow lists. Standard library only."""
    items, cur, sub = [], None, None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("- "):
            cur, sub = {}, None
            items.append(cur)
            line = "  " + line[2:]
        if cur is None:
            continue
        indent = len(line) - len(line.lstrip())
        k, _, v = line.strip().partition(":")
        if indent <= 2:
            if v.strip():
                cur[k], sub = _yaml_value(v), None
            else:
                cur[k] = sub = {}
        elif sub is not None:
            sub[k] = _yaml_value(v)
    return items


def from_erdosproblems_yaml(text, only=None):
    """Open problems from teorth/erdosproblems (data/problems.yaml, Apache-2.0): its status data only (the statements
    live on erdosproblems.com and are not copied). Each becomes a demand listing, key erdos:<n>, with no verifier of
    its own: it merges with the formal-conjectures statement of the same problem, which brings the Lean verifier."""
    out = []
    for p in parse_erdosproblems_yaml(text):
        n = str(p.get("number") or "")
        state = ((p.get("status") or {}).get("state") or "").lower()
        if not n or state != "open" or (only and n not in {str(x) for x in only}):
            continue
        tags = [t for t in (p.get("tags") or []) if isinstance(t, str)]
        out.append(normalize({
            "v": VERSION, "title": f"Erdős Problem {n}",
            "statement": (f"Erdős Problem {n} (https://www.erdosproblems.com/{n}), open. Tags: {', '.join(tags) or 'none'}. "
                          "Listed from the erdosproblems database's status data; a formal statement with a verifier "
                          "merges in from formal-conjectures where one exists."),
            "path": "math/proof", "key": f"erdos:{n}",
            "metric": {"name": "proved", "direction": "maximize", "baseline": 0.0, "target": 1.0, "final_share": 1.0,
                       "min_step": 1.0},
            "verifier": {"kind": "none"},
            "source": {"name": "erdosproblems (teorth)", "url": "https://github.com/teorth/erdosproblems",
                       "licence": "Apache-2.0", "ref": f"data/problems.yaml#{n}"}}))
    return out


# --- Yukon-style benchmark.json ------------------------------------------------------------------------------------------
YUKON_REQUIRED = ("name", "description", "category", "direction", "editablePaths", "setupCommand", "benchmarkCommand",
                  "scorePath", "runner")


def from_yukon(bench, *, baseline=None, source=None, path=None):
    """Challenges from a Yukon-style benchmark.json (schemaVersion 1, or 2 with tracks[]): one per track. Read in our
    own words from the public format: direction "+" maximizes, "-" minimizes; minScoreImprovementBips (100 = 1%)
    becomes the minimum relative step; the commands become a `command` verifier that validators run in their own
    sandbox (a node never runs submitted code). benchmark.json carries no baseline score, so pass the current best."""
    bench = json.loads(bench) if isinstance(bench, str) else bench
    if baseline is None and "baseline" not in bench:
        raise ValueError("benchmark.json carries no score: pass the best known result as the baseline (the record a "
                         "challenge starts from; pledges never pay for reaching it)")
    tracks = bench.get("tracks") if int(bench.get("schemaVersion", 1)) >= 2 else [bench]
    out = []
    for t in tracks or []:
        t = dict({k: v for k, v in bench.items() if k != "tracks"}, **t)
        missing = [k for k in YUKON_REQUIRED if k not in t]
        if missing:
            raise ValueError(f"benchmark.json is missing {missing}")
        name = str(t.get("trackName") or t.get("track") or t["name"])
        bips = t.get("minScoreImprovementBips")
        cat = re.sub(r"[^a-z]+", "_", str(t["category"]).lower()).strip("_") or "general"
        out.append(normalize({
            "v": VERSION, "title": str(t["name"]) + (f" ({name})" if name != t["name"] else ""),
            "statement": str(t["description"]),
            "path": path or f"benchmark/{cat}", "key": f"benchmark:{slug(t['name'])}:{slug(name)}",
            "metric": {"name": "score", "direction": t["direction"],
                       "baseline": float(baseline if baseline is not None else t.get("baseline", 0.0)),
                       "min_step_rel": (int(bips) / 10_000) if bips else 0.0, "min_step": 0.0,
                       "scale": abs(float(baseline or 1.0)) * 0.05 or 1.0},
            "verifier": {"id": f"command:{slug(t['name'])}:{slug(name)}@1", "kind": "command",
                         "command": {k: t[k] for k in ("editablePaths", "setupCommand", "benchmarkCommand", "scorePath",
                                                       "runner", "maxSubmissionBytes") if k in t}},
            "source": source or {"name": "benchmark.json", "format": "Yukon-style benchmark.json (schemaVersion "
                                 f"{bench.get('schemaVersion', 1)})"}}))
    return out


VERIFY_PY = '''"""Standalone verifier written by traceX for a Yukon-style benchmark: reads the submission JSON, writes
{"score": n, "metrics": {}} to the score path and exits 0, or exits 1 with no score file (invalid submission)."""
import json, math, os, sys

%s

if __name__ == "__main__":
    sub, out = sys.argv[1], sys.argv[2]
    if os.path.exists(out):
        os.remove(out)
    with open(sub, encoding="utf-8") as f:
        solution = json.load(f)
    score = %s(solution, %s)
    if score is None or not math.isfinite(score):
        sys.exit(1)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"score": score, "metrics": {}}, f)
'''


def to_yukon(c):
    """A challenge as files for a Yukon-style benchmark repository: benchmark.json (schemaVersion 1), and for a python
    verifier a standalone verify.py that keeps the score-file contract. Our own fields travel in tracex.json, beside
    it, never inside benchmark.json."""
    c = normalize(c)
    m, v = c["metric"], c["verifier"]
    cmd = dict(v.get("command") or {})
    bench = {"schemaVersion": 1, "name": slug(c["title"]), "description": c["statement"][:2000],
             "category": c["path"].replace("/", "-"), "direction": "+" if direction(m) > 0 else "-",
             "editablePaths": cmd.get("editablePaths") or ["submission/"],
             "setupCommand": cmd.get("setupCommand") or ["true"],
             "benchmarkCommand": cmd.get("benchmarkCommand") or ["python", "verify.py", "submission/solution.json",
                                                                 "score.json"],
             "scorePath": cmd.get("scorePath") or "score.json",
             "runner": cmd.get("runner") or {"provider": "github-actions", "workflow": "benchmark.yml"}}
    step = m.get("min_step_rel") or (m["min_step"] / abs(m["baseline"]) if m["baseline"] else 0)
    if step:
        bench["minScoreImprovementBips"] = max(1, int(round(step * 10_000)))
    files = {"benchmark.json": bench, "tracex.json": c}
    if v["kind"] == "python" and v["id"] in BUILTIN:
        import inspect
        from . import verifiers
        fn = BUILTIN[v["id"]]
        helpers = "\n\n".join(inspect.getsource(x) for x in (verifiers.Invalid, verifiers._finite, verifiers._rows))
        files["verify.py"] = VERIFY_PY % (helpers + "\n\n" + inspect.getsource(fn), fn.__name__,
                                          json.dumps(v["instance"]))
    return files


# --- command line -------------------------------------------------------------------------------------------------------
def load(path):
    """Challenge files from any supported source file, by its shape: a challenge (or a list), a sample sheet
    ({"challenges": [...]}), a benchmark.json, a formal-conjectures .lean file, problems.yaml, or an AlphaEvolve
    notebook (its directory names the family)."""
    text = open(path, encoding="utf-8").read()
    base = os.path.basename(path)
    if path.endswith(".lean"):
        rel = path.replace("\\", "/")
        rel = rel[rel.index("FormalConjectures/"):] if "FormalConjectures/" in rel else f"FormalConjectures/ErdosProblems/{base}"
        return from_formal_conjectures(text, file=rel)
    if path.endswith((".yaml", ".yml")):
        return from_erdosproblems_yaml(text)
    if path.endswith(".ipynb"):
        family = os.path.basename(os.path.dirname(os.path.abspath(path)))
        return [dict(x["challenge"], reference=x["reference"]) for x in from_alphaevolve_notebook(text, family)]
    obj = json.loads(text)
    if isinstance(obj, dict) and "challenges" in obj:            # a sample sheet; a `reference` solution travels along
        return [dict(normalize(x), **({"reference": x["reference"]} if "reference" in x else {})) for x in obj["challenges"]]
    if isinstance(obj, dict) and ("benchmarkCommand" in obj or "tracks" in obj):
        return from_yukon(obj)
    return [normalize(x) for x in (obj if isinstance(obj, list) else [obj])]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m traceex.challenges",
                                 description="challenge bounties: import, show and export big open problems")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="print the challenge files a source file yields")
    s.add_argument("files", nargs="+")
    i = sub.add_parser("import", help="post the challenges a source file yields (free; duplicates merge)")
    i.add_argument("files", nargs="+")
    i.add_argument("--node", required=True)
    i.add_argument("--address", required=True)
    e = sub.add_parser("export", help="write a challenge as a Yukon-style benchmark (benchmark.json, verify.py)")
    e.add_argument("id")
    e.add_argument("--node", required=True)
    e.add_argument("--out", default=".")
    a = ap.parse_args(argv)
    if a.cmd == "show":
        for f in a.files:
            print(json.dumps(load(f), indent=1, ensure_ascii=False))
        return 0
    from .client import Client
    if a.cmd == "import":
        c = Client(a.node, a.address)
        for f in a.files:
            for ch in load(f):
                r = c.post_challenge(ch)
                print(f"{ch['key']}: #{r['id']} {'merged' if r.get('merged') else r.get('status')}")
        return 0
    files = Client(a.node).challenge_export(a.id)["files"]
    os.makedirs(a.out, exist_ok=True)
    for name, body in files.items():
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as f:
            f.write(body if isinstance(body, str) else json.dumps(body, indent=2, ensure_ascii=False))
        print("wrote", os.path.join(a.out, name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
