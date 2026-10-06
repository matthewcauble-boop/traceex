"""Countdown as TROPIC runs it: one exact binary operation per step over the live numbers, deterministic and replayable.

The generator is a port of `ragen/env/countdown/generate.py` and the step semantics of
`ragen/env/countdown/tropic_env.py` from the TROPIC reference code (MIT; see NOTICE): n in {3, 4, 5, 6} inputs from
[1, 20], a target in [1, 100] reachable by combining every input exactly once with + - * / through integer
intermediates (a witness is kept), exact rational arithmetic, fractions and negatives allowed, division by zero
invalid, success when the one remaining number is the target. The state is the target and the sorted multiset of
live numbers (commuting operation orders reach the same state, as in TROPIC's snapshot key); the step count is the
graph's depth.
"""
import random
import re
from fractions import Fraction

INPUT_RANGE = (1, 20)
TARGET_RANGE = (1, 100)
SIZES = (3, 4, 5, 6)
MAX_STEPS = 5
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
       "/": lambda a, b: None if b == 0 else a / b}
ALIASES = {"x": "*", "×": "*", "÷": "/"}
NUM = r"(-?\d+(?:/\d+)?)"
ACTION = re.compile(NUM + r"\s*([+\-*/x×÷])\s*" + NUM)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


def fmt(v):
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


# --- problems: TROPIC's generator ----------------------------------------------------------------------------------
def _random_witness(rng, n):
    values = [rng.randint(*INPUT_RANGE) for _ in range(n)]
    live = [Fraction(v) for v in values]
    steps = []
    while len(live) > 1:
        i, j = rng.sample(range(len(live)), 2)
        a, b = live[i], live[j]
        options = [("+", a + b), ("-", a - b), ("*", a * b)]
        if b != 0 and (a / b).denominator == 1:
            options.append(("/", a / b))
        op, result = rng.choice(options)
        if result.denominator != 1 or abs(result) > 10_000:
            return None
        steps.append(f"{a} {op} {b}")
        live = [v for k, v in enumerate(live) if k not in (i, j)] + [result]
    target = live[0]
    if not TARGET_RANGE[0] <= target <= TARGET_RANGE[1]:
        return None
    return {"target": int(target), "nums": values, "solution": steps}


def generate(seed, per_size, sizes=SIZES, exclude=()):
    rng = random.Random(seed)
    seen = {(tuple(sorted(row["nums"])), row["target"]) for row in exclude}
    rows = []
    for n in sizes:
        count = 0
        while count < per_size:
            row = _random_witness(rng, n)
            if row is None:
                continue
            key = (tuple(sorted(row["nums"])), row["target"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            count += 1
    rng.shuffle(rows)
    return rows


def problems(per_size, seed=2026):
    """TROPIC's training split: validation drawn first with `seed`, training with `seed + 1` excluding it."""
    val = generate(seed, 128)
    return generate(seed + 1, per_size, exclude=val)


# --- the step environment -----------------------------------------------------------------------------------------
def root(problem):
    return {"target": int(problem["target"]), "nums": sorted((fmt(v) for v in problem["nums"]), key=Fraction)}


def render(state):
    """What the agent sees: the target and the live numbers only (TROPIC's single-turn observation)."""
    return (f"Target: {state['target']}\nNumbers: [{', '.join(state['nums'])}]\n"
            f"Operations left before every number is used: {len(state['nums']) - 1}")


def parse(text):
    """The one operation an answer names, as 'a op b', or None. The text inside the last <answer>...</answer> if there
    is one, else the whole reply; the first 'number op number' in it. (TROPIC's strict parser ends an episode on any
    extra text; an untrained 0.5B model rarely answers that cleanly, so this is more lenient, identically for every
    agent.)"""
    found = ANSWER.findall(text or "")
    m = ACTION.search(found[-1] if found else (text or ""))
    if not m:
        return None
    a, op, b = m.groups()
    try:
        return f"{fmt(Fraction(a))} {ALIASES.get(op, op)} {fmt(Fraction(b))}"
    except (ValueError, ZeroDivisionError):
        return None


def step(state, action):
    """The next state, or None for an invalid action (a number not live, division by zero, malformed)."""
    m = re.fullmatch(NUM + r" ([+\-*/]) " + NUM, str(action))
    if not m or len(state["nums"]) < 2:
        return None
    a, op, b = Fraction(m[1]), m[2], Fraction(m[3])
    live = [Fraction(v) for v in state["nums"]]
    if a not in live:
        return None
    live.remove(a)
    if b not in live:
        return None
    live.remove(b)
    result = OPS[op](a, b)
    if result is None:
        return None
    return {"target": state["target"], "nums": sorted((fmt(v) for v in live + [result]), key=Fraction)}


def check(state):
    return len(state["nums"]) == 1 and Fraction(state["nums"][0]) == state["target"]


def alive(state):
    return len(state["nums"]) >= 2


def expression_check(problem, actions):
    """An independent verifier: fold the actions into one expression over the original inputs, then evaluate it. It
    must use every input exactly once and equal the target."""
    live = {}
    for v in problem["nums"]:
        live.setdefault(Fraction(v), []).append((str(v), Fraction(v)))
    for act in actions:
        a, op, b = act.split(" ")
        a, b = Fraction(a), Fraction(b)
        if not live.get(a):
            return False
        ea = live[a].pop()
        if not live.get(b):
            return False
        eb = live[b].pop()
        value = OPS[op](ea[1], eb[1])
        if value is None:
            return False
        live.setdefault(value, []).append((f"({ea[0]} {op} {eb[0]})", value))
    rest = [x for xs in live.values() for x in xs]
    if len(rest) != 1:
        return False
    expr = rest[0][0]
    used = sorted(int(t) for t in re.findall(r"\d+", expr))
    if used != sorted(int(v) for v in problem["nums"]):
        return False
    value = eval(re.sub(r"(\d+)", r"Fraction(\1)", expr), {"Fraction": Fraction, "__builtins__": {}})
    return value == problem["target"]
