"""Code skeletons: what of a code change may leave the machine (used by the pytest plugin, traceex.pytest_plugin).

Source code never leaves verbatim. A skeleton keeps the *structure* of a fix and drops everything specific to a project:
  * string literals, f-strings and bytes  -> {STR_n}   (all of them: they hold paths, messages, keys, data)
  * comments                              -> removed
  * identifiers                           -> {ID_n}, unless they are Python keywords, builtins, standard-library module
                                             names, common methods of built-in types, unittest/pytest vocabulary or
                                             dunder names, which are kept (they carry the structure: len, range,
                                             append, isinstance, assertEqual, pytest.raises)
  * numbers                               -> kept when they are small integers (-10..10: off-by-one fixes stay visible),
                                             otherwise {NUM_n}
  * file paths                            -> {DIR}/ plus the file name's skeleton
Placeholders are consistent within one skeleton (the same name is the same {ID_n} in the old code, the new code, the
test and the failure message), so the diff still reads. The lexer is a tolerant regex scanner, not Python's tokenizer,
so code that does not parse (a syntax error is a common failure) is skeletonized the same way.
"""
import ast
import builtins
import difflib
import keyword
import re
import sys
import unittest

_STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"os", "sys", "re", "json", "math", "time", "datetime",
                                                         "collections", "itertools", "functools", "typing", "pathlib"}
_METHODS = set()
for _t in (str, bytes, list, dict, set, frozenset, tuple, int, float, complex, object, BaseException):
    _METHODS |= {n for n in dir(_t) if not n.startswith("_")}
_TEST_WORDS = {n for n in dir(unittest.TestCase) if not n.startswith("_")} | {
    "pytest", "raises", "approx", "fixture", "mark", "parametrize", "skip", "skipif", "xfail", "mock", "patch",
    "monkeypatch", "tmp_path", "capsys", "caplog", "request", "unittest", "TestCase", "self", "cls", "args", "kwargs",
    "test", "tests", "setup", "teardown", "expected", "actual", "result", "results", "value", "values", "data", "item",
    "items", "key", "keys", "index", "count", "total", "name", "names", "path", "line", "lines", "text", "msg", "error",
    "errors", "exc", "e", "i", "j", "k", "n", "x", "y", "z", "a", "b", "c", "s", "t", "v", "f", "fn", "func", "obj",
    "out", "res", "ret", "tmp", "num", "nums", "lst", "arr", "start", "end", "left", "right", "mid", "low", "high",
    "first", "last", "prev", "next", "node", "root", "head", "tail", "size", "length", "width", "height", "row", "col",
    "rows", "cols", "acc", "sum", "max", "min", "default", "config", "options", "client", "response", "status", "body",
    "where", "and", "or", "not", "is", "in", "assert", "where", "got", "but", "should", "when", "then", "returns",
    "return", "raise", "empty", "none", "valid", "invalid", "list", "dict", "string", "int", "float", "number", "of",
    "the", "to", "at", "with", "from", "by", "for", "on", "if", "else", "new", "old", "get", "set", "add", "remove",
    "update", "create", "delete", "parse", "load", "save", "read", "write", "open", "close", "check", "compare",
    "sort", "sorted", "reverse", "merge", "split", "join", "find", "search", "match", "replace", "format", "convert",
    "encode", "decode", "run", "call", "apply", "map", "filter", "reduce", "init", "main", "helper", "util", "utils"}
KEEP = (set(keyword.kwlist) | set(getattr(keyword, "softkwlist", ())) | set(dir(builtins)) | _STDLIB | _METHODS |
        _TEST_WORDS)
SMALL = 10
_LEX = re.compile(r"""
    (?P<comment>\#[^\n]*)
  | (?P<string>(?:[rRbBuUfF]{0,2})(?:'''[\s\S]*?(?:'''|$)|\"\"\"[\s\S]*?(?:\"\"\"|$)|'(?:\\.|[^'\\\n])*'?|"(?:\\.|[^"\\\n])*"?))
  | (?P<number>\b\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?[jJ]?\b|\b0[xXoObB][0-9a-fA-F_]+\b)
  | (?P<name>[^\W\d]\w*)
  | (?P<other>[\s\S])
""", re.X)
PLACEHOLDER = re.compile(r"\{(?:ID|STR|NUM|PATH)_\d+\}|\{DIR\}")


class CodeSkeleton:
    """One skeleton's placeholder table: share an instance across every piece of one trace."""

    def __init__(self):
        self.mapping, self.counters = {}, {}

    def _ph(self, kind, value):
        key = (kind, value)
        if key not in self.mapping:
            self.counters[kind] = self.counters.get(kind, 0) + 1
            self.mapping[key] = f"{{{kind}_{self.counters[kind]}}}"
        return self.mapping[key]

    def name(self, word):
        if word in KEEP or (word.startswith("__") and word.endswith("__")) or PLACEHOLDER.fullmatch(word):
            return word
        if word.startswith("test_") and len(word) > 5:          # a test's name keeps its shape: test_{ID_1}_empty
            return self.words(word)
        return self._ph("ID", word)

    def string(self, tok):
        """A string literal's placeholder, keyed on its value: 'NONE' in a message and "NONE" in code are one {STR_n}."""
        try:
            key = repr(ast.literal_eval(tok))
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            key = tok
        return self._ph("STR", key)

    def code(self, src):
        """Skeleton of source code (or of any code-like text: an assertion message, a traceback line)."""
        out = []
        for m in _LEX.finditer(src or ""):
            kind = m.lastgroup
            tok = m.group(0)
            if kind == "comment":
                continue
            if kind == "string":
                out.append(self.string(tok))
            elif kind == "number":
                try:
                    small = abs(int(tok.replace("_", ""), 0)) <= SMALL
                except ValueError:
                    small = False
                out.append(tok if small else self._ph("NUM", tok))
            elif kind == "name":
                out.append(self.name(tok))
            else:
                out.append(tok)
        text = "".join(out)
        return "\n".join(line.rstrip() for line in text.split("\n"))

    def words(self, name):
        """A name made of words (a test's name): each word kept or replaced on its own, 'test_{ID_1}_empty'."""
        out = []
        for w in name.split("_"):
            if not w or w in KEEP or PLACEHOLDER.fullmatch(w):
                out.append(w)
            else:
                out.append(self._ph("ID", w))
        return "_".join(out)

    def path(self, path):
        """A file path: the directory dropped, the file name's words skeletonized, the extension kept."""
        base = re.split(r"[\\/]", str(path))[-1]
        stem, dot, ext = base.rpartition(".")
        if not dot:
            stem, ext = base, ""
        return "{DIR}/" + self.words(stem) + (f".{ext}" if ext else "")

    def diff(self, old, new, label="", context=2, max_lines=80):
        """A unified diff of two versions of one file, both skeletonized with this table: only the hunks."""
        a, b = self.code(old).splitlines(), self.code(new).splitlines()
        lines = [ln for ln in difflib.unified_diff(a, b, lineterm="", n=context)][2:]   # no ---/+++ headers
        lines = ["@@" if ln.startswith("@@") else ln for ln in lines if ln.strip(" +-")]   # no line numbers, no blanks
        if label:
            lines.insert(0, f"@@ {self.path(label)} @@")
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"… {len(lines) - max_lines} more diff lines not shown"]
        return lines


def changed_sides(diff_lines):
    """(removed, added) code lines of a skeleton diff, without the +/- markers."""
    removed = [ln[1:] for ln in diff_lines if ln.startswith("-")]
    added = [ln[1:] for ln in diff_lines if ln.startswith("+")]
    return removed, added
