"""The scrubber: what of an Agent Beacon session may go into a trace (mandatory before export).

Beacon records whole agent sessions: prompts, command output, diffs, file reads. Anything a person pasted or an agent
printed is in there, keys included. Every string the exporter takes from a Beacon event goes through `Scrubber.scrub`
first, and the finished trace goes through `second_pass`; if any secret pattern survives, the trace is refused.

What is replaced (the bracketed label is what the text then holds; values are never kept, not even in the report):
  secrets
    * private key blocks (`-----BEGIN ... PRIVATE KEY-----` to its END line, or to the end of the text)
    * provider keys and tokens: `sk-...` (OpenAI, Anthropic `sk-ant-`), Stripe `sk_live_ / rk_ / pk_live_`, GitHub
      `ghp_ gho_ ghu_ ghs_ ghr_ github_pat_`, GitLab `glpat-`, AWS access key ids (`AKIA...`, `ASIA...`) and secret
      access keys, Google `AIza...`, Slack `xox?-`, Hugging Face `hf_`, npm `npm_`, SendGrid `SG.`, Shopify `shp??_`,
      fal-style `<uuid>:<hex>` keys, JWTs
    * `Authorization: Bearer|Key|Basic|Token <value>` headers, bare `Bearer <value>`, `x-api-key: <value>`
    * credentials in URLs (`scheme://user:pass@host`) and secret-named query parameters (`?token=`, `&api_key=`)
    * assignments, JSON members and flags whose name says secret (`FAL_KEY=...`, `"password": "..."`, `--token x`,
      `export CLOUDFLARE_API_TOKEN=...`); a value that is plainly a reference, not a secret (`os.environ[...]`,
      `$VAR`, `process.env.X`, `None`, a function call) is left alone
    * every value in a `.env` file's content (a diff of, or a read from, `.env`, `.env.local`, `prod.env`, ...)
    * high-entropy strings: 24+ characters of base64/base62 with upper case, lower case and digits and Shannon
      entropy of at least 4.0 bits per character; 32+ hex characters unless introduced as a hash (`sha256:`,
      `commit`, `digest`, ...)
  personal data
    * email addresses, phone numbers (North American and +international), street addresses (`742 Evergreen
      Terrace`, `PO Box 12`, with an optional unit, city, state and ZIP)
    * home directories: `C:\\Users\\<name>`, `C:/Users/<name>`, `/c/Users/<name>`, `/Users/<name>`, `/home/<name>`,
      `/root` become `~`; the user and host names the session reports (and this machine's) are removed anywhere
  paths
    * an absolute path inside the project (the session's working directory) becomes relative: `./src/app.py`
    * any other absolute file-system path is redacted (`~/[REDACTED_PATH]` under a home directory,
      `[REDACTED_PATH]` elsewhere); a third-party library frame keeps only its package part:
      `<site-packages>/requests/models.py`

`Scrubber.report()` counts replacements by kind and by field. It never holds a matched value.
Standard library only.
"""
import math
import os
import re
import socket

# --- secrets -------------------------------------------------------------------------------------------------------

# (kind, regex, label). Specific patterns first, so a key is named for what it is.
TOKENS = [
    ("private_key", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----[\s\S]*?"
                               r"(?:-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----|\Z)"), "PRIVATE_KEY"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "JWT"),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}"), "API_KEY"),
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-|admin-|live-|test-)?[A-Za-z0-9_-]{20,}"), "API_KEY"),
    ("stripe_key", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}"), "API_KEY"),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{22,})"), "TOKEN"),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}"), "TOKEN"),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[0-9A-Z]{16}\b"), "AWS_KEY"),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}"), "API_KEY"),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"), "TOKEN"),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "TOKEN"),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{30,}"), "TOKEN"),
    ("sendgrid_key", re.compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}"), "API_KEY"),
    ("shopify_token", re.compile(r"\bshp(?:at|ss|ca|pa)_[A-Fa-f0-9]{24,}"), "TOKEN"),
    ("fal_key", re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}:[0-9a-f]{24,}\b", re.I),
     "API_KEY"),
]

# (kind, regex): group 1 is kept (the name, the scheme), the last group is the value that goes.
VALUED = [
    ("aws_secret_key", re.compile(r"(?i)(\baws_?secret_?access_?key[\"']?\s*[:=]\s*[\"']?)([A-Za-z0-9/+=]{40})")),
    ("auth_header", re.compile(r"(?i)(\bauthorization[\"']?\s*[:=]\s*[\"']?\s*(?:(?:bearer|key|basic|token|bot|digest)"
                               r"\s+)?)([A-Za-z0-9._~+/:=-]{8,})")),
    ("auth_header", re.compile(r"(?i)(\bbearer\s+)([A-Za-z0-9._~+/-]{16,}=*)")),
    ("auth_header", re.compile(r"(?i)(\b(?:x-api-key|api-key|x-auth-token|x-access-token|cf-access-client-secret|"
                               r"x-goog-api-key)[\"']?\s*[:=]\s*[\"']?)([^\s\"',;]{8,})")),
    ("url_credentials", re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://[^\s/:@\"']+:)([^\s/@\"']+)(?=@)")),
    ("url_token", re.compile(r"(?i)([?&](?:token|access_token|refresh_token|id_token|api_?key|key|secret|password|"
                             r"sig|signature|auth|code|client_secret)=)([^&\s\"'#]+)")),
    # NAME = value, NAME: value, "name": "value", NAME := value, name => value. Whether NAME says secret is decided
    # by secret_name(); whether the value is a real value (not a reference) by real_value().
    ("secret_assignment", re.compile(r"(\b[A-Za-z_][A-Za-z0-9_.-]*[\"']?\s*(?::=|=>|:|=)\s*)"
                                     r"(\"[^\"\n]*\"|'[^'\n]*'|`[^`\n]*`|[^\s\"',;)}\]]+)")),
    ("secret_flag", re.compile(r"(?i)((?:^|(?<=\s))--?[a-z0-9]+(?:-[a-z0-9]+)*(?:=|\s+))(\"[^\"\n]*\"|'[^'\n]*'|"
                               r"[^\s\"'-][^\s\"']*)")),
]
LABEL = {"aws_secret_key": "SECRET", "auth_header": "AUTH", "url_credentials": "SECRET", "url_token": "SECRET",
         "secret_assignment": "SECRET", "secret_flag": "SECRET"}

_SECRET_WORDS = {"password", "passwd", "pwd", "pass", "passphrase", "secret", "secrets", "token", "apikey",
                 "credential", "credentials", "dsn", "authorization", "cookie", "privatekey", "accesskey"}
_KEY_QUALIFIERS = {"api", "access", "private", "secret", "signing", "client", "auth", "encryption", "master",
                   "service", "account", "app", "license", "session", "webhook", "publishable", "admin", "deploy",
                   "ssh", "gpg", "pgp", "rsa", "fal", "openai", "anthropic", "stripe", "aws", "gcp", "azure"}


def _words(name):
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip("\"'"))
    return [w for w in re.split(r"[_.\-]+", name.lower()) if w]


def secret_name(name):
    """True when an assignment's or flag's name says it holds a secret: FAL_KEY, db_password, apiKey, x-api-token,
    CLOUDFLARE_API_TOKEN, client_secret, --password. Not: cache_key, sort_key, max_tokens, tokenizer, author."""
    raw = name.strip("\"'-")
    words = _words(raw)
    if not words:
        return False
    if any(w in _SECRET_WORDS for w in words):
        return True
    joined = "".join(words)
    if any(s in joined for s in ("apikey", "accesskey", "privatekey", "secretkey", "authtoken", "accesstoken",
                                 "connectionstring", "databaseurl", "dburl")):
        return True
    if words[-1] == "key" and len(words) > 1:
        # env-var style FAL_KEY / STRIPE_KEY, or a qualified key: api_key, signingKey
        return raw.isupper() or words[-2] in _KEY_QUALIFIERS
    return False


_REFERENCE = re.compile(r"""^(?:None|null|nil|true|false|True|False|undefined|""|''|``|[\d.]+|
                            \$\{?[A-Za-z_]\w*\}?|%[A-Za-z_]\w*%|\{[A-Za-z_]+(?:_\d+)?\}?|
                            (?:os\.environ|os\.getenv|process\.env|env|ENV|config|settings|secrets|self|this|cfg|
                               import\.meta\.env)\b.*|[A-Za-z_][\w.]*\(.*|[A-Za-z_][\w.]*\[.*|
                            <[^>]*>|\[REDACTED_[A-Z_]+\]|.*\[REDACTED_[A-Z_]+\].*)$""", re.X)


def real_value(value):
    """False for a value that is a reference or a placeholder, not a secret: os.environ["X"], $FAL_KEY, ${TOKEN},
    process.env.KEY, getpass(), None, a number, <your-key>, an already-redacted value, a {STR_1} skeleton slot."""
    v = value.strip()
    if "REDACTED_" in v:
        return False
    inner = v[1:-1] if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'`" else v
    if not inner.strip():
        return False
    return not (_REFERENCE.match(v) or _REFERENCE.match(inner))


HASH_CONTEXT = re.compile(r"(?i)(?:sha(?:1|224|256|384|512)?|md5|blake2b?|hash|digest|commit|rev(?:ision)?|object|"
                          r"tree|parent|merge|checksum|etag|integrity|fingerprint|oid|index)[\s:=\"'#@-]*$")
HIGH = re.compile(r"(?<![A-Za-z0-9_+/-])[A-Za-z0-9_+/-]{24,}={0,2}(?![A-Za-z0-9_+/=-])")
HEX = re.compile(r"(?<![A-Za-z0-9])[0-9a-fA-F]{32,}(?![A-Za-z0-9])")


def entropy(s):
    """Shannon entropy in bits per character."""
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def looks_random(tok):
    """A 24+ character token that reads as a key, not as a word, an identifier or a path."""
    core = tok.strip("=")
    if len(core) < 24 or "REDACTED_" in core:
        return False
    if "/" in core and all(len(p) < 20 for p in core.split("/")):
        return False                                   # a path: src/components/Header
    if re.fullmatch(r"[0-9a-fA-F-]+", core):
        return False                                   # hex (and UUIDs) are judged by HEX, with their hash context
    if not (any(c.isdigit() for c in core) and any(c.isupper() for c in core) and any(c.islower() for c in core)):
        return False                                   # snake_case and camelCase identifiers, words
    parts = [p for p in re.split(r"[_\-]", core) if p]
    if len(parts) > 2 and all(re.fullmatch(r"[A-Za-z]+\d{0,3}", p) for p in parts):
        return False                                   # test_parse_v2_handles_Empty_input
    if re.fullmatch(r"(?:[A-Z]?[a-z]+|[A-Z]+|\d{1,3})+", core) and len(re.findall(r"\d", core)) <= 3:
        return False                                   # CamelCaseIdentifierWithAVersion2
    return entropy(core) >= 4.0


def _hex_is_hash(text, start):
    return bool(HASH_CONTEXT.search(text[max(0, start - 24):start]))


# --- personal data -------------------------------------------------------------------------------------------------

EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<![\w+])(?:(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}|"
                   r"\+\d{1,3}(?:[\s.-]?\(?\d{1,4}\)?){2,5}\d)(?!\w)")
STREET_TYPES = r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|Terrace|Ter|" \
               r"Parkway|Pkwy|Circle|Cir|Highway|Hwy|Trail|Trl|Square|Sq|Loop|Alley|Crescent|Close)"
ADDRESS = re.compile(
    rf"\b\d{{1,6}}[A-Za-z]?\s+(?:[NSEW]\.?\s+)?(?:[A-Z][A-Za-z'-]*\s+){{1,4}}{STREET_TYPES}\b\.?"
    r"(?:,?\s*(?:Apt|Apartment|Suite|Ste|Unit|#)\.?\s*[A-Za-z0-9-]+)?"
    r"(?:,\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})?(?:,?\s*[A-Z]{2}\b)?(?:\s+\d{5}(?:-\d{4})?)?"
    r"|\bP\.?\s?O\.?\s+Box\s+\d+\b")


def _phone_ok(m):
    digits = re.sub(r"\D", "", m.group(0))
    if m.group(0).startswith("+") and not re.search(r"[\s.()-]", m.group(0)):
        return len(digits) >= 11                       # +15125550142, not +2024 (a diff line adding 2024)
    return 10 <= len(digits) <= 15


# Home directories. A Windows user name may hold a space ("C:\Users\Jordan Example\..."): two words are taken only
# when a path separator follows them, so "C:\Users\bob is" becomes "~ is".
_N = r"[^\\/\s\"'<>|:*?`]+"
_HOME_NAME = rf"(?:{_N} {_N}(?=[\\/])|{_N})"
_SEP = r"(?:\\\\|\\|/)+"
HOME_WIN = re.compile(rf"(?i)(?:\b[a-z]:{_SEP}|(?<![\w.~/])/[a-z]/)users{_SEP}{_HOME_NAME}")
HOME_POSIX = re.compile(rf"(?<![\w.~/-])/(?:Users|home|export/home)/{_HOME_NAME}|(?<![\w.~/-])/root(?![\w.-])")

# Absolute paths left after the home collapse. A POSIX path counts only when it starts at a file-system root
# (/usr, /tmp, /var, /opt, /mnt, ...), so URL routes in code (fetch("/api/users")) are kept.
_PATHCH = r"[^\s\"'<>|*?`,;:)\]]"
WIN_PATH = re.compile(rf"(?i)\b[a-z]:(?:\\\\|\\|/){_PATHCH}*")
UNC_PATH = re.compile(r"\\\\[A-Za-z0-9_.$-]+\\[^\s\"'<>|*?`]+")
_ROOTS = r"(?:usr|tmp|var|etc|opt|mnt|media|srv|private|Volumes|System|Library|Applications|workspaces?|proc|sys|" \
         r"dev|run|snap|nix|bin|sbin|lib|lib64|data|app|code|builds|github|__w|cygdrive|[a-z])"
POSIX_PATH = re.compile(rf"(?<![\w:/.~\]-])/{_ROOTS}(?:/[\w.@+%=-]+)+/?")
HOME_PATH = re.compile(rf"(?<![\w/])~[\\/]{_PATHCH}+")
LIBRARY = re.compile(r"(?:site-packages|dist-packages|node_modules|\.cargo/registry/src|go/pkg/mod)/(.+)$")

ENV_FILE = re.compile(r"(?:^|[\\/])(?:\.env(?:\.[\w.-]+)?|[\w.-]+\.env|\.envrc|\.netrc|\.npmrc|\.pypirc|"
                      r"credentials(?:\.json)?|secrets?\.(?:json|ya?ml|toml))$", re.I)
ENV_LINE = re.compile(r"(?m)^([+\- ]?\s*(?:export\s+)?[A-Za-z_][A-Za-z0-9_.-]*\s*[=:]\s*)(\S.*?)\s*$")


class ScrubError(ValueError):
    """A secret pattern survived the second pass: the trace must not be exported."""

    def __init__(self, hits):
        super().__init__(f"secret patterns survived scrubbing: {sorted({k for k, _ in hits})}")
        self.hits = hits


def _norm(p):
    return re.sub(r"(?:\\\\|\\|/)+", "/", p).rstrip("/")


def _node_secrets():
    """The node's own secret detectors (traceex.skeleton.SECRETS), so whatever the node would refuse, pass 1 removes."""
    try:
        from traceex.skeleton import SECRETS
        return SECRETS
    except ImportError:
        return []


def _secret_hits(text):
    """(kind, start, end, keep_prefix) for every secret in text, earliest first, without overlaps."""
    spans = []
    for kind, rx, _ in TOKENS:
        spans += [(m.start(), m.end(), kind, "") for m in rx.finditer(text)]
    for kind, rx in VALUED:
        for m in rx.finditer(text):
            value = m.group(2)
            if kind == "secret_assignment":
                name = m.group(1).rstrip(" \t:=>\"'")
                if not secret_name(name) or not real_value(value):
                    continue
            elif kind == "secret_flag":
                flag = m.group(1).strip().rstrip("=")
                if not secret_name(flag.lstrip("-")) or not real_value(value):
                    continue
            elif "REDACTED_" in value or not value.strip("\"'` "):
                continue
            spans.append((m.start(2), m.end(2), kind, ""))
    for kind, rx in _node_secrets():
        for m in rx.finditer(text):
            q = re.search(r"['\"]", m.group(0)) if kind == "password" else None
            spans.append((m.start() + (q.start() if q else 0), m.end(), "node_" + kind, ""))
    for m in HEX.finditer(text):
        if not _hex_is_hash(text, m.start()):
            spans.append((m.start(), m.end(), "high_entropy", ""))
    for m in HIGH.finditer(text):
        if looks_random(m.group(0)):
            spans.append((m.start(), m.end(), "high_entropy", ""))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out, end = [], -1
    for s in spans:
        if s[0] >= end:
            out.append(s)
            end = s[1]
    return out


def _label(kind):
    for k, _, lab in TOKENS:
        if k == kind:
            return f"[REDACTED_{lab}]"
    if kind == "high_entropy":
        return "[REDACTED_HIGH_ENTROPY]"
    return f"[REDACTED_{LABEL.get(kind, 'SECRET')}]"


class Scrubber:
    """One scrubber per trace: it counts what it replaced, by kind and by field, for that trace's report.

    project: the session's working directory (absolute). Paths inside it become relative; it may be None.
    names:   extra terms to remove wherever they appear: the user and host names the session reports (user.name,
             endpoint.hostname). This machine's own user and host names are always added."""

    COMMON = {"user", "users", "root", "admin", "home", "runner", "ubuntu", "local", "localhost", "test", "dev",
              "local-user", "codespace", "vscode", "node", "app", "default", "public", "shared", "none", "unknown"}

    def __init__(self, project=None, names=()):
        self.project = _norm(project) if project else None
        terms = {str(n) for n in names if n}
        if self.project:                    # the user name in the project's own path: C:/Users/<name>/proj
            for rx in (HOME_WIN, HOME_POSIX):
                m = rx.search(self.project)
                if m:
                    terms.add(re.split(r"[\\/]", m.group(0))[-1])
        for env in ("USERNAME", "USER", "LOGNAME"):
            terms.add(os.environ.get(env, ""))
        terms.add(os.path.basename(os.path.expanduser("~")))
        try:
            terms.add(socket.gethostname())
        except OSError:
            pass
        # Short or generic names would redact ordinary words: a name must be 3+ characters and not a common word.
        self.terms = sorted((t for t in terms if len(t) >= 3 and t.lower() not in self.COMMON), key=len, reverse=True)
        self.counts, self.fields, self._field = {}, {}, None

    def _hit(self, kind, n=1):
        if n <= 0:
            return
        self.counts[kind] = self.counts.get(kind, 0) + n
        if self._field:
            f = self.fields.setdefault(self._field, {})
            f[kind] = f.get(kind, 0) + n

    def report(self):
        """Counts only, by kind and by field: what was replaced, never what it was."""
        return {"replacements": dict(sorted(self.counts.items())), "total": sum(self.counts.values()),
                "by_field": {k: dict(sorted(v.items())) for k, v in sorted(self.fields.items())}}

    def scrub(self, text, field="text", path=None):
        """Scrubbed copy of text. path: the file the text came from, if any (a .env file's values all go)."""
        if text is None:
            return None
        text = str(text)
        self._field = field
        if path and ENV_FILE.search(_norm(str(path))):
            text = self._env(text)
        text = self._secrets(text)
        text = self._personal(text)
        text = self._project_paths(text)
        text = self._terms(text)
        text = self._other_paths(text)
        self._field = None
        return text

    def _env(self, text):
        def sub(m):
            if m.group(2).startswith("[REDACTED_"):
                return m.group(0)
            self._hit("env_value")
            return m.group(1) + "[REDACTED_SECRET]"
        return ENV_LINE.sub(sub, text)

    def _secrets(self, text):
        out, last = [], 0
        for start, end, kind, _ in _secret_hits(text):
            out.append(text[last:start] + _label(kind))
            self._hit(kind)
            last = end
        return "".join(out) + text[last:]

    def _personal(self, text):
        text, n = EMAIL.subn("[REDACTED_EMAIL]", text)
        self._hit("email", n)
        text, n = ADDRESS.subn("[REDACTED_ADDRESS]", text)
        self._hit("address", n)

        def phone(m):
            if not _phone_ok(m):
                return m.group(0)
            self._hit("phone")
            return "[REDACTED_PHONE]"
        return PHONE.sub(phone, text)

    def _project_paths(self, text):
        # 1. inside the project: relative. Matched on any spelling: \, \\ or /, any case.
        if self.project:
            parts = [re.escape(p) for p in self.project.split("/")]
            rx = re.compile(r"(?i)" + _SEP.join(parts) + rf"(?:{_SEP}({_PATHCH}*))?(?![\w.-])")

            def rel(m):
                self._hit("project_path")
                rest = _norm(m.group(1) or "")
                return "./" + rest if rest else "."
            text = rx.sub(rel, text)
            # git bash spelling of the same folder: /c/Users/x/proj for C:\Users\x\proj
            drive = re.match(r"(?i)^([a-z]):/(.*)$", self.project)
            if drive:
                parts = [re.escape(p) for p in ("/" + drive.group(1).lower() + "/" + drive.group(2)).split("/")[1:]]
                rx = re.compile(r"(?i)/" + "/".join(parts) + rf"(?:/({_PATHCH}*))?(?![\w.-])")
                text = rx.sub(rel, text)
        return text

    def _other_paths(self, text):
        # 2. home directories collapse to ~
        for rx in (HOME_WIN, HOME_POSIX):
            text, n = rx.subn("~", text)
            self._hit("home_dir", n)

        # 3. any other absolute path: redacted, keeping only a third-party library's package-relative part
        def other(m):
            p = m.group(0)
            if "REDACTED_PATH" in p:
                return p
            lib = LIBRARY.search(_norm(p))
            if lib:
                self._hit("library_path")
                return "<site-packages>/" + lib.group(1)
            self._hit("outside_path")
            return "~/[REDACTED_PATH]" if p.startswith("~") else "[REDACTED_PATH]"
        for rx in (UNC_PATH, WIN_PATH, HOME_PATH, POSIX_PATH):
            text = rx.sub(other, text)
        return text

    def _terms(self, text):
        for t in self.terms:
            text, n = re.subn(rf"(?i)(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])", "[REDACTED_USER]", text)
            self._hit("user_or_host", n)
        return text


# --- the second pass -----------------------------------------------------------------------------------------------

def find_secrets(text):
    """Every secret pattern still present in text, as (kind, length) pairs (never the value)."""
    return [(kind, end - start) for start, end, kind, _ in _secret_hits(text or "")]


def second_pass(text):
    """Raise ScrubError if any secret pattern survives in text. Runs this module's patterns and traceex's own
    node-side secret detectors (traceex.skeleton.find_secrets), so a trace that passes here passes the node."""
    hits = find_secrets(text)
    try:
        from traceex.skeleton import find_secrets as node_secrets
        hits += [(k, len(v)) for k, v in node_secrets(text)]
    except ImportError:                                 # traceex not on the path: this module's patterns alone
        pass
    if hits:
        raise ScrubError(hits)
    return True
