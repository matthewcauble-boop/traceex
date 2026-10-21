# Agent Beacon -> traceX: field mapping

[Agent Beacon](https://github.com/Asymptote-Labs/agent-beacon) (Asymptote Labs, MIT) records coding-agent sessions
(Claude Code, Cursor, Codex, OpenCode, ...) as normalized JSONL events in `~/.beacon/endpoint/logs/runtime.jsonl`
(schema_version `1.0`). This adapter turns the **fix episodes** in that log into `trace/0.1` objects, scrubbed on this
machine, and writes them to the local outbox. It reads the log only. It never runs Beacon or talks to Beacon Cloud, and
it sends nothing unless you pass `--submit`.

```
python -m adapters.beacon ~/.beacon/endpoint/logs/runtime.jsonl            # dry run -> .traceex/outbox/
PYTHONPATH=sdk/python python -m traceex.outbox list                         # review
python -m adapters.beacon runtime.jsonl --submit --node https://<node> --address 0x...   # only when you mean it
```

## 1. Reading the log

| Beacon | Use |
|---|---|
| one JSON object per line, `vendor: beacon`, `event.kind: agent_runtime` | anything else is skipped and counted |
| `timestamp`, then `sequence` | event order (never file position: the hook and collector paths append at different times) |
| `event.id` | the hook's and the collector's reports of one action share it: merged into one event |
| `session.id` (else `gen_ai.conversation.id`, else `session.working_directory`, else `run.run_id`), with `harness.name` | the session an event belongs to; episodes never span sessions |

## 2. Finding a fix episode

Only fix episodes become trace candidates: **a failing check, then edits, then the same check passing.**

| Step | Beacon events | Rule |
|---|---|---|
| check | `command.executed`; also `tool.failed` / `tool.completed` carrying a command (`command.command`, or `tool.command` when `tool.name` is a shell tool, or `gen_ai.tool.call.arguments.command`) | not a check: `ls`, `cat`, `grep`, `cd`, `git`, `curl`, editors, and every install (`pip install`, `npm ci`, `uv sync`, ...) |
| fails | `command.exit_code` != 0. With no exit code: `event.action` `tool.failed`, `error.type` set, `severity` medium or worse, or `command.output` / `gen_ai.tool.call.result` reading as a failure (a traceback, `N failed`, `FAILED`, `--- FAIL:`, `error TS2304`, `panic:`, `npm ERR!`) | `cmd \| tail` exits 0 whatever `cmd` did, so a piped command with a failure in its output fails |
| edit | `file.modified`, `file.created`, `file.deleted` with `file.path`; the change from `file.diff` (unified), or rebuilt from the edit tool's `gen_ai.tool.call.arguments` (`old_string` / `new_string`, `edits[]`, `content`) | an edit with neither (Beacon in metadata-only retention keeps only `file.diff_hash`) is counted but carries no code: an episode with no retained change makes no trace |
| passes | the same check, exit code 0 (or, without one, `severity: info` and no failure in the output) | "same" = the same command once a leading `cd x &&`, env assignments (`FOO=1`, `set X=..`, `export X=..`), trailing pipes and redirects, and output flags (`-q`, `-v`, `-x`, `--tb=..`, `--color=..`) are dropped; or a broader run of the same test runner (`pytest` or `pytest tests/` after `pytest tests/test_a.py::t`) |

More failures of the same check before it passes are attempts of one episode (all edits since the first failure are
the change). A check that passes again with **no edit** in between was flaky: no trace. A failure never followed by a
pass stays unresolved: no trace. A `prompt.submitted` during the fix sets `fixed_by` to `unknown` (a person may have
steered it); otherwise it is `model`.

## 3. Building the trace

| `trace/0.1` field | From Beacon |
|---|---|
| `task` | `code.repair` (as the pytest plugin) |
| `base_model.name` | `--model`, else the last `model` (canonical, lower-case: `claude-sonnet-4-5`) / `gen_ai.response.model` / `gen_ai.request.model` seen up to the passing run, else `unknown` |
| `checker` | `<runner>@1` from the passing command: `pytest`, `unittest`, `jest`, `vitest`, `npm-test`, `go-test`, `cargo-test`, `tsc`, `mypy`, `ruff`, `eslint`, `make`, `python-run`, ..., else `shell` (exit code only) |
| `input` | one summary line (harness, the check command, attempts, exit codes, edit and file counts), `failure: <mode>`, the first failure's output excerpt (the lines around the first failure marker, then the tail, at most 25 lines), with `--open` the prompt (`prompt.text` of the last `prompt.submitted` before the first failure, first 1,500 characters), then `change:` and the diff hunks (`@@ <file> @@`, no line numbers, at most 80 lines and 5 files) |
| `model_output.code` / `verified_output.code` | the change's removed / added lines |
| `feedback` | the failure output excerpts, up to 3 attempts |
| `failure_modes.code` | from the failure output: `syntax_error`, `wrong_name` (NameError, AttributeError, ImportError, ReferenceError, TS2304, `is not defined`), `timeout`, `wrong_answer` (AssertionError, `expected ... to`, `N failed`), else `runtime_error` |
| `fixed_by.code` | `model`, or `unknown` when a prompt arrived during the fix |
| `privacy` | `open` (code is not personal data; secrets and personal data are removed by the scrub, and the node's own check still runs) |
| `created` | the passing event's `timestamp`, to the second (same log, same trace id) |
| `producer` | `--address`, else the zero address that `traceex.outbox send` fills in |

Default (no `--open`): the command, the failure output and every diff line are also skeletonized with
`traceex.codeskel` (string literals, comments, project identifiers, large numbers and directories become
placeholders; command and test-runner words such as `pytest`, `FAILED`, `Expected` are kept), and the prompt is left
out, matching what the pytest plugin sends (SPEC 4i). `--open` keeps the scrubbed text verbatim.

Local only (in the outbox file's `about`, never sent): a hash of the session key, the harness, the attempt count, the
Beacon `event.id`s of the failing and passing runs.

**Never used:** `endpoint`, `user`, `run.actor` (except as words to scrub), `raw`, `gen_ai.input/output.messages`,
`gen_ai.system_instructions`, `gen_ai.usage`, `approval`, `policy`, `mcp`, `vcs`, file reads, and every event outside
an episode.

## 4. Scrub (mandatory)

Every string taken from Beacon (command, output, diff, file path, prompt) goes through `scrub.Scrubber` before the
trace is built; `scrub.py`'s header lists every rule. In short: provider keys and tokens (`sk-`, `sk-ant-`, `ghp_`,
`github_pat_`, AWS, Google, Slack, Hugging Face, Stripe, npm, SendGrid, Shopify, fal-style `uuid:hex`, JWTs),
`Authorization: Bearer|Key|Basic` and `x-api-key` headers, URL credentials and token query parameters, secret-named
assignments and flags, every value of a `.env`-style file, private key blocks and high-entropy strings; emails, phone
numbers, street addresses; home directories collapse to `~` and the session's user and host names are removed; paths
inside the project become relative and every other absolute path is redacted (library frames keep only
`<site-packages>/pkg/file.py`).

Then the finished trace, serialized, gets a **second pass** with every secret pattern (this adapter's and the node's
`traceex.skeleton.find_secrets`), and `traceex.client.privacy_leaks`. Any hit: the trace is not written, and
`<outbox>/refused/<episode>.json` records the kinds that survived (never the values). Each written trace gets
`<outbox>/scrub/<file>.json`: replacement counts by kind and by field, and `second_pass: clean`.

Beacon redacts some secrets itself and caps strings at 4 KB; the adapter does not rely on either.
