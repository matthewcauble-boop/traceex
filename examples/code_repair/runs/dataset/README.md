---
license: cc-by-4.0
tags: [trace-exchange, verified-fixes]
---
# traceX: verified fixes for Qwen/Qwen2.5-0.5B-Instruct on MBPP

244 verified fixes from traceX: each row is a real failure of Qwen/Qwen2.5-0.5B-Instruct and the answer a checker (mbpp-tests@1) proved right.

Problems from MBPP (Austin et al. 2021, CC BY 4.0), train split only.

| producer | traces |
|---|---|
| `0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` | 100 |
| `0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` | 74 |
| `0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee` | 70 |

Failure modes: wrong_answer 190, runtime_error 43, syntax_error 7, wrong_name 4

Formats: `sft.jsonl` (chat messages), `dpo.jsonl` (prompt / chosen / rejected), `repair.jsonl` (failing answer + feedback in, fix out). Every row carries its trace id; royalties and credit follow it.
