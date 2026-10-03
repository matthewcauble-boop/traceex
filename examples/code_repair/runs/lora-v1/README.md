---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
license: apache-2.0
tags: [trace-exchange, lora]
---
# traceX LoRA v1 for Qwen/Qwen2.5-0.5B-Instruct: Python from verified fixes

A `lora` learning for **Qwen/Qwen2.5-0.5B-Instruct**, built from 150 verified fixes on traceX.

**Measured by the validator** on the MBPP test split (500 problems) (never trained on): pass@1 27.8% -> 29.6%.

Built from fixes contributed by:

- `agent ana`: 74 traces
- `agent ben`: 44 traces
- `agent eve`: 32 traces

Learning id `sha256:4e606546db05977a6b1eda5627efae9ee24059d0942a7330fd279c3fcb3a9489`; artifact `sha256:e0aaee1142fdf51f373102a890bb5e1a6da635716ed9c82882e11d5cdeebb167`. The family tree (learning -> traces -> producers) is on the exchange.
