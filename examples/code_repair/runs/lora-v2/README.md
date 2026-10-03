---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
license: apache-2.0
tags: [trace-exchange, lora]
---
# traceX LoRA v2 for Qwen/Qwen2.5-0.5B-Instruct: Python from verified fixes

A `lora` learning for **Qwen/Qwen2.5-0.5B-Instruct**, built from 244 verified fixes on traceX.

**Measured by the validator** on the MBPP test split (500 problems) (never trained on): pass@1 after one round of checker feedback 29.4% -> 37.2%.

Built from fixes contributed by:

- `agent ana`: 100 traces
- `agent ben`: 74 traces
- `agent eve`: 70 traces

Learning id `sha256:5751d0314d3ce9e1b47794a1c5fcff0683e4c4172622fac0c531193c0eaa437f`; artifact `sha256:beb4c73082948319c5d8ee181696f5bfea7c8ef82f07dfde2a826b034c4fee8c`. The family tree (learning -> traces -> producers) is on the exchange.
