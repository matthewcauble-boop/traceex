"""The trainer: buy a lot of traces, export it to training data, and train a LoRA adapter for the open-weight base model.

    python examples/code_repair/train_lora.py runs/traces.jsonl runs/lora-v1

SFT on the verified fixes (prompt -> passing code) plus the self-repair turns (failing code + traceback -> passing
code), loss on the assistant tokens only. Rank-16 LoRA on every projection of Qwen2.5-0.5B-Instruct; bf16 base, fp32
adapter; a few minutes on an RTX 3060. Seeded, so the same lot trains the same adapter.
"""
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "..", "sdk", "python")]
from traceex import export  # noqa: E402
from lm import BASE_MODEL, SYSTEM  # noqa: E402

CONFIG = {"rank": 16, "alpha": 32, "dropout": 0.05, "lr": 2e-4, "epochs": 3, "batch": 8, "max_len": 1024, "seed": 7,
          "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], "repair_turns": True}


def load_traces(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_rows(traces, repair_turns=True):
    rows = export.to_sft(traces)
    if repair_turns:
        rows += export.to_repair(traces)
    return rows


def train(traces, out_dir, config=CONFIG, log=print):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    rows = build_rows(traces, config["repair_turns"])

    def encode(row):
        msgs = [{"role": "system", "content": SYSTEM}] + row["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)
        full = prompt + msgs[-1]["content"] + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = tok(full, add_special_tokens=False)["input_ids"][:config["max_len"]]
        labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
        return ids, labels[:len(ids)]

    data = [encode(r) for r in rows]
    log(f"  {len(traces)} traces -> {len(rows)} training rows ({sum(len(d[0]) for d in data):,} tokens)")

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=config["rank"], lora_alpha=config["alpha"], lora_dropout=config["dropout"],
                                             target_modules=config["targets"], task_type="CAUSAL_LM"))
    for n, p in model.named_parameters():                 # adapter weights in fp32, base stays bf16
        if p.requires_grad:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  LoRA r={config['rank']}: {trainable:,} trainable parameters")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["lr"], weight_decay=0.0)
    steps = config["epochs"] * math.ceil(len(data) / config["batch"])
    warm = max(1, int(0.06 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    model.train()
    step, t0, hist = 0, time.time(), []
    for ep in range(config["epochs"]):
        order = list(range(len(data)))
        random.shuffle(order)
        for i in range(0, len(order), config["batch"]):
            batch = [data[j] for j in order[i:i + config["batch"]]]
            L = max(len(b[0]) for b in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id or tok.eos_token_id)
            lab = torch.full((len(batch), L), -100)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for k, (x, y) in enumerate(batch):
                ids[k, :len(x)], lab[k, :len(y)], att[k, :len(x)] = torch.tensor(x), torch.tensor(y), 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            hist.append(round(loss.item(), 4))
            if step % 10 == 0 or step == steps:
                log(f"  step {step}/{steps}  epoch {ep + 1}  loss {sum(hist[-10:]) / len(hist[-10:]):.3f}  "
                    f"({time.time() - t0:.0f}s)")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "training.json"), "w", encoding="utf-8") as f:
        json.dump({"base_model": BASE_MODEL, "config": config, "traces": len(traces), "rows": len(rows),
                   "steps": steps, "loss": hist, "seconds": round(time.time() - t0)}, f, indent=0)
    return out_dir


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs", "traces.jsonl")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "runs", "lora-v1")
    src = src if os.path.isabs(src) else os.path.join(HERE, src) if not os.path.exists(src) else src
    out = out if os.path.isabs(out) else os.path.join(HERE, out) if not out.startswith(HERE) else out
    print(f"training {os.path.basename(out)} from {src}")
    train(load_traces(src), out)
