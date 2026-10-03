"""An open-weight model (optionally with a LoRA adapter), with every generation recorded.

Each call is keyed by sha256(model, adapter, prompt, sampling, sample index) and appended to recorded/<record>.jsonl,
so the whole example replays on any machine without a GPU or a model download. A live run fills in what's missing.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(HERE, "recorded")
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"          # Apache-2.0 open weights
SYSTEM = "You are a helpful coding assistant. Answer with Python code only, in a single ```python code block."


def _key(model, adapter, prompt, temperature, idx, max_new):
    raw = json.dumps([model, adapter or "", prompt, round(float(temperature), 3), idx, max_new])
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class LM:
    def __init__(self, model_id=BASE_MODEL, adapter=None, live=None, batch_size=24, record="generations"):
        """adapter: path to a PEFT LoRA folder (its weights hash is part of every key). live: None = load the model only
        if a needed generation isn't recorded; False = replay only; True = always able to generate."""
        self.model_id, self.adapter_path, self.live, self.batch_size = model_id, adapter, live, batch_size
        self.adapter = adapter_hash(adapter) if adapter else None
        self.cache, self.record = {}, os.path.join(RECORD_DIR, record + ".jsonl")   # reads every file, writes its own
        if os.path.isdir(RECORD_DIR):
            for name in sorted(os.listdir(RECORD_DIR)):
                if name.endswith(".jsonl"):
                    with open(os.path.join(RECORD_DIR, name), encoding="utf-8") as f:
                        for line in f:
                            r = json.loads(line)
                            self.cache[r["k"]] = r["out"]
        self._model = self._tok = None
        self.generated = 0

    # -- model, loaded only when something isn't recorded ------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return
        if self.live is False:
            raise RuntimeError("replay-only, and a generation is missing from recorded/*.jsonl")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16).to("cuda").eval()
        if self.adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter_path).eval()
        self._model = model

    def chat(self, prompt):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        return self._tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def _run(self, prompts, temperature, n, max_new, on_batch):
        """Generate in batches; sampled batches are smaller so 8 returns per prompt fit in 12 GB. Each finished batch
        is handed to on_batch, so a crash loses at most one batch of work."""
        torch = self._torch
        bs = self.batch_size if temperature == 0 else max(1, self.batch_size // max(1, n // 2))
        for i in range(0, len(prompts), bs):
            chunk = prompts[i:i + bs]
            enc = self._tok([self.chat(p) for p in chunk], return_tensors="pt", padding=True).to("cuda")
            kw = dict(max_new_tokens=max_new, pad_token_id=self._tok.eos_token_id)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=0.95, num_return_sequences=n)
            else:
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            with torch.no_grad():
                gen = self._model.generate(**enc, **kw)
            texts = self._tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            k = n if temperature > 0 else 1
            on_batch(i, [texts[j * k:(j + 1) * k] for j in range(len(chunk))])
            del gen, enc
            torch.cuda.empty_cache()

    # -- public ------------------------------------------------------------------------------------------------------
    def generate(self, prompts, temperature=0.0, n=1, max_new=512):
        """-> list (per prompt) of n replies. Greedy when temperature is 0 (then n must be 1)."""
        n = 1 if temperature == 0 else n
        keys = [[_key(self.model_id, self.adapter, p, temperature, j, max_new) for j in range(n)] for p in prompts]
        todo = [i for i, ks in enumerate(keys) if any(k not in self.cache for k in ks)]
        if todo:
            self._load()
            order = sorted(todo, key=lambda i: len(prompts[i]))          # similar lengths batch better
            os.makedirs(RECORD_DIR, exist_ok=True)

            def save(offset, outs):
                with open(self.record, "a", encoding="utf-8") as f:
                    for i, o in zip(order[offset:offset + len(outs)], outs):
                        for j, k in enumerate(keys[i]):
                            self.cache[k] = o[j]
                            f.write(json.dumps({"k": k, "out": o[j]}) + "\n")
                self.generated += sum(len(o) for o in outs)
            self._run([prompts[i] for i in order], temperature, n, max_new, save)
        return [[self.cache[k] for k in ks] for ks in keys]


def adapter_hash(path):
    """Content hash of a LoRA adapter's weights: the id a learning's artifact carries."""
    h = hashlib.sha256()
    for name in sorted(os.listdir(path)):
        if name.endswith((".safetensors", ".json")):
            with open(os.path.join(path, name), "rb") as f:
                h.update(name.encode() + f.read())
    return "sha256:" + h.hexdigest()
