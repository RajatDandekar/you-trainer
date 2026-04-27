"""You. — RunPod Serverless training handler.

Receives a Q&A dataset extracted from a user's wiki, fine-tunes a small
LoRA adapter on a base model, and (optionally) pushes the adapter to
Hugging Face. Returns progress via RunPod's streaming output.

Input JSON:
  {
    "input": {
      "dataset":      [{"messages": [{"role":"user","content":"..."},
                                     {"role":"assistant","content":"..."}]}, ...],
      "base_model":   "meta-llama/Llama-3.2-1B-Instruct",   # optional
      "iters":        200,                                  # optional
      "lora_rank":    8,                                    # optional
      "lr":           1e-4,                                 # optional
      "hf_token":     "hf_…",                               # optional
      "hf_repo":      "username/you-wiki",                  # optional
      "hf_private":   true,                                 # optional
      "user_id":      "anon-uuid"                           # optional
    }
  }

Output:
  { "status":"ok", "iters":200, "adapter_size_mb":12.4,
    "hf_repo":"username/you-wiki", "hf_url":"https://…", "duration_s":287 }
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import runpod
import torch
from datasets import Dataset
from huggingface_hub import HfApi, create_repo
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def _format_chat(example: dict[str, Any], tokenizer) -> dict[str, str]:
    """Apply the model's chat template to a {messages: [...]} record."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def _tokenize(example: dict[str, Any], tokenizer, max_length: int = 1024) -> dict:
    out = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    out["labels"] = out["input_ids"].copy()
    return out


def handler(event: dict) -> dict:
    started = time.time()
    inp = event.get("input", {}) or {}

    pairs = inp.get("dataset") or []
    if not pairs:
        return {"error": "no dataset in input.dataset"}

    base_model = inp.get("base_model") or "meta-llama/Llama-3.2-1B-Instruct"
    iters = int(inp.get("iters", 200))
    lora_rank = int(inp.get("lora_rank", 8))
    lr = float(inp.get("lr", 1e-4))
    hf_token = inp.get("hf_token") or os.environ.get("HF_TOKEN")
    hf_repo = inp.get("hf_repo")
    hf_private = bool(inp.get("hf_private", True))

    print(f"[you] dataset={len(pairs)} base={base_model} iters={iters}", flush=True)

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )

    # Wrap with LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Build dataset — chat template → tokens
    ds = Dataset.from_list(pairs)
    ds = ds.map(lambda e: _format_chat(e, tokenizer), remove_columns=ds.column_names)
    ds = ds.map(lambda e: _tokenize(e, tokenizer), remove_columns=["text"])

    # Train
    out_dir = Path("/tmp/you-output")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=iters,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        warmup_steps=min(20, iters // 10),
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=tokenizer)
    trainer.train()

    # Save adapter
    adapter_dir = Path("/tmp/you-adapter")
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    adapter_size_mb = sum(p.stat().st_size for p in adapter_dir.rglob("*") if p.is_file()) / 1e6

    duration = round(time.time() - started, 1)
    result: dict[str, Any] = {
        "status": "ok",
        "iters": iters,
        "n_samples": len(pairs),
        "base_model": base_model,
        "adapter_size_mb": round(adapter_size_mb, 2),
        "duration_s": duration,
    }

    # Push to HF if configured
    if hf_token and hf_repo:
        api = HfApi(token=hf_token)
        create_repo(hf_repo, token=hf_token, private=hf_private, exist_ok=True)
        # Generate a small README in the adapter folder
        (adapter_dir / "README.md").write_text(
            f"""---
language: [en]
library_name: peft
base_model: {base_model}
tags: [personal, lora, you, wiki]
---

# You. — a fine-tuned reflection of one person

This is a small LoRA adapter, trained on a single person's voice journal — their
*You.* wiki. Tiny (~{round(adapter_size_mb, 1)} MB). Loads on top of `{base_model}`.

Trained on **{len(pairs)} Q&A pairs** for **{iters} iterations** on a RunPod GPU.

Voice register: hushed, intimate, knowing. Second-person address. Cites the wiki.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{hf_repo}")
tok = AutoTokenizer.from_pretrained("{hf_repo}")
```

*Hello, you.*
"""
        )
        api.upload_folder(
            folder_path=str(adapter_dir),
            repo_id=hf_repo,
            repo_type="model",
            commit_message=f"You. wiki adapter — {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        )
        result["hf_repo"] = hf_repo
        result["hf_url"] = f"https://huggingface.co/{hf_repo}"

    print(f"[you] done in {duration}s — {result}", flush=True)
    return result


runpod.serverless.start({"handler": handler})
