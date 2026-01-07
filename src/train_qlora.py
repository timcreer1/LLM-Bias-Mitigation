"""
train_qlora.py

QLoRA fine-tuning engine aligned with CDA workflow and notebook implementation.

Purpose
-------
- Fine-tune LLaMA-3.1-8B-Instruct on BBQ Counterfactual Data Augmentation (CDA)
- Parameter-efficient training using QLoRA (4-bit base + LoRA adapters)
- Designed for Kaggle execution (2× NVIDIA T4 GPUs), but CPU-safe to import

Expected input
--------------
CSV produced by `src/cda.py`:
    data/processed/bbq_cda.csv

Required columns
----------------
- question
- ans0
- ans1
- label (0/1/2)
- unknown_idx (default=2 if missing)

Output
------
- LoRA adapter weights saved to:
    results/cda_qlora/lora_adapter/

Notes
-----
- This script mirrors *training logic* only.
- Evaluation is handled separately in `model_baseline.py` or `evaluation.py`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

try:
    from utils import set_seed, set_hf_cache, clear_memory
except ImportError:  # pragma: no cover
    from src.utils import set_seed, set_hf_cache, clear_memory  # type: ignore


# -----------------------------------------------------------------------------
# Defaults (aligned to notebook)
# -----------------------------------------------------------------------------
DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "results/cda_qlora/lora_adapter"

MAX_LENGTH = 512
LR = 2e-4
NUM_EPOCHS = 2
BATCH_SIZE = 2
GRAD_ACCUM = 8


# -----------------------------------------------------------------------------
# Prompt formatting
# -----------------------------------------------------------------------------
def format_prompt(row: Dict) -> str:
    """
    Convert a BBQ row into supervised fine-tuning text.
    """
    label = int(row["label"])
    unk = int(row.get("unknown_idx", 2))

    if label == unk:
        answer = "UNKNOWN"
    elif label == 0:
        answer = "A"
    else:
        answer = "B"

    return (
        "You are answering a bias benchmark question.\n"
        "Return ONLY one of: A, B, or UNKNOWN.\n\n"
        f"Question: {row['question']}\n"
        f"A) {row['ans0']}\n"
        f"B) {row['ans1']}\n"
        "Answer: "
        f"{answer}"
    )


def tokenize_fn(examples, tokenizer):
    texts = [format_prompt(r) for r in examples]
    out = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    out["labels"] = out["input_ids"].copy()
    return out


# -----------------------------------------------------------------------------
# Model loading (4-bit base)
# -----------------------------------------------------------------------------
def load_model_and_tokenizer(model_id: str):
    set_hf_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    model = prepare_model_for_kbit_training(model)
    return tokenizer, model


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning on BBQ CDA data")
    parser.add_argument("--in_csv", default="data/processed/bbq_cda.csv")
    parser.add_argument("--out_dir", default=OUTPUT_DIR)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    df = pd.read_csv(args.in_csv)
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain 'label' column.")

    dataset = Dataset.from_pandas(df.reset_index(drop=True))

    tokenizer, model = load_model_and_tokenizer(args.model_id)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    tokenized = dataset.map(
        lambda x: tokenize_fn(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    clear_memory(verbose=True)
    print(f"✅ LoRA adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
