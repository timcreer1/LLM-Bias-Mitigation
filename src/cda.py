"""
cda.py

Counterfactual Data Augmentation (CDA) implementation aligned with
`3-method-1-cda.ipynb`.

This module mirrors the notebook logic:
- Operates on preprocessed BBQ data
- Uses identity-based counterfactual swaps
- Preserves labels and question structure
- Produces an augmented dataset for QLoRA fine-tuning

Design principles
-----------------
- CPU-only (safe to run locally)
- Deterministic via seed
- Explicit metadata for auditability
- No model or GPU dependencies

Expected input
--------------
CSV produced by `src/preprocess.py`, typically:
    data/processed/bbq_clean.csv

Output
------
Augmented CSV with original + counterfactual rows:
    data/processed/bbq_cda.csv

Additional columns:
- is_counterfactual
- cf_swap_type
- cf_swap_map
- source_row_id
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd

BiasType = Literal["gender", "race", "age"]

TEXT_FIELDS = ["context", "question", "ans0", "ans1", "ans2"]


# ---------------------------------------------------------------------
# Swap lexicons (aligned to notebook logic)
# ---------------------------------------------------------------------
GENDER_SWAPS = {
    "he": "she",
    "she": "he",
    "him": "her",
    "her": "him",
    "his": "her",
    "hers": "his",
    "man": "woman",
    "woman": "man",
    "male": "female",
    "female": "male",
}

AGE_SWAPS = {
    "young": "old",
    "old": "young",
    "younger": "older",
    "older": "younger",
    "child": "adult",
    "adult": "child",
    "elderly": "young",
}

RACE_SWAPS = {
    "black": "white",
    "white": "black",
    "asian": "white",
    "hispanic": "white",
    "latino": "white",
    "latina": "white",
}


def _get_swap_map(bias_type: BiasType) -> Dict[str, str]:
    if bias_type == "gender":
        return GENDER_SWAPS
    if bias_type == "age":
        return AGE_SWAPS
    if bias_type == "race":
        return RACE_SWAPS
    raise ValueError(f"Unknown bias_type: {bias_type}")


def _preserve_case(src: str, tgt: str) -> str:
    if src.isupper():
        return tgt.upper()
    if src[0].isupper():
        return tgt.capitalize()
    return tgt


def apply_swaps(text: str, swap_map: Dict[str, str]) -> (str, Dict[str, str]):
    applied = {}
    out = text

    for k, v in swap_map.items():
        pattern = re.compile(rf"\b{k}\b", flags=re.IGNORECASE)

        def repl(m):
            applied[k] = v
            return _preserve_case(m.group(0), v)

        out = pattern.sub(repl, out)

    return out, applied


def make_counterfactual(row: pd.Series, seed: int) -> Dict:
    rng = random.Random(seed + int(row.name))
    bias_type = row.get("bias_type")

    out = row.to_dict()
    out["is_counterfactual"] = True
    out["source_row_id"] = int(row.name)
    out["cf_swap_type"] = bias_type
    applied_all = {}

    if bias_type in ("gender", "age", "race"):
        swap_map = _get_swap_map(bias_type)
        for field in TEXT_FIELDS:
            text = str(out[field])
            new_text, applied = apply_swaps(text, swap_map)
            out[field] = new_text
            applied_all.update(applied)

    out["cf_swap_map"] = json.dumps(applied_all, ensure_ascii=False)
    return out


def build_cda_dataframe(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    df["is_counterfactual"] = False
    df["cf_swap_type"] = None
    df["cf_swap_map"] = "{}"
    df["source_row_id"] = df.index.astype(int)

    cf_rows: List[Dict] = []
    for _, row in df.iterrows():
        cf_rows.append(make_counterfactual(row, seed))

    df_cf = pd.DataFrame(cf_rows)
    return pd.concat([df, df_cf], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CDA-augmented BBQ dataset.")
    parser.add_argument("--in_csv", default="data/processed/bbq_clean.csv")
    parser.add_argument("--out_csv", default="data/processed/bbq_cda.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    out_df = build_cda_dataframe(df, seed=args.seed)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"✅ CDA dataset written to {out_path}")
    print(f"Rows: original={len(df)}, counterfactual={out_df['is_counterfactual'].sum()}")


if __name__ == "__main__":
    main()
