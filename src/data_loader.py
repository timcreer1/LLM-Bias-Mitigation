"""
data_loader.py

Utilities to load and lightly clean the BBQ dataset used in this project.

Design goals
------------
- Match the workflow used in `1-load-clean-eda.ipynb`:
  - load BBQ category JSONL files from Hugging Face-hosted URLs
  - combine Age, Gender_identity, Race_ethnicity subsets
  - add a short `bias_type` label: {age, gender, race}
  - produce a clean tabular dataset that downstream notebooks can reuse
- Keep this module safe to import (no heavy model code; no GPU requirement)

Notes
-----
- The notebooks were originally executed on Kaggle (2×T4). This loader is CPU-friendly.
- This project uses the public JSONL files hosted at:
    https://huggingface.co/datasets/heegyu/bbq/resolve/main/data
  (same as the notebook)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import json
import re
import unicodedata

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

BiasType = Literal["age", "gender", "race"]
ContextType = Literal["ambig", "disambig"]


# ---------------------------------------------------------------------
# Dataset source (matches your notebook)
# ---------------------------------------------------------------------
BASE_URL = "https://huggingface.co/datasets/heegyu/bbq/resolve/main/data"

# Files on the remote host + how we label them
BBQ_FILES: dict[str, BiasType] = {
    "Age": "age",
    "Gender_identity": "gender",
    "Race_ethnicity": "race",
}

DEFAULT_SEED = 42


# ---------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------
def _norm_text(x: object) -> str:
    """Normalise text while keeping punctuation/case (good for prompting)."""
    if x is None:
        return ""
    s = str(x)
    s = unicodedata.normalize("NFKC", s)
    # normalise whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


_UNKNOWN_CANON_PHRASES = {
    # Keep this list short + robust; it only supports UNKNOWN-index detection
    "not answerable",
    "not known",
    "can't answer",
    "cannot answer",
    "not enough info",
    "not enough information",
    "insufficient information",
    "unknown",
}


def _is_unknown_answer(ans: str) -> bool:
    a = _norm_text(ans).lower()
    return any(p in a for p in _UNKNOWN_CANON_PHRASES)


def _coerce_dict(x: object) -> dict:
    """Coerce dict-like values (or stringified dicts) into a dict."""
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        # Try JSON first
        try:
            v = json.loads(s)
            return v if isinstance(v, dict) else {}
        except Exception:
            pass
        # Some datasets store Python dict strings; attempt a safe-ish fallback
        # (if this fails we return empty)
        try:
            import ast

            v = ast.literal_eval(s)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load_bbq_raw(filename: str, bias_type: BiasType) -> Dataset:
    """
    Load a single BBQ JSONL file from the remote URL and add `bias_type`.

    Parameters
    ----------
    filename:
        One of: "Age", "Gender_identity", "Race_ethnicity"
    bias_type:
        Short label to add to all rows.
    """
    ds = load_dataset("json", data_files={"data": f"{BASE_URL}/{filename}.jsonl"}, split="data")
    ds = ds.add_column("bias_type", [bias_type] * len(ds))
    return ds


def load_bbq(
    bias_types: Iterable[BiasType] = ("gender", "race", "age"),
    seed: int = DEFAULT_SEED,
    shuffle: bool = True,
) -> Dataset:
    """
    Load and concatenate the BBQ subsets used in this project.

    Returns a Hugging Face `Dataset`.
    """
    chosen = set(bias_types)
    parts: list[Dataset] = []
    for fname, btype in BBQ_FILES.items():
        if btype in chosen:
            parts.append(load_bbq_raw(fname, btype))
    if not parts:
        raise ValueError("No bias_types selected. Choose from {'age','gender','race'}.")

    ds = concatenate_datasets(parts)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    return ds


# ---------------------------------------------------------------------
# Cleaning / feature creation
# ---------------------------------------------------------------------
def _infer_context_type(row: dict) -> Optional[ContextType]:
    """
    Infer context type ("ambig" vs "disambig") using dataset fields.

    The BBQ variants sometimes expose:
    - context_condition
    - context_type / context_label
    - ambiguous / is_ambiguous

    We prioritise explicit values; otherwise fall back to ambiguous flags.
    """
    # 1) Direct label if present
    for key in ("context_condition", "context_type", "context_label"):
        v = row.get(key, None)
        if v is None:
            continue
        s = str(v).strip().lower()
        if s in ("ambig", "disambig"):
            return s  # type: ignore[return-value]

    # 2) Ambiguous boolean flags
    for key in ("ambiguous", "is_ambiguous"):
        v = row.get(key, None)
        if isinstance(v, (bool, np.bool_)):
            return "ambig" if bool(v) else "disambig"
        if isinstance(v, (int, np.integer)):
            return "ambig" if int(v) == 1 else "disambig"
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "1", "yes", "y"):
                return "ambig"
            if s in ("false", "0", "no", "n"):
                return "disambig"
    return None


def _unknown_idx(ans0: str, ans1: str, ans2: str) -> Optional[int]:
    """Return the option index (0/1/2) that corresponds to UNKNOWN, if detectable."""
    answers = [ans0, ans1, ans2]
    idxs = [i for i, a in enumerate(answers) if _is_unknown_answer(a)]
    return idxs[0] if len(idxs) == 1 else None


def clean_bbq(ds: Dataset) -> Dataset:
    """
    Add standardised fields and normalise text.

    Output columns added:
    - context_type: "ambig" or "disambig" (best-effort)
    - unknown_idx: 0/1/2 if a single answer option looks like UNKNOWN, else None
    - answer_info_dict: dict version of answer_info (if present)
    """
    def _map_fn(ex: dict) -> dict:
        # Normalise text fields (keep punctuation/case)
        ex["context"] = _norm_text(ex.get("context", ""))
        ex["question"] = _norm_text(ex.get("question", ""))
        ex["ans0"] = _norm_text(ex.get("ans0", ""))
        ex["ans1"] = _norm_text(ex.get("ans1", ""))
        ex["ans2"] = _norm_text(ex.get("ans2", ""))

        # context type
        ex["context_type"] = _infer_context_type(ex)

        # unknown option index (best-effort)
        ex["unknown_idx"] = _unknown_idx(ex["ans0"], ex["ans1"], ex["ans2"])

        # answer_info coercion (optional)
        ex["answer_info_dict"] = _coerce_dict(ex.get("answer_info", None))

        # Ensure bias_type exists
        if "bias_type" not in ex or ex["bias_type"] is None:
            # try to map from category if present
            cat = str(ex.get("category", "")).strip()
            ex["bias_type"] = BBQ_FILES.get(cat, None)  # type: ignore[assignment]

        return ex

    return ds.map(_map_fn)


# ---------------------------------------------------------------------
# Convenience helpers for downstream use
# ---------------------------------------------------------------------
PROMPT_TEMPLATE = (
    "Context: {context}\n"
    "Question: {question}\n"
    "A) {ans0}\n"
    "B) {ans1}\n"
    "C) {ans2}\n"
    "Answer:"
)


def build_prompt(row: dict, template: str = PROMPT_TEMPLATE) -> str:
    """Build a single multiple-choice prompt string from a BBQ row."""
    return template.format(
        context=_norm_text(row.get("context", "")),
        question=_norm_text(row.get("question", "")),
        ans0=_norm_text(row.get("ans0", "")),
        ans1=_norm_text(row.get("ans1", "")),
        ans2=_norm_text(row.get("ans2", "")),
    )


def to_dataframe(ds: Dataset) -> pd.DataFrame:
    """Convert to pandas DataFrame (kept as a separate helper for convenience)."""
    return ds.to_pandas()


def save_processed_csv(ds: Dataset, path: str | Path = "data/processed/bbq_clean.csv") -> Path:
    """Save the dataset as a UTF-8 CSV (mirrors the notebook behaviour)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = ds.to_pandas()
    df.to_csv(path, index=False, encoding="utf-8")
    return path


# ---------------------------------------------------------------------
# Optional CLI usage
# ---------------------------------------------------------------------
def main() -> None:
    """
    Example usage:
        python -m src.data_loader
    """
    ds = load_bbq()
    ds = clean_bbq(ds)
    out = save_processed_csv(ds, "data/processed/bbq_clean.csv")
    print(f"Saved cleaned dataset to: {out}")


if __name__ == "__main__":
    main()
