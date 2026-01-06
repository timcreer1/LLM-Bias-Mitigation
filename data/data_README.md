# Data Directory

This folder contains the data used in the **Bias Mitigation in Large Language Models (LLaMA-3.1 + BBQ Benchmark)** project.

> ⚠️ **Important:** The BBQ dataset itself is **not** stored in this repository.  
> It is automatically downloaded from Hugging Face when you run the notebooks or data loader script.

---

## 📥 Dataset Source

This project uses the **BBQ (Bias Benchmark for Question Answering)** dataset:

- Hugging Face: https://huggingface.co/datasets/nyu-mll/BBQ

The code loads the dataset using the `datasets` library, for example:

```python
from datasets import load_dataset

dataset = load_dataset("nyu-mll/BBQ", split="train")
```

No manual download is required for standard usage.

---

## 📂 Expected Structure

At runtime, the project expects the following layout:

```bash
data/
├── raw/             # raw BBQ data (optional local cache)
├── processed/       # cleaned / filtered splits used in experiments
└── counterfactual/  # CDA-augmented data
```

These folders may be created automatically by the notebooks or scripts.  
They can also be created manually if you prefer.

---

## 📴 Offline Usage (Optional)

If you are working in an offline environment:

1. Download the BBQ dataset in advance from Hugging Face.
2. Place the files into:

```bash
data/raw/
```

3. Adjust paths in the notebooks or `src/data_loader.py` if needed.

---

## ℹ️ Notes

- This repository intentionally omits large data files to keep the repo lightweight.
- All **results and metrics** are preserved inside the notebook outputs rather than in separate data files.
- For full experimental details, see the project `README.md` and `report/Bias_Mitigation_Report.pdf`.
