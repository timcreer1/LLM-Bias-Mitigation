# Results Directory

This directory documents the outputs produced by the bias-mitigation experiments in this repository.

> **Note:** Due to GPU cost, runtime, and storage constraints, most experimental outputs are
> embedded directly in the Jupyter notebooks rather than saved as standalone result files.

---

## 📌 Where to Find Results

### Primary Results (Authoritative)
All key figures, tables, and analyses are contained **inside the notebooks**:

- `notebooks/2-baseline-model.ipynb`
- `notebooks/3-method-1-cda.ipynb`
- `notebooks/4-method2_Few_Shot.ipynb`
- `notebooks/5-evaluation.ipynb`

These notebooks include:
- Slice-level accuracy
- Bias metrics (sDIS, sAMB)
- Calibration metrics (AURC)
- Comparative plots across methods

---

## 📂 Expected Subdirectories (Optional)

When running scripts locally or on Kaggle, the following subfolders may be created:

```
results/
├── baseline/        # Baseline inference outputs (optional)
├── cda_qlora/       # LoRA adapter checkpoints (not committed)
├── few_shot/        # Few-shot slice metrics & predictions (optional)
└── summary.csv      # Aggregated metrics (optional)
```

These files are **not required** to interpret the project.

---

## ⚠️ Why Results Are Not Fully Stored Here

This project was originally executed on **Kaggle using 2× NVIDIA T4 GPUs**.

As a result:
- Some experiments take **hours to re-run**
- Model checkpoints exceed GitHub size limits
- Results are deterministic and reproducible from notebooks

To keep the repository lightweight and recruiter-friendly:
- ✔ Code is fully provided
- ✔ Methodology is transparent
- ✔ Results are embedded and explained in notebooks
- ❌ Large binary artifacts are excluded

---

## 🔁 Reproducing Results

To regenerate outputs locally or on Kaggle:

1. Follow setup instructions in the root `README.md`
2. Run notebooks sequentially **OR**
3. Use the `src/` scripts:
   - `model_baseline.py`
   - `cda.py`
   - `train_qlora.py`
   - `few_shot.py`
   - `evaluation.py`

---

## ✅ Reviewer Guidance

If you are reviewing this repository:
- Start with the **root README**
- Open the **notebooks in order**
- Treat notebook outputs as the source of truth

This structure mirrors common academic + applied ML workflows.
