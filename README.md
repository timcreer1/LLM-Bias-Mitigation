# Bias Mitigation in Large Language Models (LLaMA-3.1 + BBQ Benchmark)

This repository investigates social bias in Large Language Models using the BBQ benchmark.  
It provides a reproducible research pipeline including data loading, baseline evaluation, Counterfactual Data Augmentation (CDA), QLoRA fine-tuning, Few-Shot prompting, and structured fairness evaluation (sDIS, sAMB, AURC, log-odds).

> **Note:** This project was originally developed and executed on **Kaggle using 2× NVIDIA T4 GPUs**.  
> To avoid unnecessary recomputation, **all experimental outputs are preserved directly inside the notebooks** rather than regenerated or exported to standalone result files.

---

## 📌 Project Overview

Large Language Models (LLMs) can unintentionally propagate or amplify social biases.  
This project evaluates bias in **LLaMA-3.1-8B-Instruct** and explores two mitigation strategies:

1. **Counterfactual Data Augmentation (CDA)**  
2. **Few-Shot Prompting with Debiasing-Pattern Examples**

All methods are evaluated on the **BBQ Benchmark**, covering nine sensitive demographic categories.

---

## ⚙️ Execution Environment (Important)

- **Platform:** Kaggle Notebooks  
- **Hardware:** 2 × NVIDIA T4 GPUs  
- **Reason:** QLoRA fine-tuning and large-scale LLaMA inference are computationally expensive  

As a result:
- Notebooks are presented **as executed artefacts**
- Outputs (tables, metrics, plots, examples) are intentionally kept **inside the `.ipynb` files**
- Re-running the full pipeline locally is **not required** to understand the results

---

## 📂 Repository Structure

```
bias-llm-fairness/
│
├── README.md
├── environment.yml
├── LICENSE
│
├── data/
│   ├── raw/               # empty – dataset auto-downloads from URL
│   ├── processed/
│   └── counterfactual/
│   └── README.md
│
├── notebooks/
│   ├── 1-load-clean-eda.ipynb
│   ├── 2-baseline-model.ipynb
│   ├── 3-method-1-cda.ipynb
│   ├── 4-method-2-few-shot.ipynb
│   ├── 5-evaluation.ipynb
│   └── figures/
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── cda.py
│   ├── model_baseline.py
│   ├── train_qlora.py
│   ├── few_shot.py
│   ├── evaluation.py
│   └── utils.py
│
├── results/
│   └── README.md          # intentionally empty – outputs live in notebooks
│
└── report/
    ├── Bias_Mitigation_Report.pdf
    └── references.bib
```

---

## 🗂️ About Results and Outputs

To preserve compute resources and ensure transparency:

- **All key outputs** (EDA plots, bias metrics, tables, example generations)  
  are stored **directly inside the executed notebooks**
- The `results/` directory is retained only for structural completeness
- No outputs are missing — they are embedded where they were generated

This approach is common for LLM projects that rely on expensive GPU workloads.

---

## 🚀 Methods Implemented

### **1. Baseline Model Evaluation**
- LLaMA-3.1-8B-Instruct evaluated directly on BBQ
- Analysis includes:
  - Ambiguous vs. non-ambiguous cases
  - Target vs. non-target bias
  - Incorrect inference patterns

### **2. Counterfactual Data Augmentation (CDA)**
- Identity-swapping via lexical templates
- Expands dataset size and balances demographic attributes
- Fine-tuned using **QLoRA** for parameter efficiency

### **3. Few-Shot Debiasing**
- Hand-crafted exemplars embedded in prompts
- Introduces balanced reasoning patterns
- Requires **no model training**

---

## 📊 Evaluation Metrics

| Metric | Description |
|------|-------------|
| **sDIS** | Directional bias across demographic dimensions |
| **sAMB** | Bias tendencies in ambiguous cases |
| **AURC** | Area under the rejection curve (calibration) |
| **Log-Odds Ratio** | Bias magnitude between identity groups |

All metrics follow the BBQ benchmark definitions.

---

## 🧪 How to Explore the Project

### Recommended approach
Open the notebooks **in order**:

1. `1-load-clean-eda.ipynb`  
2. `2-baseline-model.ipynb`  
3. `3-method-1-cda.ipynb`  
4. `4-method-2-few-shot.ipynb`  
5. `5-evaluation.ipynb`

Each notebook:
- Contains explanatory markdown
- Preserves original outputs
- Ends with a summary of findings

---

## 🧱 Dependencies

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- datasets
- PEFT (QLoRA)
- accelerate
- bitsandbytes
- NumPy, Pandas
- matplotlib, seaborn
- scikit-learn
- tqdm

See **environment.yml** for the full specification.

---

## 📘 Report

The full academic report—including methodology, experimental design, and discussion—is available in:

```
report/Bias_Mitigation_Report.pdf
```

---

## 📄 License

MIT License (or adjust as desired).

---

## 🤝 Acknowledgements
- BBQ Benchmark authors  
- Meta LLaMA-3.1  
- Hugging Face ecosystem  
- Kaggle GPU infrastructure  
- University of Sydney — Advanced Machine Learning Coursework  

---

## ⭐ If you use this work
Please cite or link to this repository.
