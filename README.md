# Bias Mitigation in Large Language Models (LLaMA-3.1 + BBQ Benchmark)

This repository investigates social bias in Large Language Models using the BBQ benchmark.  
It provides a reproducible pipeline including data loading, baseline evaluation, Counterfactual Data Augmentation (CDA), QLoRA fine‑tuning, Few‑Shot prompting, and structured fairness evaluation (sDIS, sAMB, AURC, log‑odds).

---

## 📌 Project Overview

Large Language Models (LLMs) can unintentionally propagate or amplify social biases.  
This project evaluates bias in **LLaMA‑3.1‑8B‑Instruct** and explores two mitigation strategies:

1. **Counterfactual Data Augmentation (CDA)**  
2. **Few‑Shot Prompting with Debiasing‑Pattern Examples**  

All methods are evaluated on the **BBQ Benchmark**, covering nine sensitive categories.

---

## ⚠️ About Outputs

Because the original model runs (baseline inference, CDA augmentation, QLoRA training, and Few‑Shot prompting) required **significant GPU resources**, the final results are **not stored as separate files in the `results/` directory**.

Instead:

### ✅ **All outputs (tables, metrics, plots, comparisons) are preserved directly inside the notebook outputs (`.ipynb` files`).**

This ensures the project remains fully viewable without requiring re‑execution of expensive LLM experiments.

You can open each notebook to see:

- evaluation metrics  
- sample predictions  
- bias measurements  
- visualisations  
- summary tables  

The notebooks *are the source of truth* for all results.

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
│   ├── raw/               # empty – dataset auto-downloads
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
│   └── figures/           # optional saved images
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
├── results/               # folder exists but outputs live inside notebooks
│   └── (empty)
│
└── report/
    ├── Bias_Mitigation_Report.pdf
    └── references.bib
```

---

## 🚀 Methods Implemented

### **1. Baseline Model Evaluation**
Evaluates LLaMA‑3.1‑8B‑Instruct directly on BBQ, exploring:

- ambiguous vs non‑ambiguous cases  
- target vs non‑target bias  
- stereotype‑consistent vs inconsistency behaviour  

### **2. Counterfactual Data Augmentation (CDA)**
- Identity‑swapping templates  
- Balances dataset demographics  
- Fine‑tuned using QLoRA (parameter‑efficient training)

### **3. Few‑Shot Debiasing**
- Curated positive examples  
- Demonstrates reasoning style changes without training  

---

## 📊 Evaluation Metrics

| Metric | Description |
|-------|-------------|
| **sDIS** | Directional bias across demographic dimensions |
| **sAMB** | Bias in ambiguous questions |
| **AURC** | Calibration quality under uncertainty |
| **Log-Odds Ratio** | Identity‑based bias magnitude |

All metrics display inside the notebooks.

---

## 🧪 How to Run the Project

### **1. Create Conda environment**
```bash
conda env create -f environment.yml
conda activate llm-bias
```

### **2. Dataset Handling**
The BBQ dataset auto‑downloads using 🤗 `datasets`.  
Offline users may place a copy into:

```
data/raw/
```

### **3. Execution Notes**
Because experiments are computationally expensive, notebooks:

- show preserved results  
- do **not** require re‑running inference  
- contain all plots & metrics already computed  

---

## 📘 Report

The full research report is available at:

```
report/Bias_Mitigation_Report.pdf
```

---

## 📄 License
MIT License (or your chosen license).

---

## 🤝 Acknowledgements
- BBQ Benchmark authors  
- Meta LLaMA‑3.1  
- Hugging Face community  
- University of Sydney — Advanced Machine Learning Coursework  

---

## ⭐ If you use this work
Please cite or link to this repository.
