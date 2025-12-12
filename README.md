# Bias Mitigation in Large Language Models (LLaMA-3.1 + BBQ Benchmark)

This repository investigates social bias in Large Language Models using the BBQ benchmark.  
It provides a fully reproducible pipeline including data loading, baseline evaluation, Counterfactual Data Augmentation (CDA), QLoRA fine‑tuning, Few‑Shot prompting, and structured fairness evaluation (sDIS, sAMB, AURC, log‑odds).

---

## 📌 Project Overview

Large Language Models (LLMs) can unintentionally propagate or amplify social biases.  
This project evaluates bias in LLaMA‑3.1‑8B‑Instruct and explores two mitigation strategies:

1. **Counterfactual Data Augmentation (CDA)**  
2. **Few‑Shot Prompting with Debiasing‑Pattern Examples**  

All methods are evaluated on the **BBQ Benchmark**, covering nine sensitive categories.

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
│   ├── raw/
│   ├── processed/
│   └── counterfactual/
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
│   ├── baseline/
│   ├── cda_qlora/
│   ├── few_shot/
│   └── summary.csv
│
└── report/
    ├── Bias_Mitigation_Report.pdf
    └── references.bib
```

---

## 🚀 Methods Implemented

### **1. Baseline Model Evaluation**
- LLaMA‑3.1‑8B‑Instruct evaluated directly on BBQ.
- Analysis of:
  - Ambiguous vs. non‑ambiguous cases  
  - Target vs. non‑target bias  
  - Incorrect inference patterns  

### **2. Counterfactual Data Augmentation (CDA)**
- Identity-swapping via lexical templates  
- Augments dataset size and balances sensitive attributes  
- QLoRA used for parameter‑efficient fine‑tuning  

### **3. Few‑Shot Debiasing**
- Manual construction of balanced exemplars  
- Introduces reasoning patterns for fairer inference  

---

## 📊 Evaluation Metrics

| Metric | Description |
|-------|-------------|
| **sDIS** | Measures directional bias across demographic dimensions |
| **sAMB** | Measures ambiguous-case bias tendencies |
| **AURC** | Area under the rejection curve (confidence calibration) |
| **Log-Odds Ratio** | Bias magnitude across identity pairs |

All metrics follow the definitions from the BBQ benchmark paper.

---

## 🧪 How to Run the Project

### **1. Install environment**
```bash
conda env create -f environment.yml
conda activate llm-bias
```

### **2. Download the BBQ dataset**
Place files into:
```
data/raw/
```

### **3. Run preprocessing**
```bash
python src/preprocess.py
```

### **4. Run baseline inference**
```bash
python src/model_baseline.py
```

### **5. Run CDA + QLoRA training**
```bash
python src/train_qlora.py
```

### **6. Run Few‑Shot prompting experiments**
```bash
python src/few_shot.py
```

### **7. Evaluate**
```bash
python src/evaluation.py
```

---

## 📈 Results Summary (High-Level)

| Method | sDIS ↓ | sAMB ↓ | AURC ↑ | Notes |
|--------|--------|--------|--------|--------|
| Baseline | High bias | High | Low | Clear preference for societal stereotypes |
| CDA + QLoRA | Reduced | Reduced | Improved | Strongest overall mitigation |
| Few‑Shot | Moderate reduction | Low | Stable | Effective without training |

(Insert your actual numbers here.)

---

## 🧱 Dependencies

- Python 3.10+
- Hugging Face Transformers
- PEFT (QLoRA)
- PyTorch
- NumPy, Pandas
- matplotlib, seaborn
- tqdm
- scikit‑learn

Full list in **environment.yml**.

---

## 📘 Report

The full academic report—including methodology, metric definitions, diagrams, and result discussion—is available in:

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
- Hugging Face ecosystem  
- University of Sydney — Advanced Machine Learning Coursework

---

## ⭐ If you use this work
Please cite or link to this repository.

---
