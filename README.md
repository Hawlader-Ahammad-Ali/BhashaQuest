# BhashaQuest 2026: Multi-Pillar Code-Switched & Transliterated Bengali Safety Taxonomy Classification

[![Competition](https://img.shields.io/badge/Kaggle-BhashaQuest%202026-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/)
[![Metric](https://img.shields.io/badge/Evaluation%20Metric-Macro%20F1-success?style=flat-square)]()
[![Public LB](https://img.shields.io/badge/Public%20LB%20(30%25)-0.76336-blue?style=flat-square)]()
[![Private LB](https://img.shields.io/badge/Private%20LB%20(70%25)-0.76455-brightgreen?style=flat-square)]()

## 1. Executive Summary & Competition Results

This repository contains the complete, leak-free machine learning pipeline developed for the **BhashaQuest 2026** 4-class text classification challenge (`14AC78`, `19E1A1`, `6C0197`, `72A2C8`), evaluated on **Macro F1-Score**.

To address the challenges of noisy Banglish (Romanized Bengali), code-switched English-Bengali syntax, and severe inter-class semantic overlap, we engineered a **9-Pillar Heterogeneous Ensemble** combining diverse multilingual transformers, lexical/subword n-gram models, and gradient-boosted decision trees, topped by a **Strictly Leak-Free Level-2 Meta-Stacker**.

| Evaluation Split | Split Ratio | Macro F1-Score | Key Takeaway |
| :--- | :---: | :---: | :--- |
| **5-Fold Stratified OOF CV** | 100% Train (8,195 rows) | **0.77086** | Zero-leak fold-isolated feature & meta pipeline |
| **Public Leaderboard** | ~30% Test (614 rows) | **0.76336** | Consistent improvement over single-model baselines |
| **Private Leaderboard (Final)** | ~70% Test (1,434 rows) | **0.76455** | **+0.00119 positive shake-up** confirming strong generalization |

---

## 2. Problem Formulation & Domain Challenges

1. **Dual-Script & Transliteration Variance:** Input texts are predominantly written in informal Romanized Bengali (*Banglish*) mixed with English technical terms and spelling variations (e.g., *kibhabe / kivabe*, *dhora pora*).
2. **Fine-Grained Class Boundary Ambiguity:** While benign/creative prompts (`14AC78`) achieve >0.90 F1, adversarial jailbreaks (`72A2C8`), illicit procedural queries (`19E1A1`), and toxic/harmful statements (`6C0197`) share overlapping vocabulary, creating high confusion between `19E1A1` and `72A2C8`.
3. **30% Public / 70% Private Shake-Up Risk:** With only ~614 samples in the Public LB, optimizing post-processing thresholds on public feedback risks severe overfitting. All architectural decisions in our final pipeline were governed by **strictly out-of-fold validation** to maximize performance on the unseen 70% Private split.

---

## 3. System Architecture & Methodology

### 3.1 Dual-Script Preprocessing (`Avro` Phonetic Normalization)
Every raw Banglish sample is phonetically transliterated into native Bengali script using `avro-py` and paired with the original Romanized text (`"{raw_text} | {avro_bengali}"`). This enables multilingual backbones (e.g., `BanglaBERT`, `XLM-RoBERTa`, `mDeBERTa-v3`) to leverage both native Bengali subword representations and Latin-script semantic cues simultaneously.

### 3.2 Level-1 Diverse Model Pillars (5-Fold Stratified CV)
All Level-1 models were trained using an identical `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` split to guarantee exact Out-Of-Fold (OOF) alignment:

| Pillar | Model Family / Architecture | Key Training & Feature Engineering Details | Standalone OOF Macro F1 |
| :---: | :--- | :--- | :---: |
| **P1** | **BanglaBERT (`csebuetnlp/banglabert`)** | Dual-script input, Mean+Max pooling, Cosine warmup schedule | `0.72638` |
| **P2** | **XLM-RoBERTa-Base** | Multilingual cross-script representations, FP16 mixed precision | `0.69543` |
| **P3** | **Dual-Script Word/Subword TF-IDF** | Sublinear TF-IDF (`ngram_range=(1,3)`) + Regularized Linear/Logistic models | `0.74661` |
| **P4** | **CatBoost-v12 Multi-Variant Ensemble** | Weighted blend (`0.35 / 0.35 / 0.30`) across 3 domain-engineered feature views (`none`, `v1`, `v2`) | `0.72909` |
| **P5** | **mDeBERTa-v3-Base** | Disentangled relative positional attention across dual-script sequences | `0.71044` |
| **P6** | **BanglaBERT-Large (335M)** | Multi-Sample Dropout (5 heads), FGM adversarial embedding perturbations | `0.73400` |
| **P7** | **XLM-RoBERTa-Large (560M)** | Dual-script encoding, Mean+Max pooling, Multi-Sample Dropout + FGM (`eps=0.30`) | `0.74883` |
| **P8** | **Char+Word N-Gram Variant Model** | Word `(1,3)` + Character `char_wb (2,5)` TF-IDF with Calibrated Ridge & L2 Logistic Regression | `0.72430` |
| **P9** | **Fold-Isolated SVD + Lexical Meta-Features** | 40-component `TruncatedSVD` fit strictly inside each CV fold + structural text statistics | Feature Pillar |

### 3.3 Level-2 Leak-Free Meta-Stacker (`OOF Macro F1: 0.77086`)
Instead of simple probability averaging, we constructed a meta-feature matrix combining:
* **Base Class Probabilities:** Out-of-fold probability distributions across all base pillars.
* **Uncertainty & Calibration Signals:** Per-pillar Shannon Entropy ($H(p) = -\sum p_i \log p_i$) and Top-1 vs. Top-2 confidence margins to inform the meta-learner when specific backbones are uncertain.
* **Cross-Pillar Consensus & Dispersion:** Element-wise mean, standard deviation, min-max range across pillars, and geometric mean agreement ($\sqrt{p_{\text{XLMR-L}} \cdot p_{\text{CharWord}}}$) between orthogonal neural and lexical signals.
* **Meta-Learner Blend:** Predictions from three distinct Level-2 learners trained strictly out-of-fold:
  * **CatBoost-v13** (`depth=5`, `l2_leaf_reg=10.0`, `lr=0.025`) — OOF F1: `0.77049`
  * **LightGBM-v2** (`num_leaves=20`, `colsample_bytree=0.70`, `reg_lambda=4.0`) — OOF F1: `0.76859`
  * **Regularized Logistic Regression** (`C=0.08`, standardized features) — OOF F1: `0.77025`
  * **Final Level-2 Blend (`0.45 CB + 0.35 LGB + 0.20 LR`)** — **OOF Macro F1: `0.77086`**

---

## 4. Validation Performance Breakdown (OOF Classification Report)

```text
              precision    recall  f1-score   support

      14AC78     0.9197    0.8943    0.9068      2791
      19E1A1     0.7192    0.7297    0.7244      1994
      6C0197     0.7553    0.7131    0.7336      1255
      
      72A2C8     0.7000    0.7383    0.7186      2155

    accuracy                         0.7855      8195
   macro avg     0.7735    0.7689    0.7709      8195
weighted avg     0.7879    0.7855    0.7864      8195
