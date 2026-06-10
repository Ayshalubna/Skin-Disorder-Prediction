# Skin Disorder Prediction

Machine learning system for differential diagnosis of **erythemato-squamous skin diseases** (6 classes: psoriasis, seborrheic dermatitis, lichen planus, pityriasis rosea, chronic dermatitis, pityriasis rubra pilaris) from clinical and histopathological features.

**Domain:** Healthcare · **Dataset:** UCI Dermatology (clinical + histopathological attributes)

## Why it matters

These six conditions share overlapping clinical symptoms, and definitive diagnosis usually requires a biopsy. A reliable classifier on clinical + histopathological inputs can act as a decision-support tool, reducing unnecessary biopsies and speeding up diagnosis.

## Approach

1. **EDA** — class balance, feature distributions, correlation analysis across clinical vs histopathological attribute groups.
2. **Feature engineering** — missing-value treatment (age), encoding, scaling.
3. **Modeling** — Logistic Regression, SVM, Random Forest and Decision Tree compared under a consistent train/test protocol.
4. **Evaluation** — per-class precision/recall/F1 (critical under class imbalance), not just headline accuracy.

## Results

| Model | Accuracy |
| --- | --- |
| Logistic Regression | 95.8% |
| SVM / Random Forest | best overall performers |
| Decision Tree | slightly lower, most clinically interpretable |

Key findings: histopathological features carry most of the predictive signal; underrepresented classes are the main source of residual error; tree-based models offer the interpretability clinicians need.

## Run it

```bash
pip install pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook "PRCP-1027-Skin Disorder.ipynb"
```

## Limitations & next steps

Class imbalance affects generalisation for rare conditions; hyperparameter tuning and calibrated probabilities are the next steps before any clinical use. This is a research/portfolio project, not a medical device.
