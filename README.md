# Loan Default Prediction with Cost-Sensitive Thresholding

This project builds a machine learning pipeline to predict **loan defaulters** using a real-world imbalanced dataset (12% default rate).  
Instead of chasing accuracy, we optimize for **business cost**, balancing false positives and false negatives through threshold calibration.



##  Features

- Preprocessing with label encoding and scaling
- Handling class imbalance using XGBoost (no SMOTE needed)
- Custom cost-sensitive threshold evaluation
- SHAP-based feature interpretability (optional)
- Final model tuned for lowest financial risk, not just highest F1

---

## Dataset Overview

Each row represents a loan application with features like:

- Age, Income, LoanAmount
- CreditScore, MonthsEmployed
- LoanPurpose, EmploymentType, etc.

Target variable: `Default`  
- `1` → Defaulted  
- `0` → Paid back

---

## Business-Aware Model Evaluation

Instead of optimizing for accuracy, we minimized expected loss using this cost matrix:

| Type               | Cost      |
|--------------------|-----------|
| False Negative (missed defaulter) | ₹500,000 |
| False Positive (wrongly flagged defaulter) | ₹50,000 |

---

## Final Model

| Metric               | Value    |
|----------------------|----------|
| Model                | XGBoost  |
| Optimal Threshold    | `0.10` (based on cost minimization across folds)   |
| Recall (1)           | 70.09%   |
| Precision (1)        | 20.78%   |
| Total Cost           | ₹1.67 Billion |

---

## Getting Started

```bash
pip install -r requirements.txt 
```

Then run: 
```
# Load final model

import joblib
model = joblib.load('xgb_final_model_thresh045.pkl')
````
