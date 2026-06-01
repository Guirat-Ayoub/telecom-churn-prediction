[README.md](https://github.com/user-attachments/files/28461996/README.md)
# Customer Churn Prediction

Predicting which telecom customers are likely to cancel their subscription
using a full end-to-end machine learning pipeline — from raw data to model evaluation.

---

## Problem Statement

Customer churn is one of the most costly problems in the telecom industry.
Acquiring a new customer costs 5–7x more than retaining an existing one.
This project builds a binary classifier to identify high-risk customers
before they leave, enabling targeted retention strategies.

---

## Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers × 21 features
- **Target:** `Churn` (Yes / No) — 26.5% positive class (imbalanced)
- **Features:** Contract type, tenure, monthly charges, internet service,
  payment method, and 15 other customer attributes

---

## Pipeline Overview

```
Raw Data → Data Cleaning → Feature Engineering → Preprocessing Pipeline
→ Model Training (4 models) → Threshold Tuning → Evaluation
```

---

## Feature Engineering

Three custom features were created beyond the raw columns:

| Feature | Logic | Rationale |
|---|---|---|
| `avg_spend` | TotalCharges / (tenure + 1) | Captures spending rate, not just total spend |
| `tenure_stage` | new / mid / loyal buckets | Encodes business lifecycle stage |
| `num_services` | Count of active services | Higher services = higher switching cost |

---

## Models Compared

All models were evaluated using 5-fold Stratified Cross-Validation
with ROC-AUC as the metric (chosen over accuracy due to class imbalance).

| Model | ROC-AUC | Std |
|---|---|---|
| Logistic Regression | 0.8446 | ±0.0134 |
| Gradient Boosting | **0.8475** | ±0.0106 |
| Random Forest | 0.8272 | ±0.0126 |
| XGBoost | 0.8228 | ±0.0128 |

Gradient Boosting achieved the highest ROC-AUC.
XGBoost was selected for the final model due to speed and extensibility.

---

## Threshold Tuning

The default threshold of 0.5 was not optimal for this imbalanced problem.
A precision-recall analysis was conducted across thresholds:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.20 | 0.49 | 0.78 | 0.60 |
| **0.25** | **0.50** | **0.74** | **0.60** |
| 0.30 | 0.53 | 0.69 | 0.60 |
| 0.50 | 0.60 | 0.53 | 0.56 |

Threshold 0.25 was chosen to maximise recall. In a churn context,
missing a churner (false negative) is more costly than a false alarm.

---

## Final Model Results

Model: XGBoost — Decision threshold: 0.25

| Metric | Score |
|---|---|
| Accuracy | 73.7% |
| ROC-AUC | 0.818 |
| Recall (churn class) | 0.74 |
| Precision (churn class) | 0.50 |
| F1 (churn class) | 0.60 |

Confusion matrix: 275 true churners identified, 99 missed out of 374.

---

## Tech Stack

- Python 3.12
- pandas, NumPy
- scikit-learn (Pipeline, ColumnTransformer, StratifiedKFold)
- XGBoost, GradientBoostingClassifier
- Matplotlib, Seaborn

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/Guirat-Ayoub/customer-churn-prediction
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook directly in Google Colab:

https://colab.research.google.com/github/Guirat-Ayoub/customer-churn-prediction/blob/main/Telco_Customer_Churn.ipynb

Replace `Guirat-Ayoub` with your GitHub username. The link works automatically
once the repository is public.

---

## Key Takeaways

- Month-to-month contract customers churn at a significantly higher rate than those on annual contracts
- Customers who churn have considerably shorter tenure compared to retained customers
- Lowering the decision threshold from 0.5 to 0.25 increased recall from 53% to 74% at a manageable precision cost
- Gradient Boosting slightly outperformed XGBoost on cross-validation but XGBoost is preferred for production use

---

## Conclusion

This project aimed to predict customer churn in a telecom company using machine learning techniques.
After performing data preprocessing, exploratory data analysis, feature engineering, and model
evaluation, several classification algorithms were compared using cross-validation and ROC-AUC scores.

The results showed that Gradient Boosting and Logistic Regression achieved the best cross-validation
performance, with ROC-AUC scores around 0.84–0.85, indicating good discrimination between churning
and non-churning customers.

To improve the business value of the model, threshold tuning was performed. Instead of using the
default threshold of 0.5, lower thresholds were evaluated to increase the detection of customers at
risk of churning. A threshold of 0.25 increased churn recall from approximately 53% to 74%, allowing
the model to identify a much larger proportion of customers likely to leave.

The analysis also highlighted that customer churn is strongly associated with:

- **Short customer tenure** — new customers are the highest-risk group
- **Month-to-month contracts** — flexible contracts have far higher churn rates
- **High monthly charges** — customers paying more are more likely to question their value
- **Lack of additional services** — customers without tech support or online security show higher churn

From a business perspective, these findings suggest that retention strategies should focus on new
customers, customers with flexible contracts, and customers who do not use value-added services.

Overall, the project demonstrates a complete machine learning workflow — including data preparation,
feature engineering, model comparison, cross-validation, threshold optimisation, and business
interpretation. The final model can help telecom companies proactively identify at-risk customers
and design targeted retention campaigns to reduce churn and improve long-term revenue.
