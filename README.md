# Credit Risk Probability Model

## Project Overview

This project involves building a credit scoring model for Bati Bank's new Buy-Now-Pay-Later (BNPL) product. The goal is to estimate the Probability of Default (PD) for customers using behavioral transaction data and convert it into a credit score to determine optimal loan amounts and durations.

## Credit Scoring Business Understanding

### 1. Basel II and Model Interpretability

In a regulated banking environment like Bati Bank, adherence to **Basel II** capital accord principles is critical. Basel II emphasizes the "Internal Ratings-Based" (IRB) approach, which allows banks to use their own estimated risk parameters for capital requirements.

- **Risk Measurement**: The core requirement is the accurate estimation of **Probability of Default (PD)**, **Loss Given Default (LGD)**, and **Exposure at Default (EAD)**.
- **Interpretability**: Regulators require that models be transparent and explainable. A "black box" model where the decision logic is opaque is generally unacceptable for credit scoring. We must be able to explain _why_ a specific applicant was approved or rejected (e.g., "high utilization ratio" or "delinquency history"). This drives the preference for models where feature contributions are clear.

### 2. The Proxy Default Variable

Since the product is new, there is no historical "Default" label in the dataset. We must engineer a **Proxy Default Variable** based on transaction behavior (RFM - Recency, Frequency, Monetary).

- **Necessity**: Without a target variable, supervised learning is impossible. We define "Bad" behavior (high risk) based on patterns like:
  - No transactions for X days (Recency).
  - Consistent low-value transactions or failed payments.
  - Abnormal spending bursts followed by dormancy.
- **Business Risks**:
  - **Misclassification**: A customer labeled "Bad" by the proxy might actually be creditworthy but just inactive (churned, not defaulted).
  - **Feedback Loop**: If the model is trained on a biased proxy, it will perpetuate that bias in future lending decisions.
  - **Validation Gap**: Until actual repayment data is collected from the live BNPL product, the proxy remains a hypothesis.

### 3. Logistic Regression (WoE) vs. Tree-Based Models

| Feature              | Logistic Regression + Weight of Evidence (WoE)                                                                                 | Gradient Boosting / Random Forest                                                                                                                                |
| :------------------- | :----------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Interpretability** | **High**. Coefficients are directly linkable to odds ratios. Using WoE handles non-linearities while keeping the model linear. | **Medium**. Feature importance plots exist, but individual prediction paths (SHAP values) are more complex to explain significantly to non-technical regulators. |
| **Performance**      | **Moderate**. Assumes linear relationships (after WoE transformation). Robust but may miss complex interactions.               | **High**. Captures complex non-linear interactions automatically. Generally higher AUC/Gini.                                                                     |
| **Stability**        | **High**. Less prone to overfitting. Scorecards (points based) are standard in banking.                                        | **Lower**. Can overfit if not heavily regularized. "Slight changes in data can lead to different trees."                                                         |
| **Regulation**       | **Industry Standard**. Accepted globally by regulators for decades.                                                            | **Emerging**. careful documentation and "post-hoc" explainability (SHAP/LIME) are required to satisfy compliance.                                                |

**Recommendation**: For the initial baseline and regulatory compliance, a **Logistic Regression model with WoE binning** is the safest and most standard approach. However, we will also train a Gradient Boosting model (XGBoost/LightGBM) to benchmark performance potential and use SHAP values to explain the complex model, offering a "champion-challenger" framework.

## Project Structure

```text
credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```
