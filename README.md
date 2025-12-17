# Credit Risk Probability Model

## Project Overview

This project involves building a credit scoring model for Bati Bank's new Buy-Now-Pay-Later (BNPL) product. The goal is to estimate the Probability of Default (PD) for customers using behavioral transaction data and convert it into a credit score to determine optimal loan amounts and durations.

## Credit Scoring Business Understanding

## Credit Scoring Business Understanding

### 1. Basel II and Model Interpretability

In a regulated banking environment like Bati Bank, adherence to the **Basel II Accord** is non-negotiable. The Accord's emphasis on risk measurement directly governs our modeling choices:

- **Risk Measurement Focus**: Basel II requires banks to hold capital reserves proportional to their credit risk. This necessitates valid, robust estimates of **Probability of Default (PD)** as part of the Internal Ratings-Based (IRB) approach. The model is not just a classifier; it is a measurement tool for capital adequacy.
- **Influence on Interpretability**:
  - **Regulatory Scrutiny**: Under Basel II (and III), models must be transparent. Regulators and internal validation teams must understand the specific drivers of a score.
  - **The "Use Test"**: The model must be ingrained in the bank's day-to-day operations. Loan officers need to explain decision rationale to customers (e.g., "denied due to irregular transaction frequency"), which is straightforward with linear models but difficult with black-box algorithms.
  - **Documentation**: Every variable transformation and weight must be justified. Complex ensemble methods often fail to meet these stringent documentation and explainability standards without extensive "post-hoc" analysis.

### 2. Proxy Default Variable Justification

As Bati Bank's BNPL product is new, we lack historical data with explicit "Default" labels (people who took a loan and failed to pay).

- **Necessity**: Supervised learning requires a target variable ($Y$). We must therefore create a **Proxy Variable** using available behavioral data. We hypothesis that "Bad" behavior in the past correlates with future default risk.
- **Approximation Method (RFM)**: We will define the proxy using **Recency, Frequency, and Monetary (RFM)** metrics:
  - _High Risk ("Bad")_: Customers with long gaps between transactions (Recency), failed transactions, or consistently low balances.
  - _Low Risk ("Good")_: Customers with frequent, consistent activity and growing transaction volumes.
- **Business Risks of Using a Proxy**:
  - **Type I/II Errors**: We might classify a dormant but creditworthy user as "Bad" (Opportunity Cost) or an active but over-leveraged user as "Good" (Credit Risk).
  - **Bias Propagation**: If the proxy definition is flawed (e.g., penalizing seasonal usage), the model will learn and scale this flaw, potentially violating fair lending laws.
  - **Cold Start Problem**: The interactions are limited for new users, making the proxy definition unstable for them.

### 3. Model Trade-offs: Logistic Regression vs. Gradient Boosting

We weigh the industry-standard generic approach against modern machine learning techniques.

| Feature                   | Logistic Regression (with WoE)                                                                                                                                                                        | Gradient Boosting (XGBoost/LightGBM)                                                                                                                                                                           |
| :------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Interpretability**      | **High**. The use of Weight of Evidence (WoE) binning allows for a scorecard format where points are assigned to specific attribute ranges. This is natively understandable by business stakeholders. | **Low/Medium**. While feature importance is available, the decision path involves hundreds of trees. SHAP values can improve explainability, but are harder to explain to regulators than simple coefficients. |
| **Linearity**             | **Linear**. Assumes a linear relationship between the WoE-transformed features and the log-odds of default.                                                                                           | **Non-Linear**. Can automatically capture complex, non-linear interactions between variables without explicit feature engineering.                                                                             |
| **Stability**             | **High**. Generally more stable over time and less sensitive to small changes in training data.                                                                                                       | **Lower**. Prone to overfitting on small datasets if hyperparameters are not strictly tuned.                                                                                                                   |
| **Regulatory Acceptance** | **Gold Standard**. The preferred method for regulatory capital models due to its simplicity and track record.                                                                                         | **Challenger**. Often used as a "challenger" model to benchmark the potential predictive uplift, but rarely as the primary regulatory capital model without massive validation overheads.                      |

**Strategy**: We will prioritize **Logistic Regression** for the production scoring engine to ensure full regulatory compliance. We will develop a Gradient Boosting model in parallel to identify any predictive power left on the table (Potential Uplift) and to help refine feature engineering.

## Exploratory Data Analysis (EDA)

The detailed analysis is available in `notebooks/eda.ipynb`. The process includes:

1.  **Data Overview**: analyzing dataset shape, data types, and statistical summaries (central tendency, dispersion).
2.  **Numerical Distributions**: Visualizing `Amount` and `Value` to identify skewness and potential outliers using Histograms.
3.  **Categorical Distributions**: analyzing frequency and variability of features like `ProductCategory`, `ChannelId`, and `ProviderId`.
4.  **Correlation Analysis**: examining relationships between numerical variables to identify multicollinearity.
5.  **Missing Values**: Identifying gaps and defining imputation strategies (Median for numerical, Mode for categorical).
6.  **Outlier Detection**: Using Box plots to flag anomalous transactions that could distort the model.

## Project Structure

```text
credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── models/                     # Saved pkl models and transformers
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data_processing.py      # Feature engineering and target creation
│   ├── train.py                # Training and MLflow tracking
│   ├── predict.py
│   └── api/
│       ├── main.py             # FastAPI entry point
│       └── pydantic_models.py  # Data models for the API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Implementation Details

### 1. Feature Engineering Pipeline (`src/data_processing.py`)

A fully automated pipeline that transforms raw transaction logs into customer-level features:

- **Temporal Features**: Extraction of hour, day, month, and year from timestamps.
- **Aggregation**: Programmatic generation of `total_amount`, `avg_amount`, `transaction_count`, and `std_amount` per customer.
- **Preprocessing**: `SimpleImputer` (Median/Mode), `StandardScaler`, and `OneHotEncoder`.
- **WoE & IV**: Custom Weight of Evidence transformer to maximize interpretability. Features with Information Value (IV) below 0.02 are automatically dropped.

### 2. High-Risk Proxy Definition

In the absence of historical default labels, we engineered a deterministic target variable:

- **RFM Clustering**: KMeans clustering ($k=3$) on Recency, Frequency, and Monetary metrics.
- **Risk Assignment**: The "Least Engaged" cluster (High Recency, Low Frequency, Low monetary) is programmatically identified and labeled as `is_high_risk = 1`.

### 3. Model Training & Tracking (`src/train.py`)

- **Experiment Tracking**: Integrated with **MLflow** to log parameters, metrics, and artifacts (confusion matrices).
- **Hyperparameter Tuning**: `GridSearchCV` used for both Logistic Regression and Random Forest.
- **Best Model Selection**: Automatic selection based on ROC-AUC scores on a stratified test set.

### 4. Deployment API (`src/api/main.py`)

FastAPI service exposing a `/predict` endpoint:

- **Input**: Customer transaction behavioral summaries.
- **Output**: Classification (`is_high_risk`) and Probability of Default (PD Score).
- **Validation**: Strict schema validation using Pydantic models.

## How to Run

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized run)

### Local Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Process Data**:
   Generate features and Create high-risk proxy labels.
   ```bash
   python src/data_processing.py
   ```
3. **Train Models**:
   Run training, log to MLflow, and save the best model.
   ```bash
   python src/train.py
   ```
4. **Launch API**:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```

### Running with Docker

1. **Spin up the container**:
   ```bash
   docker-compose up --build
   ```
2. **Access the API**:
   - Prediction: `POST http://localhost:8000/predict`
   - Interactive Docs: `http://localhost:8000/docs`

## Quality Assurance

- **Linting**: Run `flake8 src tests`
- **Testing**: Run `python -m pytest tests/`
