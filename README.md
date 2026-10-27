# Explainable XGBoost for Customer Lifetime Value Prediction

## 1. Project Overview

Customer Lifetime Value (CLV) prediction estimates how much a customer is likely to spend in the future based on their historical purchasing behaviour.

This project develops an **explainable XGBoost regression model** using real-world retail transaction data. It compares XGBoost with a Linear Regression baseline and uses SHAP to interpret the model's predictions.

Two separate secondary analyses extend the project:

* Predicting whether a customer will make another purchase.
* Identifying high-value customers using predicted future spending.

The primary objective is to build a reliable customer-level prediction pipeline while maintaining clear evaluation and interpretability.

## 2. Dataset

**Dataset:** Online Retail II

The dataset contains historical retail transactions, including invoice details, product information, quantities, prices, customer identifiers, and countries.

The original Excel workbook contains two sheets:

* `Year 2009-2010`
* `Year 2010-2011`

### Dataset Summary

| Description                             |     Count |
| --------------------------------------- | --------: |
| Original transaction records            | 1,067,371 |
| Records after cleaning                  |   805,549 |
| Customers with historical data          |     4,966 |
| Customers with positive future spending |     2,581 |

The original workbook is not included in this repository. Place `online_retail_II.xlsx` in `data/raw/` before running the pipeline.

## 3. Project Methodology

### Step 1: Data Cleaning

The cleaning process:

* Removes transactions without customer identifiers.
* Excludes cancellation invoices.
* Removes transactions with non-positive quantities or prices.
* Calculates transaction revenue for subsequent customer-level analysis.

After cleaning, 805,549 transaction records remain.

### Step 2: Historical Feature Engineering

Customer features are constructed using transactions before **9 June 2011**.

The model uses the following ten features:

| Feature             | Description                                                                     |
| ------------------- | ------------------------------------------------------------------------------- |
| `recency_days`      | Days since the customer's most recent historical purchase                       |
| `tenure_days`       | Days between the customer's first historical purchase and the historical cutoff |
| `frequency`         | Number of distinct historical invoices                                          |
| `monetary_total`    | Total historical customer spending                                              |
| `monetary_avg`      | Average monetary value per historical invoice                                   |
| `monetary_std`      | Standard deviation of monetary value                                            |
| `avg_qty`           | Average quantity purchased per transaction record                               |
| `product_diversity` | Number of distinct products purchased                                           |
| `active_months`     | Number of distinct months with purchases                                        |
| `is_uk`             | Indicator for customers located in the United Kingdom                           |

All features are derived from historical transactions to avoid using future purchasing information as model input.

### Step 3: Target Definition and Data Splitting

The target is the customer's total spending between **9 June 2011 and 9 December 2011, inclusive**.

A customer-level train/test split is used:

* Training set: 80%
* Testing set: 20%
* Random seed: 42

The 99th percentile of the training target is used as the spending cap. The same cap is applied to the training and test targets.

This reduces the influence of extreme spending values while ensuring that the cap is determined without using the test target distribution.

The training-derived cap in the reported run was **£9,971.23**.

### Step 4: Model Training and Evaluation

Two regression models are evaluated:

1. Linear Regression — baseline model.
2. XGBoost Regressor — primary prediction model.

The models are evaluated using:

* **R²:** Proportion of target variance explained by the model.
* **MAE:** Mean absolute prediction error.
* **RMSE:** Root mean squared prediction error, which penalises larger errors more strongly.

### Regression Results

| Model             |         R² |    MAE (£) |   RMSE (£) |
| ----------------- | ---------: | ---------: | ---------: |
| Linear Regression |     0.4608 |     515.02 |   1,019.60 |
| **XGBoost**       | **0.5733** | **461.54** | **907.05** |

XGBoost outperformed the Linear Regression baseline on all three reported metrics.

On the test set, XGBoost achieved an R² of **0.5733**, with a mean absolute error of approximately **£461.54**.

These results correspond to the specified split and capped target; they should not be interpreted as guaranteed performance on other datasets or splits.

## 4. Explainability Using SHAP

SHAP (SHapley Additive exPlanations) is used to interpret the trained XGBoost model.

The `TreeExplainer` method calculates feature contributions to the model's predictions. The summary plot provides an overview of feature importance and the direction of feature effects.

The generated figure is available at:

`reports/figures/shap_summary.png`

The plot indicates that historical monetary value, recency, tenure, and purchase frequency are among the important features influencing predictions.

SHAP explains model behaviour; it does not establish that a feature causes a customer to spend more or less.

## 5. Secondary Analysis: Predicting Repeat Spending

A separate XGBoost classifier predicts whether a customer will have positive future spending.

This classification task is evaluated independently of the primary regression model.

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 73.94% |
| Precision | 76.12% |
| Recall    | 72.43% |
| F1-score  | 74.23% |
| ROC-AUC   | 80.84% |

The ROC-AUC of **0.8084** indicates that the classifier demonstrates useful discrimination between customers with positive and zero future spending on the evaluated test set.

## 6. Secondary Analysis: High-Value Customer Identification

The project also evaluates the identification of customers in the top 20% and top 10% of future spending.

The spending thresholds are derived from the training data. Accuracy is reported alongside precision, recall, and F1-score because the high-value groups are smaller than the remaining customer groups.

| Metric                         | Top 20% Identification | Top 10% Identification |
| ------------------------------ | ---------------------: | ---------------------: |
| Training-derived threshold (£) |                951.456 |              1,788.014 |
| Accuracy                       |                 86.32% |                 93.56% |
| Precision                      |                 63.69% |                 68.35% |
| Recall                         |                 55.87% |                 58.06% |
| F1-score                       |                 59.52% |                 62.79% |

Although accuracy is high, recall is more moderate. Therefore, the model does not identify every high-value customer.

These results are supplementary analyses and do not replace the primary regression metrics.

## 7. Project Structure

```text
clv_single_xgboost/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── linear_regression_baseline.joblib
│   ├── xgboost_clv.joblib
│   └── xgboost_will_spend_again_classifier.joblib
├── reports/
│   └── figures/
│       └── shap_summary.png
├── src/
│   ├── 01_clean_data.py
│   ├── 02_build_features.py
│   ├── 03_train_model.py
│   ├── 04_shap_explainability.py
│   └── 05_secondary_analysis.py
├── .gitignore
├── README.md
└── requirements.txt
```

Generated datasets are stored in `data/processed/`. The raw workbook and intermediate CSV files are excluded from Git tracking.

## 8. Technologies Used

* **Python** — data processing and model development
* **pandas** — transaction cleaning and feature engineering
* **NumPy** — numerical operations
* **scikit-learn** — data splitting, baseline regression, and evaluation metrics
* **XGBoost** — regression and classification
* **SHAP** — model explainability
* **Matplotlib** — visualisation

## 9. Installation and Execution

### Prerequisites

* Python 3.12
* Git
* The Online Retail II Excel workbook

Open PowerShell in the project root directory.

### Create the environment and install dependencies

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Add the dataset

Place the workbook here:

```text
data/raw/online_retail_II.xlsx
```

### Run the pipeline

Execute the scripts in the following order:

```powershell
python -u src/01_clean_data.py
python -u src/02_build_features.py
python -u src/03_train_model.py
python -u src/04_shap_explainability.py
python -u src/05_secondary_analysis.py
```

The scripts clean the transactions, build customer-level features, train the models, generate the SHAP plot, and perform the secondary analyses.

## 10. Limitations

* Customer spending patterns may change over time.
* Capping the target reduces the influence of extreme values but limits predictions of unusually large future spending.
* The evaluation uses one customer-level train/test split; results may vary with different splits.
* High-value customer identification has moderate recall, so some high-value customers may be missed.
* SHAP explains the model's learned relationships but does not demonstrate causation.
* Performance on future transactions or other retail datasets requires additional validation.

## 11. Conclusion

This project implements an explainable customer lifetime value prediction workflow using real retail transaction data.

The primary XGBoost regressor outperformed the Linear Regression baseline on the evaluated test set. SHAP was used to interpret the model, while separate classification analyses explored repeat spending and high-value customer identification.

The project demonstrates practical applications of customer-level feature engineering, regression evaluation, explainable machine learning, and supplementary classification analysis.
