# Gender Prediction ML Homework

## Overview
This project implements a machine learning pipeline for predicting customer gender based on transaction data. It demonstrates feature engineering, model training using XGBoost, and evaluation using multiple metrics.

---

## Data Description

The dataset contains **3,751,083 transaction records** with the following columns:

| Column          | Type             | Description |
|-----------------|-----------------|-------------|
| `customer_id`    | int64           | Unique identifier for each customer |
| `tr_datetime`    | datetime64[ns]  | Date and time of the transaction |
| `mcc_code`       | int64           | Merchant category code |
| `tr_type`        | int64           | Transaction type |
| `amount`         | float64         | Transaction amount (can be negative or positive) |
| `term_id`        | object          | Terminal ID where the transaction occurred (NaNs replaced with `-1`) |
| `gender`         | int64           | Target variable: 0 (male), 1 (female) |
| `tr_type_desc`   | object          | Description of transaction type (63 unique values) |
| `mcc_code_desc`  | object          | Description of merchant category (184 unique values) |
| `day`            | int64           | Days since reference date (derived from `tr_datetime`) |

---

## Methodology

### 1. Data Loading
Load the dataset into a pandas DataFrame.

### 2. Feature Engineering Reccomendations
- Fill missing `term_id` with `-1`.  
- Parse `tr_datetime` as datetime.  
- Aggregate transaction statistics per customer (`count`, `sum`, `mean`, `std`, `min`, `max`).  
- Count unique values for categorical features (`term_id`, `mcc_code`, `tr_type`).  
- Compute temporal features: earliest/latest transaction date, number of unique transaction days.  
- Compute frequencies of top merchant categories and transaction types.  
- Merge all features and prepare `X` (features) and `y` (target).

### 3. Model Training
- Split data into train, validation, and test sets using stratified sampling.  
- Train an XGBoost classifier on the training set with early evaluation on the validation set.  

### 4. Evaluation
Metrics used:

- **Accuracy** – overall correctness  
- **ROC-AUC** – discrimination ability  
- **Average Precision** – area under precision-recall curve

### 5. Notes
- Stratified splits ensure class balance during training and evaluation.  
- Aggregations are done at the customer level to avoid data leakage.  
- Classification threshold is set to 0.5 by default but can be adjusted.
