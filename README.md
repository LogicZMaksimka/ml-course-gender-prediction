# ML Course Homework: Gender Prediction from Transactions

## Task Overview

The goal of this homework is to **predict customer gender** based on their transaction history.  
You will need to **aggregate transaction data into features**, train various machine learning models, and evaluate their performance.  


- **Baseline Goal:** Achieve at least **0.86 ROC-AUC** and **0.78 accuracy** on the test set.  
---

## 📊 Dataset Details  

The dataset contains **3,751,083 transaction records** for **8,400 customers**.  
It is split into **70% train, 10% validation, and 20% test** by `customer_id`.  

| Column          | Type             | Description |
|-----------------|-----------------|-------------|
| `customer_id`   | int64           | Unique identifier for each customer |
| `tr_datetime`   | datetime64[ns]  | Date and time of the transaction |
| `mcc_code`      | int64           | Merchant category code |
| `tr_type`       | int64           | Transaction type |
| `amount`        | float64         | Transaction amount (can be negative or positive) |
| `term_id`       | object          | Terminal ID where the transaction occurred (NaNs replaced with `-1`) |
| `gender`        | int64           | Target variable: 0 (female), 1 (male) |
| `tr_type_desc`  | object          | Description of transaction type (63 unique values) |
| `mcc_code_desc` | object          | Description of merchant category (184 unique values) |
| `day`           | int64           | Days since reference date (derived from `tr_datetime`) |

**Sample records:**

| customer_id | tr_datetime         | mcc_code | tr_type | amount   | term_id | gender | tr_type_desc                                  | mcc_code_desc                                                  | gender_desc |
|------------:|:-------------------|---------:|--------:|---------:|--------:|-------:|:---------------------------------------------|:---------------------------------------------------------------|:------------|
| 0           | 2000-01-01 10:23:26 | 0       | 0       | -2245.92 | -1      | 1      | Оплата услуги. Банкоматы СБ РФ               | Звонки с использованием телефонов, считывающих магнитную ленту | мужчина     |
| 0           | 2000-01-02 10:19:29 | 1       | 1       | 56147.9  | -1      | 1      | Взнос наличных через АТМ (в своем тер.банке) | Финансовые институты — снятие наличности автоматически         | мужчина     |

➡️ Explore the dataset here: [HF Hub Link](https://huggingface.co/datasets/mks-logic/gender_prediction/tree/main)

---

## ⚙️ Setup Instructions  

### 1. Install Dependencies  

**Python >= 3.8 recommended**  

Using `venv`:  
```bash
python3.10 -m venv gender_prediction_env
source gender_prediction_env/bin/activate
pip install -r requirements.txt
````

Using `conda`:

```bash
conda create -n gender_prediction_env python=3.10
conda activate gender_prediction_env
pip install -r requirements.txt
```

### 2. Download Dataset

Activate the environment and download the dataset:

```bash
python download_dataset.py
```

This will save the dataset to `./data`.

## Full Task Description

All the code you need should be written in gender_prediction.ipynb, and all the supporting scripts and infrastructure are included here.



### 1. 🛠 Feature Engineering
- Improve the aggregation function for transactions.  
- Suggested improvements:
  - Add more amount-based statistics (mean, std, min, max, quantiles, etc.)
  - Capture temporal patterns (time of day, day of week, number of transactions per week/month)
  - Improve the representation of transaction types and MCC codes  
    - Options: categorical encoding, text embeddings  

---

### 2. 📈 Model Training & Evaluation
- Train and compare multiple classic ML algorithms:
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost / CatBoost

- Experiment with full transaction sequences instead of aggregated features:
  - RNNs (LSTM/GRU)
  - Tabular transformers ([PyTorch Tabular](https://github.com/manujosephv/pytorch_tabular)).  
- Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Visualize the results of the hyperparameter search
- Goal: achieve ROC AUC > 0.86 and accuracy > 0.78

---


## ⚙️ Baseline Functions

To help you get started, the notebook provides baseline functions for training and evaluating models:

- `baseline_aggregation(df)`  
  Aggregates transaction data into basic per-customer features (`count`, `sum`, `unique MCC/tr_type`). Returns `(X, y)`.

- `fit_predict_rf(X_train, y_train, X_val, y_val, X_test)`  
  Fits a **Random Forest classifier** on the training set and returns predicted probabilities and binary labels for the test set.

- `eval(y_true, y_pred, y_pred_proba)`  
  Computes evaluation metrics:
  - ROC-AUC
  - Accuracy
  - Precision
  - Recall