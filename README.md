# Gender Prediction ML Homework

## Task Overview

The goal of this homework is to **predict customer gender** based on their transaction history.  
You will need to **aggregate transaction data into features**, train various machine learning models, and evaluate their performance.  

**Baseline Goal:**  
- Achieve at least **0.87 ROC-AUC** on the test set.  

**Additional Challenges (Extra Points):**  
- Implement and test **RNN-based models** on raw transaction sequences.  
- Try **tabular LLM approaches** to leverage the full dataset without aggregation.  
- Implement a **custom ROC-AUC function** (instead of relying only on sklearn).  
- Add **assertions** on `X.shape` and data sanity checks.  

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
| `gender`        | int64           | Target variable: 0 (male), 1 (female) |
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

---

## 🛠 Feature Engineering Recommendations

* Fill missing values in `term_id` with `-1`.
* Parse `tr_datetime` into datetime and extract derived features (day, month, weekday, etc.).
* Aggregate statistics per customer:

  * Numeric: `count`, `sum`, `mean`, `std`, `min`, `max`.
  * Categorical: number of unique values (`term_id`, `mcc_code`, `tr_type`).
* Temporal features: earliest/latest transaction, transaction span, number of unique days.
* Frequencies of top merchant categories and transaction types.
* Merge all features into `X` and prepare `y` (target gender).

---

## 🤖 Model Training

* Split data by `customer_id` into train/val/test (70/10/20).
* Start with **XGBoost** and add **GridSearchCV** for hyperparameter tuning.
* Visualize grid search performance.
* Try different models:

  * Logistic Regression / Linear SVM
  * Decision Trees / Random Forest
  * Gradient Boosting (XGBoost, CatBoost, LightGBM)
  * RNN-based models on transaction sequences
  * Tabular LLM approaches

---

## 📈 Evaluation

Metrics: **ROC-AUC** must exceed 0.87