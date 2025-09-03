import pandas as pd
import numpy as np
from xgboost import XGBClassifier

def baseline_aggregates(df):
    # Replace NaNs with new class
    df['term_id'] = df['term_id'].fillna(-1)  # or any unique sentinel value

    # Ensure datetime is parsed
    df['tr_datetime'] = pd.to_datetime(df['tr_datetime'], errors='coerce')
    assert np.issubdtype(df['tr_datetime'].dtype, np.datetime64), "tr_datetime must be parsed"

    # ─────────────────────────────────────
    # Feature Engineering Per Customer
    # ─────────────────────────────────────

    # 1. Basic transaction stats + counts of unique categorical features
    tx_stats = df.groupby('customer_id').agg({
        'amount': ['count', 'sum', 'mean', 'std', 'min', 'max'],
        'term_id': pd.Series.nunique,
        'mcc_code': pd.Series.nunique,
        'tr_type': pd.Series.nunique,
    })

    # Flatten multiindex columns
    tx_stats.columns = [f'base_{k}_{stat}' for k, stat in tx_stats.columns]
    tx_stats.reset_index(inplace=True)

    # 3. Temporal patterns
    df['day'] = (df['tr_datetime'] - pd.to_datetime("2000-01-01")).dt.days

    temporal_stats = df.groupby('customer_id').agg({
        'day': ['min', 'max', pd.Series.nunique],
    })
    temporal_stats.columns = [f'temp_{k}_{stat}' for k, stat in temporal_stats.columns]
    temporal_stats.reset_index(inplace=True)

    # 4. MCC frequency
    top_mcc = df['mcc_code'].value_counts().index
    df_top_mcc = df[df['mcc_code'].isin(top_mcc)]
    mcc_freq = pd.crosstab(df_top_mcc['customer_id'], df_top_mcc['mcc_code'])
    mcc_freq.columns = [f'mcc_{c}' for c in mcc_freq.columns]
    mcc_freq.reset_index(inplace=True)

    # 5. Transaction type frequency
    trtype_freq = pd.crosstab(df['customer_id'], df['tr_type'])
    trtype_freq.columns = [f'trtype_{c}' for c in trtype_freq.columns]
    trtype_freq.reset_index(inplace=True)

    # ─────────────────────────────────────
    # Merge all features
    # ─────────────────────────────────────
    features = tx_stats \
        .merge(temporal_stats, on='customer_id', how='left') \
        .merge(mcc_freq, on='customer_id', how='left') \
        .merge(trtype_freq, on='customer_id', how='left') 

    # Add target
    target = df[['customer_id', 'gender']].drop_duplicates()
    features = features.merge(target, on='customer_id', how='left')

    # Final data
    X = features.drop(columns=['customer_id', 'gender'])
    y = features['gender']

    return X, y


def fit_predict_xgb(X_train, y_train, X_val, y_val, X_test):
    model = XGBClassifier()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    return y_pred_proba, y_pred