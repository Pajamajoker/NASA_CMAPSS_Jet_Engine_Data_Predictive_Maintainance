# features.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def add_rolling_features(df, sensor_cols, window=10):
    df = df.sort_values(['unit_number','time_in_cycles'])
    for c in sensor_cols:
        df[f'{c}_roll{window}'] = (
            df.groupby('unit_number')[c]
              .rolling(window, min_periods=1)
              .mean()
              .reset_index(0,drop=True)
        )
    return df

def prepare_datasets(train_df, test_df, sensor_cols=None, val_frac=0.1):
    if sensor_cols is None:
        sensor_cols = [c for c in train_df.columns if 'sensor_measurement' in c]
    # split engine IDs
    engines = train_df.unit_number.unique()
    train_ids, val_ids = train_test_split(engines, test_size=val_frac, random_state=42)
    # train set
    X_train = train_df[train_df.unit_number.isin(train_ids)][sensor_cols]
    y_train = train_df[train_df.unit_number.isin(train_ids)]['RUL']
    # val: one random row per engine
    val_rows = []
    for uid in val_ids:
        sub = train_df[train_df.unit_number==uid]
        val_rows.append(sub.sample(1, random_state=42))
    val_df = pd.concat(val_rows)
    X_val, y_val = val_df[sensor_cols], val_df['RUL']
    # test set
    X_test, y_test = test_df[sensor_cols], test_df['RUL']
    return X_train, y_train, X_val, y_val, X_test, y_test, sensor_cols

def scale_data(X_train, X_val, X_test):
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler
