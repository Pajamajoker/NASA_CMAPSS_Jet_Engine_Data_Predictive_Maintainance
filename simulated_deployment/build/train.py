# train.py
from data_loader import load_data, compute_rul
from features import add_rolling_features, prepare_datasets, scale_data
from model import train_and_save, save_scaler

if __name__=="__main__":
    # load & preprocess
    train_df, test_df, rul_df = load_data(
        "../../dataset/CMAPSSData/train_FD001.txt",
        "../../dataset/CMAPSSData/test_FD001.txt",
        "../../dataset/CMAPSSData/RUL_FD001.txt"
    )
    train_df, test_df = compute_rul(train_df, test_df, rul_df)

    # optional rolling features
    sensor_cols = [c for c in train_df.columns if 'sensor_measurement' in c]
    train_df = add_rolling_features(train_df, sensor_cols, window=20)
    test_df  = add_rolling_features(test_df,  sensor_cols, window=20)
    sensor_cols += [f"{c}_roll20" for c in sensor_cols]

    # prepare & scale
    X_tr, y_tr, X_va, y_va, X_te, y_te, _ = prepare_datasets(train_df, test_df, sensor_cols)
    X_tr_s, X_va_s, X_te_s, scaler = scale_data(X_tr, X_va, X_te)

    # train & save
    train_and_save(X_tr_s, y_tr)
    save_scaler(scaler)
    print("Training complete – model and scaler saved.")
