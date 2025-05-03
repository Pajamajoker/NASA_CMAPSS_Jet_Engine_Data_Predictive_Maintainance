# model.py
import joblib
from pathlib import Path
from sklearn.svm import SVR

MODEL_FILE  = Path("../model/svr_model.joblib")
SCALER_FILE = Path("../model/scaler.joblib")

def train_and_save(X_train, y_train):
    svr = SVR(C=0.1, epsilon=0.05, kernel='rbf')
    svr.fit(X_train, y_train)
    joblib.dump(svr, MODEL_FILE)
    return svr

def save_scaler(scaler):
    joblib.dump(scaler, SCALER_FILE)

def load_model_and_scaler():
    if not MODEL_FILE.exists() or not SCALER_FILE.exists():
        raise FileNotFoundError("Run train.py first to produce model and scaler.")
    svr    = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return svr, scaler
