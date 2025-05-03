# pipeline.py
import threading
import queue
import json
import random
import time
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from build.data_loader import load_data, compute_rul
from build.features import add_rolling_features
from build.model import load_model_and_scaler

# In-memory queue
Q = queue.Queue()
# Log file for monitor
LOG_FILE = "../model/predictions.log"

# Prepare producer to interleave engine sequences randomly
def producer(train_path, test_path, rul_path, rolling_window=False):
    # load only test data (RUL already computed in train.py)
    _, test_df, _ = load_data(train_path, test_path, rul_path)

    sensor_cols = [c for c in test_df.columns if c.startswith('sensor_measurement')]
    if rolling_window:
        test_df = add_rolling_features(test_df, sensor_cols, window=20)
        sensor_cols += [f"{c}_roll20" for c in sensor_cols]

    # group rows by engine
    groups = {}
    for row in test_df.itertuples():
        uid = int(row.unit_number)
        groups.setdefault(uid, []).append(row)

    # Clear previous log
    open(LOG_FILE, 'w').close()

    total = sum(len(lst) for lst in groups.values())
    count = 0
    print(f"[Producer] Enqueuing {total} records (interleaved)…")

    # While any engine has remaining rows, pick random engine and emit its next row
    while any(groups.values()):
        # choose among engines with data
        available = [uid for uid, lst in groups.items() if lst]
        uid = random.choice(available)
        row = groups[uid].pop(0)
        rec = {"unit": uid, "cycle": int(row.time_in_cycles), **{c: getattr(row, c) for c in sensor_cols}}
        Q.put(rec)
        count += 1
        print(f"[Producer] → enqueued {count}/{total} (Engine {uid:02d}, cycle {rec['cycle']})")
        #time.sleep(1)

    # signal consumers
    for _ in range(2):
        Q.put(None)
    print("[Producer] Done enqueuing.")


def consumer(worker_id):
    svr, scaler = load_model_and_scaler()
    feature_names = list(scaler.feature_names_in_)

    while True:
        rec = Q.get()
        if rec is None:
            Q.task_done()
            print(f"[Consumer {worker_id}] exiting.")
            break

        unit = rec.pop("unit")
        cycle = rec.pop("cycle")
        df = pd.DataFrame([rec], columns=feature_names)
        X_s = scaler.transform(df)
        pred = svr.predict(X_s)[0]

        # append to log for monitor
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps({"unit": unit, "cycle": cycle, "rul": float(pred)}) + "\n")

        print(f"[Consumer {worker_id}] Engine {unit:02d}, cycle {cycle} → RUL={pred:.2f}")
        time.sleep(0.25)
        Q.task_done()


if __name__ == "__main__":
    # start consumers
    for wid in (1, 2, 3):
        t = threading.Thread(target=consumer, args=(wid,), daemon=True)
        t.start()

    producer(
        "../../dataset/CMAPSSData/train_FD001.txt",
        "../../dataset/CMAPSSData/test_FD001.txt",
        "../../dataset/CMAPSSData/RUL_FD001.txt",
        rolling_window=True
    )

    print("[Main] Waiting for queue to drain…")
    Q.join()
    print("[Main] All records processed.")