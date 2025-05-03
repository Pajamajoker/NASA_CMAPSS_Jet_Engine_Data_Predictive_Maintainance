# pipeline.py
import threading
import queue
import json
import pandas as pd
import time  # ← Added
from simulated_deployment.build.data_loader import load_data, compute_rul
from simulated_deployment.build.features    import add_rolling_features
from simulated_deployment.build.model       import load_model_and_scaler
from results     import latest_rul, rul_lock

Q = queue.Queue()

def producer(train_path, test_path, rul_path, rolling_window=False):
    train_df, test_df, rul_df = load_data(train_path, test_path, rul_path)
    _, test_df = compute_rul(train_df, test_df, rul_df)

    sensor_cols = [c for c in test_df.columns if 'sensor_measurement' in c]
    if rolling_window:
        test_df = add_rolling_features(test_df, sensor_cols, window=20)
        sensor_cols += [f"{c}_roll20" for c in sensor_cols]

    print(f"[Producer] Enqueuing {len(test_df)} records…")
    for i, row in enumerate(test_df.itertuples(), 1):
        rec = {"unit": int(row.unit_number), **{c: getattr(row, c) for c in sensor_cols}}
        Q.put(rec)
        print(f"[Producer]  → enqueued {i}/{len(test_df)}")
        time.sleep(2)  # ← Delay between each enqueue

    # send one sentinel per consumer
    for _ in range(2):
        Q.put(None)
    print("[Producer] Done enqueuing, sent sentinels.")

def consumer(worker_id):
    svr, scaler = load_model_and_scaler()
    feature_names = list(scaler.feature_names_in_)

    while True:
        rec = Q.get()
        if rec is None:
            Q.task_done()
            break

        unit = rec.pop("unit")
        df = pd.DataFrame([rec], columns=feature_names)
        X_s = scaler.transform(df)
        pred = svr.predict(X_s)[0]

        # write to log
        with open("predictions.log", "a") as f:
            f.write(json.dumps({"unit": unit, "rul": float(pred)}) + "\n")

        with rul_lock:
            latest_rul[unit] = float(pred)
        print(f"[Consumer {worker_id}] Unit {unit:02d} → RUL={pred:.2f}")
        time.sleep(2)  # ← Delay between each consume
        Q.task_done()

if __name__ == "__main__":
    # start 2 worker threads
    for wid in (1, 2):
        t = threading.Thread(target=consumer, args=(wid,), daemon=True)
        t.start()

    producer(
        "../dataset/CMAPSSData/train_FD001.txt",
        "../dataset/CMAPSSData/test_FD001.txt",
        "../dataset/CMAPSSData/RUL_FD001.txt",
        rolling_window=True
    )

    print("[Main] Waiting for queue to drain…")
    Q.join()
    print("[Main] All records processed.")
