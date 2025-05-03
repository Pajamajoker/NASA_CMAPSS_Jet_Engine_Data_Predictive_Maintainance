# monitor.py
import time, json

LOG_FILE = "predictions.log"

def load_preds():
    d = {}
    try:
        with open(LOG_FILE) as f:
            for line in f:
                rec = json.loads(line)
                d[rec["unit"]] = rec["rul"]
    except FileNotFoundError:
        pass
    return d

def classify(r):
    if r>100: return "Healthy"
    if r>50:  return "Minor"
    if r>10:  return "Major"
    return "Broken"

if __name__=="__main__":
    print("Starting file‑based monitor…")
    while True:
        time.sleep(5)
        preds = load_preds()
        if not preds:
            print("[Monitor] No predictions yet…")
        else:
            print("[Monitor] Engine statuses:")
            for u,r in sorted(preds.items()):
                print(f"  Engine {u:02d}: RUL={r:.1f} → {classify(r)}")
        print("-"*40)
