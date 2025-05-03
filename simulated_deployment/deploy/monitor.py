import time
import os
import json

LOG_FILE = "../model/predictions.log"

# track latest per engine
latest = {}

# classification
def classify(rul):
    if rul > 100: return "Healthy"
    if rul > 50:  return "Minor"
    if rul > 20:  return "Major"
    return "Broken"

# clear console
def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    print("Starting file-based monitor… (Ctrl-C to exit)")
    try:
        while True:
            time.sleep(2)
            # reload log
            if not os.path.exists(LOG_FILE):
                print("[Monitor] No log file yet…")
                continue

            with open(LOG_FILE) as f:
                for line in f:
                    rec = json.loads(line)
                    u = rec['unit']; c = rec['cycle']; r = rec['rul']
                    prev = latest.get(u, {})
                    latest[u] = {'cycle': c, 'rul': r, 'prev': prev.get('rul')}

            clear()
            print(f"{'Engine':>6} │ {'Cycle':>5} │ {'RUL':>7} │ {'ΔRUL':>6} │ Status")
            print("──────┼───────┼─────────┼────────┼────────")
            for u in sorted(latest):
                info = latest[u]
                delta = info['rul'] - info['prev'] if info['prev'] is not None else 0.0
                status = classify(info['rul'])
                print(f"{u:>6} │ {info['cycle']:>5} │ {info['rul']:7.2f} │ {delta:6.2f} │ {status}")
    except KeyboardInterrupt:
        print("Monitor stopped.")
