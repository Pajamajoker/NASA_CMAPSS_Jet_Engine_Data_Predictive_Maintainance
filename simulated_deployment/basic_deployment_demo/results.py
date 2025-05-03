# results.py
import threading

# thread‑safe store of latest RUL per engine
latest_rul = {}
rul_lock    = threading.Lock()
