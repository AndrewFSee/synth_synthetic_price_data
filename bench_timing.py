"""
Benchmark predict() and predict_all() timing.
Simulates what the platform does: predict_all for every asset within 40s.
"""
import sys, time, numpy as np

# Test invisible-fox
sys.path.insert(0, "synth-invisible-fox")
from main import MyTracker as FoxTracker
sys.path.pop(0)

# Constants from crunch_synth
FORECAST_PROFILES = {
    "24h": {"horizon": 86400, "steps": [300, 3600, 21600, 86400], "interval": 3600},
    "1h":  {"horizon": 3600,  "steps": [60, 300, 900, 1800, 3600], "interval": 720},
}
ASSETS_24H = ["BTC", "SOL", "ETH", "XAUT", "SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"]
ASSETS_1H  = ["BTC", "SOL", "ETH", "XAUT"]

def setup_tracker(TrackerCls):
    t = TrackerCls()
    np.random.seed(42)
    n = 2016  # 7 days of 5-min data
    ts0 = int(time.time()) - n * 300
    # Simulate all 9 assets
    for asset, base_price, vol in [
        ("BTC", 98000, 200), ("ETH", 3200, 15), ("SOL", 150, 0.8),
        ("XAUT", 2700, 5), ("SPYX", 580, 0.5), ("NVDAX", 130, 0.4),
        ("TSLAX", 350, 1.0), ("AAPLX", 230, 0.35), ("GOOGLX", 180, 0.5),
    ]:
        prices = base_price + np.cumsum(np.random.randn(n) * vol)
        t.tick({asset: [(ts0 + i*300, float(prices[i])) for i in range(n)]})
    return t

print("Setting up tracker...")
fox = setup_tracker(FoxTracker)

# Time individual predict() calls
print("\n--- Individual predict() calls ---")
for asset in ["BTC", "SOL", "AAPLX"]:
    for horizon, step in [(86400, 300), (86400, 3600), (3600, 60), (3600, 3600)]:
        t0 = time.perf_counter()
        preds = fox.predict(asset, horizon, step)
        dt = time.perf_counter() - t0
        print(f"  {asset} h={horizon:>5} step={step:>5} -> {len(preds):>3} preds  {dt*1000:>8.1f} ms")

# Time full 24h round (what platform does)
print("\n--- Full 24h prediction round (9 assets × 4 steps) ---")
t0 = time.perf_counter()
for asset in ASSETS_24H:
    for step in FORECAST_PROFILES["24h"]["steps"]:
        preds = fox.predict(asset, FORECAST_PROFILES["24h"]["horizon"], step)
total_24h = time.perf_counter() - t0
print(f"  Total: {total_24h:.2f}s  (limit: 40s)  {'OK' if total_24h < 40 else 'TIMEOUT!'}")

# Time full 1h round (what platform does)
print("\n--- Full 1h prediction round (4 assets × 5 steps) ---")
t0 = time.perf_counter()
for asset in ASSETS_1H:
    for step in FORECAST_PROFILES["1h"]["steps"]:
        preds = fox.predict(asset, FORECAST_PROFILES["1h"]["horizon"], step)
total_1h = time.perf_counter() - t0
print(f"  Total: {total_1h:.2f}s  (limit: 40s)  {'OK' if total_1h < 40 else 'TIMEOUT!'}")

# Time predict_all (which is what the framework actually calls)
print("\n--- predict_all() timing ---")
t0 = time.perf_counter()
fox.predict_all("BTC", 86400, [300, 3600, 21600, 86400])
dt = time.perf_counter() - t0
print(f"  BTC predict_all(24h): {dt:.2f}s")

t0 = time.perf_counter()
fox.predict_all("BTC", 3600, [60, 300, 900, 1800, 3600])
dt = time.perf_counter() - t0
print(f"  BTC predict_all(1h): {dt:.2f}s")

# Breakdown: what's slow?
print("\n--- Bottleneck analysis ---")
recent = np.random.randn(300) * 200

t0 = time.perf_counter()
for _ in range(36):
    fox._base_features(recent)
dt = time.perf_counter() - t0
print(f"  36× _base_features: {dt*1000:.0f} ms")

t0 = time.perf_counter()
for _ in range(36):
    fox._pretrained_base(recent)
dt = time.perf_counter() - t0
print(f"  36× _pretrained_base (features+LightGBM): {dt*1000:.0f} ms")

print("\nDone.")
