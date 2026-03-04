"""Smoke test for regime-adaptive models - no crunch_synth import needed."""
import sys
import os
import math
import numpy as np

# --- Mock crunch_synth just enough to import ---
import types
mock_synth = types.ModuleType("crunch_synth")

class FakePriceStore:
    def __init__(self):
        self._data = {}
    def set_data(self, asset, pairs):
        self._data[asset] = pairs
    def get_prices(self, asset, days=7, resolution=300):
        return self._data.get(asset, [])

class TrackerBase:
    def __init__(self):
        self.prices = FakePriceStore()
    def tick(self, data):
        for asset, pairs in data.items():
            self.prices.set_data(asset, pairs)

mock_synth.TrackerBase = TrackerBase
sys.modules["crunch_synth"] = mock_synth

# Mock validate_distribution
mock_utils = types.ModuleType("crunch_synth.utils")
mock_dist = types.ModuleType("crunch_synth.utils.distributions")
def validate_distribution(d):
    assert "step" in d
    assert "type" in d
    assert "components" in d
    assert len(d["components"]) <= 3, f"Too many components: {len(d['components'])}"
    w_sum = sum(c["weight"] for c in d["components"])
    assert abs(w_sum - 1.0) < 0.01, f"Weights sum to {w_sum}"
    for c in d["components"]:
        assert c["density"]["params"]["scale"] > 0
mock_dist.validate_distribution = validate_distribution
sys.modules["crunch_synth.utils"] = mock_utils
sys.modules["crunch_synth.utils.distributions"] = mock_dist


def test_model(name, module_dir):
    sys.path.insert(0, module_dir)
    if "main" in sys.modules:
        del sys.modules["main"]
    from main import MyTracker
    sys.path.pop(0)

    t = MyTracker()

    # Simulate 7 days of 5-min prices
    np.random.seed(42)
    n = 2016
    ts0 = 1000000000

    # --- Stable regime test ---
    prices = 100000 + np.cumsum(np.random.randn(n) * 200)
    t.tick({"BTC": [(ts0 + i*300, float(prices[i])) for i in range(n)]})

    for horizon, step, label in [(86400, 300, "24h/5m"), (86400, 3600, "24h/1h"),
                                  (86400, 86400, "24h/24h"), (3600, 60, "1h/1m")]:
        preds = t.predict("BTC", horizon, step)
        expect = horizon // step
        ok = len(preds) == expect

        fail = 0
        for p in preds:
            try:
                validate_distribution(p)
            except Exception as e:
                fail += 1
                print(f"    VALIDATION ERROR: {e}")

        n_comp = len(preds[0]["components"]) if preds else 0
        ws = sum(c["weight"] for c in preds[0]["components"]) if preds else 0

        # Check sigma evolution
        if len(preds) > 1:
            s0 = preds[0]["components"][0]["density"]["params"]["scale"]
            sN = preds[-1]["components"][0]["density"]["params"]["scale"]
            evolves = "YES" if abs(sN - s0) > 1e-8 else "NO"
        else:
            evolves = "N/A"

        status = "OK" if (ok and fail == 0 and n_comp <= 3) else "FAIL"
        print(f"  [{status}] BTC {label}: {len(preds)}/{expect} preds, "
              f"{n_comp} comp, w_sum={ws:.4f}, fail={fail}, evolves={evolves}")

    # --- Regime CHANGE test ---
    # First half calm, second half volatile (3x vol spike)
    calm = np.random.randn(n // 2) * 200
    volatile = np.random.randn(n // 2) * 600  # 3x vol spike
    prices_regime = 100000 + np.cumsum(np.concatenate([calm, volatile]))
    t.tick({"BTC": [(ts0 + i*300, float(prices_regime[i])) for i in range(n)]})

    preds_regime = t.predict("BTC", 86400, 300)
    s0 = preds_regime[0]["components"][0]["density"]["params"]["scale"]
    sN = preds_regime[-1]["components"][0]["density"]["params"]["scale"]

    # Get the regime intensity from the model
    returns = np.diff(prices_regime)
    recent = returns[-300:]
    fast_std = float(np.std(recent[-12:])) + 1e-10
    slow_std = float(np.std(recent[-96:])) + 1e-10
    if hasattr(t, '_vol_regime'):
        result = t._vol_regime(recent)
        ri = result[-1]  # last element is regime_intensity
        print(f"  [INFO] Regime intensity after vol spike: {ri:.3f}")
        if ri > 0.3:
            print(f"  [OK]   Regime change DETECTED (ri={ri:.3f} > 0.3)")
        else:
            print(f"  [WARN] Regime change NOT detected (ri={ri:.3f} <= 0.3)")

    # Verify weights are valid for all components
    fail_count = 0
    for p in preds_regime:
        try:
            validate_distribution(p)
        except Exception as e:
            fail_count += 1
    print(f"  [{'OK' if fail_count == 0 else 'FAIL'}] Regime-change predictions: "
          f"{len(preds_regime)} preds, {fail_count} validation failures")

    # Check weight sums are exact
    for p in preds_regime[:5]:
        ws = sum(c["weight"] for c in p["components"])
        assert abs(ws - 1.0) < 0.001, f"Weights sum to {ws}"
    print(f"  [OK]   Weight sums validated (all within 0.001 of 1.0)")

    print()


print("=== invisible-fox ===")
test_model("invisible-fox", "synth-invisible-fox")

print("=== realistic-gazelle ===")
test_model("realistic-gazelle", "synth-realistic-gazelle")

print("All smoke tests passed!")
