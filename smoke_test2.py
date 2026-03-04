"""Quick targeted smoke test — avoids full import overhead."""
import sys, os, time, math
import numpy as np

def test_one(name, module_dir):
    # Temporarily add path and import
    sys.path.insert(0, module_dir)
    if "main" in sys.modules:
        del sys.modules["main"]
    from main import MyTracker
    sys.path.pop(0)

    t = MyTracker()

    # Simulate 7 days of 5-min BTC prices (~100k level)
    np.random.seed(42)
    n = 2016  # 7 days * 288 per day
    prices = 100000 + np.cumsum(np.random.randn(n) * 200)
    ts0 = int(time.time()) - n * 300
    t.tick({"BTC": [(ts0 + i*300, float(prices[i])) for i in range(n)]})

    # Also add SOL (~150 level)
    sol_prices = 150 + np.cumsum(np.random.randn(n) * 0.8)
    t.tick({"SOL": [(ts0 + i*300, float(sol_prices[i])) for i in range(n)]})

    from crunch_synth.utils.distributions import validate_distribution

    for asset in ["BTC", "SOL"]:
        for horizon, step, label in [(86400, 300, "24h/5m"), (3600, 60, "1h/1m"), (86400, 86400, "24h/24h")]:
            preds = t.predict(asset, horizon, step)
            expect = horizon // step
            ok = len(preds) == expect

            # Validate all
            fail = 0
            for p in preds:
                try:
                    validate_distribution(p)
                except Exception:
                    fail += 1

            n_comp = len(preds[0]["components"]) if preds else 0
            ws = sum(c["weight"] for c in preds[0]["components"]) if preds else 0

            # Check sigma evolution (first vs last)
            if len(preds) > 1:
                s0 = preds[0]["components"][0]["density"]["params"]["scale"]
                sN = preds[-1]["components"][0]["density"]["params"]["scale"]
                evolves = "YES" if abs(sN - s0) > 1e-8 else "NO"
            else:
                evolves = "N/A"

            status = "OK" if (ok and fail == 0 and n_comp <= 3) else "FAIL"
            print(f"  [{status}] {asset} {label}: {len(preds)}/{expect} preds, "
                  f"{n_comp} components, w_sum={ws:.4f}, fail={fail}, evolves={evolves}")

    print()


print("=== invisible-fox ===")
test_one("invisible-fox", "synth-invisible-fox")

print("=== realistic-gazelle ===")
test_one("realistic-gazelle", "synth-realistic-gazelle")

print("All done.")
