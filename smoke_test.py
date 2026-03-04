"""Smoke test for both trackers."""
import sys, time, numpy as np

def test_tracker(name, module_path):
    sys.path.insert(0, module_path)
    # Force reimport
    if "main" in sys.modules:
        del sys.modules["main"]
    from main import MyTracker
    sys.path.pop(0)

    t = MyTracker()
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(500) * 10 + 100000)
    ts_base = int(time.time()) - 500 * 300
    ticks = {a: [(ts_base + i * 300, float(prices[i])) for i in range(500)]
             for a in ["BTC", "SOL", "ETH", "AAPLX"]}
    t.tick(ticks)

    from crunch_synth.utils.distributions import validate_distribution
    from crunch_synth.constants import MAX_DISTRIBUTION_COMPONENTS

    print(f"\n=== {name} (MAX_COMPONENTS={MAX_DISTRIBUTION_COMPONENTS}) ===")

    for asset, horizon, step in [("BTC", 86400, 300), ("SOL", 3600, 60), ("AAPLX", 86400, 3600)]:
        preds = t.predict(asset, horizon, step)
        print(f"\n  {asset} h={horizon}s step={step}s -> {len(preds)} preds (expect {horizon//step})")

        if not preds:
            print("  EMPTY!")
            continue

        p = preds[0]
        n_comp = len(p["components"])
        ws = [c["weight"] for c in p["components"]]
        print(f"  step={p['step']}, type={p['type']}, components={n_comp}, weights={ws}, sum={sum(ws):.4f}")
        for c in p["components"]:
            d = c["density"]
            print(f"    {d['name']}: {d['params']}")

        # Validate ALL predictions
        failed = 0
        for pred in preds:
            try:
                validate_distribution(pred)
            except Exception as e:
                failed += 1
                if failed == 1:
                    print(f"  VALIDATION FAILED: {e}")
        if failed:
            print(f"  {failed}/{len(preds)} predictions failed validation!")
        else:
            print(f"  All {len(preds)} predictions validated OK")


test_tracker("invisible-fox", "synth-invisible-fox")
test_tracker("realistic-gazelle", "synth-realistic-gazelle")
print("\nDone.")
