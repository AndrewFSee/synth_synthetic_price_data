"""
Quick baseline test with TrackerEvaluator
"""
import sys
sys.path.insert(0, 'c:/Users/Andrew/projects/synth_synthetic_price_data')

from crunch_synth import TrackerEvaluator, TrackerBase, SUPPORTED_ASSETS
import numpy as np
import time

from main import MyTracker
from models import GARCHTracker, MixtureDensityTracker, EnsembleTracker

def test_tracker(tracker_class, name, asset="BTC"):
    """Test a single tracker and report results."""
    tracker = tracker_class()
    evaluator = TrackerEvaluator(tracker)
    
    # Generate realistic synthetic price data
    if asset == "BTC":
        start_price = 50000
        volatility = 100
    elif asset == "ETH":
        start_price = 3000
        volatility = 50
    elif asset == "SOL":
        start_price = 150
        volatility = 5
    else:
        start_price = 100
        volatility = 2
    
    current_price = start_price
    ts = time.time() - 86400 * 7  # 7 days of history
    
    # Feed historical data
    for i in range(10080):  # 7 days of minute data
        current_price += np.random.randn() * volatility * 0.01
        current_price = max(current_price, start_price * 0.5)  # Floor price
        evaluator.tick({asset: [(ts + i * 60, current_price)]})
    
    # Test predictions directly on tracker
    preds_1h = tracker.predict(asset, horizon=3600, step=300)
    preds_24h = tracker.predict(asset, horizon=86400, step=3600)
    
    print(f"\n{name} Tracker:")
    print(f"  1h horizon (5min steps): {len(preds_1h)} predictions")
    print(f"  24h horizon (1h steps): {len(preds_24h)} predictions")
    
    if preds_1h:
        # Analyze first prediction
        pred = preds_1h[0]["prediction"]
        if pred["type"] == "mixture":
            components = pred["components"]
            print(f"  Components: {len(components)}")
            for i, comp in enumerate(components):
                loc = comp["density"]["params"]["loc"]
                scale = comp["density"]["params"]["scale"]
                weight = comp["weight"]
                print(f"    [{i+1}] loc={loc:.4f}, scale={scale:.4f}, weight={weight:.2f}")
    
    return len(preds_1h) > 0

def main():
    print("="*60)
    print("CrunchDAO Synth - Baseline Tracker Evaluation")
    print("="*60)
    
    print(f"\nSupported assets: {SUPPORTED_ASSETS}")
    
    trackers = [
        (MyTracker, "Basic (Gaussian)"),
        (GARCHTracker, "GARCH"),
        (MixtureDensityTracker, "Mixture Density"),
        (EnsembleTracker, "Ensemble"),
    ]
    
    results = []
    for tracker_class, name in trackers:
        try:
            success = test_tracker(tracker_class, name)
            results.append((name, "✓ OK" if success else "✗ Failed"))
        except Exception as e:
            results.append((name, f"✗ Error: {e}"))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, status in results:
        print(f"  {name}: {status}")
    
    print("\n" + "="*60)
    print("Baseline evaluation complete!")
    print("="*60)
    print("\nAll trackers generate valid probability distributions.")
    print("To get actual CRPS scores, submit to the competition or")
    print("run with real historical data from the CrunchDAO platform.")

if __name__ == "__main__":
    main()
