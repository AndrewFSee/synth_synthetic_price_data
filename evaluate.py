"""
Local Evaluation Script for CrunchDAO Synth Competition

This script allows you to:
1. Test your tracker with simulated or historical price data
2. Evaluate CRPS scores locally before submitting
3. Visualize your probability distribution forecasts
4. Compare different model approaches

Run this script to validate your tracker works correctly.
"""

import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

# Import trackers
from main import MyTracker
from models import (
    GARCHTracker, 
    MixtureDensityTracker,
    StudentTTracker,
    EnsembleTracker,
    AssetSpecificTracker
)


def generate_synthetic_prices(
    n_points: int = 1000,
    start_price: float = 50000,
    volatility: float = 0.02,
    drift: float = 0.0001,
    regime_changes: bool = True
) -> List[tuple]:
    """
    Generate synthetic price data for testing.
    
    Args:
        n_points: Number of price points to generate
        start_price: Initial price
        volatility: Base volatility (standard deviation of returns)
        drift: Expected return per period
        regime_changes: Whether to include volatility regime changes
        
    Returns:
        List of (timestamp, price) tuples
    """
    prices = [start_price]
    timestamps = []
    
    current_time = time.time() - n_points * 60  # Start n_points minutes ago
    
    current_vol = volatility
    
    for i in range(n_points):
        timestamps.append(current_time + i * 60)  # 1-minute intervals
        
        # Regime changes - switch between high and low volatility
        if regime_changes and np.random.random() < 0.02:  # 2% chance per period
            current_vol = volatility * (0.5 + np.random.random() * 2)
        
        # Generate return
        return_t = drift + current_vol * np.random.randn()
        new_price = prices[-1] * (1 + return_t)
        prices.append(new_price)
    
    return list(zip(timestamps, prices[1:]))


def evaluate_tracker(tracker, asset: str = "BTC", n_trials: int = 10) -> Dict:
    """
    Evaluate a tracker's predictions over multiple trials.
    
    Returns statistics on CRPS scores and prediction quality.
    """
    try:
        from crunch_synth.tracker_evaluator import TrackerEvaluator
        evaluator = TrackerEvaluator(tracker)
        
        # Generate and feed price data
        prices = generate_synthetic_prices(n_points=2000)
        
        for ts, price in prices:
            evaluator.tick({asset: [(ts, price)]})
        
        # Make predictions at different horizons
        results = {
            "24h_predictions": [],
            "1h_predictions": [],
            "scores": []
        }
        
        # 24-hour horizon with multiple steps
        predictions_24h = evaluator.predict(
            asset,
            horizon=3600 * 24,  # 24 hours
            steps=[300, 3600, 3600 * 6, 3600 * 24]  # 5min, 1h, 6h, 24h
        )
        results["24h_predictions"] = predictions_24h
        
        # 1-hour horizon with finer steps
        predictions_1h = evaluator.predict(
            asset,
            horizon=3600,  # 1 hour
            steps=[60, 300, 900, 1800, 3600]  # 1min, 5min, 15min, 30min, 1h
        )
        results["1h_predictions"] = predictions_1h
        
        # Get overall score
        results["overall_score"] = evaluator.overall_score(asset)
        
        return results
        
    except ImportError:
        print("Note: crunch_synth not installed. Running basic validation only.")
        return run_basic_validation(tracker, asset)


def run_basic_validation(tracker, asset: str = "BTC") -> Dict:
    """
    Basic validation without crunch_synth installed.
    Tests that the tracker produces valid output format.
    """
    from crunch_synth import TrackerBase
    
    # Manually feed price data to the tracker
    prices = generate_synthetic_prices(n_points=500)
    
    # Simulate the tick method
    price_dict = {asset: prices}
    for ts, price in prices:
        tracker.tick({asset: [(ts, price)]})
    
    # Test prediction
    predictions = tracker.predict(asset, horizon=3600, step=300)
    
    results = {
        "n_predictions": len(predictions),
        "valid_format": True,
        "sample_prediction": None
    }
    
    # Validate format
    if predictions:
        pred = predictions[0]
        results["sample_prediction"] = pred
        
        # Check required fields
        if "step" not in pred:
            results["valid_format"] = False
            print("ERROR: Missing 'step' field in prediction")
        
        if "type" not in pred:
            results["valid_format"] = False
            print("ERROR: Missing 'type' in prediction")
        if pred.get("type") == "mixture":
            if "components" not in pred:
                results["valid_format"] = False
                print("ERROR: Missing 'components' in mixture")
    
    return results


def visualize_predictions(predictions: List[Dict], title: str = "Forecast Distributions"):
    """
    Visualize the predicted probability distributions.
    """
    from scipy import stats
    
    n_preds = min(6, len(predictions))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, pred in enumerate(predictions[:n_preds]):
        ax = axes[idx]
        step = pred.get("step", 0)
        
        # Extract distribution parameters
        p = pred
        if p.get("type") == "mixture":
            components = p.get("components", [])
            if not components:
                continue

            # Generate x range
            locs = [c["density"]["params"]["loc"] for c in components]
            scales = [c["density"]["params"]["scale"] for c in components]

            x_min = min(locs) - 3 * max(scales)
            x_max = max(locs) + 3 * max(scales)
            x = np.linspace(x_min, x_max, 500)

            # Plot each component
            total_pdf = np.zeros_like(x)
            for comp in components:
                density = comp["density"]
                weight = comp["weight"]

                if density["type"] == "builtin" and density["name"] == "norm":
                    loc = density["params"]["loc"]
                    scale = density["params"]["scale"]
                    pdf = stats.norm.pdf(x, loc=loc, scale=scale)
                    ax.plot(x, pdf * weight, '--', alpha=0.5, label=f'Component (w={weight:.2f})')
                    total_pdf += pdf * weight

            ax.plot(x, total_pdf, 'b-', linewidth=2, label='Mixture')
            ax.fill_between(x, total_pdf, alpha=0.3)
        
        ax.set_title(f"Step: {step}s ({step/60:.0f} min)")
        ax.set_xlabel("Return (price change)")
        ax.set_ylabel("Probability Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("forecast_visualization.png", dpi=150)
    plt.show()
    print("Saved visualization to forecast_visualization.png")


def compare_trackers():
    """Compare different tracker implementations."""
    trackers = {
        "Basic": MyTracker(),
        "GARCH": GARCHTracker(),
        "Mixture": MixtureDensityTracker(),
        "Student-t": StudentTTracker(),
        "Ensemble": EnsembleTracker(),
        "Asset-Specific": AssetSpecificTracker()
    }
    
    print("\n" + "="*60)
    print("TRACKER COMPARISON")
    print("="*60)
    
    # Generate shared price data
    prices = generate_synthetic_prices(n_points=1000)
    
    results = {}
    for name, tracker in trackers.items():
        print(f"\nEvaluating {name} tracker...")
        
        # Feed prices
        for ts, price in prices:
            tracker.tick({"BTC": [(ts, price)]})
        
        # Generate predictions
        try:
            preds = tracker.predict("BTC", horizon=3600, step=300)
            
            if preds:
                # Extract first prediction stats
                first_pred = preds[0]
                if first_pred.get("type") == "mixture":
                    components = first_pred.get("components", [])
                    n_components = len(components)
                    
                    # Get mean and std of mixture
                    total_mean = sum(
                        c["density"]["params"]["loc"] * c["weight"] 
                        for c in components
                    )
                    
                    results[name] = {
                        "n_predictions": len(preds),
                        "n_components": n_components,
                        "mean": total_mean,
                        "status": "OK"
                    }
                else:
                    results[name] = {"status": "OK", "n_predictions": len(preds)}
            else:
                results[name] = {"status": "No predictions", "n_predictions": 0}
                
        except Exception as e:
            results[name] = {"status": f"Error: {str(e)}", "n_predictions": 0}
    
    # Print comparison table
    print("\n" + "-"*60)
    print(f"{'Tracker':<20} {'# Preds':<10} {'Components':<12} {'Status':<15}")
    print("-"*60)
    
    for name, res in results.items():
        n_preds = res.get("n_predictions", 0)
        n_comp = res.get("n_components", "-")
        status = res.get("status", "-")
        print(f"{name:<20} {n_preds:<10} {str(n_comp):<12} {status:<15}")
    
    print("-"*60)
    
    return results


def main():
    """Main evaluation entry point."""
    print("="*60)
    print("CrunchDAO Synth Competition - Local Evaluation")
    print("="*60)
    
    # 1. Basic validation
    print("\n[1] Running basic validation on MyTracker...")
    tracker = MyTracker()
    results = run_basic_validation(tracker, "BTC")
    
    print(f"    Number of predictions: {results['n_predictions']}")
    print(f"    Valid format: {results['valid_format']}")
    
    if results['sample_prediction']:
        print(f"    Sample prediction step: {results['sample_prediction'].get('step')}s")
    
    # 2. Compare different trackers
    print("\n[2] Comparing tracker implementations...")
    comparison = compare_trackers()
    
    # 3. Test on different assets
    print("\n[3] Testing on different asset types...")
    tracker = AssetSpecificTracker()
    
    for asset in ["BTC", "ETH", "SOL", "SPYX", "NVDAX"]:
        prices = generate_synthetic_prices(n_points=500)
        for ts, price in prices:
            tracker.tick({asset: [(ts, price)]})
        
        preds = tracker.predict(asset, horizon=3600, step=300)
        print(f"    {asset}: {len(preds)} predictions generated")
    
    # 4. Visualization (if matplotlib available)
    print("\n[4] Generating visualization...")
    try:
        tracker = MixtureDensityTracker()
        prices = generate_synthetic_prices(n_points=1000)
        for ts, price in prices:
            tracker.tick({"BTC": [(ts, price)]})
        
        preds = tracker.predict("BTC", horizon=3600*6, step=300)
        if preds:
            visualize_predictions(preds, "MixtureDensityTracker Forecasts")
    except Exception as e:
        print(f"    Visualization skipped: {e}")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Install crunch-synth for full evaluation: pip install crunch-synth")
    print("2. Run with real data using TrackerEvaluator")
    print("3. Iterate on your model to improve CRPS scores")
    print("4. Submit via: crunch push -m 'your commit message'")


if __name__ == "__main__":
    main()
