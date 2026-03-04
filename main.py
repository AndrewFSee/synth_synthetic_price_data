"""
CrunchDAO Synth Competition - Main Tracker

This is your main entry point for the Synth competition.
Your tracker must predict probability distributions of incremental price changes
(not raw prices) for crypto and tokenized stock assets.

Key concepts:
- r_{t,k} = P_t - P_{t-k} : incremental return (price difference)
- You predict a full PDF for each step in the forecast horizon
- Scoring is via CRPS (Continuous Ranked Probability Score) - lower is better
"""

import numpy as np
from typing import List, Dict, Any
from crunch_synth import TrackerBase


class MyTracker(TrackerBase):
    """
    Your custom tracker for the Synth competition.
    
    The tracker must implement the predict() method which returns probability 
    distributions for future incremental returns at each forecast step.
    
    Assets to model:
    - Crypto: BTC, ETH, SOL, XAUT
    - Tokenized stocks: SPYX, NVDAX, TSLAX, AAPLX, GOOGLX
    
    Forecast horizons:
    - 24-hour horizon (hourly trigger): steps of 5min, 1h, 6h, 24h
    - 1-hour horizon (12-min trigger): steps of 1min, 5min, 15min, 30min, 1h
    """
    
    def __init__(self):
        super().__init__()
        # Initialize any model parameters or state here
        # self.model = ...
        
    def predict(self, asset: str, horizon: int, step: int) -> List[Dict[str, Any]]:
        """
        Generate probability distribution forecasts for incremental returns.
        
        Args:
            asset: Asset ticker (e.g., "BTC", "SOL", "ETH")
            horizon: Total forecast horizon in seconds (e.g., 86400 for 24h)
            step: Time interval between predictions in seconds (e.g., 300 for 5min)
            
        Returns:
            List of dicts with 'step' and 'prediction' (density specification)
            Each prediction covers the return between t + (k-1)*step and t + k*step
        """
        # Get historical prices from the built-in PriceStore
        # The framework automatically updates prices via tick() before predict()
        resolution = 300  # 5-minute resolution for historical data
        pairs = self.prices.get_prices(asset, days=5, resolution=resolution)
        
        if not pairs:
            return []
        
        _, past_prices = zip(*pairs)
        
        if len(past_prices) < 10:
            return []
        
        # Compute historical incremental returns (price differences)
        returns = np.diff(past_prices)
        
        # Estimate distribution parameters from historical data
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))
        
        if sigma <= 0:
            sigma = 1e-6  # Prevent zero variance
        
        # Number of forecast segments
        num_segments = horizon // step
        
        # Build predictions - one distribution per step
        distributions = []
        for k in range(1, num_segments + 1):
            # Scale parameters based on the step duration
            # Drift scales linearly, volatility scales with sqrt(time)
            scale_factor = step / resolution
            scaled_mu = scale_factor * mu
            scaled_sigma = np.sqrt(scale_factor) * sigma
            
            # Create the prediction dictionary following density_pdf specification
            # Options: "builtin" (fast), "scipy", or "mixture" for complex distributions
            distributions.append({
                "step": k * step,  # Time offset from forecast origin (seconds)
                "type": "mixture",
                "components": [
                    {
                        "density": {
                            "type": "builtin",
                            "name": "norm",
                            "params": {
                                "loc": scaled_mu,      # Mean of the distribution
                                "scale": scaled_sigma  # Standard deviation
                            }
                        },
                        "weight": 1.0
                    }
                ]
            })
        
        return distributions


# For local testing
if __name__ == "__main__":
    from crunch_synth.tracker_evaluator import TrackerEvaluator
    import time
    
    # Initialize tracker and evaluator
    tracker = MyTracker()
    evaluator = TrackerEvaluator(tracker)
    
    # Simulate some price ticks
    current_price = 100000  # Starting BTC price
    ts = time.time()
    
    for i in range(100):
        # Simulate price movement
        current_price += np.random.randn() * 100
        evaluator.tick({"BTC": [(ts + i * 60, current_price)]})
    
    # Generate predictions
    predictions = evaluator.predict(
        "BTC", 
        horizon=3600 * 24,  # 24 hours
        steps=[300, 3600, 3600 * 6, 3600 * 24]  # 5min, 1h, 6h, 24h
    )
    
    print(f"Generated {len(predictions)} predictions")
    print(f"Overall CRPS score: {evaluator.overall_score('BTC'):.4f}")
