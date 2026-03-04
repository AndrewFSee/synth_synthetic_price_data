"""
Advanced Tracker Models for CrunchDAO Synth Competition

This module contains more sophisticated model approaches for probability 
distribution forecasting of crypto/stock price returns.

Key approaches:
1. GARCHTracker - Volatility modeling with heteroskedasticity
2. MixtureDensityTracker - Multi-modal distributions for regime changes
3. AdaptiveTracker - Online learning with regime detection
4. EnsembleTracker - Combine multiple model predictions
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from crunch_synth import TrackerBase
from scipy import stats


class GARCHTracker(TrackerBase):
    """
    GARCH-based tracker for volatility forecasting.
    
    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models
    capture time-varying volatility - essential for financial data where 
    volatility clusters (high vol follows high vol).
    
    The model:
    - Return: r_t = mu + epsilon_t
    - Variance: sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}
    
    This is particularly good for crypto assets with frequent volatility regimes.
    """
    
    def __init__(self, omega: float = 0.1, alpha: float = 0.1, beta: float = 0.8):
        super().__init__()
        self.omega = omega  # Base variance
        self.alpha = alpha  # ARCH term (reaction to shocks)
        self.beta = beta    # GARCH term (persistence)
        self.last_variance = {}  # Track variance per asset
        
    def _estimate_garch_params(self, returns: np.ndarray) -> Tuple[float, float, float]:
        """
        Simple method-of-moments estimation for GARCH(1,1) parameters.
        For production, consider using the `arch` library for MLE.
        """
        if len(returns) < 20:
            return self.omega, self.alpha, self.beta
            
        # Estimate unconditional variance
        var = np.var(returns)
        
        # Estimate autocorrelation of squared returns
        sq_returns = returns ** 2
        if len(sq_returns) > 1:
            acf1 = np.corrcoef(sq_returns[:-1], sq_returns[1:])[0, 1]
            acf1 = max(0.01, min(0.99, acf1))  # Bound the correlation
        else:
            acf1 = 0.5
        
        # Rough parameter estimates
        alpha = 0.05 + 0.2 * acf1
        beta = 0.7 + 0.2 * (1 - acf1)
        omega = var * (1 - alpha - beta)
        omega = max(0.0001, omega)  # Ensure positive
        
        return omega, alpha, beta
    
    def _forecast_variance(self, returns: np.ndarray, steps_ahead: int) -> np.ndarray:
        """Forecast variance for multiple steps ahead."""
        if len(returns) < 2:
            return np.ones(steps_ahead) * 1.0
            
        omega, alpha, beta = self._estimate_garch_params(returns)
        
        # Current variance estimate
        current_var = np.var(returns[-20:]) if len(returns) >= 20 else np.var(returns)
        last_shock = (returns[-1] - np.mean(returns)) ** 2
        
        variances = []
        var_t = current_var
        
        for _ in range(steps_ahead):
            var_t = omega + alpha * last_shock + beta * var_t
            variances.append(var_t)
            last_shock = var_t  # Expected shock for future
            
        return np.array(variances)
    
    def predict(self, asset: str, horizon: int, step: int) -> List[Dict[str, Any]]:
        resolution = 300
        pairs = self.prices.get_prices(asset, days=10, resolution=resolution)
        
        if not pairs or len(pairs) < 20:
            return []
        
        _, past_prices = zip(*pairs)
        returns = np.diff(past_prices)
        
        mu = float(np.mean(returns))
        num_segments = horizon // step
        
        # Forecast variances for each step
        variances = self._forecast_variance(returns, num_segments)
        
        distributions = []
        for k in range(1, num_segments + 1):
            scale_factor = step / resolution
            scaled_mu = scale_factor * mu
            # Use GARCH variance forecast, scaled appropriately
            scaled_var = variances[k-1] * scale_factor
            scaled_sigma = np.sqrt(max(scaled_var, 1e-10))
            
            distributions.append({
                "step": k * step,
                "type": "mixture",
                "components": [{
                    "density": {
                        "type": "builtin",
                        "name": "norm",
                        "params": {"loc": scaled_mu, "scale": scaled_sigma}
                    },
                    "weight": 1.0
                }]
            })
        
        return distributions


class MixtureDensityTracker(TrackerBase):
    """
    Gaussian Mixture Model tracker for capturing multi-modal return distributions.
    
    Financial returns often exhibit:
    - Fat tails (more extreme events than normal distribution)
    - Skewness (asymmetric upside/downside moves)
    - Regime changes (calm vs volatile periods)
    
    A mixture of Gaussians can capture these better than a single Gaussian.
    
    The model uses 2-3 components:
    - Main component: normal market conditions
    - Tail components: capturing jumps/crashes and rallies
    """
    
    def __init__(self, n_components: int = 3):
        super().__init__()
        self.n_components = n_components
        
    def _fit_mixture(self, returns: np.ndarray) -> List[Dict]:
        """
        Simple mixture fitting using quantile-based approach.
        For production, use sklearn.mixture.GaussianMixture or pomegranate.
        """
        if len(returns) < 30:
            # Fallback to single Gaussian
            return [{
                "weight": 1.0,
                "loc": float(np.mean(returns)),
                "scale": float(np.std(returns)) + 1e-6
            }]
        
        # Identify regimes by volatility
        rolling_var = []
        window = min(20, len(returns) // 3)
        for i in range(window, len(returns)):
            rolling_var.append(np.var(returns[i-window:i]))
        
        if len(rolling_var) < 10:
            return [{
                "weight": 1.0,
                "loc": float(np.mean(returns)),
                "scale": float(np.std(returns)) + 1e-6
            }]
        
        # Split into low/medium/high volatility regimes
        vol_percentiles = np.percentile(rolling_var, [33, 67])
        
        components = []
        
        # Low volatility regime
        low_vol_mask = np.array(rolling_var) <= vol_percentiles[0]
        if np.sum(low_vol_mask) > 5:
            low_vol_returns = returns[window:][low_vol_mask]
            components.append({
                "weight": 0.5,  # Will be normalized
                "loc": float(np.mean(low_vol_returns)),
                "scale": float(np.std(low_vol_returns)) + 1e-6
            })
        
        # Medium volatility (normal) regime  
        med_vol_mask = (np.array(rolling_var) > vol_percentiles[0]) & \
                       (np.array(rolling_var) <= vol_percentiles[1])
        if np.sum(med_vol_mask) > 5:
            med_vol_returns = returns[window:][med_vol_mask]
            components.append({
                "weight": 0.35,
                "loc": float(np.mean(med_vol_returns)),
                "scale": float(np.std(med_vol_returns)) + 1e-6
            })
        
        # High volatility regime
        high_vol_mask = np.array(rolling_var) > vol_percentiles[1]
        if np.sum(high_vol_mask) > 5:
            high_vol_returns = returns[window:][high_vol_mask]
            components.append({
                "weight": 0.15,
                "loc": float(np.mean(high_vol_returns)),
                "scale": float(np.std(high_vol_returns)) + 1e-6
            })
        
        if not components:
            components = [{
                "weight": 1.0,
                "loc": float(np.mean(returns)),
                "scale": float(np.std(returns)) + 1e-6
            }]
        
        # Normalize weights
        total_weight = sum(c["weight"] for c in components)
        for c in components:
            c["weight"] /= total_weight
        
        return components
    
    def predict(self, asset: str, horizon: int, step: int) -> List[Dict[str, Any]]:
        resolution = 300
        pairs = self.prices.get_prices(asset, days=14, resolution=resolution)
        
        if not pairs or len(pairs) < 30:
            return []
        
        _, past_prices = zip(*pairs)
        returns = np.diff(past_prices)
        
        # Fit mixture model
        components = self._fit_mixture(returns)
        
        num_segments = horizon // step
        distributions = []
        
        for k in range(1, num_segments + 1):
            scale_factor = step / resolution
            
            # Scale each component
            scaled_components = []
            for comp in components:
                scaled_components.append({
                    "density": {
                        "type": "builtin",
                        "name": "norm",
                        "params": {
                            "loc": comp["loc"] * scale_factor,
                            "scale": comp["scale"] * np.sqrt(scale_factor)
                        }
                    },
                    "weight": comp["weight"]
                })
            
            distributions.append({
                "step": k * step,
                "type": "mixture",
                "components": scaled_components
            })
        
        return distributions


class StudentTTracker(TrackerBase):
    """
    Student's t-distribution tracker for heavy-tailed returns.
    
    Crypto returns often have fatter tails than a normal distribution.
    Student's t with low degrees of freedom captures this well.
    
    Lower degrees of freedom = fatter tails = more probability of extreme events.
    """
    
    def __init__(self, df_range: Tuple[float, float] = (3.0, 30.0)):
        super().__init__()
        self.df_range = df_range
        
    def _estimate_df(self, returns: np.ndarray) -> float:
        """Estimate degrees of freedom from kurtosis."""
        if len(returns) < 20:
            return 10.0
        
        kurtosis = stats.kurtosis(returns, fisher=False)
        
        # Kurtosis of t-distribution: 3 + 6/(df-4) for df > 4
        # Solve for df: df = 4 + 6/(kurtosis - 3)
        if kurtosis > 3:
            df = 4 + 6 / (kurtosis - 3)
        else:
            df = 30.0  # Near-normal
        
        return max(self.df_range[0], min(self.df_range[1], df))
    
    def predict(self, asset: str, horizon: int, step: int) -> List[Dict[str, Any]]:
        resolution = 300
        pairs = self.prices.get_prices(asset, days=10, resolution=resolution)
        
        if not pairs or len(pairs) < 20:
            return []
        
        _, past_prices = zip(*pairs)
        returns = np.diff(past_prices)
        
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))
        df = self._estimate_df(returns)
        
        if sigma <= 0:
            sigma = 1e-6
        
        num_segments = horizon // step
        distributions = []
        
        for k in range(1, num_segments + 1):
            scale_factor = step / resolution
            
            # Note: For t-distribution, we use scipy name "t"
            # Parameters: df, loc, scale
            distributions.append({
                "step": k * step,
                "type": "mixture",
                "components": [{
                    "density": {
                        "type": "scipy",  # Use scipy for t-distribution
                        "name": "t",
                        "params": {
                            "df": df,
                            "loc": mu * scale_factor,
                            "scale": sigma * np.sqrt(scale_factor)
                        }
                    },
                    "weight": 1.0
                }]
            })
        
        return distributions


class EnsembleTracker(TrackerBase):
    """
    Ensemble tracker that combines multiple model predictions.
    
    Averaging predictions from diverse models typically improves robustness
    and reduces variance compared to any single model.
    
    The ensemble creates a mixture distribution from individual model outputs.
    """
    
    def __init__(self, weights: Optional[List[float]] = None):
        super().__init__()
        # Initialize sub-trackers
        self.trackers = [
            GARCHTracker(),
            MixtureDensityTracker(),
            StudentTTracker()
        ]
        self.weights = weights or [1/3, 1/3, 1/3]
        
    def tick(self, data):
        """Forward price ticks to all sub-trackers."""
        super().tick(data)
        for tracker in self.trackers:
            tracker.tick(data)
    
    def predict(self, asset: str, horizon: int, step: int) -> List[Dict[str, Any]]:
        # Collect predictions from all trackers
        all_predictions = []
        for tracker in self.trackers:
            pred = tracker.predict(asset, horizon, step)
            if pred:
                all_predictions.append(pred)
        
        if not all_predictions:
            return []
        
        # Combine predictions at each step
        num_segments = horizon // step
        combined = []
        
        for k in range(num_segments):
            components = []
            
            for tracker_idx, predictions in enumerate(all_predictions):
                if k < len(predictions):
                    pred = predictions[k]
                    # Extract components and scale weights
                    if pred.get("type") == "mixture":
                        for comp in pred.get("components", []):
                            scaled_comp = comp.copy()
                            scaled_comp["weight"] = comp["weight"] * self.weights[tracker_idx]
                            components.append(scaled_comp)
            
            if components:
                # Normalize weights
                total = sum(c["weight"] for c in components)
                if total > 0:
                    for c in components:
                        c["weight"] /= total
                
                combined.append({
                    "step": (k + 1) * step,
                    "type": "mixture",
                    "components": components
                })
        
        return combined


class AssetSpecificTracker(TrackerBase):
    """
    Tracker that uses different models for different asset types.
    
    Crypto assets (BTC, ETH, SOL) may behave differently from tokenized 
    stocks (SPYX, NVDAX, etc.), so we can use specialized models for each.
    """
    
    CRYPTO_ASSETS = {"BTC", "ETH", "SOL", "XAUT"}
    STOCK_ASSETS = {"SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"}
    
    def __init__(self):
        super().__init__()
        # Crypto: higher volatility, use GARCH + mixture
        self.crypto_tracker = GARCHTracker(omega=0.15, alpha=0.15, beta=0.75)
        # Stocks: typically more stable, use mixture for regime changes
        self.stock_tracker = MixtureDensityTracker(n_components=2)
        
    def tick(self, data):
        super().tick(data)
        self.crypto_tracker.tick(data)
        self.stock_tracker.tick(data)
    
    def predict(self, asset: str, horizon: int, step: int) -> List[Dict[str, Any]]:
        if asset in self.CRYPTO_ASSETS:
            return self.crypto_tracker.predict(asset, horizon, step)
        elif asset in self.STOCK_ASSETS:
            return self.stock_tracker.predict(asset, horizon, step)
        else:
            # Fallback to crypto model for unknown assets
            return self.crypto_tracker.predict(asset, horizon, step)
