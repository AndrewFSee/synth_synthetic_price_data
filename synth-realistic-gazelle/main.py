"""
CrunchDAO Synth Competition -- Adaptive Multi-scale + Laplace (realistic-gazelle)

Regime-adaptive probabilistic forecasting:
- Three-scale volatility blend (1.5h / 4h / 12h windows)
- EWMA variance seeding (lambda=0.97, smoother than fox for diversity)
- Adaptive GARCH with regime-dependent reaction speed
- Dynamic weights: more Laplace, less pretrained during transitions
- Multi-step GARCH evolution (correct variance growth for all time steps)
MAX 3 leaf components per density (framework limit).
"""

import math
import numpy as np
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from crunch_synth import TrackerBase

# Asset-specific 5-min sigma defaults (rough calibration from CRPS bounds)
_DEFAULT_SIGMA_5M: Dict[str, float] = {
    "BTC": 300.0, "ETH": 18.0, "SOL": 0.9, "XAUT": 7.0,
    "SPYX": 0.7, "NVDAX": 0.5, "TSLAX": 1.3, "AAPLX": 0.45, "GOOGLX": 0.7,
}


class MyTracker(TrackerBase):
    """Regime-aware tracker with Laplace tails.  ≤ 3 components always."""

    def __init__(self):
        super().__init__()
        self._quantile_models = None
        self._feature_cols: List[str] = []
        self._model_quantiles: List[float] = []
        self._feature_index_cache: Dict[str, int] = {}
        self._load_pretrained()

    # ── pretrained quantile model helpers ─────────────────────────────

    def _load_pretrained(self) -> None:
        path = Path(__file__).resolve().parent / "resources" / "quantile_models.pkl"
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._quantile_models = data.get("models")
            self._feature_cols = list(data.get("feature_cols") or [])
            self._model_quantiles = list(data.get("quantiles") or [])
        except Exception:
            self._quantile_models = None

    def _feature_index_for_col(self, col: str) -> int:
        cached = self._feature_index_cache.get(col)
        if cached is not None:
            return cached
        # FIX: use \d+ (digit class), NOT \\d+ (literal backslash-d)
        match = re.search(r"Feature_(\d+)", col)
        idx = int(match.group(1)) if match else 1
        self._feature_index_cache[col] = idx
        return idx

    def _base_features(self, returns: np.ndarray) -> List[float]:
        arr = self._winsorize(returns)
        feats: List[float] = []
        mean = float(np.mean(arr))
        std = float(np.std(arr)) + 1e-8
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))

        feats.extend([
            mean, std, med, mad,
            float(np.min(arr)), float(np.max(arr)),
            float(np.quantile(arr, 0.1)), float(np.quantile(arr, 0.25)),
            float(np.quantile(arr, 0.75)), float(np.quantile(arr, 0.9)),
            float(np.mean(np.abs(arr))), float(np.sqrt(np.mean(arr ** 2))),
        ])
        centered = arr - mean
        feats.append(float(np.mean(centered ** 3) / (std ** 3)))  # skew
        feats.append(float(np.mean(centered ** 4) / (std ** 4)))  # kurt

        for w in [3, 6, 12, 24, 48, 96, 144, 192]:
            chunk = arr[-w:] if len(arr) >= w else arr
            cstd = float(np.std(chunk)) + 1e-8
            feats.extend([
                float(np.mean(chunk)), cstd, float(np.median(chunk)),
                float(chunk[-1]), float(np.mean(np.abs(chunk))),
                float(np.quantile(chunk, 0.1)), float(np.quantile(chunk, 0.9)),
                float(cstd / std),
            ])

        abs_arr = np.abs(arr)
        for w in [6, 12, 24, 48, 96]:
            chunk = abs_arr[-w:] if len(abs_arr) >= w else abs_arr
            feats.extend([
                float(np.mean(chunk)), float(np.std(chunk)),
                float(np.quantile(chunk, 0.5)),
            ])

        for lag in range(1, 33):
            feats.append(self._safe_autocorr(arr, lag))

        while len(feats) < 220:
            idx = len(feats) + 1
            feats.append(float(math.tanh(feats[-1]) + 0.0001 * idx))
        return feats

    def _vector_for_pretrained(self, returns: np.ndarray) -> np.ndarray:
        base = self._base_features(returns)
        n_base = len(base)
        row = np.zeros(len(self._feature_cols), dtype=float)
        for i, col in enumerate(self._feature_cols):
            idx = self._feature_index_for_col(col)
            row[i] = base[(idx - 1) % n_base]
        return row.reshape(1, -1)

    def _pretrained_base(self, returns: np.ndarray) -> Optional[Dict[str, float]]:
        """Compute pretrained loc/scale at the BASE resolution (300 s)."""
        if not self._quantile_models or not self._feature_cols:
            return None
        try:
            x = self._vector_for_pretrained(returns)
            locs, scales = [], []
            for _, by_q in self._quantile_models.items():
                if not isinstance(by_q, dict):
                    continue
                qp = {}
                for q in self._model_quantiles:
                    m = by_q.get(q)
                    if m is not None:
                        qp[float(q)] = float(m.predict(x)[0])
                if 0.5 in qp and 0.9 in qp and 0.1 in qp:
                    locs.append(qp[0.5])
                    scales.append(max((qp[0.9] - qp[0.1]) / 2.563, 1e-6))
            if not locs:
                return None
            loc = float(np.median(locs))
            scale = float(np.median(scales))
            if not (np.isfinite(loc) and np.isfinite(scale)):
                return None
            return {"loc": loc, "scale": max(scale, 1e-6)}
        except Exception:
            return None

    # ── helpers ───────────────────────────────────────────────────────

    def _winsorize(self, v: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
        if len(v) < 10:
            return v
        return np.clip(v, float(np.quantile(v, lo)), float(np.quantile(v, hi)))

    def _safe_autocorr(self, v: np.ndarray, lag: int) -> float:
        if len(v) <= lag + 5:
            return 0.0
        x, y = v[:-lag], v[lag:]
        if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
            return 0.0
        c = float(np.corrcoef(x, y)[0, 1])
        return 0.0 if np.isnan(c) else c

    def _ewma_var(self, returns: np.ndarray, lam: float = 0.97) -> float:
        """EWMA variance — λ=0.97 gives effective memory of ~33 bars (~2.75h).

        Smoother than fox (λ=0.94) for model diversity.
        """
        if len(returns) < 5:
            return float(np.var(returns)) + 1e-10
        seed_n = min(10, len(returns))
        var = float(np.mean(returns[:seed_n] ** 2))
        for r in returns[seed_n:]:
            var = lam * var + (1 - lam) * r * r
        return max(var, 1e-10)

    def _vol_regime(self, returns: np.ndarray) -> Tuple[float, float, float, float]:
        """Three-scale regime detection (broader windows than fox for diversity).

        Returns (fast_std, med_std, slow_std, regime_intensity):
            fast_std:  ~1.5h (18 bars)
            med_std:   ~4h   (48 bars)
            slow_std:  ~12h  (144 bars)
            regime_intensity: 0 = stable, 1 = extreme regime shift
        """
        n = len(returns)
        fast_std = float(np.std(returns[-min(18, n):])) + 1e-10
        med_std  = float(np.std(returns[-min(48, n):])) + 1e-10
        slow_std = float(np.std(returns[-min(144, n):])) + 1e-10
        vol_ratio = fast_std / slow_std
        log_ratio = abs(math.log(max(vol_ratio, 0.01)))
        regime_intensity = min(1.0, log_ratio / math.log(3.0))
        return fast_std, med_std, slow_std, regime_intensity

    def _estimate_garch_adaptive(self, returns: np.ndarray, regime_intensity: float):
        """GARCH(1,1) with adaptive alpha — reacts faster during regime changes."""
        if len(returns) < 20:
            return 0.1, 0.15, 0.75
        var = float(np.var(returns))
        sq = returns ** 2
        acf1 = float(np.corrcoef(sq[:-1], sq[1:])[0, 1]) if len(sq) > 1 else 0.5
        acf1 = max(0.01, min(0.99, acf1))
        base_alpha = 0.05 + 0.2 * acf1
        base_beta = 0.7 + 0.2 * (1 - acf1)
        alpha = base_alpha + 0.15 * regime_intensity
        beta = max(0.5, base_beta - 0.15 * regime_intensity)
        if alpha + beta >= 0.99:
            s = 0.98 / (alpha + beta)
            alpha *= s
            beta *= s
        omega = max(var * (1 - alpha - beta), 1e-6)
        return omega, alpha, beta

    # ── main prediction ──────────────────────────────────────────────

    def predict(self, asset: str, horizon: int, step: int) -> List[Dict[str, Any]]:
        resolution = 300
        pairs = self.prices.get_prices(asset, days=7, resolution=resolution)

        if not pairs or len(pairs) < 20:
            return self._default_predictions(asset, horizon, step)

        _, prices = zip(*pairs)
        returns = np.diff(prices)
        last_price = float(prices[-1])

        if len(returns) < 10:
            return self._default_predictions(asset, horizon, step)

        # ── Horizon-aware lookback ──
        # 1h profile: shorter lookback, slightly faster reaction
        is_short = (horizon <= 3600)
        lookback = 144 if is_short else 300        # 12h vs 25h (broader than fox)
        ewma_lam = 0.95 if is_short else 0.97     # slightly faster
        garch_win = 48 if is_short else 72         # 4h vs 6h

        recent_raw = returns[-lookback:] if len(returns) >= lookback else returns
        recent = self._winsorize(recent_raw)

        # ── Three-scale regime detection ──
        fast_std, med_std, slow_std, ri = self._vol_regime(recent)
        w_fast = 0.2 + 0.5 * ri
        w_med  = 0.3
        w_slow = 0.5 - 0.5 * ri
        sigma_base = w_fast * fast_std + w_med * med_std + w_slow * slow_std

        # ── Adaptive GARCH ──
        gw = recent[-garch_win:] if len(recent) >= garch_win else recent
        omega, alpha, beta = self._estimate_garch_adaptive(gw, ri)
        current_var = self._ewma_var(gw, lam=ewma_lam)
        last_shock = float((recent[-1] - np.mean(recent[-18:])) ** 2)

        # Clamps
        if last_price > 0:
            min_scale = 0.0003 * last_price
            max_scale = 0.015 * last_price * (1 + 0.5 * ri)
        else:
            min_scale, max_scale = 1e-6, sigma_base * 8.0

        # Pretrained: compute ONCE at base resolution
        pre_base = self._pretrained_base(recent)

        # ── Dynamic weights ──
        # 1h: slightly less pretrained, keep Laplace close to 24h
        if is_short:
            if pre_base:
                tail_w = 0.25 + 0.10 * ri
                pre_w  = 0.18 - 0.10 * ri
                core_w = 1.0 - tail_w - pre_w
            else:
                tail_w = 0.38 + 0.07 * ri
                core_w = 1.0 - tail_w
        else:
            if pre_base:
                tail_w = 0.30 + 0.10 * ri
                pre_w  = 0.20 - 0.15 * ri
                core_w = 1.0 - tail_w - pre_w
            else:
                tail_w = 0.40 + 0.10 * ri
                core_w = 1.0 - tail_w

        scale_factor = step / resolution
        sqrt_sf = math.sqrt(max(scale_factor, 1e-6))
        n_sub = max(1, int(scale_factor))
        num_segments = horizon // step
        distributions: List[Dict[str, Any]] = []

        blend_fast = 0.3 + 0.5 * ri

        var_t = current_var
        for k in range(1, num_segments + 1):
            var_sum = 0.0
            if n_sub >= 1:
                for _ in range(n_sub):
                    var_t = omega + alpha * last_shock + beta * var_t
                    last_shock = var_t
                    var_sum += var_t
                avg_var = var_sum / n_sub
            else:
                var_t = omega * scale_factor + alpha * last_shock * scale_factor + beta * var_t
                last_shock = var_t
                avg_var = var_t

            garch_sig = math.sqrt(max(avg_var * scale_factor, 1e-10))
            sigma = blend_fast * garch_sig + (1 - blend_fast) * (sigma_base * sqrt_sf)
            sigma = max(min_scale * sqrt_sf, min(max_scale * sqrt_sf, sigma))

            # Laplace scale: keep same multiplier for both horizons
            lap_mult = 1.5 + 0.8 * ri
            lap_scale = sigma * lap_mult

            if pre_base:
                pre_loc = pre_base["loc"] * scale_factor
                pre_scale = max(pre_base["scale"] * sqrt_sf, 1e-6)
                components = [
                    {"density": {"type": "builtin", "name": "norm",
                                 "params": {"loc": 0.0, "scale": sigma}},
                     "weight": core_w},
                    {"density": {"type": "builtin", "name": "laplace",
                                 "params": {"loc": 0.0, "scale": lap_scale}},
                     "weight": tail_w},
                    {"density": {"type": "builtin", "name": "norm",
                                 "params": {"loc": pre_loc, "scale": pre_scale}},
                     "weight": pre_w},
                ]
            else:
                components = [
                    {"density": {"type": "builtin", "name": "norm",
                                 "params": {"loc": 0.0, "scale": sigma}},
                     "weight": core_w},
                    {"density": {"type": "builtin", "name": "laplace",
                                 "params": {"loc": 0.0, "scale": lap_scale}},
                     "weight": tail_w},
                ]

            distributions.append({
                "step": k * step,
                "type": "mixture",
                "components": components,
            })

        return distributions

    def _default_predictions(self, asset: str, horizon: int, step: int) -> List[Dict[str, Any]]:
        resolution = 300
        sigma_5m = _DEFAULT_SIGMA_5M.get(asset, 1.0)
        num_segments = horizon // step
        distributions: List[Dict[str, Any]] = []
        for k in range(1, num_segments + 1):
            sqrt_sf = math.sqrt(step / resolution)
            sigma = sigma_5m * sqrt_sf
            distributions.append({
                "step": k * step,
                "type": "mixture",
                "components": [
                    {"density": {"type": "builtin", "name": "norm",
                                 "params": {"loc": 0.0, "scale": sigma}},
                     "weight": 0.70},
                    {"density": {"type": "builtin", "name": "laplace",
                                 "params": {"loc": 0.0, "scale": sigma * 1.3}},
                     "weight": 0.30},
                ],
            })
        return distributions
