"""
CrunchDAO Synth Competition -- Calibrated GARCH + Student-t (invisible-fox)

v20 — Empirically calibrated from 90-day pricedb backtest.
Per-asset optimal parameters from CRPS grid search:
  - t_scale_mult 0.3–0.7 × sigma (NARROWER Student-t creates leptokurtic peak)
  - Crypto: low df (3–7), higher ewma reactivity
  - Stocks: high df (~30, near-Gaussian tails), shorter lookback
GARCH multi-step evolution for longer horizons.
MAX 3 leaf components per density (framework limit).
"""

import math
import numpy as np
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from crunch_synth import TrackerBase

# Asset-specific 5-min sigma defaults (fallback when no data)
_DEFAULT_SIGMA_5M: Dict[str, float] = {
    "BTC": 300.0, "ETH": 18.0, "SOL": 0.9, "XAUT": 7.0,
    "SPYX": 0.7, "NVDAX": 0.5, "TSLAX": 1.3, "AAPLX": 0.45, "GOOGLX": 0.7,
}

# ── Empirically calibrated per-asset density parameters ──────────────────────
# From calibrate.py CRPS grid search on 90-day pricedb data.
# Keys: tail_w, tsm (t_scale_mult), df, ewma_lam, lookback
_CAL: Dict[str, Dict[str, float]] = {
    "BTC":    {"tail_w": 0.20, "tsm": 0.30, "df": 4.2, "lam": 0.88, "lb": 250},
    "ETH":    {"tail_w": 0.39, "tsm": 0.50, "df": 3.0, "lam": 0.92, "lb": 200},
    "XAUT":   {"tail_w": 0.34, "tsm": 0.50, "df": 6.8, "lam": 0.86, "lb":  50},
    "SOL":    {"tail_w": 0.34, "tsm": 0.50, "df": 3.4, "lam": 0.92, "lb": 200},
    "SPYX":   {"tail_w": 0.39, "tsm": 0.50, "df": 30., "lam": 0.86, "lb": 175},
    "NVDAX":  {"tail_w": 0.39, "tsm": 0.50, "df": 30., "lam": 0.86, "lb": 100},
    "TSLAX":  {"tail_w": 0.37, "tsm": 0.70, "df": 30., "lam": 0.86, "lb": 100},
    "AAPLX":  {"tail_w": 0.35, "tsm": 0.30, "df": 30., "lam": 0.94, "lb": 250},
    "GOOGLX": {"tail_w": 0.39, "tsm": 0.50, "df": 30., "lam": 0.88, "lb":  75},
}
_CAL_DEFAULT = {"tail_w": 0.35, "tsm": 0.50, "df": 8.0, "lam": 0.90, "lb": 200}


class MyTracker(TrackerBase):
    """GARCH + Student-t tracker.  ≤ 3 components always."""

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

        # Pad to ≥ 220
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
        """Compute pretrained loc/scale at the BASE resolution (300 s).
        Caller is responsible for scaling to the target step."""
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

    def _ewma_var(self, returns: np.ndarray, lam: float = 0.94) -> float:
        """EWMA variance — reacts to regime changes ~6x faster than window var.

        λ=0.94 gives effective memory of ~17 bars (~85 min at 5-min resolution).
        """
        if len(returns) < 5:
            return float(np.var(returns)) + 1e-10
        seed_n = min(10, len(returns))
        var = float(np.mean(returns[:seed_n] ** 2))
        for r in returns[seed_n:]:
            var = lam * var + (1 - lam) * r * r
        return max(var, 1e-10)

    def _vol_regime(self, returns: np.ndarray) -> Tuple[float, float, float]:
        """Detect regime changes via short/long volatility ratio.

        Returns (fast_std, slow_std, regime_intensity):
            fast_std:  ~1h window (12 bars) — responsive to recent moves
            slow_std:  ~8h window (96 bars) — stable baseline
            regime_intensity: 0 = stable, 1 = extreme regime shift
        """
        n = len(returns)
        fast_std = float(np.std(returns[-min(12, n):])) + 1e-10
        slow_std = float(np.std(returns[-min(96, n):])) + 1e-10
        vol_ratio = fast_std / slow_std
        log_ratio = abs(math.log(max(vol_ratio, 0.01)))
        regime_intensity = min(1.0, log_ratio / math.log(3.0))
        return fast_std, slow_std, regime_intensity

    def _estimate_garch_adaptive(self, returns: np.ndarray, regime_intensity: float):
        """GARCH(1,1) with adaptive alpha — reacts faster during regime changes.

        Stable  (ri≈0): alpha~0.10, beta~0.85 → slow, smooth
        Changing (ri≈1): alpha~0.25, beta~0.70 → fast, reactive
        """
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

        # ── Calibrated per-asset parameters ──
        cal = _CAL.get(asset, _CAL_DEFAULT)
        tail_w_cal = cal["tail_w"]
        t_scale_mult = cal["tsm"]
        df_cal = cal["df"]
        ewma_lam = cal["lam"]
        lookback = int(cal["lb"])

        recent_raw = returns[-lookback:] if len(returns) >= lookback else returns
        recent = self._winsorize(recent_raw)

        # ── Regime detection (kept light — sigma blending + GARCH adaptation) ──
        fast_std, slow_std, ri = self._vol_regime(recent)
        blend_fast = 0.3 + 0.5 * ri
        sigma_base = blend_fast * fast_std + (1 - blend_fast) * slow_std

        # ── EWMA variance with calibrated lambda ──
        garch_win = min(48, len(recent))
        gw = recent[-garch_win:]
        current_var = self._ewma_var(gw, lam=ewma_lam)
        omega, alpha, beta = self._estimate_garch_adaptive(gw, ri)
        last_shock = float((recent[-1] - np.mean(recent[-12:])) ** 2)

        # ── Sigma clamps ──
        if last_price > 0:
            min_scale = 0.0003 * last_price
            max_scale = 0.015 * last_price * (1 + 0.5 * ri)
        else:
            min_scale, max_scale = 1e-6, sigma_base * 8.0

        # ── Pretrained baseline ──
        pre_base = self._pretrained_base(recent)

        # ── Component weights ──
        # Small ri adjustment: slightly more tail during regime changes
        tail_w = min(0.50, tail_w_cal + 0.05 * ri)

        if pre_base:
            pre_w = max(0.05, 0.25 - 0.15 * ri)
            core_w = max(0.10, 1.0 - tail_w - pre_w)
            # Re-normalize if needed
            s = core_w + tail_w + pre_w
            core_w, tail_w, pre_w = core_w / s, tail_w / s, pre_w / s
        else:
            tail_w = min(0.55, tail_w + 0.05)  # extra tail when no pretrained
            core_w = 1.0 - tail_w

        # ── df: small regime adjustment for crypto ──
        df = max(2.5, df_cal - 1.0 * ri) if df_cal < 25 else df_cal

        scale_factor = step / resolution
        sqrt_sf = math.sqrt(max(scale_factor, 1e-6))
        num_segments = horizon // step
        distributions: List[Dict[str, Any]] = []

        n_sub = max(1, int(scale_factor))

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

            t_scale = sigma * t_scale_mult

            if pre_base:
                pre_loc = pre_base["loc"] * scale_factor
                pre_scale = max(pre_base["scale"] * sqrt_sf, 1e-6)
                components = [
                    {"density": {"type": "builtin", "name": "norm",
                                 "params": {"loc": 0.0, "scale": sigma}},
                     "weight": core_w},
                    {"density": {"type": "builtin", "name": "t",
                                 "params": {"df": df, "loc": 0.0, "scale": t_scale}},
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
                    {"density": {"type": "builtin", "name": "t",
                                 "params": {"df": df, "loc": 0.0, "scale": t_scale}},
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
                    {"density": {"type": "builtin", "name": "t",
                                 "params": {"df": 5.0, "loc": 0.0, "scale": sigma * 2.0}},
                     "weight": 0.30},
                ],
            })
        return distributions
