"""
CrunchDAO Synth Competition -- GARCH + Adaptive Tails (invisible-fox)

v22 — v21 improvements + external volatility signals:
  v21: stock params from v2 calibration, Laplace tails, step-adaptive sigma
  v22: VIX (Yahoo Finance) for stocks, Fear & Greed (alternative.me) for crypto
       Both cached 1h, graceful fallback to neutral (1.0) on failure.
MAX 3 leaf components per density (framework limit).
"""

import json
import math
import numpy as np
import pickle
import re
import time
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from crunch_synth import TrackerBase

# ── External volatility signals ──────────────────────────────────────────────
_STOCK_ASSETS = {"SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"}
_CRYPTO_ASSETS = {"BTC", "ETH", "SOL", "XAUT"}
_VIX_BASELINE = 20.0   # long-term average VIX
_FNG_NEUTRAL = 50       # Fear & Greed neutral point


class _ExternalSignals:
    """Cached VIX + Fear & Greed fetcher with staleness decay.

    Tokenized stocks trade 24/7 but VIX only updates during US market hours
    (~6.5h/day).  Fear & Greed updates once daily.  To avoid applying a stale
    signal as if it were fresh, multipliers decay toward 1.0 (neutral) as
    the data ages:
      - VIX:  full weight for 1h, linear decay to neutral over 6h
      - F&G:  full weight for 6h, linear decay to neutral over 24h
    """

    _VIX_TTL = 900      # re-fetch every 15 min (captures intraday moves)
    _FNG_TTL = 3600      # re-fetch every 1h (only changes daily anyway)
    _VIX_FRESH = 3600    # full signal weight for 1h after fetch
    _VIX_STALE = 21600   # decays to neutral by 6h after fetch
    _FNG_FRESH = 21600   # full signal weight for 6h
    _FNG_STALE = 86400   # decays to neutral by 24h

    def __init__(self):
        self._vix: float = _VIX_BASELINE
        self._vix_ts: float = 0.0          # last successful fetch time
        self._vix_fetch_ts: float = 0.0    # last fetch attempt time
        self._fng: int = _FNG_NEUTRAL
        self._fng_ts: float = 0.0
        self._fng_fetch_ts: float = 0.0

    @staticmethod
    def _get_json(url: str, timeout: int = 5) -> Any:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (compatible; SynthTracker)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    @staticmethod
    def _staleness_weight(age: float, fresh: float, stale: float) -> float:
        """1.0 when age < fresh, linear decay to 0.0 at age >= stale."""
        if age <= fresh:
            return 1.0
        if age >= stale:
            return 0.0
        return (stale - age) / (stale - fresh)

    # ── VIX (Yahoo Finance) ──────────────────────────────────────────

    def _refresh_vix(self) -> None:
        now = time.time()
        if now - self._vix_fetch_ts < self._VIX_TTL:
            return
        self._vix_fetch_ts = now
        try:
            data = self._get_json(
                "https://query1.finance.yahoo.com/v8/finance/chart/"
                "%5EVIX?interval=1d&range=2d"
            )
            v = float(data["chart"]["result"][0]["meta"]["regularMarketPrice"])
            if 5.0 <= v <= 100.0:
                self._vix = v
                self._vix_ts = now   # mark data as fresh
        except Exception:
            pass

    def stock_vol_mult(self) -> float:
        """VIX-based sigma multiplier for stock assets with staleness decay.

        Fresh VIX=30 → 1.22;  6h-stale VIX=30 → 1.0 (neutral).
        Clamped to [0.7, 1.6] for safety.
        """
        self._refresh_vix()
        raw_mult = math.sqrt(self._vix / _VIX_BASELINE)
        raw_mult = max(0.7, min(1.6, raw_mult))
        # Decay toward 1.0 as data ages
        age = time.time() - self._vix_ts
        w = self._staleness_weight(age, self._VIX_FRESH, self._VIX_STALE)
        return 1.0 + w * (raw_mult - 1.0)

    # ── Fear & Greed Index (alternative.me) ──────────────────────────

    def _refresh_fng(self) -> None:
        now = time.time()
        if now - self._fng_fetch_ts < self._FNG_TTL:
            return
        self._fng_fetch_ts = now
        try:
            data = self._get_json("https://api.alternative.me/fng/?limit=1")
            v = int(data["data"][0]["value"])
            if 0 <= v <= 100:
                self._fng = v
                self._fng_ts = now
        except Exception:
            pass

    def crypto_vol_mult(self) -> float:
        """Fear & Greed-based sigma multiplier with staleness decay.

        Extreme fear (0-20) → up to 1.25×, decays to neutral over 24h.
        """
        self._refresh_fng()
        fng = self._fng
        if fng < 20:
            raw_mult = 1.0 + 0.0125 * (20 - fng)     # 1.0–1.25
        elif fng < 40:
            raw_mult = 1.0 + 0.005 * (40 - fng)       # 1.0–1.10
        elif fng <= 60:
            raw_mult = 1.0                              # neutral
        elif fng <= 80:
            raw_mult = max(0.92, 1.0 - 0.004 * (fng - 60))  # 0.92–1.0
        else:
            raw_mult = 1.0 + 0.005 * (fng - 80)       # 1.0–1.10
        # Decay toward 1.0 as data ages
        age = time.time() - self._fng_ts
        w = self._staleness_weight(age, self._FNG_FRESH, self._FNG_STALE)
        return 1.0 + w * (raw_mult - 1.0)


_ext = _ExternalSignals()

# Asset-specific 5-min sigma defaults (fallback when no data)
_DEFAULT_SIGMA_5M: Dict[str, float] = {
    "BTC": 300.0, "ETH": 18.0, "SOL": 0.9, "XAUT": 7.0,
    "SPYX": 0.7, "NVDAX": 0.5, "TSLAX": 1.3, "AAPLX": 0.45, "GOOGLX": 0.7,
}

# ── Empirically calibrated per-asset density parameters ──────────────────────
# Crypto: v20 params (performing well on 1h, rank 34).
# Stocks: v2 train/test calibration (longer lb, smoother lam for stability).
# Keys: tail_w, tsm (t_scale_mult), df, ewma_lam, lookback
_CAL: Dict[str, Dict[str, float]] = {
    # Crypto — keep v20 params
    "BTC":    {"tail_w": 0.20, "tsm": 0.30, "df": 4.2, "lam": 0.88, "lb": 250},
    "ETH":    {"tail_w": 0.39, "tsm": 0.50, "df": 3.0, "lam": 0.92, "lb": 200},
    "XAUT":   {"tail_w": 0.34, "tsm": 0.50, "df": 6.8, "lam": 0.86, "lb":  50},
    "SOL":    {"tail_w": 0.34, "tsm": 0.50, "df": 3.4, "lam": 0.92, "lb": 200},
    # Stocks — v2 calibration (longer lookback, smoother EWMA)
    "SPYX":   {"tail_w": 0.29, "tsm": 0.30, "df": 16., "lam": 0.92, "lb": 250},
    "NVDAX":  {"tail_w": 0.24, "tsm": 0.40, "df": 30., "lam": 0.86, "lb": 225},
    "TSLAX":  {"tail_w": 0.39, "tsm": 0.60, "df": 30., "lam": 0.91, "lb": 175},
    "AAPLX":  {"tail_w": 0.39, "tsm": 0.40, "df": 30., "lam": 0.90, "lb": 375},
    "GOOGLX": {"tail_w": 0.39, "tsm": 0.30, "df": 30., "lam": 0.90, "lb": 175},
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

        # ── External vol signal ──
        if asset in _STOCK_ASSETS:
            sigma_base *= _ext.stock_vol_mult()
        elif asset in _CRYPTO_ASSETS:
            sigma_base *= _ext.crypto_vol_mult()

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
            # Step-adaptive blending: GARCH converges to unconditional variance
            # over many sub-steps, losing current regime info.  For long steps,
            # rely more on sigma_base (current vol) scaled by sqrt(time).
            # sf=1 → gd=0.98; sf=12 → 0.80; sf=72 → 0.40; sf=288 → 0.14
            garch_decay = 1.0 / (1.0 + scale_factor / 48.0)
            eff_blend = blend_fast * garch_decay
            sigma = eff_blend * garch_sig + (1 - eff_blend) * (sigma_base * sqrt_sf)
            sigma = max(min_scale * sqrt_sf, min(max_scale * sqrt_sf, sigma))

            t_scale = sigma * t_scale_mult

            # Tail component: Laplace for stocks (df>=20 -> Student-t ~ Normal,
            # wastes component slot).  Keep Student-t for crypto (genuine fat tails).
            if df_cal >= 20:
                tail_comp = {"density": {"type": "builtin", "name": "laplace",
                                          "params": {"loc": 0.0, "scale": t_scale}},
                              "weight": tail_w}
            else:
                tail_comp = {"density": {"type": "builtin", "name": "t",
                                          "params": {"df": df, "loc": 0.0, "scale": t_scale}},
                              "weight": tail_w}

            if pre_base:
                pre_loc = pre_base["loc"] * scale_factor
                pre_scale = max(pre_base["scale"] * sqrt_sf, 1e-6)
                components = [
                    {"density": {"type": "builtin", "name": "norm",
                                 "params": {"loc": 0.0, "scale": sigma}},
                     "weight": core_w},
                    tail_comp,
                    {"density": {"type": "builtin", "name": "norm",
                                 "params": {"loc": pre_loc, "scale": pre_scale}},
                     "weight": pre_w},
                ]
            else:
                components = [
                    {"density": {"type": "builtin", "name": "norm",
                                 "params": {"loc": 0.0, "scale": sigma}},
                     "weight": core_w},
                    tail_comp,
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
