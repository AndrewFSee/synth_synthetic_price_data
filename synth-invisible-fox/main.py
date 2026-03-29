"""
CrunchDAO Synth Competition -- GARCH + Adaptive Tails (invisible-fox)

v25 — Time-of-day sigma adjustment for 24h profile:
  v24: Add derivatives data (Deribit implied vol), leverage metrics (OKX funding),
       and macro signals (10Y Treasury yield) for better sigma calibration.
  v25: Per-segment sigma scaling based on intraday volatility seasonality.
       Crypto vol varies 2-3× through the day (peak at US open 14-16 UTC).
       For 24h predictions with >=12 segments spanning >=12h, adjust each
       segment's scale by time-of-day multiplier (60% blend).
       Also thin tails at step>=21600 (CLT at longer horizons).
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
_TNX_BASELINE = 4.0     # long-term 10Y yield baseline

# OKX perpetual swap instrument IDs for funding rate
_OKX_FUNDING_MAP = {"BTC": "BTC-USDT-SWAP", "ETH": "ETH-USDT-SWAP", "SOL": "SOL-USDT-SWAP"}

# ── Intraday volatility seasonality (smoothed 3-hour rolling avg) ────────────
# Measured from 450 days of 5-min data.  Peak ~14-16 UTC (US open), trough ~10-12.
_TOD_MULT = {
    "BTC": {0: 0.996, 1: 0.996, 2: 0.933, 3: 0.829, 4: 0.773, 5: 0.741, 6: 0.755, 7: 0.757, 8: 0.786, 9: 0.757, 10: 0.737, 11: 0.700, 12: 0.803, 13: 1.081, 14: 1.350, 15: 1.456, 16: 1.359, 17: 1.250, 18: 1.171, 19: 1.098, 20: 0.976, 21: 0.920, 22: 0.961, 23: 0.975},
    "ETH": {0: 0.983, 1: 0.983, 2: 0.941, 3: 0.837, 4: 0.774, 5: 0.742, 6: 0.775, 7: 0.749, 8: 0.773, 9: 0.745, 10: 0.691, 11: 0.687, 12: 0.791, 13: 1.069, 14: 1.314, 15: 1.422, 16: 1.334, 17: 1.305, 18: 1.224, 19: 1.032, 20: 0.981, 21: 0.959, 22: 0.958, 23: 0.972},
    "SOL": {0: 1.120, 1: 1.120, 2: 0.957, 3: 0.837, 4: 0.781, 5: 0.764, 6: 0.771, 7: 0.794, 8: 0.801, 9: 0.749, 10: 0.708, 11: 0.693, 12: 0.784, 13: 1.020, 14: 1.259, 15: 1.377, 16: 1.317, 17: 1.255, 18: 1.172, 19: 1.020, 20: 0.957, 21: 0.955, 22: 0.946, 23: 1.108},
    "XAUT": {0: 1.109, 1: 1.135, 2: 1.071, 3: 0.822, 4: 0.737, 5: 0.770, 6: 0.816, 7: 0.821, 8: 0.831, 9: 0.856, 10: 0.833, 11: 0.799, 12: 0.936, 13: 1.084, 14: 1.335, 15: 1.371, 16: 1.109, 17: 1.068, 18: 1.126, 19: 0.923, 20: 0.810, 21: 0.821, 22: 0.918, 23: 1.134},
}
_TOD_BLEND = 0.6  # blend factor: 0=no adjustment, 1=full adjustment


class _ExternalSignals:
    """Multi-source signal fetcher with staleness decay.

    Sources: VIX, Fear & Greed, Deribit DVOL (implied vol),
    OKX funding rate (leverage), 10Y Treasury yield (macro).
    All signals decay toward neutral as data ages.
    """

    _VIX_TTL = 900       # re-fetch every 15 min
    _FNG_TTL = 3600      # re-fetch every 1h
    _VIX_FRESH = 3600    # full signal weight for 1h after fetch
    _VIX_STALE = 21600   # decays to neutral by 6h after fetch
    _FNG_FRESH = 21600   # full signal weight for 6h
    _FNG_STALE = 86400   # decays to neutral by 24h
    # Deribit DVOL (crypto implied volatility)
    _DVOL_TTL = 900      # re-fetch every 15 min
    _DVOL_FRESH = 3600   # full weight for 1h
    _DVOL_STALE = 21600  # decay to neutral by 6h
    # OKX funding rate (crypto leverage)
    _FUNDING_TTL = 1800  # re-fetch every 30 min
    _FUNDING_FRESH = 7200   # full weight for 2h
    _FUNDING_STALE = 28800  # decay by 8h (funding settles every 8h)
    # 10Y Treasury yield (macro)
    _TNX_TTL = 3600      # re-fetch every 1h
    _TNX_FRESH = 3600
    _TNX_STALE = 21600

    def __init__(self):
        self._vix: float = _VIX_BASELINE
        self._vix_ts: float = 0.0
        self._vix_fetch_ts: float = 0.0
        self._fng: int = _FNG_NEUTRAL
        self._fng_ts: float = 0.0
        self._fng_fetch_ts: float = 0.0
        # Deribit DVOL
        self._btc_dvol: Optional[float] = None
        self._eth_dvol: Optional[float] = None
        self._dvol_ts: float = 0.0
        self._dvol_fetch_ts: float = 0.0
        # OKX funding rate
        self._funding: Dict[str, float] = {}
        self._funding_ts: float = 0.0
        self._funding_fetch_ts: float = 0.0
        # 10Y Treasury yield
        self._tnx: float = _TNX_BASELINE
        self._tnx_prev: float = _TNX_BASELINE
        self._tnx_ts: float = 0.0
        self._tnx_fetch_ts: float = 0.0

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

    # ── Deribit DVOL (crypto implied volatility) ─────────────────────

    def _refresh_dvol(self) -> None:
        now = time.time()
        if now - self._dvol_fetch_ts < self._DVOL_TTL:
            return
        self._dvol_fetch_ts = now
        now_ms = int(now * 1000)
        start_ms = now_ms - 7200000  # last 2h
        for currency, attr in [("BTC", "_btc_dvol"), ("ETH", "_eth_dvol")]:
            try:
                data = self._get_json(
                    "https://www.deribit.com/api/v2/public/"
                    "get_volatility_index_data"
                    f"?currency={currency}&resolution=3600"
                    f"&start_timestamp={start_ms}&end_timestamp={now_ms}"
                )
                points = data.get("result", {}).get("data", [])
                if points:
                    val = float(points[-1][4])  # close of latest candle
                    if 5.0 <= val <= 200.0:
                        setattr(self, attr, val)
                        self._dvol_ts = now
            except Exception:
                pass

    def crypto_dvol_mult(self, asset: str, realized_annual_vol: float) -> float:
        """Deribit implied vol vs realized vol → sigma multiplier.

        If options market implies higher vol than GARCH estimates,
        widen sigma (forward-looking adjustment).
        """
        self._refresh_dvol()
        dvol = self._eth_dvol if asset == "ETH" else self._btc_dvol
        if dvol is None:
            return 1.0
        if realized_annual_vol < 5.0:
            return 1.0
        ratio = dvol / realized_annual_vol
        raw_mult = max(0.85, min(1.20, math.sqrt(ratio)))
        age = time.time() - self._dvol_ts
        w = self._staleness_weight(age, self._DVOL_FRESH, self._DVOL_STALE)
        return 1.0 + w * (raw_mult - 1.0)

    # ── OKX Funding Rate (crypto leverage) ───────────────────────────

    def _refresh_funding(self) -> None:
        now = time.time()
        if now - self._funding_fetch_ts < self._FUNDING_TTL:
            return
        self._funding_fetch_ts = now
        for asset, inst_id in _OKX_FUNDING_MAP.items():
            try:
                data = self._get_json(
                    "https://www.okx.com/api/v5/public/funding-rate"
                    f"?instId={inst_id}"
                )
                if data.get("code") == "0":
                    items = data.get("data", [])
                    if items:
                        rate = float(items[0].get("fundingRate", 0))
                        self._funding[asset] = rate
                        self._funding_ts = now
            except Exception:
                pass

    def funding_tail_adj(self, asset: str) -> float:
        """Extreme funding rate → wider tails (leverage risk).

        |rate| > 0.0002 starts adding tail weight, max +0.04.
        """
        self._refresh_funding()
        rate = self._funding.get(asset)
        if rate is None:
            return 0.0
        raw_adj = min(0.04, max(0.0, (abs(rate) - 0.0002) * 40))
        age = time.time() - self._funding_ts
        w = self._staleness_weight(age, self._FUNDING_FRESH, self._FUNDING_STALE)
        return raw_adj * w

    # ── 10Y Treasury Yield (macro signal for stocks) ─────────────────

    def _refresh_tnx(self) -> None:
        now = time.time()
        if now - self._tnx_fetch_ts < self._TNX_TTL:
            return
        self._tnx_fetch_ts = now
        try:
            data = self._get_json(
                "https://query1.finance.yahoo.com/v8/finance/chart/"
                "%5ETNX?interval=1d&range=5d"
            )
            meta = data["chart"]["result"][0]["meta"]
            price = meta.get("regularMarketPrice")
            if price is not None and 0.5 <= float(price) <= 20.0:
                self._tnx_prev = self._tnx if self._tnx_ts > 0 else float(price)
                self._tnx = float(price)
                self._tnx_ts = now
        except Exception:
            pass

    def treasury_vol_mult(self) -> float:
        """10Y yield change → stock vol multiplier.

        Rapid yield increases (+20bps) → 1.08× vol (rate hikes disruptive).
        Small moves (<3bps) → neutral.
        """
        self._refresh_tnx()
        if self._tnx_ts == 0:
            return 1.0
        change_bps = (self._tnx - self._tnx_prev) * 100
        if abs(change_bps) < 3:
            raw_mult = 1.0
        elif change_bps > 0:
            raw_mult = 1.0 + min(0.08, change_bps * 0.004)
        else:
            raw_mult = 1.0 + max(-0.05, change_bps * 0.0025)
        raw_mult = max(0.93, min(1.10, raw_mult))
        age = time.time() - self._tnx_ts
        w = self._staleness_weight(age, self._TNX_FRESH, self._TNX_STALE)
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
        # v23: pretrained quantile model disabled (28% CRPS improvement)
        # self._load_pretrained()

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
            sigma_base *= _ext.treasury_vol_mult()
        elif asset in _CRYPTO_ASSETS:
            sigma_base *= _ext.crypto_vol_mult()
            # Deribit implied vol: forward-looking sigma correction
            if asset in _OKX_FUNDING_MAP and last_price > 0 and slow_std > 1e-10:
                realized_annual = (slow_std / last_price) * math.sqrt(288 * 365.25) * 100
                sigma_base *= _ext.crypto_dvol_mult(asset, realized_annual)

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
        # Funding rate: extreme leverage → wider tails
        funding_adj = _ext.funding_tail_adj(asset) if asset in _OKX_FUNDING_MAP else 0.0
        tail_w = min(0.50, tail_w_cal + 0.05 * ri + funding_adj)

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

        # ── v25: Tail thinning at long steps (CLT → more Normal) ─────────
        if step >= 21600:
            tail_red = 0.7 if step < 86400 else 0.5
            for d in distributions:
                comps = d.get("components", [])
                for c in comps:
                    dn = c.get("density", {}).get("name", "")
                    if dn in ("t", "laplace"):
                        c["weight"] *= tail_red
                tw = sum(c["weight"] for c in comps)
                if tw > 0:
                    for c in comps:
                        c["weight"] /= tw

        # ── v25: Time-of-day sigma for 24h-spanning predictions ──────────
        tod = _TOD_MULT.get(asset)
        if tod and num_segments >= 12 and num_segments * step >= 43200:
            current_ts = float(pairs[-1][0])
            for k, d in enumerate(distributions, 1):
                seg_hour = int(((current_ts + k * step) % 86400) // 3600)
                raw = tod.get(seg_hour, 1.0)
                m = 1.0 + _TOD_BLEND * (raw - 1.0)
                for c in d.get("components", []):
                    p = c.get("density", {}).get("params", {})
                    if "scale" in p:
                        p["scale"] *= m

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
