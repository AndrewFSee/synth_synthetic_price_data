"""
CrunchDAO Synth Competition -- Fox-architecture Adaptive (realistic-gazelle)

v30 — Enhanced external data signals:
  v29: Fox-full architecture (-22% CRPS over 55 days vs v26)
  v30: Add derivatives data (Deribit implied vol), leverage metrics (OKX funding),
       and macro signals (10Y Treasury yield) for better sigma calibration.
MAX 3 leaf components per density (framework limit).
"""

import json
import math
import numpy as np
import time
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from crunch_synth import TrackerBase

# ── External volatility signals ──────────────────────────────────────────────
_STOCK_ASSETS = {"SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"}
_CRYPTO_ASSETS = {"BTC", "ETH", "SOL", "XAUT"}
_VIX_BASELINE = 20.0
_FNG_NEUTRAL = 50
_TNX_BASELINE = 4.0     # long-term 10Y yield baseline

# OKX perpetual swap instrument IDs for funding rate
_OKX_FUNDING_MAP = {"BTC": "BTC-USDT-SWAP", "ETH": "ETH-USDT-SWAP", "SOL": "SOL-USDT-SWAP"}


class _ExternalSignals:
    """Multi-source signal fetcher with staleness decay.

    Sources: VIX, Fear & Greed, Deribit DVOL (implied vol),
    OKX funding rate (leverage), 10Y Treasury yield (macro).
    """

    _VIX_TTL = 900
    _FNG_TTL = 3600
    _VIX_FRESH = 3600
    _VIX_STALE = 21600
    _FNG_FRESH = 21600
    _FNG_STALE = 86400
    # Deribit DVOL
    _DVOL_TTL = 900
    _DVOL_FRESH = 3600
    _DVOL_STALE = 21600
    # OKX funding rate
    _FUNDING_TTL = 1800
    _FUNDING_FRESH = 7200
    _FUNDING_STALE = 28800
    # 10Y Treasury yield
    _TNX_TTL = 3600
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
        if age <= fresh:
            return 1.0
        if age >= stale:
            return 0.0
        return (stale - age) / (stale - fresh)

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
                self._vix_ts = now
        except Exception:
            pass

    def stock_vol_mult(self) -> float:
        self._refresh_vix()
        raw_mult = math.sqrt(self._vix / _VIX_BASELINE)
        raw_mult = max(0.7, min(1.6, raw_mult))
        age = time.time() - self._vix_ts
        w = self._staleness_weight(age, self._VIX_FRESH, self._VIX_STALE)
        return 1.0 + w * (raw_mult - 1.0)

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
        self._refresh_fng()
        fng = self._fng
        if fng < 20:
            raw_mult = 1.0 + 0.0125 * (20 - fng)
        elif fng < 40:
            raw_mult = 1.0 + 0.005 * (40 - fng)
        elif fng <= 60:
            raw_mult = 1.0
        elif fng <= 80:
            raw_mult = max(0.92, 1.0 - 0.004 * (fng - 60))
        else:
            raw_mult = 1.0 + 0.005 * (fng - 80)
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
                    val = float(points[-1][4])
                    if 5.0 <= val <= 200.0:
                        setattr(self, attr, val)
                        self._dvol_ts = now
            except Exception:
                pass

    def crypto_dvol_mult(self, asset: str, realized_annual_vol: float) -> float:
        """Deribit implied vol vs realized vol → sigma multiplier."""
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

    # ── OKX Funding Rate (crypto leverage) ───────────────────────

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
        """Extreme funding rate → wider tails (leverage risk)."""
        self._refresh_funding()
        rate = self._funding.get(asset)
        if rate is None:
            return 0.0
        raw_adj = min(0.04, max(0.0, (abs(rate) - 0.0002) * 40))
        age = time.time() - self._funding_ts
        w = self._staleness_weight(age, self._FUNDING_FRESH, self._FUNDING_STALE)
        return raw_adj * w

    # ── 10Y Treasury Yield (macro signal for stocks) ───────────────

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
        """10Y yield change → stock vol multiplier."""
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

# Asset-specific 5-min sigma defaults (rough calibration from CRPS bounds)
_DEFAULT_SIGMA_5M: Dict[str, float] = {
    "BTC": 300.0, "ETH": 18.0, "SOL": 0.9, "XAUT": 7.0,
    "SPYX": 0.7, "NVDAX": 0.5, "TSLAX": 1.3, "AAPLX": 0.45, "GOOGLX": 0.7,
}

# ── Per-asset calibrated parameters ──────────────────────────────────────────
# Fox v23 proven params.  All assets use tsm + df.
# Stocks with df>=20 get Laplace tail (Student-t ≈ Normal at high df, wastes slot).
# Keys: tail_w, tsm (t_scale_mult), df, ewma_lam, lookback
_CAL: Dict[str, Dict[str, float]] = {
    # Crypto
    "BTC":    {"tail_w": 0.20, "tsm": 0.30, "df": 4.2, "lam": 0.88, "lb": 250},
    "ETH":    {"tail_w": 0.39, "tsm": 0.50, "df": 3.0, "lam": 0.92, "lb": 200},
    "XAUT":   {"tail_w": 0.34, "tsm": 0.50, "df": 6.8, "lam": 0.86, "lb":  50},
    "SOL":    {"tail_w": 0.34, "tsm": 0.50, "df": 3.4, "lam": 0.92, "lb": 200},
    # Stocks
    "SPYX":   {"tail_w": 0.29, "tsm": 0.30, "df": 16., "lam": 0.92, "lb": 250},
    "NVDAX":  {"tail_w": 0.24, "tsm": 0.40, "df": 30., "lam": 0.86, "lb": 225},
    "TSLAX":  {"tail_w": 0.39, "tsm": 0.60, "df": 30., "lam": 0.91, "lb": 175},
    "AAPLX":  {"tail_w": 0.39, "tsm": 0.40, "df": 30., "lam": 0.90, "lb": 375},
    "GOOGLX": {"tail_w": 0.39, "tsm": 0.30, "df": 30., "lam": 0.90, "lb": 175},
}
_CAL_DEFAULT = {"tail_w": 0.35, "tsm": 0.50, "df": 8.0, "lam": 0.90, "lb": 200}


class MyTracker(TrackerBase):
    """Fox-architecture adaptive tracker.  ≤ 3 components always."""

    def __init__(self):
        super().__init__()



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
        """EWMA variance — λ=0.94 (fox-proven default)."""
        if len(returns) < 5:
            return float(np.var(returns)) + 1e-10
        seed_n = min(10, len(returns))
        var = float(np.mean(returns[:seed_n] ** 2))
        for r in returns[seed_n:]:
            var = lam * var + (1 - lam) * r * r
        return max(var, 1e-10)

    def _vol_regime(self, returns: np.ndarray) -> Tuple[float, float, float]:
        """Two-scale regime detection (fox architecture).

        Returns (fast_std, slow_std, regime_intensity):
            fast_std:  ~1h  (12 bars)
            slow_std:  ~8h  (96 bars)
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

        # ── Per-asset calibrated parameters ──
        cal = _CAL.get(asset, _CAL_DEFAULT)
        tail_w_cal = cal["tail_w"]
        t_scale_mult = cal["tsm"]
        df_cal = cal["df"]
        ewma_lam = cal["lam"]
        lookback = int(cal["lb"])

        recent_raw = returns[-lookback:] if len(returns) >= lookback else returns
        recent = self._winsorize(recent_raw)

        # ── Two-scale regime detection ──
        fast_std, slow_std, ri = self._vol_regime(recent)
        blend_fast = 0.3 + 0.5 * ri
        sigma_base = blend_fast * fast_std + (1 - blend_fast) * slow_std

        # ── External volatility signal ──
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

        # ── Component weights ──
        funding_adj = _ext.funding_tail_adj(asset) if asset in _OKX_FUNDING_MAP else 0.0
        tail_w = min(0.50, tail_w_cal + 0.05 * ri + funding_adj)
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
            garch_decay = 1.0 / (1.0 + scale_factor / 48.0)
            eff_blend = blend_fast * garch_decay
            sigma = eff_blend * garch_sig + (1 - eff_blend) * (sigma_base * sqrt_sf)
            sigma = max(min_scale * sqrt_sf, min(max_scale * sqrt_sf, sigma))

            t_scale = sigma * t_scale_mult

            # Tail: Laplace for stocks (df>=20), Student-t for crypto
            if df_cal >= 20:
                tail_comp = {"density": {"type": "builtin", "name": "laplace",
                                          "params": {"loc": 0.0, "scale": t_scale}},
                              "weight": tail_w}
            else:
                tail_comp = {"density": {"type": "builtin", "name": "t",
                                          "params": {"df": df, "loc": 0.0, "scale": t_scale}},
                              "weight": tail_w}

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
                    {"density": {"type": "builtin", "name": "laplace",
                                 "params": {"loc": 0.0, "scale": sigma * 1.3}},
                     "weight": 0.30},
                ],
            })
        return distributions
