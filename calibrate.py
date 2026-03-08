"""
Empirical parameter calibration for Crunch Synth models.

Uses pricedb historical data to:
1. Compute per-asset distributional statistics (sigma, skewness, kurtosis, vol-of-vol)
2. Fit GARCH(1,1) parameters per asset via MLE-style grid search
3. Backtest different density configurations against real data via CRPS
4. Output calibrated parameters ready to bake into model code

Usage:
    python calibrate.py stats          # Phase 1: fetch data & compute statistics
    python calibrate.py backtest       # Phase 2: grid-search density params via CRPS
    python calibrate.py full           # Both phases
    python calibrate.py eval           # Full evaluation using TrackerEvaluator
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

from crunch_synth.price_provider import pricedb
from crunch_synth.constants import SUPPORTED_ASSETS, FORECAST_PROFILES, CRPS_BOUNDS

# ─── Configuration ────────────────────────────────────────────────────────────

# How much history to fetch for calibration
HISTORY_DAYS = 90       # 3 months of data
RESOLUTION_S = 300      # 5-min bars (native resolution)

# CRPS backtest window: last N days of the fetched history
BACKTEST_DAYS = 14      # 2 weeks for backtesting
WARMUP_DAYS = 30        # warm-up before backtest window

# Output
OUT_DIR = Path(__file__).resolve().parent / "calibration_results"

# ─── Data Fetching ────────────────────────────────────────────────────────────

def fetch_asset_history(asset: str, days: int = HISTORY_DAYS) -> List[Tuple[float, float]]:
    """Fetch historical price data from pricedb."""
    to = datetime.now(timezone.utc)
    from_ = to - timedelta(days=days)
    print(f"  Fetching {asset}: {from_.date()} to {to.date()} ({days}d)...", end=" ", flush=True)
    t0 = time.time()
    data = pricedb.get_price_history(asset=asset, from_=from_, to=to, timeout=60)
    elapsed = time.time() - t0
    print(f"{len(data)} ticks ({elapsed:.1f}s)")
    return data


def resample_to_resolution(data: List[Tuple[float, float]], res: int = RESOLUTION_S) -> np.ndarray:
    """Resample raw tick data to fixed-resolution price array."""
    if not data:
        return np.array([])
    timestamps, prices = zip(*data)
    ts_arr = np.array(timestamps)
    px_arr = np.array(prices, dtype=float)

    # Create regular grid
    t_start = int(ts_arr[0] // res) * res
    t_end = int(ts_arr[-1] // res) * res
    grid = np.arange(t_start, t_end + res, res)

    # Nearest-neighbor resample
    sampled = np.interp(grid, ts_arr, px_arr)
    return sampled


def compute_returns(prices: np.ndarray) -> np.ndarray:
    """Compute price increments (not log returns — matching competition format)."""
    return np.diff(prices)


# ─── Phase 1: Statistical Analysis ───────────────────────────────────────────

def winsorize(v: np.ndarray, lo: float = 0.005, hi: float = 0.995) -> np.ndarray:
    if len(v) < 20:
        return v
    return np.clip(v, float(np.quantile(v, lo)), float(np.quantile(v, hi)))


def compute_asset_stats(returns: np.ndarray, asset: str) -> Dict[str, Any]:
    """Compute comprehensive distributional statistics for one asset."""
    r = winsorize(returns)
    n = len(r)

    mu = float(np.mean(r))
    sigma = float(np.std(r))
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))

    # Higher moments
    centered = r - mu
    skew = float(np.mean(centered ** 3) / (sigma ** 3 + 1e-12))
    kurt = float(np.mean(centered ** 4) / (sigma ** 4 + 1e-12))  # raw kurtosis
    excess_kurt = kurt - 3.0  # excess kurtosis (0 = Gaussian)

    # Tail statistics
    q01 = float(np.quantile(r, 0.01))
    q05 = float(np.quantile(r, 0.05))
    q25 = float(np.quantile(r, 0.25))
    q75 = float(np.quantile(r, 0.75))
    q95 = float(np.quantile(r, 0.95))
    q99 = float(np.quantile(r, 0.99))
    iqr = q75 - q25

    # Volatility of volatility (vol-of-vol)
    # Rolling 1h (12 bars) standard deviation, then std of that
    window = 12
    if n > window + 5:
        rolling_vol = np.array([
            float(np.std(r[i:i+window]))
            for i in range(n - window)
        ])
        vol_of_vol = float(np.std(rolling_vol) / (np.mean(rolling_vol) + 1e-10))
    else:
        vol_of_vol = 0.0

    # Autocorrelation of squared returns (GARCH indicator)
    sq = r ** 2
    if len(sq) > 5:
        acf1_sq = float(np.corrcoef(sq[:-1], sq[1:])[0, 1])
        acf1_sq = 0.0 if np.isnan(acf1_sq) else acf1_sq
    else:
        acf1_sq = 0.0

    # Multi-scale volatility
    windows = {"1h": 12, "4h": 48, "12h": 144, "24h": 288}
    vol_scales = {}
    for label, w in windows.items():
        chunk = r[-min(w, n):]
        vol_scales[label] = float(np.std(chunk))

    # Regime detection: ratio of fast vs slow vol
    fast_std = vol_scales.get("1h", sigma)
    slow_std = vol_scales.get("12h", sigma)
    vol_ratio = fast_std / (slow_std + 1e-10)

    # CRPS integration context
    crps_t = CRPS_BOUNDS["t"].get(asset, 1.0)
    # What fraction of returns fall within CRPS bounds?
    in_bounds = float(np.mean(np.abs(r) < crps_t))

    return {
        "asset": asset,
        "n_returns": n,
        "mean": mu,
        "std": sigma,
        "median": med,
        "mad": mad,
        "skewness": skew,
        "excess_kurtosis": excess_kurt,
        "q01": q01, "q05": q05, "q25": q25, "q75": q75, "q95": q95, "q99": q99,
        "iqr": iqr,
        "vol_of_vol": vol_of_vol,
        "acf1_squared": acf1_sq,
        "vol_scales": vol_scales,
        "vol_ratio_fast_slow": vol_ratio,
        "crps_t": crps_t,
        "pct_in_crps_bounds": in_bounds,
    }


def fit_garch_grid(returns: np.ndarray) -> Dict[str, float]:
    """Fit GARCH(1,1) via grid search minimizing variance prediction error.

    Finds omega, alpha, beta that minimize the MSE between
    predicted variance (GARCH one-step) and realized squared return.
    """
    r = winsorize(returns[-2000:])  # Use last ~7 days for fitting
    n = len(r)
    if n < 50:
        return {"omega": 1e-6, "alpha": 0.10, "beta": 0.85}

    sq = r ** 2
    var_emp = float(np.var(r))

    best_mse = float("inf")
    best_params = {"omega": 1e-6, "alpha": 0.10, "beta": 0.85}

    for alpha in np.arange(0.02, 0.35, 0.02):
        for beta in np.arange(0.50, 0.96, 0.02):
            if alpha + beta >= 0.99:
                continue
            omega = var_emp * (1 - alpha - beta)
            if omega < 0:
                continue

            # Run GARCH forward
            var_t = var_emp
            mse = 0.0
            for i in range(1, n):
                var_t = omega + alpha * sq[i-1] + beta * var_t
                mse += (var_t - sq[i]) ** 2

            mse /= (n - 1)
            if mse < best_mse:
                best_mse = mse
                best_params = {"omega": float(omega), "alpha": float(alpha), "beta": float(beta)}

    return best_params


def compute_optimal_student_t_df(returns: np.ndarray) -> float:
    """Estimate optimal Student-t df from excess kurtosis.

    For Student-t: excess_kurtosis = 6 / (df - 4) for df > 4.
    Inverting: df = 4 + 6 / excess_kurtosis.
    """
    r = winsorize(returns)
    mu = np.mean(r)
    sigma = np.std(r)
    centered = r - mu
    kurt = float(np.mean(centered ** 4) / (sigma ** 4 + 1e-12))
    excess = kurt - 3.0

    if excess <= 0.1:
        return 30.0  # Near-Gaussian, very high df
    df = 4.0 + 6.0 / excess
    return max(2.5, min(30.0, df))


# ─── Phase 2: CRPS Backtest (Vectorized, CDF-cached) ─────────────────────────

SQRT_2PI = math.sqrt(2.0 * math.pi)

# Subsample: every 6th bar = ~30-min spacing (sufficient for CRPS estimation)
EVAL_SUBSAMPLE = 6

if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz


def _precompute_sigmas(
    returns: np.ndarray,
    lookback: int,
    ewma_lam: float,
    start_idx: int,
    end_idx: int,
    step: int = EVAL_SUBSAMPLE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute EWMA sigma for each evaluation bar (subsampled)."""
    indices = list(range(start_idx, end_idx, step))
    sigmas = np.empty(len(indices))
    observed = np.empty(len(indices))
    for j, i in enumerate(indices):
        lb_start = max(0, i - lookback)
        window = returns[lb_start:i]
        if len(window) < 20:
            sigmas[j] = np.nan
            observed[j] = 0.0
            continue
        seed_n = min(10, len(window))
        var = float(np.mean(window[:seed_n] ** 2))
        for r in window[seed_n:]:
            var = ewma_lam * var + (1 - ewma_lam) * r * r
        sigmas[j] = math.sqrt(max(var, 1e-12))
        observed[j] = float(returns[i])
    mask = ~np.isnan(sigmas)
    return sigmas[mask], observed[mask]


def _score_with_cached_cdfs(
    norm_cdf: np.ndarray,
    t_cdf: np.ndarray,
    indicator: np.ndarray,
    ts: np.ndarray,
    crps_t: float,
    tail_w: float,
) -> float:
    """Compute mean normalized CRPS from precomputed Normal and Student-t CDFs.

    This is the innermost scoring function — only does array blending + trapezoidal.
    Everything else is precomputed.
    """
    cdf = (1.0 - tail_w) * norm_cdf + tail_w * t_cdf    # (N, P)
    np.clip(cdf, 0.0, 1.0, out=cdf)
    integrand = (cdf - indicator) ** 2
    crps_vals = _trapz(integrand, ts, axis=1)
    return float(np.mean(crps_vals / crps_t))


def grid_search_density_params(
    returns: np.ndarray,
    asset: str,
    garch_params: Dict[str, float],
    optimal_df: float,
) -> Dict[str, Any]:
    """Grid search with triple-level caching for speed.

    Cache hierarchy:
      Level 1  (lb, lam)         → sigmas, norm_cdf, indicator  [precomputed once]
      Level 2  (lb, lam, tsm, df) → t_cdf                       [reused across tail_w]
      Level 3  (tail_w)           → blend CDFs & score           [cheap array ops]

    Two-phase strategy:
      Phase A — full coarse grid.
      Phase B — fine refinement around the Phase A winner.
    """
    crps_t = CRPS_BOUNDS["t"][asset]
    num_pts = CRPS_BOUNDS["num_points"]
    n = len(returns)

    eval_bars = min(BACKTEST_DAYS * 288, n - 300 - 10)
    if eval_bars < 50:
        print(f"  {asset}: not enough data for backtest ({eval_bars} bars)")
        return {}
    start_idx = n - eval_bars

    ts = np.linspace(-crps_t, crps_t, num_pts)  # (P,)
    dt = ts[1] - ts[0]

    # ── Parameter grids (extended to cover low tsm range) ──
    tail_weights = [0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35]
    t_scale_mults = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0, 1.3, 1.6, 2.0]
    dfs_raw = [max(2.5, optimal_df - 1.5), optimal_df,
               min(30, optimal_df + 1.5), 5.0, 8.0, 15.0, 30.0]
    dfs = sorted(set(round(d, 1) for d in dfs_raw))
    ewma_lams = [0.88, 0.91, 0.94, 0.97]
    lookbacks = [100, 175, 250, 350]

    total = len(lookbacks) * len(ewma_lams) * len(t_scale_mults) * len(dfs) * len(tail_weights)
    print(f"  {asset}: Phase A — {total} configs "
          f"({len(lookbacks)}×{len(ewma_lams)}×{len(t_scale_mults)}×{len(dfs)}×{len(tail_weights)})",
          flush=True)

    best_score = float("inf")
    best_config: Dict[str, Any] = {}
    t0 = time.time()
    batch_done = 0
    total_batches = len(lookbacks) * len(ewma_lams)

    # Level-1 cache: (lb, lam) → (sigmas, norm_cdf, indicator)
    l1_cache: Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for lb in lookbacks:
        for lam in ewma_lams:
            # ── Level 1: precompute sigmas, norm_cdf, indicator ──
            sigmas, observed = _precompute_sigmas(returns, lb, lam, start_idx, n, EVAL_SUBSAMPLE)
            N = len(sigmas)
            if N < 50:
                batch_done += 1
                continue

            s = sigmas[:, np.newaxis]                                    # (N, 1)
            z_norm = ts[np.newaxis, :] / s                               # (N, P)
            norm_pdfs = np.exp(-0.5 * z_norm * z_norm) / (s * SQRT_2PI)  # (N, P)
            norm_cdf = np.cumsum(norm_pdfs, axis=1) * dt                 # (N, P)
            indicator = (ts[np.newaxis, :] >= observed[:, np.newaxis]).astype(float)

            l1_cache[(lb, lam)] = (sigmas, norm_cdf, indicator)

            # ── Level 2: iterate (tsm, df) → t_cdf ──
            for tsm in t_scale_mults:
                t_scale = s * tsm                                         # (N, 1)
                z_t = ts[np.newaxis, :] / t_scale                         # (N, P)

                for df in dfs:
                    log_c = (math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
                             - 0.5 * math.log(df * math.pi))
                    coeff = math.exp(log_c)
                    t_pdfs = coeff * (1 + z_t * z_t / df) ** (-(df + 1) / 2) / t_scale
                    t_cdf = np.cumsum(t_pdfs, axis=1) * dt                # (N, P)

                    # ── Level 3: iterate tail_w → blend & score ──
                    for tw in tail_weights:
                        score = _score_with_cached_cdfs(
                            norm_cdf, t_cdf, indicator, ts, crps_t, tw
                        )
                        if score < best_score:
                            best_score = score
                            best_config = {
                                "tail_type": "t", "tail_w": tw,
                                "t_scale_mult": tsm, "df": df,
                                "ewma_lam": lam, "lookback": lb,
                                "crps": score,
                            }

            batch_done += 1
            elapsed = time.time() - t0
            print(f"    [{batch_done}/{total_batches}]  ({elapsed:.0f}s)  best={best_score:.6f}", flush=True)

    elapsed = time.time() - t0
    print(f"  {asset} Phase A done in {elapsed:.0f}s — best CRPS={best_score:.6f}")

    # ── Phase B: refine around best ──
    if best_config:
        b = best_config
        refine_tw = sorted(set(
            float(np.clip(b["tail_w"] + d, 0.04, 0.50))
            for d in [-0.04, -0.02, -0.01, 0, 0.01, 0.02, 0.04]
        ))
        refine_tsm = sorted(set(
            float(np.clip(b["t_scale_mult"] + d, 0.5, 4.0))
            for d in [-0.3, -0.15, 0, 0.15, 0.3]
        ))
        refine_df = sorted(set(
            round(float(np.clip(b["df"] + d, 2.5, 30.0)), 1)
            for d in [-1.0, -0.5, -0.2, 0, 0.2, 0.5, 1.0]
        ))
        refine_lam = sorted(set(
            float(np.clip(b["ewma_lam"] + d, 0.80, 0.99))
            for d in [-0.02, -0.01, 0, 0.01, 0.02]
        ))
        refine_lb = sorted(set(
            max(40, b["lookback"] + d) for d in [-50, -25, 0, 25, 50]
        ))

        total_b = len(refine_lb) * len(refine_lam) * len(refine_tsm) * len(refine_df) * len(refine_tw)
        print(f"  {asset} Phase B — {total_b} refinement configs ...", flush=True)

        for lb in refine_lb:
            for lam in refine_lam:
                key = (lb, lam)
                if key in l1_cache:
                    sigmas_r, norm_cdf_r, indicator_r = l1_cache[key]
                else:
                    sigmas_r, observed_r = _precompute_sigmas(returns, lb, lam, start_idx, n, EVAL_SUBSAMPLE)
                    if len(sigmas_r) < 50:
                        continue
                    s_r = sigmas_r[:, np.newaxis]
                    z_nr = ts[np.newaxis, :] / s_r
                    norm_cdf_r = np.cumsum(np.exp(-0.5 * z_nr * z_nr) / (s_r * SQRT_2PI), axis=1) * dt
                    indicator_r = (ts[np.newaxis, :] >= observed_r[:, np.newaxis]).astype(float)
                    l1_cache[key] = (sigmas_r, norm_cdf_r, indicator_r)

                s_r = sigmas_r[:, np.newaxis]
                for tsm in refine_tsm:
                    t_sc = s_r * tsm
                    z_tr = ts[np.newaxis, :] / t_sc
                    for df in refine_df:
                        log_c = (math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
                                 - 0.5 * math.log(df * math.pi))
                        t_pdfs_r = math.exp(log_c) * (1 + z_tr * z_tr / df) ** (-(df + 1) / 2) / t_sc
                        t_cdf_r = np.cumsum(t_pdfs_r, axis=1) * dt

                        for tw in refine_tw:
                            score = _score_with_cached_cdfs(
                                norm_cdf_r, t_cdf_r, indicator_r, ts, crps_t, tw
                            )
                            if score < best_score:
                                best_score = score
                                best_config = {
                                    "tail_type": "t", "tail_w": float(tw),
                                    "t_scale_mult": float(tsm), "df": float(df),
                                    "ewma_lam": float(lam), "lookback": int(lb),
                                    "crps": score,
                                }

        elapsed = time.time() - t0
        print(f"  {asset} Phase B done — best CRPS={best_score:.6f} (total {elapsed:.0f}s)")

    print(f"    ✓ {asset}: CRPS={best_score:.6f}  tail_w={best_config.get('tail_w'):.3f}  "
          f"tsm={best_config.get('t_scale_mult'):.2f}  df={best_config.get('df'):.1f}  "
          f"λ={best_config.get('ewma_lam'):.3f}  lb={best_config.get('lookback')}")
    return best_config


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_stats(assets: List[str] = None):
    """Phase 1: Fetch data and compute statistics."""
    if assets is None:
        assets = SUPPORTED_ASSETS

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    all_garch = {}
    all_df = {}
    all_returns = {}

    print("=" * 60)
    print("PHASE 1: Fetching data and computing statistics")
    print("=" * 60)

    for asset in assets:
        raw_data = fetch_asset_history(asset, days=HISTORY_DAYS)
        prices = resample_to_resolution(raw_data, RESOLUTION_S)
        returns = compute_returns(prices)

        if len(returns) < 100:
            print(f"  WARNING: {asset} has only {len(returns)} returns, skipping")
            continue

        all_returns[asset] = returns

        # Statistics
        stats = compute_asset_stats(returns, asset)
        all_stats[asset] = stats

        # GARCH fit
        garch = fit_garch_grid(returns)
        all_garch[asset] = garch

        # Optimal Student-t df
        opt_df = compute_optimal_student_t_df(returns)
        all_df[asset] = opt_df

        print(f"  {asset}: sigma_5m={stats['std']:.4f}, skew={stats['skewness']:.3f}, "
              f"ex_kurt={stats['excess_kurtosis']:.3f}, vol_of_vol={stats['vol_of_vol']:.3f}, "
              f"GARCH a={garch['alpha']:.3f} b={garch['beta']:.3f}, opt_df={opt_df:.1f}")

    # Save stats
    stats_path = OUT_DIR / "asset_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "generated": datetime.now(timezone.utc).isoformat(),
            "history_days": HISTORY_DAYS,
            "resolution_s": RESOLUTION_S,
            "assets": {k: _json_safe(v) for k, v in all_stats.items()},
            "garch_params": all_garch,
            "optimal_df": all_df,
        }, f, indent=2)
    print(f"\nStatistics saved to {stats_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Asset':<8} {'σ_5m':>10} {'Skew':>8} {'ExKurt':>8} {'VoV':>8} "
          f"{'α_GARCH':>8} {'β_GARCH':>8} {'opt_df':>8} {'%InBounds':>10}")
    print("-" * 90)
    for asset in assets:
        s = all_stats.get(asset)
        g = all_garch.get(asset)
        d = all_df.get(asset)
        if not s:
            continue
        print(f"{asset:<8} {s['std']:>10.4f} {s['skewness']:>8.3f} {s['excess_kurtosis']:>8.3f} "
              f"{s['vol_of_vol']:>8.3f} {g['alpha']:>8.3f} {g['beta']:>8.3f} "
              f"{d:>8.1f} {s['pct_in_crps_bounds']:>9.1%}")

    return all_returns, all_stats, all_garch, all_df


def run_backtest(all_returns=None, all_stats=None, all_garch=None, all_df=None,
                 assets: List[str] = None):
    """Phase 2: Grid search density params via CRPS backtest."""
    if assets is None:
        assets = SUPPORTED_ASSETS

    # Load stats if not provided
    if all_returns is None:
        print("Fetching data for backtest...")
        all_returns = {}
        all_garch = {}
        all_df = {}
        for asset in assets:
            raw = fetch_asset_history(asset, days=HISTORY_DAYS)
            prices = resample_to_resolution(raw, RESOLUTION_S)
            rets = compute_returns(prices)
            if len(rets) < 100:
                continue
            all_returns[asset] = rets
            all_garch[asset] = fit_garch_grid(rets)
            all_df[asset] = compute_optimal_student_t_df(rets)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PHASE 2: CRPS Grid Search Backtest")
    print("=" * 60)

    results = {}
    for asset in assets:
        if asset not in all_returns:
            continue
        best = grid_search_density_params(
            all_returns[asset], asset,
            all_garch.get(asset, {}),
            all_df.get(asset, 5.0),
        )
        results[asset] = best

    # Save results
    results_path = OUT_DIR / "backtest_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "generated": datetime.now(timezone.utc).isoformat(),
            "backtest_days": BACKTEST_DAYS,
            "results": {k: _json_safe(v) for k, v in results.items()},
        }, f, indent=2)
    print(f"\nBacktest results saved to {results_path}")

    # Print optimal params
    print("\n" + "=" * 60)
    print("OPTIMAL DENSITY PARAMETERS")
    print("=" * 60)
    print(f"{'Asset':<8} {'CRPS':>8} {'tail_w':>8} {'t_scale':>8} {'df':>6} "
          f"{'ewma_λ':>8} {'lookback':>8}")
    print("-" * 60)
    for asset in assets:
        r = results.get(asset)
        if not r:
            continue
        print(f"{asset:<8} {r.get('crps',0):>8.5f} {r.get('tail_w',0):>8.2f} "
              f"{r.get('t_scale_mult',0):>8.2f} {r.get('df',0):>6.1f} "
              f"{r.get('ewma_lam',0):>8.3f} {r.get('lookback',0):>8d}")

    # Generate code snippet
    print("\n" + "=" * 60)
    print("CODE SNIPPET: _ASSET_TUNING for fox model")
    print("=" * 60)
    print("_CALIBRATED_PARAMS: Dict[str, Dict[str, float]] = {")
    for asset in assets:
        r = results.get(asset)
        if not r:
            continue
        print(f'    "{asset}": {{"tail_w": {r.get("tail_w", 0.20):.2f}, '
              f'"t_scale_mult": {r.get("t_scale_mult", 1.8):.2f}, '
              f'"df": {r.get("df", 5.0):.1f}, '
              f'"ewma_lam": {r.get("ewma_lam", 0.94):.2f}, '
              f'"lookback": {r.get("lookback", 300)}}},')
    print("}")

    return results


def run_eval(assets: List[str] = None, horizon_key: str = "24h", days: int = 5):
    """Run full TrackerEvaluator simulation (same as example tracker)."""
    from crunch_synth import (
        TrackerEvaluator,
        load_test_prices_once,
        load_initial_price_histories_once,
        count_evaluations,
    )
    # Import our fox tracker
    sys.path.insert(0, str(Path(__file__).resolve().parent / "synth-invisible-fox"))
    from main import MyTracker

    if assets is None:
        assets = SUPPORTED_ASSETS

    tracker = MyTracker()
    evaluator = TrackerEvaluator(tracker)

    HORIZON = FORECAST_PROFILES[horizon_key]["horizon"]
    STEPS = FORECAST_PROFILES[horizon_key]["steps"]
    INTERVAL = FORECAST_PROFILES[horizon_key]["interval"]

    evaluation_end = datetime.now(timezone.utc) - timedelta(days=1)
    days_history = 30

    print(f"Loading test data ({days}d ending {evaluation_end.date()})...")
    test_prices = load_test_prices_once(assets, evaluation_end, days=days)
    print(f"Loading warm-up history ({days_history}d)...")
    initial_hist = load_initial_price_histories_once(
        assets, evaluation_end, days_history=days_history, days_offset=days
    )

    for asset in assets:
        if asset not in test_prices:
            continue
        history = test_prices[asset]
        evaluator.tick({asset: initial_hist.get(asset, [])})

        prev_ts = 0
        count = 0
        total_evals = count_evaluations(history, HORIZON, INTERVAL)
        print(f"\n[{asset}] Evaluating (~{total_evals} rounds)...")

        for ts, price in history:
            evaluator.tick({asset: [(ts, price)]})
            if ts - prev_ts >= INTERVAL:
                prev_ts = ts
                result = evaluator.predict(asset, HORIZON, STEPS)
                if result:
                    count += 1
                    if count % 10 == 0:
                        print(f"  [{asset}] {count}/{total_evals}: "
                              f"overall={evaluator.overall_score_asset(asset):.4f} "
                              f"recent={evaluator.recent_score_asset(asset):.4f}")

        print(f"  [{asset}] FINAL: overall={evaluator.overall_score_asset(asset):.4f} "
              f"recent={evaluator.recent_score_asset(asset):.4f}")

    print(f"\nGLOBAL SCORE: {evaluator.overall_score():.4f}")
    return evaluator


def _json_safe(obj):
    """Make object JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(description="Calibrate Synth model parameters from historical data.")
    parser.add_argument("mode", choices=["stats", "backtest", "full", "eval"],
                        help="stats=statistics only, backtest=CRPS grid search, full=both, eval=TrackerEvaluator")
    parser.add_argument("--assets", nargs="+", default=None,
                        help=f"Assets to calibrate (default: all). Options: {', '.join(SUPPORTED_ASSETS)}")
    parser.add_argument("--horizon", choices=["24h", "1h"], default="24h",
                        help="Horizon for eval mode (default: 24h)")
    parser.add_argument("--days", type=int, default=5,
                        help="Days of test data for eval mode (default: 5)")
    args = parser.parse_args()

    assets = args.assets or SUPPORTED_ASSETS

    if args.mode == "stats":
        run_stats(assets)
    elif args.mode == "backtest":
        all_returns, all_stats, all_garch, all_df = run_stats(assets)
        run_backtest(all_returns, all_stats, all_garch, all_df, assets)
    elif args.mode == "full":
        all_returns, all_stats, all_garch, all_df = run_stats(assets)
        run_backtest(all_returns, all_stats, all_garch, all_df, assets)
    elif args.mode == "eval":
        run_eval(assets, args.horizon, args.days)


if __name__ == "__main__":
    main()
