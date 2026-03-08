"""Quick script to check if calibration hit boundary effects."""
import numpy as np
import math
from calibrate import (
    _precompute_sigmas, _score_with_cached_cdfs, SQRT_2PI,
    CRPS_BOUNDS, EVAL_SUBSAMPLE, BACKTEST_DAYS,
    fetch_asset_history, resample_to_resolution, compute_returns,
    RESOLUTION_S, HISTORY_DAYS,
)

# Optimal (lam, lb) from calibration
ASSET_PARAMS = {
    "SOL":  {"lam": 0.93, "lb": 250},
    "BTC":  {"lam": 0.92, "lb": 250},
    "ETH":  {"lam": 0.93, "lb": 250},
    "SPYX": {"lam": 0.88, "lb": 150},
}

for asset in ["SOL", "BTC", "ETH", "SPYX"]:
    raw = fetch_asset_history(asset, days=HISTORY_DAYS)
    prices = resample_to_resolution(raw, RESOLUTION_S)
    returns = compute_returns(prices)
    n = len(returns)
    eval_bars = min(BACKTEST_DAYS * 288, n - 300 - 10)
    start_idx = n - eval_bars
    crps_t = CRPS_BOUNDS["t"][asset]
    num_pts = CRPS_BOUNDS["num_points"]
    ts = np.linspace(-crps_t, crps_t, num_pts)
    dt = ts[1] - ts[0]

    p = ASSET_PARAMS[asset]
    sigmas, observed = _precompute_sigmas(returns, p["lb"], p["lam"], start_idx, n, EVAL_SUBSAMPLE)
    s = sigmas[:, np.newaxis]
    z_norm = ts[np.newaxis, :] / s
    norm_cdf = np.cumsum(np.exp(-0.5 * z_norm * z_norm) / (s * SQRT_2PI), axis=1) * dt
    indicator = (ts[np.newaxis, :] >= observed[:, np.newaxis]).astype(float)

    print(f"\n{'='*60}")
    print(f"  {asset}: Extended grid search (lam={p['lam']}, lb={p['lb']})")
    print(f"{'='*60}")

    # Pure Normal baseline
    pure_norm = _score_with_cached_cdfs(norm_cdf, norm_cdf, indicator, ts, crps_t, 0.0)
    print(f"  Pure Normal (tw=0):          CRPS = {pure_norm:.6f}")

    # Extended grid
    best = (float("inf"), None)
    results = []
    for tw in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
        for tsm in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.6, 2.0]:
            for df in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 30.0]:
                t_scale = s * tsm
                z_t = ts[np.newaxis, :] / t_scale
                log_c = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
                t_pdfs = math.exp(log_c) * (1 + z_t * z_t / df) ** (-(df + 1) / 2) / t_scale
                t_cdf = np.cumsum(t_pdfs, axis=1) * dt
                score = _score_with_cached_cdfs(norm_cdf, t_cdf, indicator, ts, crps_t, tw)
                results.append((score, tw, tsm, df))
                if score < best[0]:
                    best = (score, (tw, tsm, df))

    # Sort and show top 10
    results.sort(key=lambda x: x[0])
    print(f"\n  Top 10 configs:")
    print(f"  {'Rank':>4} {'tw':>6} {'tsm':>6} {'df':>6}  {'CRPS':>10}")
    for i, (sc, tw, tsm, df) in enumerate(results[:10]):
        print(f"  {i+1:>4} {tw:>6.2f} {tsm:>6.2f} {df:>6.1f}  {sc:>10.6f}")

    # Show how CRPS varies with tw (at best tsm/df)
    _, best_tw, best_tsm, best_df = results[0]
    print(f"\n  CRPS vs tail_w (tsm={best_tsm}, df={best_df}):")
    for tw in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]:
        t_scale = s * best_tsm
        z_t = ts[np.newaxis, :] / t_scale
        log_c = math.lgamma((best_df + 1) / 2) - math.lgamma(best_df / 2) - 0.5 * math.log(best_df * math.pi)
        t_pdfs = math.exp(log_c) * (1 + z_t * z_t / best_df) ** (-(best_df + 1) / 2) / t_scale
        t_cdf = np.cumsum(t_pdfs, axis=1) * dt
        score = _score_with_cached_cdfs(norm_cdf, t_cdf, indicator, ts, crps_t, tw)
        marker = " ◄" if abs(tw - best_tw) < 0.001 else ""
        print(f"    tw={tw:.2f}: {score:.6f}{marker}")
