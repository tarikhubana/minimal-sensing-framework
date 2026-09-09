"""
Heat pump analysis: operation detection and thermal identification.

Steps
-----
1. Detect compressor operation and mode from the load-side temperature
   differential dT = T_sup - T_ret (1-min data):
     heating ON : dT >= +1.0 K   (5-min rolling median)
     cooling ON : dT <= -1.8 K
     OFF        : otherwise (idle baseline dT ~ -0.55 K, October reference)
2. Extract free-decay (OFF) segments of the return temperature and fit a
   first-order model  T(t) = Tinf + (T0 - Tinf) * exp(-t/tau)  per segment.
   The ensemble of segment estimates provides an empirical distribution of
   the dominant time constant tau (identification uncertainty from
   temperature-only measurements).

Outputs: cache/hp_ops.parquet, cache/decay_fits.csv, results/hp_stats.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "code" / "cache"
RES = ROOT / "code" / "results"
RES.mkdir(exist_ok=True)

HEAT_THR = 1.0
COOL_THR = -1.8
MIN_SEG_MIN = 90       # decay segments: min length (minutes)
MAX_SEG_MIN = 720
MIN_DROP = 3.0         # require at least 3 K of decay for a well-posed fit


def detect_ops(hp: pd.DataFrame) -> pd.DataFrame:
    """Season-aware mode detection.

    The load-side differential alone is ambiguous: during free decay after
    heating, T_sup falls below T_ret and mimics the cooling signature.
    The plant is a seasonal-changeover system, so mode eligibility is
    restricted by month (heating: Nov-Apr, cooling: Jun-Sep, shoulder
    months May/Oct: disambiguated by absolute supply temperature).
    """
    dt = (hp["T_sup"] - hp["T_ret"]).rolling(5, center=True).median()
    month = hp.index.month
    heat_season = np.isin(month, [11, 12, 1, 2, 3, 4])
    cool_season = np.isin(month, [6, 7, 8, 9])
    shoulder = ~heat_season & ~cool_season

    heat_on = (dt >= HEAT_THR) & (heat_season
                                  | (shoulder & (hp["T_sup"] > 28)))
    cool_on = (dt <= COOL_THR) & (cool_season
                                  | (shoulder & (hp["T_sup"] < 20)))

    ops = pd.DataFrame(index=hp.index)
    ops["dT"] = dt
    ops["mode"] = "off"
    ops.loc[heat_on, "mode"] = "heat"
    ops.loc[cool_on & ~heat_on, "mode"] = "cool"
    ops["on"] = ops["mode"] != "off"
    return ops


def decay_segments(hp, ops):
    """Yield free-decay (compressor-off) runs.

    Segment boundaries are taken from the load-side differential, which
    marks the instant heat delivery ceases. The revised mode detector
    (analysis_validation.detect_v2) uses absolute circuit temperature and
    therefore stays 'on' well into the decay, which would truncate the
    most informative early part of each relaxation.
    """
    off = (~ops["on_dT"]).astype(int)
    grp = (off.diff() != 0).cumsum()
    for _, idx in hp.groupby(grp).groups.items():
        seg = hp.loc[idx]
        if not (~ops.loc[idx, "on_dT"]).all():
            continue
        dur = (seg.index[-1] - seg.index[0]).total_seconds() / 60
        if dur < MIN_SEG_MIN or dur > MAX_SEG_MIN:
            continue
        yield seg


def fit_decay(seg: pd.DataFrame):
    y = seg["T_ret"].to_numpy(float)
    if np.isnan(y).any():
        return None
    drop = y[0] - y[-1]
    warming = drop < 0  # cooling-season segment relaxes upward
    if abs(drop) < MIN_DROP:
        return None
    t = (seg.index - seg.index[0]).total_seconds().to_numpy() / 3600.0  # h
    def model(t, tinf, a, tau):
        return tinf + a * np.exp(-t / tau)
    try:
        p0 = (y[-1] - (5 if warming else -5), y[0] - y[-1], 2.0)
        popt, pcov = curve_fit(model, t, y, p0=p0, maxfev=10000,
                               bounds=([-20, -60, 0.1], [60, 60, 48]))
    except Exception:
        return None
    resid = y - model(t, *popt)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    if rmse > 0.5:               # reject poor fits
        return None
    return {
        "start": seg.index[0], "dur_h": t[-1],
        "season": "cooling" if warming else "heating",
        "T0": float(y[0]), "Tinf": float(popt[0]),
        "tau_h": float(popt[2]), "rmse": rmse,
        "tau_sd": float(np.sqrt(pcov[2, 2])),
    }


def main():
    hp = pd.read_parquet(CACHE / "hp_temps.parquet")
    ops = detect_ops(hp)
    ops.to_parquet(CACHE / "hp_ops.parquet")   # v1, kept for the comparison
    # The revised, electrically validated detector (analysis_validation.py)
    # is used for all downstream results.
    from analysis_validation import detect_v2
    ops_dT = ops
    ops = detect_v2(hp)
    ops["on_dT"] = ops_dT["on"]      # differential-based on/off, for decays
    ops.to_parquet(CACHE / "hp_ops_v2.parquet")

    # ---- operation statistics
    per_day = ops.resample("D").agg(
        on_frac=("on", "mean"),
        starts=("on", lambda s: int(((s.astype(int).diff()) == 1).sum())),
    )
    mode_month = (ops.groupby([ops.index.month, "mode"]).size()
                  .unstack(fill_value=0))
    mode_month = mode_month.div(mode_month.sum(axis=1), axis=0)

    # ---- decay fits
    fits = []
    for seg in decay_segments(hp, ops):
        r = fit_decay(seg)
        if r:
            fits.append(r)
    fits = pd.DataFrame(fits)
    fits.to_csv(RES / "decay_fits.csv", index=False)

    stats = {
        "n_minutes": int(len(hp)),
        "period": [str(hp.index.min()), str(hp.index.max())],
        "duty_cycle_overall": float(ops["on"].mean()),
        "duty_cycle_heat": float((ops["mode"] == "heat").mean()),
        "duty_cycle_cool": float((ops["mode"] == "cool").mean()),
        "mean_starts_per_active_day": float(
            per_day.loc[per_day["on_frac"] > 0.01, "starts"].mean()),
        "n_decay_segments": int(len(fits)),
    }
    for season in ("heating", "cooling"):
        f = fits[fits["season"] == season]["tau_h"]
        if len(f):
            stats[f"tau_{season}"] = {
                "n": int(len(f)),
                "median": float(f.median()),
                "q05": float(f.quantile(0.05)),
                "q95": float(f.quantile(0.95)),
                "iqr": [float(f.quantile(0.25)), float(f.quantile(0.75))],
            }
    with open(RES / "hp_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, default=str)
    per_day.to_csv(RES / "hp_daily.csv")
    mode_month.to_csv(RES / "hp_mode_month.csv")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
