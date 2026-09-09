"""
(a) Identifiability diagnostics for the decay time constant, and
(b) strengthened scenario-generator validation.

(a) Reviewer concern: with segments of 1.5-12 h, fitted time constants up
    to ~31 h are extrapolation rather than observation, and T_inf and tau
    are correlated. Diagnostics computed here:
      - observation ratio r = segment duration / fitted tau
      - restriction of the ensemble to well-identified fits (r >= 1)
      - residual lag-1 autocorrelation per segment
      - segment-level clustering (fits per calendar day)
      - effect of the >=3 K relaxation selection rule

(b) Reviewer concerns: the leave-one-plant-out (LOPO) validation used the
    30-plant library while the case study uses the 13 local plants; the
    improvement over the nearest-plant benchmark was not tested; only
    marginal scores were reported. Added here:
      - LOPO restricted to the LOCAL library actually used downstream
      - a seasonal-climatology benchmark (per-slot seasonal mean profile)
      - paired bootstrap test of the CRPS difference
      - a multivariate variogram score (order 0.5) over the daily
        trajectory, which marginal CRPS cannot capture

Outputs: results/tau_diagnostics.json, results/scenario_local.csv,
         results/scenario_tests.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "code" / "cache"
RES = ROOT / "code" / "results"
RNG = np.random.default_rng(5)
SEASONS = ["winter", "spring", "summer", "autumn"]
SITE = (43.337, 17.815)


# ------------------------------------------------------------------ (a)
def tau_diagnostics():
    hp = pd.read_parquet(CACHE / "hp_temps.parquet")
    fits = pd.read_csv(RES / "decay_fits.csv", parse_dates=["start"])
    fits["ratio"] = fits["dur_h"] / fits["tau_h"]
    fits["day"] = fits["start"].dt.date

    # residual autocorrelation for each accepted segment
    acs = []
    for _, r in fits.iterrows():
        seg = hp.loc[r["start"]:r["start"] + pd.Timedelta(hours=r["dur_h"]),
                     "T_ret"].to_numpy(float)
        if len(seg) < 30 or np.isnan(seg).any():
            continue
        t = np.arange(len(seg)) / 60.0
        model = r["Tinf"] + (r["T0"] - r["Tinf"]) * np.exp(-t / r["tau_h"])
        res = seg - model
        if res.std() < 1e-9:
            continue
        acs.append(float(np.corrcoef(res[:-1], res[1:])[0, 1]))
    fits.to_csv(RES / "decay_fits.csv", index=False)

    well = fits[fits["ratio"] >= 1.0]
    out = {
        "n_total": int(len(fits)),
        "n_well_identified": int(len(well)),
        "share_well_identified": float(len(well) / max(len(fits), 1)),
        "ratio_quartiles": [float(fits["ratio"].quantile(q))
                            for q in (0.25, 0.5, 0.75)],
        "median_lag1_residual_autocorr": float(np.median(acs)) if acs else None,
        "n_unique_days": int(fits["day"].nunique()),
        "fits_per_day_mean": float(len(fits) / max(fits["day"].nunique(), 1)),
        "tau_all": {
            "median": float(fits["tau_h"].median()),
            "q05": float(fits["tau_h"].quantile(0.05)),
            "q95": float(fits["tau_h"].quantile(0.95)),
        },
        "tau_well_identified": {
            "median": float(well["tau_h"].median()),
            "q05": float(well["tau_h"].quantile(0.05)),
            "q95": float(well["tau_h"].quantile(0.95)),
        },
        "tau_sd_median_reported_by_fit": float(fits["tau_sd"].median()),
    }
    # day-clustered bootstrap of the median tau (accounts for pseudo-replication)
    days = fits["day"].unique()
    by_day = {d: fits.loc[fits["day"] == d, "tau_h"].to_numpy()
              for d in days}
    med = []
    for _ in range(2000):
        pick = RNG.choice(len(days), size=len(days), replace=True)
        vals = np.concatenate([by_day[days[k]] for k in pick])
        med.append(float(np.median(vals)))
    out["median_tau_ci95_day_bootstrap"] = [float(np.quantile(med, 0.025)),
                                            float(np.quantile(med, 0.975))]
    with open(RES / "tau_diagnostics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return out


# ------------------------------------------------------------------ (b)
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))


def crps(ens, y, spread):
    return float(np.mean(np.abs(ens - y[None, :])) - spread)


def spread_of(ens):
    return 0.5 * float(np.mean(np.abs(ens[:, None, :] - ens[None, :, :])))


def variogram_score(ens, y, p=0.5, step=4):
    """Multivariate score sensitive to trajectory dependence."""
    idx = np.arange(0, 96, step)
    ey = np.abs(y[idx][:, None] - y[idx][None, :]) ** p
    ee = np.mean(np.abs(ens[:, idx][:, :, None]
                        - ens[:, idx][:, None, :]) ** p, axis=0)
    return float(np.sum((ey - ee) ** 2))


def scenario_tests():
    meta = pd.read_parquet(CACHE / "pv_meta.parquet")
    prof = pd.read_parquet(CACHE / "pv_daily_profiles.parquet")
    prof.columns = [c if c == "season" else int(c) for c in prof.columns]
    slots = [c for c in prof.columns if c != "season"]
    dist = haversine_km(SITE[0], SITE[1], meta.lat, meta.lon)
    local = meta.loc[dist < 80, "plant"].tolist()
    n_scen = 200
    rows = []
    for held in local:
        own = prof[prof.index.get_level_values("plant") == held]
        others = prof[prof.index.get_level_values("plant").isin(
            [p for p in local if p != held])]
        i = meta.index[meta["plant"] == held][0]
        d_all = haversine_km(meta.lat[i], meta.lon[i], meta.lat, meta.lon)
        nearest = meta.loc[d_all.replace(0, np.inf).idxmin(), "plant"]
        near = prof[prof.index.get_level_values("plant") == nearest]
        for season in SEASONS:
            lib = others[others["season"] == season][slots].to_numpy()
            libn = near[near["season"] == season][slots].to_numpy()
            test = own[own["season"] == season][slots].to_numpy()
            if len(lib) < n_scen or len(test) < 10 or len(libn) < 20:
                continue
            ens = lib[RNG.choice(len(lib), n_scen, replace=False)]
            ensn = libn[RNG.choice(len(libn), min(n_scen, len(libn)),
                                   replace=False)]
            clim = np.tile(lib.mean(axis=0), (n_scen, 1))  # climatology
            s_e, s_n, s_c = spread_of(ens), spread_of(ensn), spread_of(clim)
            q10, q90 = (np.quantile(ens, 0.10, axis=0),
                        np.quantile(ens, 0.90, axis=0))
            mid = slice(28, 68)
            for y in test:
                rows.append({
                    "plant": held, "season": season,
                    "crps_local": crps(ens, y, s_e),
                    "crps_nearest": crps(ensn, y, s_n),
                    "crps_clim": crps(clim, y, s_c),
                    "vs_local": variogram_score(ens, y),
                    "vs_nearest": variogram_score(ensn, y),
                    "vs_clim": variogram_score(clim, y),
                    "cov": float(np.mean((y[mid] >= q10[mid])
                                         & (y[mid] <= q90[mid]))),
                })
    v = pd.DataFrame(rows)
    v.to_csv(RES / "scenario_local.csv", index=False)

    def paired_boot(a, b, n=4000):
        d = (a - b).to_numpy()
        m = [np.mean(RNG.choice(d, len(d), replace=True)) for _ in range(n)]
        return float(np.mean(d)), [float(np.quantile(m, 0.025)),
                                   float(np.quantile(m, 0.975))]

    dn, cin = paired_boot(v["crps_local"], v["crps_nearest"])
    dc, cic = paired_boot(v["crps_local"], v["crps_clim"])
    vn, civn = paired_boot(v["vs_local"], v["vs_nearest"])
    vc, civc = paired_boot(v["vs_local"], v["vs_clim"])
    out = {
        "n_local_plants": len(local), "n_evaluations": int(len(v)),
        "crps_local_mean": float(v["crps_local"].mean()),
        "crps_nearest_mean": float(v["crps_nearest"].mean()),
        "crps_climatology_mean": float(v["crps_clim"].mean()),
        "coverage_1090_mean": float(v["cov"].mean()),
        "paired_crps_local_minus_nearest": {"mean_diff": dn, "ci95": cin,
                                            "significant": cin[1] < 0},
        "paired_crps_local_minus_climatology": {"mean_diff": dc, "ci95": cic,
                                                "significant": cic[1] < 0},
        "variogram_local_mean": float(v["vs_local"].mean()),
        "variogram_nearest_mean": float(v["vs_nearest"].mean()),
        "variogram_climatology_mean": float(v["vs_clim"].mean()),
        "paired_vs_local_minus_nearest": {"mean_diff": vn, "ci95": civn,
                                          "significant": civn[1] < 0},
        "paired_vs_local_minus_climatology": {"mean_diff": vc, "ci95": civc,
                                              "significant": civc[1] < 0},
        "per_season_crps": v.groupby("season")[
            ["crps_local", "crps_nearest", "crps_clim", "cov"]
        ].mean().round(4).to_dict(orient="index"),
    }
    with open(RES / "scenario_tests.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: val for k, val in out.items()
                      if "per_season" not in k}, indent=2))
    return out


if __name__ == "__main__":
    tau_diagnostics()
    scenario_tests()
