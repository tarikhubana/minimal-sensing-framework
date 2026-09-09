"""
PV fleet analysis, scenario generation, and uncertainty propagation.

1. Spatial structure: pairwise correlation of 15-min normalized power vs
   inter-plant distance (daytime intervals only).
2. Scenario generator: for a target site, the scenario library consists of
   whole normalized daily profiles (96 x 15-min) drawn from fleet plants,
   stratified by season. Leave-one-plant-out validation: scenarios built
   from the remaining 29 plants are scored against each held-out plant
   with the empirical CRPS and 10-90 % band coverage; a nearest-plant
   baseline is included for comparison.
3. Uncertainty propagation: PV scenarios and the empirical time-constant
   distribution (decay_fits.csv) are jointly propagated by Monte Carlo
   into the solar coverage of detected heat pump operation, with and
   without tau-limited load shifting, for several rooftop capacities.

Outputs: results/*.csv, results/pv_stats.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "code" / "cache"
RES = ROOT / "code" / "results"
RES.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

SEASONS = {
    "winter": [12, 1, 2], "spring": [3, 4, 5],
    "summer": [6, 7, 8], "autumn": [9, 10, 11],
}
SITE = (43.337, 17.815)  # Faculty of Mechanical Engineering, Mostar


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))


def season_of(month):
    for s, ms in SEASONS.items():
        if month in ms:
            return s
    raise ValueError


def daily_profiles(pu: pd.DataFrame) -> pd.DataFrame:
    """long-form: (plant, date, season) -> 96-vector rows"""
    df = pu.copy()
    df["date"] = df.index.date
    df["slot"] = df.index.hour * 4 + df.index.minute // 15
    long = df.melt(id_vars=["date", "slot"], var_name="plant",
                   value_name="pu", ignore_index=False)
    wide = long.pivot_table(index=["plant", "date"], columns="slot",
                            values="pu")
    wide = wide.dropna()
    months = pd.to_datetime(wide.index.get_level_values("date")).month
    wide["season"] = [season_of(m) for m in months]
    return wide


def crps_ensemble(ens: np.ndarray, y: np.ndarray, term2: float) -> float:
    """Empirical CRPS averaged over the vector y. ens: (n_scen, T);
    term2 = 0.5 E|X-X'| precomputed for the fixed ensemble."""
    term1 = np.mean(np.abs(ens - y[None, :]))
    return float(term1 - term2)


def spread_term(ens: np.ndarray) -> float:
    return 0.5 * float(np.mean(np.abs(ens[:, None, :] - ens[None, :, :])))


def main():
    pu = pd.read_parquet(CACHE / "pv_pu.parquet")
    meta = pd.read_parquet(CACHE / "pv_meta.parquet")

    # ------------------------------------------------ 1 spatial structure
    day = pu[pu.mean(axis=1) > 0.02]  # daytime fleet-active intervals
    corr = day.corr()
    rows = []
    for i, pi in enumerate(meta["plant"]):
        for j, pj in enumerate(meta["plant"]):
            if j <= i:
                continue
            d = haversine_km(meta.lat[i], meta.lon[i], meta.lat[j], meta.lon[j])
            rows.append({"a": pi, "b": pj, "dist_km": d,
                         "corr": corr.loc[pi, pj]})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(RES / "corr_distance.csv", index=False)

    # ------------------------------------------------ 2 scenarios + LOPO
    prof_file = CACHE / "pv_daily_profiles.parquet"
    if prof_file.exists():
        prof = pd.read_parquet(prof_file)
    else:
        prof = daily_profiles(pu)
        prof.columns = [str(c) for c in prof.columns]
        prof.to_parquet(prof_file)
    prof.columns = [c if c == "season" else int(c) for c in prof.columns]
    slots = [c for c in prof.columns if c != "season"]
    n_scen = 200
    val = []
    for held in meta["plant"]:
        others = prof[prof.index.get_level_values("plant") != held]
        own = prof[prof.index.get_level_values("plant") == held]
        # nearest-plant baseline
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
            test = test[RNG.choice(len(test), min(40, len(test)),
                                   replace=False)]
            ens = lib[RNG.choice(len(lib), n_scen, replace=False)]
            ensn = libn[RNG.choice(len(libn), min(n_scen, len(libn)),
                                   replace=False)]
            t2, t2n = spread_term(ens), spread_term(ensn)
            q10 = np.quantile(ens, 0.10, axis=0)
            q90 = np.quantile(ens, 0.90, axis=0)
            mid = slice(28, 68)  # 07:00-17:00
            c_fleet, c_near, cov = [], [], []
            for y in test:
                c_fleet.append(crps_ensemble(ens, y, t2))
                c_near.append(crps_ensemble(ensn, y, t2n))
                cov.append(np.mean((y[mid] >= q10[mid]) & (y[mid] <= q90[mid])))
            val.append({"plant": held, "season": season,
                        "crps_fleet": np.mean(c_fleet),
                        "crps_nearest": np.mean(c_near),
                        "coverage_1090": np.mean(cov)})
    val = pd.DataFrame(val)
    val.to_csv(RES / "scenario_validation.csv", index=False)

    # ------------------------------------------------ 3 MC propagation
    ops = pd.read_parquet(CACHE / "hp_ops_v2.parquet")
    on15 = ops["on"].resample("15min").mean() > 0.5
    fits = pd.read_csv(RES / "decay_fits.csv")
    taus = fits["tau_h"].to_numpy()

    # scenario library near Mostar (< 80 km) for site realism
    d_site = haversine_km(SITE[0], SITE[1], meta.lat, meta.lon)
    local = meta.loc[d_site < 80, "plant"].tolist()
    lib_local = prof[prof.index.get_level_values("plant").isin(local)]

    on = pd.DataFrame({"on": on15})
    on["date"] = on.index.date
    on["slot"] = on.index.hour * 4 + on.index.minute // 15
    onw = on.pivot_table(index="date", columns="slot", values="on",
                         aggfunc="first").fillna(False)
    onw = onw[onw.sum(axis=1) >= 8]  # days with >= 2 h operation
    months = pd.to_datetime(onw.index).month

    n_mc = 500
    out = []
    for season in SEASONS:
        days = onw[[season_of(m) == season for m in months]]
        lib = lib_local[lib_local["season"] == season][slots].to_numpy()
        if not len(days) or len(lib) < 50:
            continue
        for _ in range(n_mc):
            drow = days.iloc[RNG.integers(len(days))].to_numpy(bool)
            pv = lib[RNG.integers(len(lib))]
            tau = taus[RNG.integers(len(taus))]
            avail = pv > 0.20
            base = (drow & avail).sum() / max(drow.sum(), 1)
            smax = int(np.clip(tau / 4.0, 0.5, 4.0) * 4)  # slots
            best = base
            for shift in range(-smax, smax + 1):
                sh = np.roll(drow, shift)
                best = max(best, (sh & avail).sum() / max(sh.sum(), 1))
            out.append({"season": season, "coverage": base,
                        "coverage_shift": best, "tau": tau})
    mc = pd.DataFrame(out)
    mc.to_csv(RES / "mc_coverage.csv", index=False)

    stats = {
        "corr_pairs": len(pairs),
        "corr_at_0_50km": float(pairs[pairs.dist_km < 50]["corr"].mean()),
        "corr_over_200km": float(pairs[pairs.dist_km > 200]["corr"].mean()),
        "crps_fleet_mean": float(val["crps_fleet"].mean()),
        "crps_nearest_mean": float(val["crps_nearest"].mean()),
        "coverage_1090_mean": float(val["coverage_1090"].mean()),
        "local_plants": local,
        "mc_summary": {
            s: {
                "base_med": float(g["coverage"].median()),
                "base_q10": float(g["coverage"].quantile(0.1)),
                "base_q90": float(g["coverage"].quantile(0.9)),
                "shift_med": float(g["coverage_shift"].median()),
            } for s, g in mc.groupby("season")
        },
    }
    with open(RES / "pv_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
