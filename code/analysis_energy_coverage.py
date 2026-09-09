"""
Energy-weighted solar coverage and validation of the schedule-based proxy
(Section 4.6 of the paper).

The framework itself reports solar coverage of heat pump OPERATING TIME,
because temperature channels carry no power information. With the
independent electrical record available for this building, the same
quantity can be computed as coverage of heat pump ENERGY, weighting each
interval by the measured heat pump power instead of counting it equally.

Comparing the two answers the question a reviewer will ask: how much does
the schedule-based proxy distort the result for buildings where the
energy-weighted version cannot be computed?

Outputs: results/energy_coverage.csv, results/energy_coverage.json,
         figures/fig10_energy_coverage.png/.pdf
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "code" / "cache"
RES = ROOT / "code" / "results"
FIG = ROOT / "figures"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "savefig.dpi": 600, "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
BLUE, GREEN, RED, ORANGE, GREY = ("#0072B2", "#009E73", "#D55E00",
                                  "#E69F00", "#7F7F7F")

SEASONS = {"winter": [12, 1, 2], "spring": [3, 4, 5],
           "summer": [6, 7, 8], "autumn": [9, 10, 11]}
ORDER = ["winter", "spring", "summer", "autumn"]
SITE = (43.337, 17.815)
RNG = np.random.default_rng(23)
N_MC = 500
PV_THR = 0.20


def season_of(m):
    for s, ms in SEASONS.items():
        if m in ms:
            return s


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))


def main():
    from analysis_validation import (load_elec, occupancy_baseline,
                                     ground_truth)

    el = load_elec()
    gt = ground_truth(el, occupancy_baseline(el))       # has hp_kw
    ops = pd.read_parquet(CACHE / "hp_ops_v2.parquet")
    on15 = (ops["on"].resample("15min").mean() > 0.5).rename("on")

    d = pd.DataFrame({"hp_kw": gt["hp_kw"]}).join(on15, how="inner").dropna()
    d["date"] = d.index.date
    d["slot"] = d.index.hour * 4 + d.index.minute // 15

    # daily matrices: detected operation (bool) and measured HP power (kW)
    onw = d.pivot_table(index="date", columns="slot", values="on",
                        aggfunc="first").fillna(False)
    kww = d.pivot_table(index="date", columns="slot", values="hp_kw",
                        aggfunc="first").fillna(0.0)
    keep = (onw.sum(axis=1) >= 8) & (kww.sum(axis=1) > 0)
    onw, kww = onw[keep], kww[keep]
    months = pd.to_datetime(onw.index).month

    # PV scenario library, local plants
    meta = pd.read_parquet(CACHE / "pv_meta.parquet")
    prof = pd.read_parquet(CACHE / "pv_daily_profiles.parquet")
    prof.columns = [c if c == "season" else int(c) for c in prof.columns]
    slots = [c for c in prof.columns if c != "season"]
    dist = haversine_km(SITE[0], SITE[1], meta.lat, meta.lon)
    local = meta.loc[dist < 80, "plant"].tolist()
    lib_all = prof[prof.index.get_level_values("plant").isin(local)]

    rows = []
    for season in ORDER:
        sel = [season_of(m) == season for m in months]
        on_s, kw_s = onw[sel].to_numpy(bool), kww[sel].to_numpy(float)
        lib = lib_all[lib_all["season"] == season][slots].to_numpy()
        if not len(on_s) or len(lib) < 50:
            continue
        for _ in range(N_MC):
            i = RNG.integers(len(on_s))
            drow, krow = on_s[i], kw_s[i]
            pv = lib[RNG.integers(len(lib))]
            avail = pv > PV_THR
            time_cov = (drow & avail).sum() / max(drow.sum(), 1)
            e_tot = krow.sum()
            en_cov = krow[avail].sum() / e_tot if e_tot > 0 else np.nan
            rows.append({"season": season, "time_cov": time_cov,
                         "energy_cov": en_cov})
    mc = pd.DataFrame(rows).dropna()
    mc.to_csv(RES / "energy_coverage.csv", index=False)

    summ = mc.groupby("season").agg(
        time_med=("time_cov", "median"),
        time_q10=("time_cov", lambda s: s.quantile(0.1)),
        time_q90=("time_cov", lambda s: s.quantile(0.9)),
        en_med=("energy_cov", "median"),
        en_q10=("energy_cov", lambda s: s.quantile(0.1)),
        en_q90=("energy_cov", lambda s: s.quantile(0.9)),
    ).reindex(ORDER)
    summ["bias_pp"] = 100 * (summ["en_med"] - summ["time_med"])
    out = {
        "per_season": summ.round(4).to_dict(orient="index"),
        "overall_time_med": float(mc["time_cov"].median()),
        "overall_energy_med": float(mc["energy_cov"].median()),
        "mean_abs_bias_pp": float(summ["bias_pp"].abs().mean()),
        "max_abs_bias_pp": float(summ["bias_pp"].abs().max()),
        "pearson_r": float(mc["time_cov"].corr(mc["energy_cov"])),
    }
    with open(RES / "energy_coverage.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(summ.round(3).to_string())
    print(json.dumps({k: v for k, v in out.items() if k != "per_season"},
                     indent=2))

    # -------------------------------------------------------- figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.9))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.17,
                        wspace=0.28)
    x = np.arange(len(ORDER))
    w = 0.36
    ax1.bar(x - w / 2, summ["time_med"], w, color=BLUE, alpha=0.75,
            label="Operating-time coverage (framework)")
    ax1.errorbar(x - w / 2, summ["time_med"],
                 yerr=[summ["time_med"] - summ["time_q10"],
                       summ["time_q90"] - summ["time_med"]],
                 fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
    ax1.bar(x + w / 2, summ["en_med"], w, color=ORANGE, alpha=0.85,
            label="Energy coverage (measured power)")
    ax1.errorbar(x + w / 2, summ["en_med"],
                 yerr=[summ["en_med"] - summ["en_q10"],
                       summ["en_q90"] - summ["en_med"]],
                 fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.capitalize() for s in ORDER])
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Solar coverage (–)")
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=1,
               frameon=False, columnspacing=0.9, handlelength=1.2,
               handletextpad=0.4, borderaxespad=0.0, fontsize=6.5)
    ax1.set_title("(a) Proxy vs. energy-weighted coverage", loc="left",
                  fontweight="bold", pad=22)

    sub = mc.sample(min(1200, len(mc)), random_state=1)
    for season, c in zip(ORDER, (BLUE, GREEN, ORANGE, RED)):
        q = sub[sub["season"] == season]
        ax2.scatter(q["time_cov"], q["energy_cov"], s=4, alpha=0.35, c=c,
                    linewidths=0, label=season.capitalize())
    lims = [0, 1.02]
    ax2.plot(lims, lims, color="black", linewidth=0.7, linestyle="--",
             label="1:1 line")
    ax2.set_xlim(*lims)
    ax2.set_ylim(*lims)
    ax2.set_xlabel("Operating-time coverage (–)")
    ax2.set_ylabel("Energy coverage (–)")
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=5,
               frameon=False, columnspacing=0.7, handlelength=1.0,
               handletextpad=0.3, borderaxespad=0.0, fontsize=6,
               markerscale=2)
    ax2.set_title(f"(b) Agreement (r = {out['pearson_r']:.2f})", loc="left",
                  fontweight="bold", pad=22)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=0.6)
    fig.savefig(FIG / "fig10_energy_coverage.pdf")
    fig.savefig(FIG / "fig10_energy_coverage.png")
    print("fig10 saved")


if __name__ == "__main__":
    main()
