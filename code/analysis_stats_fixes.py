"""
Dependence-aware statistics requested in the second review round.

(1) PV benchmark significance with clustered resampling.
    The 4,576 evaluated plant-days are not independent: they cluster within
    held-out plant, within season, and across plants sharing regional
    weather on the same date. Three progressively more conservative tests
    are reported:
      a) naive paired bootstrap over plant-days (as previously reported)
      b) bootstrap clustered by held-out plant
      c) the 13 per-plant mean differences treated as the sample, tested
         by sign test and by a t-interval on 13 observations

(2) Interpretation of the uncertainty intervals.
    One Monte Carlo draw = one (day, PV scenario) pair, so the previously
    reported 10-90 % intervals are DAILY predictive intervals. For planning
    the relevant quantity is the annual PV-served fraction, obtained by
    aggregating a full synthetic year of days before forming the interval.
    Both are computed here so the paper can state the difference.

(3) Detector uncertainty by episode-block resampling.
    Interval-independent perturbation is replaced by resampling complete
    daily detector-error profiles from the held-out period, preserving the
    block structure of errors around starts, stops and free decay.

Outputs: results/stats_fixes.json, figures/fig9_shifting.png/.pdf
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
RNG = np.random.default_rng(101)
ORDER = ["winter", "spring", "summer", "autumn"]
C_REF = 30

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


# ------------------------------------------------------------------ (1)
def pv_significance():
    v = pd.read_csv(RES / "scenario_local.csv")
    out = {"n_plant_days": int(len(v)), "n_plants": int(v["plant"].nunique())}
    for lab, col in (("nearest", "crps_nearest"), ("climatology", "crps_clim")):
        d = (v["crps_local"] - v[col]).to_numpy()
        # (a) naive
        naive = [np.mean(RNG.choice(d, len(d), replace=True))
                 for _ in range(3000)]
        # (b) clustered by held-out plant
        plants = v["plant"].unique()
        by_plant = {p: (v.loc[v["plant"] == p, "crps_local"]
                        - v.loc[v["plant"] == p, col]).to_numpy()
                    for p in plants}
        clus = []
        for _ in range(3000):
            pick = RNG.choice(len(plants), len(plants), replace=True)
            clus.append(np.mean(np.concatenate(
                [by_plant[plants[k]] for k in pick])))
        # (c) 13 per-plant means
        pm = np.array([by_plant[p].mean() for p in plants])
        n = len(pm)
        se = pm.std(ddof=1) / np.sqrt(n)
        t95 = 2.179 if n == 13 else 2.0        # t_{0.975, 12}
        n_neg = int((pm < 0).sum())
        out[f"local_minus_{lab}"] = {
            "mean_diff": float(d.mean()),
            "naive_ci95": [float(np.quantile(naive, .025)),
                           float(np.quantile(naive, .975))],
            "plant_clustered_ci95": [float(np.quantile(clus, .025)),
                                     float(np.quantile(clus, .975))],
            "per_plant_mean": float(pm.mean()),
            "per_plant_ci95": [float(pm.mean() - t95 * se),
                               float(pm.mean() + t95 * se)],
            "n_plants_favouring_local": n_neg,
            "n_plants": n,
            "sign_test_note": f"{n_neg}/{n} plants favour the local library",
            "significant_clustered": bool(np.quantile(clus, .975) < 0),
            "significant_per_plant": bool(pm.mean() + t95 * se < 0),
        }
    return out


# ------------------------------------------------------------------ (2)
def daily_vs_annual():
    ov = pd.read_csv(RES / "overlap_mc.csv")
    caps = [10, 30, 50]
    out = {"draw_definition": ("one draw = one observed operating day paired "
                               "with one sampled PV scenario day"),
           "daily": {}, "annual": {}}
    # season weights = share of annual HP energy, from the measured record
    from analysis_overlap import build_days, season_of
    hp, oth, on, months, tmean = build_days()
    seasons = np.array([season_of(m) for m in months])
    e_by_season = {s: hp[seasons == s].sum() for s in ORDER}
    tot = sum(e_by_season.values())
    w = {s: e_by_season[s] / tot for s in ORDER}
    out["season_energy_weights"] = {s: float(w[s]) for s in ORDER}

    for C in caps:
        col = f"scf_hp_{C}"
        out["daily"][C] = {
            "median": float(ov[col].median()),
            "q10": float(ov[col].quantile(0.1)),
            "q90": float(ov[col].quantile(0.9)),
        }
        # annual: energy-weighted mean over a synthetic year of sampled days
        ann = []
        for _ in range(2000):
            val = 0.0
            for s in ORDER:
                g = ov.loc[ov["season"] == s, col].to_numpy()
                if len(g) == 0:
                    continue
                # ~90 days per season, sampled with replacement
                val += w[s] * float(np.mean(RNG.choice(g, 90, replace=True)))
            ann.append(val)
        out["annual"][C] = {
            "median": float(np.median(ann)),
            "q10": float(np.quantile(ann, 0.1)),
            "q90": float(np.quantile(ann, 0.9)),
        }
    return out


# ------------------------------------------------------------------ (3)
def detector_episode_blocks():
    """Resample whole daily error profiles from the held-out period."""
    from analysis_validation import (load_elec, occupancy_baseline,
                                     ground_truth, detect_v2)
    from analysis_overlap import build_days, pv_library, season_of
    el = load_elec()
    hpt = pd.read_parquet(CACHE / "hp_temps.parquet")
    gt = ground_truth(el, occupancy_baseline(el))
    ops = detect_v2(hpt)
    on15 = (ops["on"].resample("15min").mean() > 0.5).rename("det")
    j = gt.join(on15, how="inner").dropna(subset=["hp_on", "det"])
    TEST = [(2025, 8), (2025, 9), (2026, 1), (2026, 2), (2026, 3), (2026, 4)]
    ym = list(zip(j.index.year, j.index.month))
    j = j[[t in TEST for t in ym]]
    j["date"] = j.index.date
    j["slot"] = j.index.hour * 4 + j.index.minute // 15
    # per-day error profile: +1 false positive, -1 false negative, 0 correct
    errp = j.assign(e=np.where(j["det"] & ~j["hp_on"], 1,
                               np.where(~j["det"] & j["hp_on"], -1, 0))
                    ).pivot_table(index="date", columns="slot", values="e",
                                  aggfunc="first").reindex(
        columns=range(96)).fillna(0).to_numpy()

    hp, oth, on, months, tmean = build_days()
    libs = pv_library()
    seasons = np.array([season_of(m) for m in months])
    rows = []
    for season in ORDER:
        idx = np.where(seasons == season)[0]
        lib = libs[season]
        if not len(idx) or len(lib) < 30:
            continue
        for _ in range(400):
            i = idx[RNG.integers(len(idx))]
            pv = lib[RNG.integers(len(lib))] * C_REF
            e_hp = hp[i].sum()
            if e_hp <= 0:
                continue
            nominal = np.minimum(hp[i], pv).sum() / e_hp
            ep = errp[RNG.integers(len(errp))]        # whole-day block
            typ = np.median(hp[i][hp[i] > 0]) if (hp[i] > 0).any() else 0.0
            pert = hp[i].copy()
            pert[ep < 0] = 0.0            # missed operation removed
            pert[(ep > 0) & (pert == 0)] = typ   # spurious operation added
            if pert.sum() <= 0:
                continue
            rows.append({"season": season, "nominal": nominal,
                         "perturbed": np.minimum(pert, pv).sum() / pert.sum()})
    pr = pd.DataFrame(rows)
    pr.to_csv(RES / "detector_episode_blocks.csv", index=False)
    return {s: {"nominal_med": float(g["nominal"].median()),
                "perturbed_med": float(g["perturbed"].median()),
                "perturbed_q10": float(g["perturbed"].quantile(0.1)),
                "perturbed_q90": float(g["perturbed"].quantile(0.9)),
                "shift_pp": float(100 * (g["perturbed"].median()
                                         - g["nominal"].median()))}
            for s, g in pr.groupby("season")}, pr


def main():
    out = {"pv_significance": pv_significance(),
           "uncertainty_interpretation": daily_vs_annual()}
    blocks, pr = detector_episode_blocks()
    out["detector_episode_blocks"] = blocks
    with open(RES / "stats_fixes.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2)[:2400])

    # ---------------- Fig 9: shifting sensitivity + detector uncertainty
    sens = pd.read_csv(RES / "overlap_shift_summary.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.85))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.83, bottom=0.17,
                        wspace=0.3)
    base = sens[(sens.cap_lo == 0.5) & (sens.cap_hi == 4.0)]
    xs = np.arange(3)
    w = 0.26
    for k, direction in enumerate(("earlier", "both", "later")):
        vals = [base[(base.divisor == d)
                     & (base.direction == direction)]["median"].iloc[0]
                for d in (8, 4, 2)]
        ax1.bar(xs + (k - 1) * w, vals, w, color=[GREY, BLUE, GREEN][k],
                alpha=0.85, label=direction.capitalize())
    ax1.set_xticks(xs)
    ax1.set_xticklabels([r"$\tau/8$", r"$\tau/4$", r"$\tau/2$"])
    ax1.set_xlabel("Assumed displacement bound")
    ax1.set_ylabel("Median gain (pp of PV-served fraction)")
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=3,
               frameon=False, handlelength=1.2, handletextpad=0.4,
               columnspacing=0.9, borderaxespad=0.0, fontsize=6.5)
    ax1.set_title("(a) Sensitivity to the shift rule", loc="left",
                  fontweight="bold", pad=22)

    xs2 = np.arange(len(ORDER))
    nom = [blocks[s]["nominal_med"] for s in ORDER]
    per = [blocks[s]["perturbed_med"] for s in ORDER]
    lo = [blocks[s]["nominal_med"] - blocks[s]["perturbed_q10"] for s in ORDER]
    hi = [blocks[s]["perturbed_q90"] - blocks[s]["nominal_med"] for s in ORDER]
    ax2.bar(xs2 - 0.19, nom, 0.38, color=BLUE, alpha=0.8, label="Nominal")
    ax2.bar(xs2 + 0.19, per, 0.38, color=ORANGE, alpha=0.85,
            label="Detector episodes resampled")
    ax2.errorbar(xs2 + 0.19, per,
                 yerr=[np.maximum(np.array(per) - np.array(
                     [blocks[s]["perturbed_q10"] for s in ORDER]), 0),
                     np.maximum(np.array(
                         [blocks[s]["perturbed_q90"] for s in ORDER])
                         - np.array(per), 0)],
                 fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels([s.capitalize() for s in ORDER])
    ax2.set_ylabel("PV-served fraction at 30 kWp (–)")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2,
               frameon=False, handlelength=1.2, handletextpad=0.4,
               columnspacing=0.9, borderaxespad=0.0, fontsize=6.5)
    ax2.set_title("(b) Detector-error sensitivity", loc="left",
                  fontweight="bold", pad=22)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=0.6)
    fig.savefig(FIG / "fig9_shifting.pdf")
    fig.savefig(FIG / "fig9_shifting.png")
    print("fig9_shifting saved")


if __name__ == "__main__":
    main()
