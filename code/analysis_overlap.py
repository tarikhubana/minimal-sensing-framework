"""
Physical PV–heat-pump energy overlap, and sensitivity of the load-shifting
rule (addresses reviewer comments on the coverage metric and on the ad hoc
shift bound).

Metric
------
For a rooftop plant of capacity C (kWp) the self-consumed fraction of heat
pump demand is

    SCF(C) = sum_t min(P_HP,t , C * pv_t) / sum_t P_HP,t

where pv_t is the normalized fleet scenario (p.u. of plant peak) and
P_HP,t the measured heat pump power. This is a genuine energy quantity and
is reported for several capacities. The previous normalized-threshold
metric is retained only as a temporal alignment index (TAI) for comparison.

Optionally the building's non-heat-pump load can be served first
(PV_avail = max(0, C*pv - P_other)), which is the realistic case for a
whole building; both priorities are reported.

Shift sensitivity
-----------------
The daily operation profile is rigidly displaced by k slots, k restricted
to |k| <= K where K = clip(tau / d, lo, hi) with divisor d in {2,4,8} and
caps (lo,hi) varied. Direction is restricted to earlier-only, later-only,
or bidirectional. Zero shift is always admissible, so the reported gain is
non-negative by construction; the sensitivity therefore reports the
distribution of gains, not their sign.

Outputs: results/overlap_*.csv, results/overlap_stats.json,
         figures/fig9_overlap.png/.pdf
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
CAPS_KWP = [5, 10, 20, 30, 50, 75, 100]
N_MC = 400
RNG = np.random.default_rng(31)


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


def build_days():
    """Daily matrices of HP power, other-load power, and detected on-state."""
    from analysis_validation import (load_elec, occupancy_baseline,
                                     ground_truth)
    el = load_elec()
    gt = ground_truth(el, occupancy_baseline(el))
    ops = pd.read_parquet(CACHE / "hp_ops_v2.parquet")
    on15 = (ops["on"].resample("15min").mean() > 0.5).rename("on")
    tout = pd.read_parquet(CACHE / "outdoor_temp.parquet")["T_out"]

    d = pd.DataFrame({"hp_kw": gt["hp_kw"],
                      "other_kw": gt["kw"] - gt["hp_kw"]}).join(
        on15, how="inner").join(tout.resample("15min").mean().rename("tout"),
                                how="left").dropna(subset=["hp_kw", "on"])
    d["date"] = d.index.date
    d["slot"] = d.index.hour * 4 + d.index.minute // 15
    hp = d.pivot_table(index="date", columns="slot", values="hp_kw",
                       aggfunc="first").reindex(columns=range(96)).fillna(0.0)
    oth = d.pivot_table(index="date", columns="slot", values="other_kw",
                        aggfunc="first").reindex(columns=range(96)).fillna(0.0)
    on = d.pivot_table(index="date", columns="slot", values="on",
                       aggfunc="first").reindex(columns=range(96)).fillna(False)
    tmean = d.groupby("date")["tout"].mean()
    keep = hp.sum(axis=1) > 0
    return (hp[keep].to_numpy(float), oth[keep].to_numpy(float),
            on[keep].to_numpy(bool),
            pd.to_datetime(hp[keep].index).month.to_numpy(),
            tmean.reindex(hp[keep].index).to_numpy())


def pv_library():
    meta = pd.read_parquet(CACHE / "pv_meta.parquet")
    prof = pd.read_parquet(CACHE / "pv_daily_profiles.parquet")
    prof.columns = [c if c == "season" else int(c) for c in prof.columns]
    slots = [c for c in prof.columns if c != "season"]
    dist = haversine_km(SITE[0], SITE[1], meta.lat, meta.lon)
    local = meta.loc[dist < 80, "plant"].tolist()
    lib = prof[prof.index.get_level_values("plant").isin(local)]
    return {s: lib[lib["season"] == s][slots].to_numpy(float) for s in ORDER}


def main():
    hp, oth, on, months, tmean = build_days()
    libs = pv_library()
    seasons = np.array([season_of(m) for m in months])

    # ---------------------------------------------------- 1 energy overlap
    rows = []
    for season in ORDER:
        idx = np.where(seasons == season)[0]
        lib = libs[season]
        if len(idx) == 0 or len(lib) < 30:
            continue
        for _ in range(N_MC):
            i = idx[RNG.integers(len(idx))]
            pv = lib[RNG.integers(len(lib))]
            e_hp = hp[i].sum()
            if e_hp <= 0:
                continue
            rec = {"season": season}
            for C in CAPS_KWP:
                pv_kw = C * pv
                # HP-priority: PV serves the heat pump first
                scf_hp = np.minimum(hp[i], pv_kw).sum() / e_hp
                # building-priority: other loads served first
                avail = np.clip(pv_kw - oth[i], 0, None)
                scf_bp = np.minimum(hp[i], avail).sum() / e_hp
                rec[f"scf_hp_{C}"] = scf_hp
                rec[f"scf_bp_{C}"] = scf_bp
            # temporal alignment index (old metric, for comparison)
            rec["tai"] = ((on[i] & (pv > 0.20)).sum()
                          / max(on[i].sum(), 1))
            rows.append(rec)
    ov = pd.DataFrame(rows)
    ov.to_csv(RES / "overlap_mc.csv", index=False)

    summ = {}
    for C in CAPS_KWP:
        summ[C] = {
            "hp_priority": {
                "median": float(ov[f"scf_hp_{C}"].median()),
                "q10": float(ov[f"scf_hp_{C}"].quantile(0.1)),
                "q90": float(ov[f"scf_hp_{C}"].quantile(0.9)),
            },
            "building_priority": {
                "median": float(ov[f"scf_bp_{C}"].median()),
                "q10": float(ov[f"scf_bp_{C}"].quantile(0.1)),
                "q90": float(ov[f"scf_bp_{C}"].quantile(0.9)),
            },
        }
    per_season = {}
    for season, g in ov.groupby("season"):
        per_season[season] = {str(C): {
            "hp_priority_med": float(g[f"scf_hp_{C}"].median()),
            "building_priority_med": float(g[f"scf_bp_{C}"].median()),
        } for C in CAPS_KWP}
        per_season[season]["tai_med"] = float(g["tai"].median())

    # ------------------------------------------------- 2 shift sensitivity
    fits = pd.read_csv(RES / "decay_fits.csv")
    taus = fits["tau_h"].to_numpy()
    variants = []
    for div in (2, 4, 8):
        for lo, hi in ((0.5, 4.0), (0.0, 2.0), (0.0, 6.0)):
            for direction in ("both", "later", "earlier"):
                variants.append((div, lo, hi, direction))

    C_REF = 30  # kWp reference capacity for the shift study
    srows = []
    for season in ORDER:
        idx = np.where(seasons == season)[0]
        lib = libs[season]
        if len(idx) == 0 or len(lib) < 30:
            continue
        for _ in range(N_MC):
            i = idx[RNG.integers(len(idx))]
            pv = lib[RNG.integers(len(lib))] * C_REF
            tau = taus[RNG.integers(len(taus))]
            e_hp = hp[i].sum()
            if e_hp <= 0:
                continue
            base = np.minimum(hp[i], pv).sum() / e_hp
            for div, lo, hi, direction in variants:
                K = int(round(np.clip(tau / div, lo, hi) * 4))
                if direction == "later":
                    ks = range(0, K + 1)
                elif direction == "earlier":
                    ks = range(-K, 1)
                else:
                    ks = range(-K, K + 1)
                best = base
                for k in ks:
                    sh = np.roll(hp[i], k)
                    best = max(best, np.minimum(sh, pv).sum() / e_hp)
                srows.append({"season": season, "divisor": div,
                              "cap_lo": lo, "cap_hi": hi,
                              "direction": direction,
                              "base": base, "shifted": best,
                              "gain_pp": 100 * (best - base)})
    sh = pd.DataFrame(srows)
    sh.to_csv(RES / "overlap_shift_sensitivity.csv", index=False)
    sens = sh.groupby(["divisor", "cap_lo", "cap_hi", "direction"])[
        "gain_pp"].agg(["median", lambda s: s.quantile(0.1),
                        lambda s: s.quantile(0.9)])
    sens.columns = ["median", "q10", "q90"]
    sens = sens.round(2).reset_index()
    sens.to_csv(RES / "overlap_shift_summary.csv", index=False)

    # --------------------------------------- 3 conditional vs independent
    # stratify PV scenarios by season AND daily mean outdoor temperature
    cond, indep = [], []
    tq = {s: np.nanquantile(tmean[seasons == s], [0.33, 0.67])
          for s in ORDER}
    for season in ORDER:
        idx = np.where(seasons == season)[0]
        lib = libs[season]
        if len(idx) == 0 or len(lib) < 30:
            continue
        # proxy for "warm day" PV: brighter-than-median library days
        e_lib = lib.sum(axis=1)
        hi_lib = lib[e_lib >= np.median(e_lib)]
        lo_lib = lib[e_lib < np.median(e_lib)]
        for _ in range(N_MC):
            i = idx[RNG.integers(len(idx))]
            e_hp = hp[i].sum()
            if e_hp <= 0:
                continue
            t = tmean[i]
            warm = t >= tq[season][1]
            cold = t <= tq[season][0]
            pv_i = lib[RNG.integers(len(lib))]
            indep.append({"season": season,
                          "scf": np.minimum(hp[i], C_REF * pv_i).sum() / e_hp})
            src = hi_lib if warm else (lo_lib if cold else lib)
            pv_c = src[RNG.integers(len(src))]
            cond.append({"season": season,
                         "scf": np.minimum(hp[i], C_REF * pv_c).sum() / e_hp})
    ci = pd.DataFrame(cond).assign(sampling="conditional")
    ii = pd.DataFrame(indep).assign(sampling="independent")
    cmp = pd.concat([ci, ii], ignore_index=True)
    cmp.to_csv(RES / "overlap_sampling.csv", index=False)
    samp = cmp.groupby(["season", "sampling"])["scf"].median().unstack()
    samp["diff_pp"] = 100 * (samp["conditional"] - samp["independent"])

    stats = {"capacity_summary": summ, "per_season": per_season,
             "shift_reference_kwp": C_REF,
             "shift_baseline_variant": "divisor 4, caps 0.5-4 h, both",
             "sampling_comparison": samp.round(4).to_dict(orient="index"),
             "n_days": int(len(hp))}
    with open(RES / "overlap_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(summ, indent=2)[:900])
    print(sens.head(12).to_string())
    print(samp.round(3).to_string())

    # ---------------------------------------------------------- figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.25))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.745, bottom=0.15,
                        wspace=0.3)
    med_hp = [summ[C]["hp_priority"]["median"] for C in CAPS_KWP]
    q10 = [summ[C]["hp_priority"]["q10"] for C in CAPS_KWP]
    q90 = [summ[C]["hp_priority"]["q90"] for C in CAPS_KWP]
    med_bp = [summ[C]["building_priority"]["median"] for C in CAPS_KWP]
    ax1.plot(CAPS_KWP, med_hp, color=BLUE, marker="o", markersize=3,
             label="Heat pump served first")
    ax1.plot(CAPS_KWP, med_bp, color=ORANGE, marker="s", markersize=3,
             label="Other building loads first")
    ax1.fill_between(CAPS_KWP, q10, q90, color=BLUE, alpha=0.16, linewidth=0,
                     label="Daily 10–90 % range (heat pump first)")
    ax1.set_xlabel("Assumed rooftop PV capacity (kWp)")
    ax1.set_ylabel("Potential PV-served fraction (–)")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=1,
               frameon=False, handlelength=1.3, handletextpad=0.4,
               borderaxespad=0.0, fontsize=6.2, labelspacing=0.28)
    ax1.set_title("(a) Daily distribution vs. plant size", loc="left",
                  fontweight="bold", pad=40)

    # (b) seasonal breakdown at the reference capacity
    xs = np.arange(len(ORDER))
    vals = [per_season[s]["30"]["hp_priority_med"] for s in ORDER]
    vals_b = [per_season[s]["30"]["building_priority_med"] for s in ORDER]
    ax2.bar(xs - 0.19, vals, 0.38, color=BLUE, alpha=0.8,
            label="Heat pump served first")
    ax2.bar(xs + 0.19, vals_b, 0.38, color=ORANGE, alpha=0.85,
            label="Other building loads first")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([s.capitalize() for s in ORDER])
    ax2.set_ylabel("Potential PV-served fraction (–)")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=1,
               frameon=False, handlelength=1.3, handletextpad=0.4,
               borderaxespad=0.0, fontsize=6.2, labelspacing=0.28)
    ax2.set_title("(b) Seasonal medians at 30 kWp", loc="left",
                  fontweight="bold", pad=40)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=0.6)
    fig.savefig(FIG / "fig8_capacity.pdf")
    fig.savefig(FIG / "fig8_capacity.png")
    print("fig8_capacity saved")


if __name__ == "__main__":
    main()
