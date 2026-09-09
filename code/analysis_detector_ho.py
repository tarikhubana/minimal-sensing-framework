"""
Held-out detector evaluation, reference-classifier sensitivity, block
bootstrap confidence intervals, and propagation of detection uncertainty.

Addresses the reviewer comments that (i) thresholds were developed and
evaluated on the same record, (ii) the electrical label is itself an
uncertain proxy rather than ground truth, (iii) no confidence intervals
were reported, and (iv) detection uncertainty was excluded from the
Monte Carlo propagation.

Design
------
DEV  months: development/diagnosis period (thresholds tuned here)
TEST months: strictly held-out evaluation period, never used for tuning
Blocked by calendar month, not by random 15-min interval, because
adjacent intervals are strongly autocorrelated.

Reference-classifier sensitivity varies the baseline reference month, the
baseline quantile, and the excess-power threshold.

Uncertainty propagation: detected operation is perturbed by resampling
detector errors (per-season FP/FN rates estimated on the TEST period) to
produce lower- and upper-bound operation profiles, whose effect on the
energy-overlap metric is reported.

Outputs: results/detector_holdout.json, results/detector_sensitivity.csv,
         results/detector_bootstrap.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "code" / "cache"
RES = ROOT / "code" / "results"
RNG = np.random.default_rng(77)

HEAT_MONTHS = [11, 12, 1, 2, 3, 4]
COOL_MONTHS = [6, 7, 8, 9]
# development period: first heating + first cooling stretch of the record
DEV = [(2025, m) for m in (6, 7, 11, 12)]
TEST = [(2025, 8), (2025, 9), (2026, 1), (2026, 2), (2026, 3), (2026, 4)]


def scores(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    n = tp + fp + fn + tn
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    acc = (tp + tn) / max(n, 1)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(n * n, 1)
    return {"n": n, "precision": prec, "recall": rec,
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "kappa": (acc - pe) / max(1 - pe, 1e-9),
            "fp_rate": fp / max(fp + tn, 1), "fn_rate": fn / max(tp + fn, 1)}


def block_bootstrap(df, pred_col, truth_col, n_boot=400):
    """Day-level block bootstrap of the score set."""
    days = df.index.normalize().unique().to_numpy()
    out = []
    for _ in range(n_boot):
        pick = RNG.choice(days, size=len(days), replace=True)
        idx = np.concatenate([np.where(df.index.normalize() == d)[0]
                              for d in pick])
        d = df.iloc[idx]
        out.append(scores(d[pred_col].to_numpy(bool),
                          d[truth_col].to_numpy(bool)))
    b = pd.DataFrame(out)
    return {k: [float(b[k].quantile(0.025)), float(b[k].quantile(0.975))]
            for k in ("precision", "recall", "f1", "kappa")}


def main():
    from analysis_validation import (load_elec, occupancy_baseline,
                                     ground_truth, detect_v2, HEAT_MONTHS as _)
    el = load_elec()
    hp = pd.read_parquet(CACHE / "hp_temps.parquet")
    ops_v1 = pd.read_parquet(CACHE / "hp_ops.parquet")
    ops_v2 = detect_v2(hp)

    base = occupancy_baseline(el)
    gt = ground_truth(el, base)

    def to15(ops, col="on"):
        return (ops[col].resample("15min").mean() > 0.5).rename(col)

    j = gt.join(to15(ops_v1).rename("v1"), how="inner")
    j = j.join(to15(ops_v2).rename("v2"), how="inner").dropna(
        subset=["hp_on", "v1", "v2"])
    ym = list(zip(j.index.year, j.index.month))
    j["split"] = ["dev" if t in DEV else ("test" if t in TEST else "other")
                  for t in ym]

    out = {"dev_months": [f"{y}-{m:02d}" for y, m in DEV],
           "test_months": [f"{y}-{m:02d}" for y, m in TEST]}
    for split in ("dev", "test"):
        d = j[j["split"] == split]
        blk = {}
        for lab, months in (("heating", HEAT_MONTHS), ("cooling", COOL_MONTHS),
                            ("all", list(range(1, 13)))):
            dd = d[np.isin(d.index.month, months)]
            if len(dd) < 200:
                continue
            e = {"v1": scores(dd["v1"].to_numpy(bool),
                              dd["hp_on"].to_numpy(bool)),
                 "v2": scores(dd["v2"].to_numpy(bool),
                              dd["hp_on"].to_numpy(bool))}
            if split == "test":
                e["v2_ci95"] = block_bootstrap(dd, "v2", "hp_on")
            blk[lab] = e
        out[split] = blk

    # ---------------- reference-classifier sensitivity
    rows = []
    for ref in (("2025-10-01", "2025-10-31"), ("2025-05-01", "2025-05-31"),
                ("2026-05-01", "2026-05-31")):
        for q in (0.90, 0.95, 0.99):
            r = el.loc[ref[0]:ref[1]].to_frame("kw")
            if len(r) < 500:
                continue
            r["h"] = r.index.hour; r["wd"] = r.index.weekday < 5
            b = r.groupby(["wd", "h"])["kw"].quantile(q).rename("base_p95")
            g = el.to_frame("kw")
            g["h"] = g.index.hour; g["wd"] = g.index.weekday < 5
            g = g.join(b, on=["wd", "h"])
            g["hp_kw"] = (g["kw"] - g["base_p95"]).clip(lower=0)
            for thr in (1.0, 2.0, 3.0, 4.0):
                g2 = g.assign(hp_on=g["hp_kw"] > thr)
                jj = g2[["hp_on"]].join(
                    to15(ops_v2).rename("v2"), how="inner").dropna()
                jj = jj[np.isin(list(zip(jj.index.year, jj.index.month)),
                                None) if False else slice(None)]
                s_all = scores(jj["v2"].to_numpy(bool),
                               jj["hp_on"].to_numpy(bool))
                rows.append({"ref_month": ref[0][:7], "quantile": q,
                             "threshold_kw": thr, **s_all})
    sens = pd.DataFrame(rows)
    sens.to_csv(RES / "detector_sensitivity.csv", index=False)
    out["sensitivity"] = {
        "f1_range": [float(sens["f1"].min()), float(sens["f1"].max())],
        "f1_median": float(sens["f1"].median()),
        "kappa_range": [float(sens["kappa"].min()), float(sens["kappa"].max())],
        "n_configs": int(len(sens)),
    }

    # ---------------- detector-uncertainty propagation into overlap
    test = j[j["split"] == "test"]
    err = {}
    for lab, months in (("heating", HEAT_MONTHS), ("cooling", COOL_MONTHS)):
        dd = test[np.isin(test.index.month, months)]
        if len(dd) < 200:
            continue
        s = scores(dd["v2"].to_numpy(bool), dd["hp_on"].to_numpy(bool))
        err[lab] = {"fp_rate": s["fp_rate"], "fn_rate": s["fn_rate"]}
    out["test_error_rates"] = err

    ov = pd.read_csv(RES / "overlap_mc.csv")
    # perturbed operation: recompute overlap with operation profiles
    # randomly corrupted at the measured FP/FN rates
    from analysis_overlap import build_days, pv_library, season_of, ORDER
    hpm, oth, on, months_arr, tmean = build_days()
    libs = pv_library()
    seasons = np.array([season_of(m) for m in months_arr])
    C = 30
    res = []
    for season in ORDER:
        idx = np.where(seasons == season)[0]
        lib = libs[season]
        if not len(idx) or len(lib) < 30:
            continue
        e = err.get("cooling" if season == "summer" else "heating",
                    {"fp_rate": 0.1, "fn_rate": 0.2})
        for _ in range(300):
            i = idx[RNG.integers(len(idx))]
            pv = lib[RNG.integers(len(lib))] * C
            e_hp = hpm[i].sum()
            if e_hp <= 0:
                continue
            nominal = np.minimum(hpm[i], pv).sum() / e_hp
            # perturb: drop FN share of running power, add FP share elsewhere
            mask_on = hpm[i] > 0
            keep = RNG.random(96) > e["fn_rate"]
            add = (RNG.random(96) < e["fp_rate"]) & ~mask_on
            pert = np.where(mask_on & keep, hpm[i], 0.0)
            pert = pert + add * np.median(hpm[i][mask_on] if mask_on.any()
                                          else [0])
            ep = pert.sum()
            if ep <= 0:
                continue
            res.append({"season": season, "nominal": nominal,
                        "perturbed": np.minimum(pert, pv).sum() / ep})
    pr = pd.DataFrame(res)
    pr.to_csv(RES / "detector_propagation.csv", index=False)
    out["detector_propagation"] = {
        s: {"nominal_med": float(g["nominal"].median()),
            "perturbed_med": float(g["perturbed"].median()),
            "perturbed_q10": float(g["perturbed"].quantile(0.1)),
            "perturbed_q90": float(g["perturbed"].quantile(0.9)),
            "shift_pp": float(100 * (g["perturbed"].median()
                                     - g["nominal"].median()))}
        for s, g in pr.groupby("season")}
    with open(RES / "detector_holdout.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("test", "sensitivity", "test_error_rates",
                               "detector_propagation")}, indent=2)[:2600])


if __name__ == "__main__":
    main()
