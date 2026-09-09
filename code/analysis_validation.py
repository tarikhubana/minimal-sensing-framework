"""
Validation of temperature-only detection against whole-building electrical
measurements (Section 4.2 of the paper).

The electrical meter is a WHOLE-BUILDING meter, so heat pump operation must
be separated from the occupancy (non-HP) load. October 2025 is used as the
reference month: the heat pump is inactive throughout, so the measured power
in that month defines the occupancy baseline as a function of hour-of-day and
day-type. An interval is labelled "heat pump running" (electrical ground
truth) when measured power exceeds the 95th percentile of that baseline by
more than MARGIN kW.

Two detectors are compared against this ground truth:
  v1 (original)  heating: dT >= +1.0 K ; cooling: dT <= -1.8 K
  v2 (revised)   heating: dT >  +0.5 K and T_sup > 30 degC
                 cooling: T_sup < 14 degC          (chilled-water signature)

Outputs: cache/hp_ops_v2.parquet, results/validation.json,
         results/validation_table.csv, figures/fig5_validation.png/.pdf
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
RES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "savefig.dpi": 600, "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
BLUE, ORANGE, GREEN, RED, GREY = ("#0072B2", "#E69F00", "#009E73",
                                  "#D55E00", "#7F7F7F")

ELEC_FILE = ROOT / "Electricity_measurement.xlsx"
REF_MONTH = ("2025-10-01", "2025-10-31")   # heat pump inactive
MARGIN = 2.0                               # kW above baseline p95
HEAT_MONTHS = [11, 12, 1, 2, 3, 4]
COOL_MONTHS = [6, 7, 8, 9]


def load_elec() -> pd.Series:
    cached = CACHE / "elec.parquet"
    if cached.exists():
        return pd.read_parquet(cached)["kw"]
    df = pd.read_excel(ELEC_FILE, engine="calamine", header=0).iloc[:, [1, 4]]
    df.columns = ["ts", "kw"]
    df["ts"] = pd.to_datetime(df["ts"], format="%d/%m/%Y %H:%M:%S",
                              errors="coerce")
    df["kw"] = pd.to_numeric(df["kw"], errors="coerce")
    s = df.dropna().drop_duplicates("ts").set_index("ts").sort_index()["kw"]
    s.to_frame().to_parquet(cached)
    return s


def occupancy_baseline(el: pd.Series) -> pd.DataFrame:
    ref = el.loc[REF_MONTH[0]:REF_MONTH[1]].to_frame("kw")
    ref["h"] = ref.index.hour
    ref["wd"] = ref.index.weekday < 5
    return ref.groupby(["wd", "h"])["kw"].quantile(0.95).rename("base_p95")


def ground_truth(el: pd.Series, base: pd.DataFrame) -> pd.DataFrame:
    d = el.to_frame("kw")
    d["h"] = d.index.hour
    d["wd"] = d.index.weekday < 5
    d = d.join(base, on=["wd", "h"])
    d["hp_kw"] = (d["kw"] - d["base_p95"]).clip(lower=0)
    d["hp_on"] = d["hp_kw"] > MARGIN
    return d


def detect_v2(hp: pd.DataFrame) -> pd.DataFrame:
    """Revised, mode-specific detection (validated in this script)."""
    dT = (hp["T_sup"] - hp["T_ret"]).rolling(5, center=True).median()
    month = hp.index.month
    heat_season = np.isin(month, HEAT_MONTHS)
    cool_season = np.isin(month, COOL_MONTHS)
    heat_on = heat_season & (dT > 0.5) & (hp["T_sup"] > 30)
    cool_on = cool_season & (hp["T_sup"] < 14)
    ops = pd.DataFrame(index=hp.index)
    ops["dT"] = dT
    ops["mode"] = "off"
    ops.loc[heat_on, "mode"] = "heat"
    ops.loc[cool_on & ~heat_on, "mode"] = "cool"
    ops["on"] = ops["mode"] != "off"
    return ops


def scores(pred: pd.Series, truth: pd.Series) -> dict:
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    # Cohen's kappa
    n = tp + fp + fn + tn
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(n * n, 1)
    kappa = (acc - pe) / max(1 - pe, 1e-9)
    return {"n": n, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": prec, "recall": rec,
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "accuracy": acc, "kappa": kappa}


def main():
    el = load_elec()
    hp = pd.read_parquet(CACHE / "hp_temps.parquet")
    ops_v1 = pd.read_parquet(CACHE / "hp_ops.parquet")
    ops_v2 = detect_v2(hp)
    ops_v2.to_parquet(CACHE / "hp_ops_v2.parquet")

    base = occupancy_baseline(el)
    gt = ground_truth(el, base)

    def to15(ops):
        return pd.DataFrame({
            "on": ops["on"].resample("15min").mean() > 0.5,
            "mode": ops["mode"].resample("15min").agg(
                lambda s: s.mode()[0] if len(s) else "off"),
        })

    j = gt.join(to15(ops_v1).add_suffix("_v1"), how="inner")
    j = j.join(to15(ops_v2).add_suffix("_v2"), how="inner").dropna()

    rows, out = [], {}
    for season, months in (("Heating (Nov-Apr)", HEAT_MONTHS),
                           ("Cooling (Jun-Sep)", COOL_MONTHS),
                           ("Full year", list(range(1, 13)))):
        d = j[np.isin(j.index.month, months)]
        s1 = scores(d["on_v1"], d["hp_on"])
        s2 = scores(d["on_v2"], d["hp_on"])
        out[season] = {"v1": s1, "v2": s2}
        rows.append({
            "season": season, "n": s1["n"],
            "v1_precision": s1["precision"], "v1_recall": s1["recall"],
            "v1_f1": s1["f1"], "v1_kappa": s1["kappa"],
            "v2_precision": s2["precision"], "v2_recall": s2["recall"],
            "v2_f1": s2["f1"], "v2_kappa": s2["kappa"],
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "validation_table.csv", index=False)

    # power signature by detected mode (revised detector)
    sig = j.groupby("mode_v2")["hp_kw"].agg(["count", "median", "mean"])
    # energy attribution
    total_kwh = float(j["kw"].sum() / 4)
    hp_kwh = float(j["hp_kw"].sum() / 4)
    out["energy"] = {"total_kwh": total_kwh, "hp_kwh": hp_kwh,
                     "hp_share": hp_kwh / total_kwh}
    out["baseline_night_kw"] = float(base.min())
    out["baseline_peak_kw"] = float(base.max())
    out["power_signature"] = sig.to_dict()
    # mean HP power when running (for energy-weighted coverage)
    run = j[j["hp_on"]]
    out["mean_hp_kw_running"] = {
        "heating": float(run[np.isin(run.index.month, HEAT_MONTHS)]["hp_kw"].mean()),
        "cooling": float(run[np.isin(run.index.month, COOL_MONTHS)]["hp_kw"].mean()),
    }
    # hourly profile of HP power by season (for the corrected narrative)
    prof = {}
    for lab, months in (("heating", HEAT_MONTHS), ("cooling", COOL_MONTHS)):
        d = j[np.isin(j.index.month, months)]
        prof[lab] = d.groupby(d.index.hour)["hp_kw"].mean().round(3).to_dict()
    out["hourly_hp_kw"] = prof
    with open(RES / "validation.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(tab.round(3).to_string())
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("energy", "mean_hp_kw_running",
                               "baseline_night_kw", "baseline_peak_kw")},
                     indent=2))

    # ---------------------------------------------------------- figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.95))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.83, bottom=0.16,
                        wspace=0.28)

    hrs = np.arange(24)
    for lab, c, months in (("Heating season", RED, HEAT_MONTHS),
                           ("Cooling season", BLUE, COOL_MONTHS)):
        d = j[np.isin(j.index.month, months)]
        ax1.plot(hrs, d.groupby(d.index.hour)["hp_kw"].mean().reindex(hrs),
                 color=c, label=lab)
    ax1.set_xlim(0, 23)
    ax1.set_xticks(range(0, 24, 6))
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Inferred heat pump power (kW)")
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2,
               frameon=False, columnspacing=0.9, handlelength=1.2,
               handletextpad=0.4, borderaxespad=0.0, fontsize=6.5)
    ax1.set_title("(a) Inferred heat pump load profile", loc="left",
                  fontweight="bold", pad=16)

    labels = ["Heating\n(Nov–Apr)", "Cooling\n(Jun–Sep)", "Full year"]
    x = np.arange(3)
    w = 0.36
    ax2.bar(x - w / 2, tab["v1_f1"], w, color=GREY, alpha=0.8,
            label="Differential-only detector")
    ax2.bar(x + w / 2, tab["v2_f1"], w, color=GREEN, alpha=0.85,
            label="Mode-specific detector")
    for xi, (a, b) in enumerate(zip(tab["v1_f1"], tab["v2_f1"])):
        ax2.text(xi - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=6)
        ax2.text(xi + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("F1 score vs. electrical ground truth")
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2,
               frameon=False, columnspacing=0.9, handlelength=1.2,
               handletextpad=0.4, borderaxespad=0.0, fontsize=6.5)
    ax2.set_title("(b) Detection accuracy", loc="left", fontweight="bold",
                  pad=16)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=0.6)
    fig.savefig(FIG / "fig5_validation.pdf")
    fig.savefig(FIG / "fig5_validation.png")
    print("fig5_validation saved")


if __name__ == "__main__":
    main()
