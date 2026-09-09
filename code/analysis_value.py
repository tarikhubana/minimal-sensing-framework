"""
Techno-economic value assessment (Section 4.5).

Combines the actual detected operating hours per season with the Monte
Carlo coverage distributions to obtain annual solar-covered operating
hours, as operated and with tau-limited shifting. Because no electrical
submetering exists, monetary value is parameterized by the heat pump
electrical demand P_el and the electricity price: savings = gained
covered hours x P_el x price.

Outputs: results/value_stats.json, results/value_seasonal.csv,
         figures/fig9_value.png/.pdf
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
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


def season_of(m):
    for s, ms in SEASONS.items():
        if m in ms:
            return s


daily = pd.read_csv(RES / "hp_daily.csv", index_col=0, parse_dates=True)
daily = daily[daily["on_frac"] > 0.01]
daily["season"] = [season_of(m) for m in daily.index.month]
daily["on_h"] = daily["on_frac"] * 24

mc = pd.read_csv(RES / "mc_coverage.csv")

rows = []
rng = np.random.default_rng(11)
n_draw = 2000
annual_base = np.zeros(n_draw)
annual_shift = np.zeros(n_draw)
for season in ["winter", "spring", "summer", "autumn"]:
    d = daily[daily["season"] == season]
    hours = d["on_h"].sum()
    g = mc[mc["season"] == season]
    idx = rng.integers(len(g), size=n_draw)  # paired sampling
    cb = g["coverage"].to_numpy()[idx]
    cs = g["coverage_shift"].to_numpy()[idx]
    base_h = hours * cb
    shift_h = hours * cs
    annual_base += base_h
    annual_shift += shift_h
    rows.append({
        "season": season,
        "active_days": int(len(d)),
        "on_hours": hours,
        "cov_base_med": float(np.median(cb)),
        "cov_shift_med": float(np.median(cs)),
        "base_h_med": float(np.median(base_h)),
        "base_h_q10": float(np.quantile(base_h, 0.1)),
        "base_h_q90": float(np.quantile(base_h, 0.9)),
        "shift_h_med": float(np.median(shift_h)),
        "shift_h_q10": float(np.quantile(shift_h, 0.1)),
        "shift_h_q90": float(np.quantile(shift_h, 0.9)),
    })
val = pd.DataFrame(rows)
val.to_csv(RES / "value_seasonal.csv", index=False)

gain = annual_shift - annual_base
stats = {
    "total_on_hours": float(daily["on_h"].sum()),
    "annual_base_med": float(np.median(annual_base)),
    "annual_base_q10q90": [float(np.quantile(annual_base, 0.1)),
                           float(np.quantile(annual_base, 0.9))],
    "annual_shift_med": float(np.median(annual_shift)),
    "annual_gain_med": float(np.median(gain)),
    "annual_gain_q10q90": [float(np.quantile(gain, 0.1)),
                           float(np.quantile(gain, 0.9))],
}
with open(RES / "value_stats.json", "w") as fh:
    json.dump(stats, fh, indent=2)
print(json.dumps(stats, indent=2))
print(val.round(1).to_string())

# ================================================================ Fig 8
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.9))
fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.17, wspace=0.3)

order = ["winter", "spring", "summer", "autumn"]
x = np.arange(len(order))
w = 0.36
v = val.set_index("season").loc[order]
ax1.bar(x - w / 2, v["base_h_med"], w, color=BLUE, alpha=0.75,
        label="As operated")
ax1.errorbar(x - w / 2, v["base_h_med"],
             yerr=[v["base_h_med"] - v["base_h_q10"],
                   v["base_h_q90"] - v["base_h_med"]],
             fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
ax1.bar(x + w / 2, v["shift_h_med"], w, color=GREEN, alpha=0.75,
        label="With τ-limited shifting")
ax1.errorbar(x + w / 2, v["shift_h_med"],
             yerr=[v["shift_h_med"] - v["shift_h_q10"],
                   v["shift_h_q90"] - v["shift_h_med"]],
             fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
ax1.set_xticks(x)
ax1.set_xticklabels([s.capitalize() for s in order])
ax1.set_ylabel("Operating hours per season (h)")
ax1.legend(frameon=False, loc="upper left")
ax1.set_title("(a) Solar-covered operating hours (median, 10–90 %)",
              loc="left", fontweight="bold")

# (b) decision-oriented screening output: CDF of the annual gain
gs = np.sort(gain)
cdf = np.arange(1, len(gs) + 1) / len(gs)
ax2.plot(gs, cdf, color=BLUE)
for q, lab, c in ((0.10, "10 % quantile", GREEN),
                  (0.50, "median", ORANGE),
                  (0.90, "90 % quantile", RED)):
    v_q = np.quantile(gs, q)
    ax2.axvline(v_q, color=c, linewidth=0.8, linestyle="--",
                label=f"{lab}: {v_q:.0f} h")
ax2.set_xlabel("Annual gain in solar-covered operation (h/yr)")
ax2.set_ylabel("Cumulative probability")
ax2.set_ylim(0, 1)
ax2.legend(frameon=False, loc="lower right")
ax2.set_title("(b) Screening output: distribution of annual gain",
              loc="left", fontweight="bold")
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
fig.savefig(FIG / "fig9_value.pdf")
fig.savefig(FIG / "fig9_value.png")
print("fig8 saved")
