"""Publication figures for the revised paper (ASEJ style, serif, 600 dpi)."""

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "code" / "cache"
RES = ROOT / "code" / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "lines.linewidth": 0.9,
    "savefig.dpi": 600, "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
BLUE, ORANGE, GREEN, RED, GREY = ("#0072B2", "#E69F00", "#009E73",
                                  "#D55E00", "#7F7F7F")
SKY, PINK = "#56B4E9", "#CC79A7"

hp = pd.read_parquet(CACHE / "hp_temps.parquet")
tout = pd.read_parquet(CACHE / "outdoor_temp.parquet")
ops = pd.read_parquet(CACHE / "hp_ops_v2.parquet")
pu = pd.read_parquet(CACHE / "pv_pu.parquet")
meta = pd.read_parquet(CACHE / "pv_meta.parquet")
fits = pd.read_csv(RES / "decay_fits.csv")
pairs = pd.read_csv(RES / "corr_distance.csv")
mc = pd.read_csv(RES / "mc_coverage.csv")
SITE = (43.337, 17.815)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


# ================================================================ Fig 1
fig = plt.figure(figsize=(7.1, 5.6))
gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.25, left=0.07,
                      right=0.99, top=0.91, bottom=0.07)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, :])

td = tout["T_out"].resample("D").agg(["mean", "min", "max"]).dropna()
ax_a.fill_between(td.index, td["min"], td["max"], color=BLUE, alpha=0.25,
                  linewidth=0, label="Daily range")
ax_a.plot(td.index, td["mean"], color=BLUE, label="Daily mean")
ax_a.axhline(0, color=GREY, linewidth=0.5, linestyle=":")
ax_a.set_ylabel("Outdoor temperature (°C)")
ax_a.legend(frameon=False, loc="upper right")
ax_a.set_title("(a) Outdoor temperature", loc="left", fontweight="bold")

hd = hp.resample("D").mean()
ax_b.plot(hd.index, hd["T_sup"], color=RED, label="Supply $T_{sup}$")
ax_b.plot(hd.index, hd["T_ret"], color=SKY, label="Return $T_{ret}$")
ax_b.plot(hd.index, hd["T_gw"], color=GREEN, label="Groundwater $T_{gw}$")
ax_b.set_ylabel("Temperature (°C)")
ax_b.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=3,
            frameon=False, columnspacing=0.9, handlelength=1.2,
            handletextpad=0.4, borderaxespad=0.0, fontsize=6.5)
ax_b.set_title("(b) Hydronic and groundwater temperatures (daily mean)",
               loc="left", fontweight="bold", pad=16)
for ax in (ax_a, ax_b):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlim(td.index.min(), td.index.max())

win = hp.loc["2026-01-12":"2026-01-15"]
owin = ops.loc["2026-01-12":"2026-01-15"]
ax_c.plot(win.index, win["T_sup"], color=RED, label="Supply $T_{sup}$",
          linewidth=0.7)
ax_c.plot(win.index, win["T_ret"], color=SKY, label="Return $T_{ret}$",
          linewidth=0.7)
ax_c.plot(win.index, win["T_gw"], color=GREEN, label="Groundwater $T_{gw}$",
          linewidth=0.7)
onmask = (owin["mode"] == "heat").to_numpy()
ax_c.fill_between(win.index, 8, 46, where=onmask, color=ORANGE, alpha=0.15,
                  linewidth=0, label="Detected operation")
ax_c.set_ylim(8, 46)
ax_c.set_ylabel("Temperature (°C)")
ax_c.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=4,
            frameon=False, columnspacing=0.9, handlelength=1.2,
            handletextpad=0.4, borderaxespad=0.0, fontsize=6.5)
ax_c.xaxis.set_major_locator(mdates.DayLocator())
ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
ax_c.set_xlim(win.index.min(), win.index.max())
ax_c.set_title("(c) 1-min resolution close-up, winter, with detected heat "
               "pump operation", loc="left", fontweight="bold", pad=16)
for ax in (ax_a, ax_b, ax_c):
    style(ax)
fig.savefig(FIG / "fig1_hp_data.pdf")
fig.savefig(FIG / "fig1_hp_data.png")
plt.close(fig)

# ================================================================ Fig 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.05))
fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.15, wspace=0.28)
ax1.scatter(meta["lon"], meta["lat"], s=meta["rated_kw"] / 4 + 8,
            c=BLUE, alpha=0.65, edgecolors="white", linewidths=0.4,
            label="PV plant (size ∝ rated power)")
ax1.scatter(SITE[1], SITE[0], marker="o", s=45, c=RED, zorder=5,
            edgecolors="white", linewidths=0.5,
            label="Case-study building (Mostar)")
ax1.set_xlabel("Longitude (°E)")
ax1.set_ylabel("Latitude (°N)")
ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2,
           frameon=False, columnspacing=0.9, handlelength=1.2,
           handletextpad=0.4, borderaxespad=0.0, fontsize=6.5)
ax1.set_title("(a) 30-plant fleet, Bosnia and Herzegovina", loc="left",
              fontweight="bold", pad=16)

pup = pu.copy()
tod = pup.index.hour + pup.index.minute / 60
month = pup.index.month
for lab, months, c in (("Summer (JJA)", [6, 7, 8], ORANGE),
                       ("Winter (DJF)", [12, 1, 2], BLUE)):
    sel = pup[np.isin(month, months)]
    stack = sel.stack().rename("v").reset_index()
    ts = pd.to_datetime(stack["timestamp"])
    stack["t"] = ts.dt.hour + ts.dt.minute / 60
    g = stack.groupby("t")["v"]
    med, q10, q90 = g.median(), g.quantile(0.1), g.quantile(0.9)
    ax2.fill_between(med.index, q10, q90, color=c, alpha=0.2, linewidth=0)
    ax2.plot(med.index, med, color=c, label=f"{lab}, median")
ax2.set_xlim(0, 24)
ax2.set_xticks(range(0, 25, 6))
ax2.set_xlabel("Hour of day")
ax2.set_ylabel("Normalized power (p.u.)")
ax2.legend(frameon=False, loc="upper left")
ax2.set_title("(b) Fleet profiles (median, 10–90 % band)", loc="left",
              fontweight="bold")
for ax in (ax1, ax2):
    style(ax)
fig.savefig(FIG / "fig2_fleet.pdf")
fig.savefig(FIG / "fig2_fleet.png")
plt.close(fig)

# ================================================================ Fig 4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.7))
fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18, wspace=0.3)
sgn = ops["mode"].map({"heat": 1.0, "cool": -1.0, "off": 0.0})
grid = sgn.groupby([sgn.index.month, sgn.index.hour]).mean().unstack()
im = ax1.imshow(grid.reindex(range(1, 13)), aspect="auto", cmap="RdBu_r",
                vmin=-1, vmax=1, origin="upper",
                extent=[0, 24, 12.5, 0.5])
ax1.set_yticks(range(1, 13))
ax1.set_yticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                     "Aug", "Sep", "Oct", "Nov", "Dec"])
ax1.set_xlabel("Hour of day")
cb = fig.colorbar(im, ax=ax1, fraction=0.04, pad=0.02)
cb.set_label("← cooling      heating →", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
ax1.set_title("(a) Mean operating state (month × hour)", loc="left",
              fontweight="bold")

daily = pd.read_csv(RES / "hp_daily.csv", index_col=0, parse_dates=True)
act = daily[daily["on_frac"] > 0.01]
ax2.hist(act["starts"], bins=np.arange(0, 40, 2), color=BLUE, alpha=0.8,
         edgecolor="white")
ax2.axvline(act["starts"].median(), color=RED, linewidth=1,
            label=f"Median = {act['starts'].median():.0f}")
ax2.set_xlabel("Compressor starts per active day")
ax2.set_ylabel("Days")
ax2.legend(frameon=False)
style(ax2)
ax2.set_title("(b) Daily cycling", loc="left", fontweight="bold")
fig.savefig(FIG / "fig4_operation.pdf")
fig.savefig(FIG / "fig4_operation.png")
plt.close(fig)

# ============================================= fig6_identification
# Manuscript Figure 2: free-decay fit, tau ensemble, identifiability.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.1, 2.75))
fig.subplots_adjust(left=0.07, right=0.99, top=0.74, bottom=0.19, wspace=0.34)

seg_row = fits.sort_values("rmse").iloc[2]
t0 = pd.Timestamp(seg_row["start"])
seg = hp.loc[t0:t0 + pd.Timedelta(hours=seg_row["dur_h"])]
th = (seg.index - seg.index[0]).total_seconds() / 3600
ax1.plot(th, seg["T_ret"], color=SKY, linewidth=1.2, label="Measured $T_{ret}$")
yfit = seg_row["Tinf"] + (seg_row["T0"] - seg_row["Tinf"]) * np.exp(
    -th / seg_row["tau_h"])
ax1.plot(th, yfit, color=RED, linestyle="--",
         label=f"First-order fit, $\\tau$ = {seg_row['tau_h']:.1f} h")
ax1.set_xlabel("Time since compressor stop (h)")
ax1.set_ylabel("Temperature (°C)")
ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=1,
           frameon=False, handlelength=1.4, handletextpad=0.4,
           borderaxespad=0.0, fontsize=6.2, labelspacing=0.25)
ax1.set_title("(a) Example free-decay segment", loc="left",
              fontweight="bold", pad=30)

bins = np.linspace(0, 48, 33)
for season, c in (("heating", RED), ("cooling", SKY)):
    f = fits[fits["season"] == season]["tau_h"]
    ax2.hist(f, bins=bins, alpha=0.55, color=c, edgecolor="white",
             label=f"{season.capitalize()}: n = {len(f)}, "
                   f"median {f.median():.1f} h")
ax2.set_xlabel("Identified time constant $\\tau$ (h)")
ax2.set_ylabel("Segments")
ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=1,
           frameon=False, handlelength=1.4, handletextpad=0.4,
           borderaxespad=0.0, fontsize=6.2, labelspacing=0.25)
ax2.set_title("(b) Ensemble distribution of $\\tau$", loc="left",
              fontweight="bold", pad=30)

ratio = (fits["dur_h"] / fits["tau_h"]).to_numpy()
share = float((ratio >= 1).mean())
ax3.hist(ratio, bins=np.linspace(0, 2, 41), color=GREY, alpha=0.85,
         edgecolor="white", label="Accepted fits")
ax3.axvline(1.0, color=RED, linewidth=1.0, linestyle="--",
            label=f"Ratio = 1: only {100*share:.0f} % well identified")
ax3.set_xlabel("Segment duration / fitted $\\tau$ (–)")
ax3.set_ylabel("Segments")
ax3.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=1,
           frameon=False, handlelength=1.4, handletextpad=0.4,
           borderaxespad=0.0, fontsize=6.2, labelspacing=0.25)
ax3.set_title("(c) Identifiability diagnostic", loc="left",
              fontweight="bold", pad=30)
for ax in (ax1, ax2, ax3):
    style(ax)
fig.savefig(FIG / "fig6_identification.pdf")
fig.savefig(FIG / "fig6_identification.png")
plt.close(fig)

# ================================================================ Fig 7
fig, ax = plt.subplots(figsize=(5.2, 2.9))
fig.subplots_adjust(left=0.11, right=0.98, top=0.92, bottom=0.16)
order = ["winter", "spring", "summer", "autumn"]
pos = np.arange(len(order))
w = 0.35
for off, col, c, lab in ((-w / 2, "coverage", BLUE, "As operated"),
                         (w / 2, "coverage_shift", GREEN,
                          "With τ-limited shifting")):
    data = [mc[mc["season"] == s][col].to_numpy() for s in order]
    bp = ax.boxplot(data, positions=pos + off, widths=w * 0.9,
                    showfliers=False, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1},
                    boxprops={"facecolor": c, "alpha": 0.6,
                              "linewidth": 0.6},
                    whiskerprops={"linewidth": 0.6},
                    capprops={"linewidth": 0.6})
    ax.plot([], [], color=c, linewidth=6, alpha=0.6, label=lab)
ax.set_xticks(pos)
ax.set_xticklabels([s.capitalize() for s in order])
ax.set_ylabel("Solar coverage of HP operation (–)")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.01),
          ncols=2, borderaxespad=0.0)
style(ax)
fig.subplots_adjust(top=0.88)
fig.savefig(FIG / "fig8_uncertainty.pdf")
fig.savefig(FIG / "fig8_uncertainty.png")
plt.close(fig)

print("figures written to", FIG)
