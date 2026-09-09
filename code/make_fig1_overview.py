"""
Figure 1 — dataset overview for the HP + PV paper.

Panels:
  (a) outdoor temperature, daily mean and min-max range, full year
  (b) buffer tank temperature (ST1), daily mean and min-max range
  (c) minute-resolution close-up of hydronic temperatures (winter days)
  (d) normalized PV fleet daily profiles, seasonal median + quantile bands

Run load_data.py first (creates ./cache/*.parquet).
Outputs: ../figures/fig1_overview.pdf / .png (600 dpi)
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "code" / "cache"
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- styling
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 0.9,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

C_OUT = "#0072B2"     # blue
C_BUF = "#D55E00"     # vermillion
C_PV = "#E69F00"      # orange
C_GREY = "#7F7F7F"

CLOSEUP = ("2026-01-12", "2026-01-15")  # 3 winter days, minute resolution

# ---------------------------------------------------------------- data
tout = pd.read_parquet(CACHE / "outdoor_temp.parquet")
hp = pd.read_parquet(CACHE / "hp_temps.parquet")
pv = pd.read_parquet(CACHE / "pv_fleet.parquet")

tout_d = tout["T_out"].resample("D").agg(["mean", "min", "max"]).dropna()
buf_d = hp["T_buffer"].resample("D").agg(["mean", "min", "max"]).dropna()

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(7.48, 5.2))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22,
                      left=0.065, right=0.985, top=0.965, bottom=0.09)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# (a) outdoor temperature -------------------------------------------------
ax_a.fill_between(tout_d.index, tout_d["min"], tout_d["max"],
                  color=C_OUT, alpha=0.25, linewidth=0, label="Daily range")
ax_a.plot(tout_d.index, tout_d["mean"], color=C_OUT, label="Daily mean")
ax_a.axhline(0, color=C_GREY, linewidth=0.5, linestyle=":")
ax_a.set_ylabel("Outdoor temperature (°C)")
ax_a.legend(frameon=False, loc="upper right", ncols=1)
ax_a.set_title("(a) Outdoor temperature", loc="left", fontweight="bold")

# (b) buffer tank temperature --------------------------------------------
ax_b.fill_between(buf_d.index, buf_d["min"], buf_d["max"],
                  color=C_BUF, alpha=0.25, linewidth=0, label="Daily range")
ax_b.plot(buf_d.index, buf_d["mean"], color=C_BUF, label="Daily mean")
ax_b.set_ylabel("Buffer tank temperature (°C)")
ax_b.legend(loc="lower right", frameon=True, framealpha=0.85,
            edgecolor="none")
ax_b.set_title("(b) Buffer tank temperature (ST1)", loc="left",
               fontweight="bold")

for ax in (ax_a, ax_b):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlim(tout_d.index.min(), tout_d.index.max())

# (c) minute-resolution close-up -----------------------------------------
win = hp.loc[CLOSEUP[0]:CLOSEUP[1]]
series = [
    ("T_buffer", "Buffer (ST1)", C_BUF),
    ("T_sup_load", "Supply load", "#009E73"),
    ("T_ret_load", "Return load", "#56B4E9"),
    ("T_sup_source", "Supply source", "#CC79A7"),
    ("T_ret_source", "Return source", "#999999"),
]
for col, lab, c in series:
    ax_c.plot(win.index, win[col], color=c, label=lab, linewidth=0.7)
ax_c.set_ylabel("Temperature (°C)")
ax_c.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=5,
            frameon=False, columnspacing=0.7, handlelength=1.0,
            handletextpad=0.4, borderaxespad=0.0, fontsize=6)
ax_c.xaxis.set_major_locator(mdates.DayLocator())
ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax_c.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
ax_c.set_xlim(win.index.min(), win.index.max())
ax_c.set_title("(c) Hydronic temperatures, 1-min resolution (winter)",
               loc="left", fontweight="bold", pad=16)

# (d) PV fleet normalized daily profiles ---------------------------------
pvp = pv.copy()
pvp["month"] = pvp["timestamp"].dt.month
pvp["tod"] = (pvp["timestamp"].dt.hour * 60
              + pvp["timestamp"].dt.minute) / 60.0
seasons = {
    "Summer (JJA)": ([6, 7, 8], C_PV),
    "Winter (DJF)": ([12, 1, 2], C_OUT),
}
for lab, (months, c) in seasons.items():
    sel = pvp[pvp["month"].isin(months)]
    grp = sel.groupby("tod")["power_pu"]
    med, q10, q90 = grp.median(), grp.quantile(0.10), grp.quantile(0.90)
    ax_d.fill_between(med.index, q10, q90, color=c, alpha=0.22, linewidth=0)
    ax_d.plot(med.index, med, color=c, label=f"{lab}, median")
ax_d.set_xlim(0, 24)
ax_d.set_xticks(range(0, 25, 6))
ax_d.set_xlabel("Hour of day")
ax_d.set_ylabel("PV power (p.u. of peak)")
ax_d.legend(frameon=False, loc="upper left")
ax_d.set_title("(d) PV fleet profiles, 8 plants (10–90% band)",
               loc="left", fontweight="bold")

for ax in (ax_a, ax_b, ax_c, ax_d):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)

fig.savefig(FIGDIR / "fig1_overview.pdf")
fig.savefig(FIGDIR / "fig1_overview.png")
print("saved to", FIGDIR)
