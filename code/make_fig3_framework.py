"""Figure 3 — methodological framework block diagram (journal style)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "savefig.dpi": 600,
})

EDGE = "black"
FILL = "white"
FILL_HDR = "#F2F2F2"
FILL_ELEC = "#E4EEF7"

fig, ax = plt.subplots(figsize=(7.0, 3.4))
ax.set_xlim(-0.3, 10.05)
ax.set_ylim(-0.55, 4.75)
ax.axis("off")


def box(x, y, w, h, title, sub=None, fill=FILL):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.04,rounding_size=0.03",
                                facecolor=fill, edgecolor=EDGE,
                                linewidth=0.8))
    if sub:
        ax.text(x + w / 2, y + h / 2 + 0.13, title, ha="center",
                va="center", fontsize=7.5, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.17, sub, ha="center", va="center",
                fontsize=6.8, style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=7.5, fontweight="bold")


def arrow(x1, y1, x2, y2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=8, linewidth=0.8,
                                 linestyle=ls, color=EDGE,
                                 shrinkA=1.5, shrinkB=1.5))


# ---- layer headers
for xc, lab in ((1.5, "I. Measurements"), (4.95, "II. Models"),
                (8.45, "III. Propagation")):
    ax.text(xc, 4.55, lab, fontsize=8, fontweight="bold", ha="center")
ax.plot([-0.3, 10.05], [4.38, 4.38], color=EDGE, linewidth=0.5)

# ---- measurement layer
box(0.2, 3.50, 2.6, 0.72, "Hydronic temperatures",
    "$T_{sup}$, $T_{ret}$, $T_{gw}$ (1-min)", FILL_HDR)
box(0.2, 2.62, 2.6, 0.72, "Outdoor temperature", "$T_{out}$ (15-min)",
    FILL_HDR)
box(0.2, 1.42, 2.6, 0.86, "Whole-building electricity",
    "15-min active power", FILL_ELEC)
box(0.2, 0.30, 2.6, 0.72, "PV fleet metering",
    "30 plants, 15-min (utility)", FILL_HDR)

# ---- model layer
box(3.6, 3.50, 2.7, 0.72, "Operation detection",
    "mode from $\\Delta T$, $T_{sup}$, Eqs. (1)–(2)")
box(3.6, 2.62, 2.7, 0.72, "Decay indicator",
    "free-decay ensemble, Eq. (3)")
box(3.6, 1.42, 2.7, 0.86, "Load disaggregation",
    "occupancy baseline\n$\\rightarrow$ inferred $P_{HP}$", FILL_ELEC)
box(3.6, 0.30, 2.7, 0.72, "Scenario generator",
    "seasonal profiles, Eq. (4)")

# ---- propagation layer
box(7.15, 1.95, 2.6, 1.45, "Monte Carlo propagation",
    "PV scenarios × operating days\n× decay draws")
box(7.15, 0.30, 2.6, 0.95, "Potential PV-served\nfraction, Eq. (5)",
    "by capacity and priority")

# ---- arrows
arrow(2.8, 3.86, 3.6, 3.86)
arrow(2.8, 2.98, 3.6, 2.98)
arrow(2.8, 1.85, 3.6, 1.85)
arrow(2.8, 0.66, 3.6, 0.66)
arrow(4.95, 3.50, 4.95, 3.34)                 # detection -> decay segments
arrow(4.95, 1.42, 4.95, 1.10, style="-")      # disagg -> detector validation
arrow(4.95, 1.10, 6.6, 1.10, style="-")
arrow(6.6, 1.10, 6.6, 3.86, style="-")
arrow(6.6, 3.86, 6.3, 3.86, style="-|>")
ax.text(5.55, 0.98, "validates detector (one-off)", fontsize=6,
        style="italic", ha="center")
arrow(6.3, 3.86, 7.15, 3.1)
arrow(6.3, 2.98, 7.15, 2.75)
arrow(6.3, 1.85, 7.15, 2.35)                  # inferred P_HP -> propagation
arrow(6.3, 0.66, 7.15, 2.05)
arrow(8.45, 1.95, 8.45, 1.25)
# feedback
arrow(8.45, 0.30, 8.45, -0.25, style="-", ls=(0, (4, 3)))
arrow(8.45, -0.25, -0.05, -0.25, style="-", ls=(0, (4, 3)))
arrow(-0.05, -0.25, -0.05, 2.98, style="-", ls=(0, (4, 3)))
arrow(-0.05, 2.98, 0.2, 2.98, style="-|>", ls=(0, (4, 3)))
ax.text(4.3, -0.47, "triage and scheduling recommendations",
        fontsize=6.5, ha="center", style="italic")

fig.tight_layout(pad=0.2)
fig.savefig(FIGDIR / "fig3_framework.pdf")
fig.savefig(FIGDIR / "fig3_framework.png")
print("saved")
