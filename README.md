# Minimal-sensing framework for heat pump flexibility and solar alignment

Analysis code for the paper:

> **A minimal-sensing framework for quantifying heat pump flexibility and solar alignment without submetering: a university building case study**
> Damir Špago, Tarik Hubana, Emir Nezirić, Mirza Šarić
> *Under review, Ain Shams Engineering Journal.*

The framework characterises heat pump operation and its alignment with photovoltaic
(PV) generation using only data that existing buildings already produce: hydronic
temperature trends from the building management system (BMS), a whole-building
electricity meter, and regional PV settlement metering. No dedicated heat pump
submeter, heat meter or irradiance sensor is required.

The case study is the Faculty of Mechanical Engineering, University "Džemal Bijedić"
of Mostar, Bosnia and Herzegovina — a reversible groundwater heat pump serving a
hydronic circuit in seasonal changeover.

---

## What this repository contains

Python scripts that reproduce every number, table and figure in the manuscript,
given the input measurement files. Each stage writes intermediate results to disk,
so stages can be re-run independently.

The pipeline covers four blocks:

1. **Operation detection** — mode-specific compressor detection from the load-side
   temperature differential, developed on one set of calendar months and evaluated
   on strictly held-out months against an electrical reference classifier.
2. **Hydronic decay identification** — first-order free-decay fits after compressor
   stop, with explicit identifiability diagnostics.
3. **Empirical PV scenario generation** — whole-day normalised profiles drawn from a
   30-plant utility fleet, validated leave-one-plant-out against a nearest-plant and
   a seasonal-climatology benchmark.
4. **Uncertainty propagation** — Monte Carlo combination of operating days, PV
   scenarios and decay draws into the potential PV-served fraction of heat pump
   energy.

---

## Data availability

**The measurement data is not included in this repository.** The BMS records belong
to the University "Džemal Bijedić" of Mostar and the PV settlement metering belongs
to the distribution system operator; neither is ours to redistribute. Requests can be
directed to the corresponding author.

To run the pipeline, place three Excel files in the repository root:

### `Heat pump measurements.xlsx`

| Sheet | Contents |
|---|---|
| `Sheet1` | 1-min BMS trends. Columns `Date`, `Time of day`, plus the three channels used: `Home > 0.2.1   > Inputs: Temp polaz DT potr` (supply, `T_sup`), `... Temp povrat DT potr` (return, `T_ret`), `... ST4- Temp polaz bu` (groundwater, `T_gw`). 526,484 valid records, 8 May 2025 – 9 May 2026. |
| `Sheet2` | Outdoor temperature at 15 min. Header row located by finding the cell `Date`; first three columns are date, time of day, temperature. 35,461 records. |

### `Solar plants measurements.xlsx`

| Sheet | Contents |
|---|---|
| `Data` | Timestamp in the first column, then 30 meter columns of **average power in kW** per 15-min interval. 1 Aug 2023 – 1 Aug 2024. |
| `Ulazni podaci` | One row per plant: `#`, `Naziv elektrane`, `Podružnica`, `Angažovana snaga (kW)`, `Decimal degrees (DD)` (as `lat,lon`), `Energija iz profila`, `Broj nedostajućih podataka`. |

Two idiosyncrasies of this file are handled in `code/load_data.py` and are worth
knowing about if you adapt the code to another dataset:

- The meter columns in `Data` are **not** in the same order as the rows of
  `Ulazni podaci`, and the identifiers do not match. The assignment is recovered
  bijectively by matching each column's annual energy to `Energija iz profila`; the
  match is asserted to within 0.5 %, so a mismatched file fails loudly rather than
  silently producing wrong plant names.
- Despite `kWh` appearing in some labels, the values are **average power (kW)**;
  comparing observed peaks against contracted power confirms this.

### `Electricity_measurement.xlsx`

Single sheet, whole-building active power at 15 min. The timestamp is read from
column 2 (format `dd/mm/YYYY HH:MM:SS`) and power in kW from column 5. 55,117
records, 1 Jan 2025 – 29 Jul 2026, ≈60,500 kWh/year.

Because this is a whole-building meter and not a submeter, heat pump power is
separated from the occupancy load using October 2025 as a reference month (the heat
pump is inactive throughout). Everything downstream that uses heat pump power uses a
**disaggregated estimate**, not a measurement.

---

## Requirements

Python 3.10 or newer.

```bash
pip install pandas numpy scipy matplotlib python-calamine pyarrow
```

`python-calamine` is the Excel reader (much faster than openpyxl on the 500k-row
sheets); `pyarrow` backs the Parquet cache.

---

## Layout

```
.
├── README.md
├── LICENSE
├── .gitignore
├── code/
│   ├── load_data.py                 # stage 0  data ingestion and caching
│   ├── analysis_hp.py               # stage 1  detection + decay segments
│   ├── analysis_validation.py       # stage 2  electrical reference classifier
│   ├── analysis_detector_ho.py      # stage 3  held-out detector evaluation
│   ├── analysis_pv.py               # stage 4  PV scenarios and propagation
│   ├── analysis_tau_scen.py         # stage 5  identifiability + scenario tests
│   ├── analysis_overlap.py          # stage 6  PV-served fraction, shift sweep
│   ├── analysis_stats_fixes.py      # stage 7  dependence-aware statistics
│   ├── make_figs.py                 # stage 8  manuscript figures
│   ├── analysis_energy_coverage.py  # supplementary (see below)
│   ├── analysis_value.py            # supplementary (see below)
│   ├── make_fig1_overview.py        # supplementary (see below)
│   ├── make_fig3_framework.py       # supplementary (see below)
│   ├── cache/                       # generated Parquet cache (not tracked)
│   └── results/                     # generated JSON/CSV results (not tracked)
└── figures/                         # generated figures (not tracked)
```

The three `.xlsx` inputs go in the repository root, alongside `README.md`. Scripts
resolve paths relative to their own location, so run them from anywhere.

---

## Running the pipeline

Run in order — later stages consume the cache written by earlier ones.

```bash
python code/load_data.py
python code/analysis_hp.py
python code/analysis_validation.py
python code/analysis_detector_ho.py
python code/analysis_pv.py
python code/analysis_tau_scen.py
python code/analysis_overlap.py
python code/analysis_stats_fixes.py
python code/make_figs.py
```

| Stage | Script | Produces | Used for |
|---|---|---|---|
| 0 | `load_data.py` | `cache/hp_temps.parquet`, `outdoor_temp.parquet`, `pv_kw.parquet`, `pv_pu.parquet`, `pv_meta.parquet` | Sections 2.1, 2.3 |
| 1 | `analysis_hp.py` | `cache/hp_ops.parquet`, `cache/decay_fits.csv`, `results/hp_stats.json` | Sections 3.1, 3.2 |
| 2 | `analysis_validation.py` | `cache/elec.parquet`, `cache/hp_ops_v2.parquet`, `results/validation.json`, `validation_table.csv`, `figures/fig5_validation.*` | Sections 2.2, 4.1 |
| 3 | `analysis_detector_ho.py` | `results/detector_holdout.json`, `detector_sensitivity.csv`, `detector_bootstrap.csv` | **Table 1** |
| 4 | `analysis_pv.py` | `results/pv_stats.json`, `results/*.csv` | Sections 3.3, 4.3 |
| 5 | `analysis_tau_scen.py` | `results/tau_diagnostics.json`, `scenario_local.csv`, `scenario_tests.json` | Sections 4.2, 4.3 |
| 6 | `analysis_overlap.py` | `results/overlap_*.csv`, `overlap_stats.json`, `figures/fig8_capacity.*` | **Figure 3**, Section 4.4 |
| 7 | `analysis_stats_fixes.py` | `results/stats_fixes.json`, `figures/fig9_shifting.*` | Section 4.4 (sensitivity) |
| 8 | `make_figs.py` | `figures/fig1_hp_data.*`, `fig2_fleet.*`, `fig4_operation.*`, `fig6_identification.*`, `fig8_uncertainty.*` | **Figures 1 and 2** |

### Figure mapping

The manuscript was condensed to nine pages and uses three of the generated figures.
File names retain their original numbering from the drafting process:

| Manuscript | Generated file | Script |
|---|---|---|
| Figure 1 — detected operation | `figures/fig4_operation.png` | `make_figs.py` |
| Figure 2 — decay identification | `figures/fig6_identification.png` | `make_figs.py` |
| Figure 3 — potential PV-served fraction | `figures/fig8_capacity.png` | `analysis_overlap.py` |

All figures are written at 600 dpi in both PDF and PNG.

### Supplementary scripts

These run correctly and are kept for completeness, but their output does not appear
in the manuscript:

- `analysis_energy_coverage.py` — coverage of heat pump *energy* versus *operating
  time*, used to check the schedule-based proxy. Superseded by the energy-weighted
  metric now built into `analysis_overlap.py`.
- `analysis_value.py` — techno-economic value in monetary terms. Removed from the
  manuscript, which reports physical quantities only.
- `make_fig1_overview.py` — earlier version of the dataset overview figure,
  superseded by the `fig1_hp_data` block of `make_figs.py`.
- `make_fig3_framework.py` — framework block diagram, dropped when the paper was cut
  to nine pages.

---

## Reproducibility

Random seeds are fixed in every stochastic stage: `analysis_pv.py` (42),
`analysis_overlap.py` (31), `analysis_detector_ho.py` (77),
`analysis_energy_coverage.py` (23), `analysis_stats_fixes.py` (101). With the same
input files the pipeline is deterministic.

Development and evaluation of the detector are separated **by calendar month**, not
by random interval, because adjacent 15-min intervals are strongly autocorrelated:

- development months: Jun, Jul, Nov, Dec 2025
- held-out months: Aug, Sep 2025 and Jan–Apr 2026

Thresholds were fixed on the development months and never revisited.

### Headline results

| Quantity | Value |
|---|---|
| Duty cycle | 21.2 % (11.2 % heating, 10.0 % cooling), 241 of 367 days active |
| Heat pump share of building electricity | 47 % |
| Detector, held-out F1 | 0.62 overall (0.65 heating, 0.55 cooling), κ = 0.52 |
| Differential-only detector, held-out F1 | 0.35 overall, 0.11 cooling (no better than chance) |
| Decay segments | 208 (178 heating, 30 cooling) |
| Decay indicator | median 10.2 h (day-clustered 95 % CI 9.6–12.1 h); only 7 % well identified; well-identified median 2.8 h |
| Scenario CRPS | 0.056 p.u. local library, 0.059 nearest plant, 0.084 climatology |
| PV-served fraction, 30 kWp | 0.65 heat-pump-first, 0.49 building-first (daily median) |
| PV-served fraction, annual, 30 kWp | 0.57 (10–90 % interval 0.55–0.59) |
| Hypothetical shifting gain | 0.0–4.8 percentage points across 27 rule variants |
| Detector-error effect on seasonal result | up to ≈10 percentage points |

---

## Scope and limitations

The framework is a **screening and decision-support instrument**, not a validated
quantification of thermal flexibility. In particular:

- The decay indicator describes how fast the *distribution circuit* relaxes. It is
  confounded between water volume, pipe and buffer insulation, pump behaviour after
  shutdown, sensor position and envelope dynamics, and it is weakly identified.
  Demonstrating that operation can be displaced without violating comfort requires
  indoor temperature measurement or a controlled experiment, neither of which was
  available here.
- The electrical reference is a disaggregated proxy, not a submeter. Detector scores
  express agreement with that classifier rather than absolute accuracy, and vary
  between 0.19 and 0.68 across 36 plausible reference definitions.
- The PV record (Aug 2023 – Jul 2024) and the heat pump record (May 2025 – May 2026)
  do not overlap, so the scenarios are climatological rather than synchronous. This
  was tested directly and shifts seasonal medians by at most 4.9 percentage points.
- Several settings are specific to this installation — the calendar-based seasonal
  changeover, the 14 °C cooling threshold, the single-compressor topology, the
  October reference month. These need re-derivation for systems with simultaneous
  heating and cooling, weather-compensated supply temperatures, multiple generators
  or buffer storage.
- Results come from one building and one year. Transferability is argued from the
  structure of the method, not demonstrated.

---

## Citation

Please cite the paper rather than this repository:

```bibtex
@article{spago2026minimalsensing,
  author  = {\v{S}pago, Damir and Hubana, Tarik and Neziri\'{c}, Emir and \v{S}ari\'{c}, Mirza},
  title   = {A minimal-sensing framework for quantifying heat pump flexibility
             and solar alignment without submetering: a university building case study},
  journal = {Ain Shams Engineering Journal},
  year    = {2026},
  note    = {Under review}
}
```

## Contact

Tarik Hubana — tarik.hubana@gmail.com

## License

MIT — see [LICENSE](LICENSE).
