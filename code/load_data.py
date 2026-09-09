"""
Load and clean measurement data for the HP + PV paper (v2).

Case study: Faculty of Mechanical Engineering, University "Dzemal Bijedic"
Mostar, Bosnia and Herzegovina. Reversible water-source heat pump
(groundwater well, "busotina").

Relevant HP channels (per system owner):
  - Temp polaz DT potr  -> T_sup   supply to consumers (all heating), degC
  - Temp povrat DT potr -> T_ret   return from consumers, degC
  - ST4- Temp polaz bu  -> T_gw    groundwater (well) temperature, degC
  - Sheet2              -> T_out   outdoor temperature (15-min)

PV fleet: 30 utility-metered plants across Bosnia and Herzegovina.
  - sheet 'Data': DateTime + 30 meter columns, average power in kW per
    15-min interval (despite 'kWh' labels elsewhere; peak vs rated confirms)
  - sheet 'Ulazni podaci': one row per plant, in the same order as the
    'Data' columns (plant #1..#30), with name, rated power, coordinates.

Outputs cached in ./cache as parquet.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "code" / "cache"
CACHE.mkdir(exist_ok=True)

HP_FILE = ROOT / "Heat pump measurements.xlsx"
PV_FILE = ROOT / "Solar plants measurements.xlsx"

HP_COLS = {
    "Home > 0.2.1   > Inputs: Temp polaz DT potr": "T_sup",
    "Home > 0.2.1   > Inputs: Temp povrat DT potr": "T_ret",
    "Home > 0.2.1   > Inputs: ST4- Temp polaz bu": "T_gw",
}


def _combine_dt(df: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(df["Date"]).dt.normalize()
    tod = pd.to_timedelta(df["Time of day"].astype(str))
    return date + tod


def load_hp_temps() -> pd.DataFrame:
    df = pd.read_excel(HP_FILE, sheet_name="Sheet1", engine="calamine")
    df["timestamp"] = _combine_dt(df)
    df = df.rename(columns=HP_COLS)
    df = df[["timestamp"] + list(HP_COLS.values())]
    for c in HP_COLS.values():
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return (df.dropna(subset=["timestamp"])
              .drop_duplicates(subset="timestamp")
              .set_index("timestamp").sort_index())


def load_outdoor_temp() -> pd.DataFrame:
    raw = pd.read_excel(HP_FILE, sheet_name="Sheet2", engine="calamine",
                        header=None)
    hdr = raw.index[raw[0] == "Date"][0]
    df = raw.iloc[hdr + 1:, :3].copy()
    df.columns = ["Date", "Time of day", "T_out"]
    df["timestamp"] = _combine_dt(df)
    df["T_out"] = pd.to_numeric(df["T_out"], errors="coerce")
    return (df[["timestamp", "T_out"]]
              .dropna(subset=["timestamp"])
              .drop_duplicates(subset="timestamp")
              .set_index("timestamp").sort_index())


def load_pv_meta() -> pd.DataFrame:
    meta = pd.read_excel(PV_FILE, sheet_name="Ulazni podaci",
                         engine="calamine")
    meta = meta.rename(columns={
        "#": "idx",
        "Naziv elektrane": "plant",
        "Angažovana snaga (kW)": "rated_kw",
        "Decimal degrees (DD)": "coords",
        "Broj nedostajućih podataka": "n_missing",
        "Podružnica": "branch",
    })
    meta = meta.dropna(subset=["plant"]).copy()
    latlon = meta["coords"].astype(str).str.split(",", expand=True)
    meta["lat"] = pd.to_numeric(latlon[0], errors="coerce")
    meta["lon"] = pd.to_numeric(latlon[1], errors="coerce")
    keep = ["idx", "plant", "branch", "rated_kw", "lat", "lon", "n_missing"]
    return meta[keep].reset_index(drop=True)


def load_pv_fleet() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """30-plant fleet from the 'Data' sheet (wide kW power matrix).

    The 'Data' columns are meter IDs whose order does NOT match the
    metadata order. Each column is matched to its plant through the total
    profile energy ('Energija iz profila' in 'Ulazni podaci'), which gives
    an exact one-to-one mapping (verified to < 1 kWh tolerance).
    """
    meta = load_pv_meta()
    meta_raw = pd.read_excel(PV_FILE, sheet_name="Ulazni podaci",
                             engine="calamine")
    target = meta_raw.set_index("Naziv elektrane")["Energija iz profila"]

    df = pd.read_excel(PV_FILE, sheet_name="Data", engine="calamine")
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated()]
    df = df.apply(pd.to_numeric, errors="coerce")

    col_energy = df.sum() / 4.0  # 15-min kW -> kWh
    mapping = {}
    for col, e in col_energy.items():
        diffs = (target - e).abs()
        plant = diffs.idxmin()
        assert diffs.min() < 0.005 * max(e, 1.0), \
            f"no energy match for column {col}: {e}"
        assert plant not in mapping.values(), f"duplicate match: {plant}"
        mapping[col] = plant
    df.columns = [mapping[c] for c in df.columns]
    df = df[meta["plant"].tolist()]  # metadata order

    p_ref = df.quantile(0.999)
    pu = df / p_ref
    meta = meta.copy()
    meta["p_ref_kw"] = p_ref.reindex(meta["plant"]).values
    return df, pu, meta


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main() -> None:
    print("HP temperatures (1-min, 3 channels) ...")
    hp = load_hp_temps()
    hp.to_parquet(CACHE / "hp_temps.parquet")
    print(f"  {len(hp):,} rows  {hp.index.min()} -> {hp.index.max()}")

    print("Outdoor temperature (15-min) ...")
    tout = load_outdoor_temp()
    tout.to_parquet(CACHE / "outdoor_temp.parquet")
    print(f"  {len(tout):,} rows")

    print("PV fleet, 30 plants ('Data' sheet) ...")
    kw, pu, meta = load_pv_fleet()
    kw.to_parquet(CACHE / "pv_kw.parquet")
    pu.to_parquet(CACHE / "pv_pu.parquet")
    meta.to_parquet(CACHE / "pv_meta.parquet")
    print(f"  {kw.shape[0]:,} intervals x {kw.shape[1]} plants")
    print(meta[["plant", "rated_kw", "p_ref_kw", "lat", "lon"]].to_string())


if __name__ == "__main__":
    main()
