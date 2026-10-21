"""
Project-wide settings: paths, file locations, case split, dates and scoring choices.
Every other module imports from here, so a setting is changed in ONE place only.
"""
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]      # repo root (works from scripts AND notebooks)
DATA_RAW = ROOT / "data" / "raw"                # original Excel files (never committed)
DATA_PROCESSED = ROOT / "data" / "processed"    # parquet outputs = the team's data contract
GRID_DIR = DATA_PROCESSED / "grid"              # one parquet per case (3D grid, optional)
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

def raw_file(prefix: str) -> Path:
    """Find a raw Excel file by its number prefix, e.g. raw_file("02") -> 02_Field_Production.xlsx.
    Matching on the prefix means slightly different file names on each laptop still work."""
    matches = sorted(DATA_RAW.glob(f"{prefix}*.xlsx"))
    if not matches:
        raise FileNotFoundError(f"No file starting with '{prefix}' in {DATA_RAW}")
    return matches[0]

# Raw file prefixes (as provided by the organisers)
INTRO_PREFIX = "01"          # Nomenclature + Uncertainty Values
PRODUCTION_PREFIX = "02"     # simulated curves (Case_1..100) + RAMP_History
GRID_PREFIXES = ["03", "04", "05", "06", "07", "08"]   # 3D grid per case (train x4, validation, test)
TEMPLATE_PREFIX = "09"       # deliverable template

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- cases & split (fixed by organisers)
N_CASES = 100
SPLIT = {
    "train": range(1, 71),         # cases 1-70
    "validation": range(71, 86),   # cases 71-85
    "test": range(86, 101),        # cases 86-100  -> use ONCE, never for tuning/optimisation
}

def split_of(case: int) -> str:
    for name, cases in SPLIT.items():
        if case in cases:
            return name
    raise ValueError(f"Case {case} is outside 1-{N_CASES}")

# ---------------------------------------------------------------- the 4 uncertainty parameters (X)
# Excel column name -> short name used in code
PARAMS = {
    "Fault Transmissibility": "fault_trans",
    "Porosity Multiplier": "poro_mult",
    "Permeability Multiplier": "perm_mult",
    "Aquifer Pore Volume": "aquifer_pv",
}
PARAM_COLS = list(PARAMS.values())
# NOTE: allowed search ranges for optimisation are computed from the data (min/max of the 100 cases),
# not hard-coded here, so they can never drift from the real values.

# ---------------------------------------------------------------- production streams (y)
# Cumulative curves: short name -> sheet name in the 02 Field Production workbook
CUM_SHEETS = {
    "oil_cum": "Oil production cumulative",      # STB
    "gas_cum": "Gas production cumulative",      # MSCF
    "water_cum": "Water production cumulative",  # STB
}
CUM_STREAMS = list(CUM_SHEETS)
# Rates are NOT read from the rate sheets: the simulator reports the rate of the period ENDING on a date,
# while the real history reports the rate of the quarter STARTING on that date (one quarter apart).
# So ALL rates (cases and history) are derived from the cumulative curves with ONE rule:
#   rate(t) = average daily rate over the quarter ENDING at t = (cum(t) - cum(t-1)) / days   [STB/d, MSCF/d]
RATE_STREAMS = ["oil_rate", "gas_rate", "water_rate"]
STREAMS = CUM_STREAMS + RATE_STREAMS
HISTORY_COL = "RAMP_History"          # the REAL field data column. Never a training target.
HISTORY_SHEET = "Field Production"    # same real data as rates (+ water injection)
GRID_NA = -999.25                     # sentinel for inactive grid cells -> NaN

# ---------------------------------------------------------------- dates
# The 41 quarterly dates where the real data was measured (1998-01-01 ... 2008-01-01)
HISTORY_DATES = pd.date_range("1998-01-01", "2008-01-01", freq="QS")
# Scoring starts when production starts (the first 3 quarters are 0 for every case)
SCORE_START = pd.Timestamp("1998-10-01")
# Forecast dates required by the template (2008-04-01 ... 2028-01-01, 80 quarters)
FORECAST_DATES = pd.date_range("2008-04-01", "2028-01-01", freq="QS")

# ---------------------------------------------------------------- scoring (team decision, 10 Oct)
# Main score: cumulative streams, each NRMSE = RMSE / max(history of that stream)
SCORE_STREAMS = CUM_STREAMS
SCORE_WEIGHTS = {"oil_cum": 1 / 3, "gas_cum": 1 / 3, "water_cum": 1 / 3}
# Alternative to discuss: gas is ~0.363 x oil in this data, so equal weights count oil twice.
# SCORE_WEIGHTS = {"oil_cum": 0.5, "gas_cum": 0.0, "water_cum": 0.5}
TOP_K = 10

# ---------------------------------------------------------------- T3 optimisation settings
T3_N_SAMPLES = 50_000        # number of random parameter sets to sample
T3_SEED = 42                 # random seed for reproducibility
T3_TIE_BAND = 0.0003         # score distance from best; candidates within this are "near-optimal"
T3_TOP_K = 500               # number of top candidates to save to outputs/
T3_N_SELECT = 10             # number of diverse candidates to pick from the near-optimal pool
T3_ARPS_FIT_START = "2003-01-01"  # start of Arps decline fit window (plain string, no pandas import)
T3_ARPS_B_MAX = 1.0               # upper bound on Arps b (harmonic = physically valid limit)