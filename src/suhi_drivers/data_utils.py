"""
Data loading and variable selection utilities for national-scale SUHI analysis.
"""

import re
from pathlib import Path
from typing import List, Union
import pandas as pd

SEASONS = ("Annual", "Winter", "Spring", "Summer", "Autumn")
PERIODS = ("Day", "Night")
CLIMATE_ZONES = ("Arid", "Semi-arid", "Humid")

OUTCOME_PATTERN = re.compile(r"^SUHI_(Annual|Winter|Spring|Summer|Autumn)_(Day|Night)$")

# Season-dependent hydro-climatic and vegetation variables
SEASON_VARYING_VARS = (
    "ΔNDVI_{season}",
    "TMP_{season}",
    "VS_{season}",
    "Pre_{season}",
    "ET_{season}",
)

# Static topographic, socioeconomic, and 2D/3D morphological indicators
INVARIANT_VARS = (
    "ΔDEM", "ΔPOP", "ΔFTC", "ΔGDP",
    "PLAND", "PD", "ED", "LSI", "FRAC",
    "BH_mn", "BH_std", "BS_max", "BS_std", "BV_max",
)


def load_suhi_dataset(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load the city-level SUHI dataset and standardize column names.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [col.strip() for col in df.columns]
    return df


def get_available_outcomes(df: pd.DataFrame) -> List[str]:
    """
    Return ordered list of SUHI target columns present in the dataframe.
    """
    ordered = [f"SUHI_{season}_{period}" for season in SEASONS for period in PERIODS]
    return [col for col in ordered if col in df.columns]


def get_predictors_for_outcome(df: pd.DataFrame, outcome: str) -> List[str]:
    """
    Match the 5 season-specific variables and 14 static variables to a given SUHI outcome.
    """
    match = OUTCOME_PATTERN.match(outcome)
    if not match:
        raise ValueError(f"Outcome '{outcome}' does not match expected pattern 'SUHI_<Season>_<Period>'.")

    season = match.group(1)
    predictors = []

    for template in SEASON_VARYING_VARS:
        col = template.format(season=season)
        if col in df.columns:
            predictors.append(col)

    for col in INVARIANT_VARS:
        if col in df.columns:
            predictors.append(col)

    return predictors


def format_driver_labels(series: pd.Series) -> pd.Series:
    """
    Strip seasonal suffixes and standardize variable abbreviations for plotting and tables.
    """
    cleaned = series.copy()
    for season in SEASONS:
        cleaned = cleaned.str.replace(f"_{season}", "", regex=False)

    # Map ERA5 wind speed column code (VS) to manuscript notation (WS)
    cleaned = cleaned.str.replace("^VS$", "WS", regex=True)
    cleaned = cleaned.str.replace("^ΔVS$", "ΔWS", regex=True)
    return cleaned
