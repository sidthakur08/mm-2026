"""Centralized data loading module for March Madness prediction project.

Provides convenience functions to load CSV datasets from the data/ directory
as pandas DataFrames. Files are prefixed with M (men's) or W (women's),
with a few shared files (Cities, Conferences, SampleSubmission).
"""

import re
from pathlib import Path
from typing import Literal

import pandas as pd

# Absolute path to the data directory, resolved relative to this file's location.
DATA_DIR: Path = (Path(__file__).resolve().parent.parent / "data")

Gender = Literal["M", "W"]


def load_csv(filename: str) -> pd.DataFrame:
    """Read a CSV file from DATA_DIR and return it as a DataFrame."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Gender-specific loaders
# ---------------------------------------------------------------------------

def load_teams(gender: Gender = "M") -> pd.DataFrame:
    """Load MTeams.csv or WTeams.csv."""
    return load_csv(f"{gender}Teams.csv")


def load_seasons(gender: Gender = "M") -> pd.DataFrame:
    """Load MSeasons.csv or WSeasons.csv."""
    return load_csv(f"{gender}Seasons.csv")


def load_seeds(gender: Gender = "M") -> pd.DataFrame:
    """Load MNCAATourneySeeds.csv or WNCAATourneySeeds.csv."""
    return load_csv(f"{gender}NCAATourneySeeds.csv")


def load_regular_season_results(
    gender: Gender = "M", detailed: bool = False
) -> pd.DataFrame:
    """Load regular-season results (compact or detailed)."""
    kind = "Detailed" if detailed else "Compact"
    return load_csv(f"{gender}RegularSeason{kind}Results.csv")


def load_tourney_results(
    gender: Gender = "M", detailed: bool = False
) -> pd.DataFrame:
    """Load NCAA tournament results (compact or detailed)."""
    kind = "Detailed" if detailed else "Compact"
    return load_csv(f"{gender}NCAATourney{kind}Results.csv")


def load_tourney_slots(gender: Gender = "M") -> pd.DataFrame:
    """Load MNCAATourneySlots.csv or WNCAATourneySlots.csv."""
    return load_csv(f"{gender}NCAATourneySlots.csv")


def load_team_conferences(gender: Gender = "M") -> pd.DataFrame:
    """Load MTeamConferences.csv or WTeamConferences.csv."""
    return load_csv(f"{gender}TeamConferences.csv")


def load_game_cities(gender: Gender = "M") -> pd.DataFrame:
    """Load MGameCities.csv or WGameCities.csv."""
    return load_csv(f"{gender}GameCities.csv")


# ---------------------------------------------------------------------------
# Shared / single-gender loaders
# ---------------------------------------------------------------------------

def load_massey_ordinals() -> pd.DataFrame:
    """Load MMasseyOrdinals.csv (men's only, ~5.7M rows). May be slow."""
    return load_csv("MMasseyOrdinals.csv")


def load_conferences() -> pd.DataFrame:
    """Load Conferences.csv (shared across genders)."""
    return load_csv("Conferences.csv")


def load_coaches() -> pd.DataFrame:
    """Load MTeamCoaches.csv (men's only)."""
    return load_csv("MTeamCoaches.csv")


def load_cities() -> pd.DataFrame:
    """Load Cities.csv."""
    return load_csv("Cities.csv")


def load_sample_submission(stage: int = 1) -> pd.DataFrame:
    """Load SampleSubmissionStage1.csv or SampleSubmissionStage2.csv."""
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage}")
    return load_csv(f"SampleSubmissionStage{stage}.csv")


# ---------------------------------------------------------------------------
# Composite loaders
# ---------------------------------------------------------------------------

def load_all_results(
    gender: Gender = "M", detailed: bool = False
) -> pd.DataFrame:
    """Concatenate regular-season and tourney results with an `is_tourney` column."""
    regular = load_regular_season_results(gender, detailed)
    tourney = load_tourney_results(gender, detailed)

    regular = regular.assign(is_tourney=False)
    tourney = tourney.assign(is_tourney=True)

    return pd.concat([regular, tourney], ignore_index=True)


# ---------------------------------------------------------------------------
# Seed parsing
# ---------------------------------------------------------------------------

# Pattern: one letter region, two-digit seed, optional play-in suffix (a/b).
_SEED_PATTERN = re.compile(r"^([WXYZ])(\d{2})([a-b])?$")


def parse_seed(seed_str: str) -> dict:
    """Parse a seed string like 'W01' or 'X16a'.

    Returns:
        dict with keys:
            region     (str)  - one of W, X, Y, Z
            seed_number (int) - 1-16
            play_in    (bool) - True if the seed has a play-in suffix
    """
    match = _SEED_PATTERN.match(seed_str)
    if not match:
        raise ValueError(f"Invalid seed string: '{seed_str}'")

    region, number, suffix = match.groups()
    return {
        "region": region,
        "seed_number": int(number),
        "play_in": suffix is not None,
    }
