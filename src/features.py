"""Feature engineering module for March Madness NCAA basketball predictions.

Transforms raw game results into per-team aggregated statistics, merges
external ranking and seed data, and constructs matchup-level training
data suitable for binary classification models.

Features include Dean-Oliver Four Factors, efficiency metrics, tempo-adjusted
rate stats, and situational performance indicators tuned for tournament play.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column mappings used to "unpivot" winner/loser perspectives
# ---------------------------------------------------------------------------

_WINNER_COL_MAP: dict[str, str] = {
    "WTeamID": "TeamID",
    "WScore": "Score",
    "LScore": "OppScore",
    "WFGM": "FGM",
    "WFGA": "FGA",
    "WFGM3": "FGM3",
    "WFGA3": "FGA3",
    "WFTM": "FTM",
    "WFTA": "FTA",
    "WOR": "OR",
    "WDR": "DR",
    "WAst": "Ast",
    "WTO": "TO",
    "WStl": "Stl",
    "WBlk": "Blk",
    "WPF": "PF",
}

_LOSER_COL_MAP: dict[str, str] = {
    "LTeamID": "TeamID",
    "LScore": "Score",
    "WScore": "OppScore",
    "LFGM": "FGM",
    "LFGA": "FGA",
    "LFGM3": "FGM3",
    "LFGA3": "FGA3",
    "LFTM": "FTM",
    "LFTA": "FTA",
    "LOR": "OR",
    "LDR": "DR",
    "LAst": "Ast",
    "LTO": "TO",
    "LStl": "Stl",
    "LBlk": "Blk",
    "LPF": "PF",
}

# Opponent box-score columns carried through unpivot for rate calculations
_WINNER_OPP_MAP: dict[str, str] = {
    "LOR": "OppOR",
    "LDR": "OppDR",
    "LFGA": "OppFGA",
    "LFGA3": "OppFGA3",
    "LFGM": "OppFGM",
    "LFGM3": "OppFGM3",
    "LFTA": "OppFTA",
    "LFTM": "OppFTM",
    "LTO": "OppTO",
    "LStl": "OppStl",
    "LBlk": "OppBlk",
    "LAst": "OppAst",
}

_LOSER_OPP_MAP: dict[str, str] = {
    "WOR": "OppOR",
    "WDR": "OppDR",
    "WFGA": "OppFGA",
    "WFGA3": "OppFGA3",
    "WFGM": "OppFGM",
    "WFGM3": "OppFGM3",
    "WFTA": "OppFTA",
    "WFTM": "OppFTM",
    "WTO": "OppTO",
    "WStl": "OppStl",
    "WBlk": "OppBlk",
    "WAst": "OppAst",
}

_COMMON_COLS = ["Season"]
_UNIFIED_COLS = ["Season", "TeamID", "Score", "OppScore",
                 "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
                 "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]

_OPP_COLS = ["OppOR", "OppDR", "OppFGA", "OppFGA3", "OppFGM", "OppFGM3",
             "OppFTA", "OppFTM", "OppTO", "OppStl", "OppBlk", "OppAst"]

_EXTRA_COLS = ["DayNum", "Location", "OppTeamID"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unpivot_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape game-level results into team-level rows.

    Each game produces two rows -- one from the winner's perspective
    (``Win=1``) and one from the loser's (``Win=0``).

    Carries forward DayNum, game location from the team's perspective,
    opponent TeamID, and opponent box-score stats needed for rate
    calculations (OppOR, OppDR, OppFGA, etc.).
    """
    has_detail = "WFGM" in results_df.columns
    has_daynum = "DayNum" in results_df.columns
    has_wloc = "WLoc" in results_df.columns

    # --- Winner rows ---
    winner_cols = {**{c: c for c in _COMMON_COLS}, **_WINNER_COL_MAP}
    winners = results_df[list(winner_cols.keys())].rename(columns=winner_cols)
    winners["Win"] = 1

    if has_daynum:
        winners["DayNum"] = results_df["DayNum"].values

    if has_wloc:
        winners["Location"] = results_df["WLoc"].values
    else:
        winners["Location"] = "N"

    winners["OppTeamID"] = results_df["LTeamID"].values

    if has_detail:
        for src, dst in _WINNER_OPP_MAP.items():
            if src in results_df.columns:
                winners[dst] = results_df[src].values

    # --- Loser rows ---
    loser_cols = {**{c: c for c in _COMMON_COLS}, **_LOSER_COL_MAP}
    losers = results_df[list(loser_cols.keys())].rename(columns=loser_cols)
    losers["Win"] = 0

    if has_daynum:
        losers["DayNum"] = results_df["DayNum"].values

    if has_wloc:
        # Flip location for the loser: H->A, A->H, N->N
        loc_map = {"H": "A", "A": "H", "N": "N"}
        losers["Location"] = results_df["WLoc"].map(loc_map).values
    else:
        losers["Location"] = "N"

    losers["OppTeamID"] = results_df["WTeamID"].values

    if has_detail:
        for src, dst in _LOSER_OPP_MAP.items():
            if src in results_df.columns:
                losers[dst] = results_df[src].values

    # Determine which columns to include in final output
    output_cols = _UNIFIED_COLS + ["Win"]

    if has_daynum:
        output_cols = output_cols + ["DayNum"]
    output_cols = output_cols + ["Location", "OppTeamID"]

    if has_detail:
        available_opp = [c for c in _OPP_COLS if c in winners.columns]
        output_cols = output_cols + available_opp

    unified = pd.concat(
        [winners[output_cols], losers[output_cols]],
        ignore_index=True,
    )
    return unified


def _estimate_possessions(
    fga: pd.Series,
    off_reb: pd.Series,
    to: pd.Series,
    fta: pd.Series,
) -> pd.Series:
    """Estimate possessions using the standard formula: FGA - OR + TO + 0.475*FTA."""
    return fga - off_reb + to + 0.475 * fta


# ---------------------------------------------------------------------------
# 1. Per-team, per-season aggregated stats (with advanced basketball metrics)
# ---------------------------------------------------------------------------

def build_team_season_stats(
    results_df: pd.DataFrame,
    compact_results_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute per-team, per-season aggregated stats from game results.

    When detailed box-score data is available (results_df), computes the full
    suite of advanced basketball analytics. For seasons with only compact
    results (score-only), produces a reduced feature set from the compact data.

    Parameters
    ----------
    results_df : pd.DataFrame
        Detailed regular-season results with box-score columns
        (WFGM, WFGA, ... and corresponding L* columns).
    compact_results_df : pd.DataFrame, optional
        Compact regular-season results (Season, DayNum, WTeamID, WScore,
        LTeamID, LScore, WLoc, NumOT) for seasons without detailed data.

    Returns
    -------
    pd.DataFrame
        One row per (Season, TeamID) with aggregated performance metrics.
        Indexed by (Season, TeamID).
    """
    has_detail = "WFGM" in results_df.columns

    if has_detail:
        stats = _build_detailed_stats(results_df)
    else:
        stats = _build_compact_stats(results_df)

    # If compact results provided for additional seasons, merge them in
    if compact_results_df is not None:
        detail_seasons = set(results_df["Season"].unique())
        compact_only = compact_results_df[
            ~compact_results_df["Season"].isin(detail_seasons)
        ]
        if len(compact_only) > 0:
            compact_stats = _build_compact_stats(compact_only)
            stats = pd.concat([stats, compact_stats])

    return stats


def _build_compact_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build basic team stats from compact (score-only) results."""
    # Minimal unpivot for compact results (no box-score columns)
    rows = []
    for _, row in results_df.iterrows():
        season = row["Season"]
        daynum = row.get("DayNum", 0)
        wloc = row.get("WLoc", "N")
        loc_map = {"H": "A", "A": "H", "N": "N"}

        rows.append({
            "Season": season,
            "TeamID": row["WTeamID"],
            "Score": row["WScore"],
            "OppScore": row["LScore"],
            "Win": 1,
            "DayNum": daynum,
            "Location": wloc,
            "OppTeamID": row["LTeamID"],
        })
        rows.append({
            "Season": season,
            "TeamID": row["LTeamID"],
            "Score": row["LScore"],
            "OppScore": row["WScore"],
            "Win": 0,
            "DayNum": daynum,
            "Location": loc_map.get(wloc, "N"),
            "OppTeamID": row["WTeamID"],
        })

    team_games = pd.DataFrame(rows)
    team_games["PointDiff"] = team_games["Score"] - team_games["OppScore"]

    grouped = team_games.groupby(["Season", "TeamID"], sort=True)

    stats = pd.DataFrame({
        "games_played": grouped["Win"].count(),
        "wins": grouped["Win"].sum(),
        "points_scored_avg": grouped["Score"].mean(),
        "points_allowed_avg": grouped["OppScore"].mean(),
        "point_diff_avg": grouped["PointDiff"].mean(),
    })

    stats["losses"] = stats["games_played"] - stats["wins"]
    stats["win_pct"] = stats["wins"] / stats["games_played"]

    return stats


def _build_detailed_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build the full advanced stats suite from detailed box-score results."""
    team_games = _unpivot_results(results_df)

    # ------------------------------------------------------------------
    # Per-game derived columns (vectorised before grouping)
    # ------------------------------------------------------------------
    team_games["PointDiff"] = team_games["Score"] - team_games["OppScore"]
    team_games["TotalReb"] = team_games["OR"] + team_games["DR"]

    # Possessions (own and opponent)
    team_games["Poss"] = _estimate_possessions(
        team_games["FGA"], team_games["OR"], team_games["TO"], team_games["FTA"]
    )
    team_games["OppPoss"] = _estimate_possessions(
        team_games["OppFGA"], team_games["OppOR"], team_games["OppTO"],
        team_games["OppFTA"],
    )

    # --- Per-game advanced metrics ---

    # Effective FG% (offensive and defensive)
    team_games["eFG_pct"] = (
        (team_games["FGM"] + 0.5 * team_games["FGM3"]) / team_games["FGA"]
    )
    team_games["opp_eFG_pct"] = (
        (team_games["OppFGM"] + 0.5 * team_games["OppFGM3"]) / team_games["OppFGA"]
    )

    # True Shooting %
    team_games["ts_pct"] = team_games["Score"] / (
        2.0 * (team_games["FGA"] + 0.475 * team_games["FTA"])
    )

    # Assist-to-Turnover Ratio
    team_games["ast_to_ratio"] = team_games["Ast"] / team_games["TO"].replace(0, np.nan)

    # Offensive Rebound %: OR / (OR + OppDR)
    team_games["off_reb_pct"] = team_games["OR"] / (
        team_games["OR"] + team_games["OppDR"]
    )

    # Defensive Rebound %: DR / (DR + OppOR)
    team_games["def_reb_pct"] = team_games["DR"] / (
        team_games["DR"] + team_games["OppOR"]
    )

    # Free Throw Rate: FTA / FGA
    team_games["ft_rate"] = team_games["FTA"] / team_games["FGA"]

    # 3-Point Attempt Rate: FGA3 / FGA
    team_games["fg3_rate"] = team_games["FGA3"] / team_games["FGA"]

    # Turnover Rate: TO / Poss
    team_games["to_rate"] = team_games["TO"] / team_games["Poss"]

    # Steal Rate: Stl / OppPoss
    team_games["stl_rate"] = team_games["Stl"] / team_games["OppPoss"]

    # Block Rate: Blk / OppPoss
    team_games["blk_rate"] = team_games["Blk"] / team_games["OppPoss"]

    # Per-game offensive and defensive efficiency
    team_games["game_off_eff"] = team_games["Score"] / team_games["Poss"] * 100
    team_games["game_def_eff"] = team_games["OppScore"] / team_games["OppPoss"] * 100

    # ------------------------------------------------------------------
    # Aggregate to season level
    # ------------------------------------------------------------------
    grouped = team_games.groupby(["Season", "TeamID"], sort=True)

    stats = pd.DataFrame({
        "games_played": grouped["Win"].count(),
        "wins": grouped["Win"].sum(),
        # Shooting totals (used for season-level percentages, dropped later)
        "total_fgm": grouped["FGM"].sum(),
        "total_fga": grouped["FGA"].sum(),
        "total_fgm3": grouped["FGM3"].sum(),
        "total_fga3": grouped["FGA3"].sum(),
        "total_ftm": grouped["FTM"].sum(),
        "total_fta": grouped["FTA"].sum(),
        # Opponent shooting totals (for defensive eFG%)
        "total_opp_fgm": grouped["OppFGM"].sum(),
        "total_opp_fga": grouped["OppFGA"].sum(),
        "total_opp_fgm3": grouped["OppFGM3"].sum(),
        # Per-game averages
        "points_scored_avg": grouped["Score"].mean(),
        "points_allowed_avg": grouped["OppScore"].mean(),
        "point_diff_avg": grouped["PointDiff"].mean(),
        "off_reb_avg": grouped["OR"].mean(),
        "def_reb_avg": grouped["DR"].mean(),
        "total_reb_avg": grouped["TotalReb"].mean(),
        "ast_avg": grouped["Ast"].mean(),
        "to_avg": grouped["TO"].mean(),
        "stl_avg": grouped["Stl"].mean(),
        "blk_avg": grouped["Blk"].mean(),
        "pf_avg": grouped["PF"].mean(),
        "possessions_avg": grouped["Poss"].mean(),
        # Totals for efficiency calculations
        "total_points_scored": grouped["Score"].sum(),
        "total_points_allowed": grouped["OppScore"].sum(),
        "total_poss": grouped["Poss"].sum(),
        "total_opp_poss": grouped["OppPoss"].sum(),
        # Advanced per-game metrics (season averages)
        "eFG_pct": grouped["eFG_pct"].mean(),
        "opp_eFG_pct": grouped["opp_eFG_pct"].mean(),
        "ts_pct": grouped["ts_pct"].mean(),
        "ast_to_ratio": grouped["ast_to_ratio"].mean(),
        "off_reb_pct": grouped["off_reb_pct"].mean(),
        "def_reb_pct": grouped["def_reb_pct"].mean(),
        "ft_rate": grouped["ft_rate"].mean(),
        "fg3_rate": grouped["fg3_rate"].mean(),
        "to_rate": grouped["to_rate"].mean(),
        "stl_rate": grouped["stl_rate"].mean(),
        "blk_rate": grouped["blk_rate"].mean(),
        # Scoring consistency (lower = more consistent)
        "scoring_consistency": grouped["PointDiff"].std(),
    })

    # ------------------------------------------------------------------
    # Derived season-level stats
    # ------------------------------------------------------------------
    stats["losses"] = stats["games_played"] - stats["wins"]
    stats["win_pct"] = stats["wins"] / stats["games_played"]

    # Season-level shooting percentages (total makes / total attempts)
    stats["fg_pct"] = stats["total_fgm"] / stats["total_fga"]
    stats["fg3_pct"] = stats["total_fgm3"] / stats["total_fga3"]
    stats["ft_pct"] = stats["total_ftm"] / stats["total_fta"]

    # Season-level effective FG% (from totals, more stable than avg of per-game)
    stats["eFG_pct_season"] = (
        (stats["total_fgm"] + 0.5 * stats["total_fgm3"]) / stats["total_fga"]
    )
    stats["opp_eFG_pct_season"] = (
        (stats["total_opp_fgm"] + 0.5 * stats["total_opp_fgm3"]) / stats["total_opp_fga"]
    )

    # Efficiency: points per 100 possessions
    stats["off_efficiency"] = stats["total_points_scored"] / stats["total_poss"] * 100
    stats["def_efficiency"] = stats["total_points_allowed"] / stats["total_opp_poss"] * 100

    # Efficiency margin -- the single best predictor in college basketball
    stats["efficiency_margin"] = stats["off_efficiency"] - stats["def_efficiency"]

    # Drop intermediate total columns used only for ratio calculations
    stats.drop(
        columns=[
            "total_fgm", "total_fga", "total_fgm3", "total_fga3",
            "total_ftm", "total_fta",
            "total_opp_fgm", "total_opp_fga", "total_opp_fgm3",
            "total_points_scored", "total_points_allowed",
            "total_poss", "total_opp_poss",
        ],
        inplace=True,
    )

    return stats


# ---------------------------------------------------------------------------
# 1b. Advanced team stats (game-level situational analysis)
# ---------------------------------------------------------------------------

def build_advanced_team_stats(
    results_df: pd.DataFrame,
    compact_results_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute situational and game-context features requiring game-level analysis.

    Produces features like close-game win %, neutral-site win %, late-season
    form, blowout rate, and upset vulnerability -- indicators that capture
    "tournament DNA" beyond raw efficiency.

    Parameters
    ----------
    results_df : pd.DataFrame
        Detailed regular-season results (preferred) or compact results.
    compact_results_df : pd.DataFrame, optional
        Additional compact results for seasons without detailed data.

    Returns
    -------
    pd.DataFrame
        One row per (Season, TeamID), indexed by (Season, TeamID), with
        situational performance columns.
    """
    all_frames = []

    if "WFGM" in results_df.columns:
        team_games = _unpivot_results(results_df)
    else:
        # Build a minimal unpivoted frame from compact results
        team_games = _compact_to_team_games(results_df)

    all_frames.append(team_games)

    if compact_results_df is not None:
        detail_seasons = set(results_df["Season"].unique())
        compact_only = compact_results_df[
            ~compact_results_df["Season"].isin(detail_seasons)
        ]
        if len(compact_only) > 0:
            all_frames.append(_compact_to_team_games(compact_only))

    team_games = pd.concat(all_frames, ignore_index=True)
    team_games["PointDiff"] = team_games["Score"] - team_games["OppScore"]
    team_games["AbsPointDiff"] = team_games["PointDiff"].abs()

    grouped = team_games.groupby(["Season", "TeamID"], sort=True)

    # --- Close Game Win % (decided by 5 or fewer points) ---
    close_mask = team_games["AbsPointDiff"] <= 5
    close_games = team_games[close_mask].groupby(["Season", "TeamID"])
    close_win_pct = close_games["Win"].mean().rename("close_game_win_pct")
    close_game_count = close_games["Win"].count().rename("close_game_count")

    # --- Neutral Site Win % ---
    neutral_mask = team_games["Location"] == "N"
    neutral_games = team_games[neutral_mask].groupby(["Season", "TeamID"])
    neutral_win_pct = neutral_games["Win"].mean().rename("neutral_win_pct")

    # --- Away Win % ---
    away_mask = team_games["Location"] == "A"
    away_games = team_games[away_mask].groupby(["Season", "TeamID"])
    away_win_pct = away_games["Win"].mean().rename("away_win_pct")

    # --- Blowout Rate (won by 15+ points) ---
    blowout_wins = team_games[(team_games["Win"] == 1) & (team_games["PointDiff"] >= 15)]
    blowout_rate = (
        blowout_wins.groupby(["Season", "TeamID"])["Win"].count()
        / grouped["Win"].count()
    ).rename("blowout_rate")

    # --- Upset Vulnerability ---
    # Approximated as: % of losses where team took more shots (FGA) than opponent
    # indicating they had opportunities but failed to convert (poor efficiency)
    if "FGA" in team_games.columns and "OppFGA" in team_games.columns:
        loss_mask = team_games["Win"] == 0
        losses = team_games[loss_mask]
        upset_losses = losses[losses["FGA"] > losses["OppFGA"]]
        loss_count = losses.groupby(["Season", "TeamID"])["Win"].count()
        upset_count = upset_losses.groupby(["Season", "TeamID"])["Win"].count()
        upset_vulnerability = (upset_count / loss_count).rename("upset_vulnerability")
    else:
        upset_vulnerability = pd.Series(dtype=float, name="upset_vulnerability")

    # --- Late Season Form (last 10 games by DayNum) ---
    if "DayNum" in team_games.columns:
        late_form = _compute_late_season_form(team_games, n_games=10)
    else:
        late_form = pd.DataFrame()

    # --- Assemble the output ---
    adv_stats = pd.DataFrame(index=grouped.size().index)
    adv_stats.index.names = ["Season", "TeamID"]

    for series in [close_win_pct, close_game_count, neutral_win_pct,
                   away_win_pct, blowout_rate, upset_vulnerability]:
        if len(series) > 0:
            adv_stats = adv_stats.join(series, how="left")

    if len(late_form) > 0:
        adv_stats = adv_stats.join(late_form, how="left")

    # Fill NaN situational stats with reasonable defaults
    adv_stats["close_game_win_pct"] = adv_stats.get("close_game_win_pct", 0.5)
    adv_stats["close_game_count"] = adv_stats.get("close_game_count", 0).fillna(0)
    adv_stats["neutral_win_pct"] = adv_stats.get("neutral_win_pct", np.nan)
    adv_stats["away_win_pct"] = adv_stats.get("away_win_pct", np.nan)
    adv_stats["blowout_rate"] = adv_stats.get("blowout_rate", 0).fillna(0)
    adv_stats["upset_vulnerability"] = adv_stats.get("upset_vulnerability", 0).fillna(0)

    return adv_stats


def _compact_to_team_games(df: pd.DataFrame) -> pd.DataFrame:
    """Convert compact results into a team-game-level DataFrame (no box scores)."""
    loc_map = {"H": "A", "A": "H", "N": "N"}

    winners = pd.DataFrame({
        "Season": df["Season"].values,
        "TeamID": df["WTeamID"].values,
        "Score": df["WScore"].values,
        "OppScore": df["LScore"].values,
        "Win": 1,
        "DayNum": df["DayNum"].values if "DayNum" in df.columns else 0,
        "Location": df["WLoc"].values if "WLoc" in df.columns else "N",
        "OppTeamID": df["LTeamID"].values,
    })

    losers = pd.DataFrame({
        "Season": df["Season"].values,
        "TeamID": df["LTeamID"].values,
        "Score": df["LScore"].values,
        "OppScore": df["WScore"].values,
        "Win": 0,
        "DayNum": df["DayNum"].values if "DayNum" in df.columns else 0,
        "Location": (
            df["WLoc"].map(loc_map).values if "WLoc" in df.columns else "N"
        ),
        "OppTeamID": df["WTeamID"].values,
    })

    return pd.concat([winners, losers], ignore_index=True)


def _compute_late_season_form(
    team_games: pd.DataFrame,
    n_games: int = 10,
) -> pd.DataFrame:
    """Compute win % and point differential in the last N games of the season."""
    # Sort by DayNum within each team-season, then take the tail
    sorted_games = team_games.sort_values(
        ["Season", "TeamID", "DayNum"], ascending=True
    )
    late_games = sorted_games.groupby(["Season", "TeamID"]).tail(n_games)
    late_games = late_games.copy()
    late_games["PointDiff"] = late_games["Score"] - late_games["OppScore"]

    late_grouped = late_games.groupby(["Season", "TeamID"])
    late_form = pd.DataFrame({
        "late_season_win_pct": late_grouped["Win"].mean(),
        "late_season_point_diff": late_grouped["PointDiff"].mean(),
    })
    return late_form


# ---------------------------------------------------------------------------
# 2. Strength of Schedule (two-pass computation)
# ---------------------------------------------------------------------------

def add_strength_of_schedule(
    team_stats: pd.DataFrame,
    results_df: pd.DataFrame,
    compact_results_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add Strength of Schedule and opponent-adjusted efficiency features.

    SOS is defined as the average win_pct of all opponents faced during the
    season. Requires a two-pass approach: first compute basic team stats,
    then for each team-season, look up opponent win percentages.

    Parameters
    ----------
    team_stats : pd.DataFrame
        Indexed by (Season, TeamID). Must include ``win_pct`` and
        (optionally) ``efficiency_margin`` columns.
    results_df : pd.DataFrame
        Detailed regular-season results.
    compact_results_df : pd.DataFrame, optional
        Compact results to cover all seasons.

    Returns
    -------
    pd.DataFrame
        ``team_stats`` with additional columns: ``sos`` (strength of
        schedule) and ``opp_adj_efficiency`` (opponent-adjusted efficiency
        margin).
    """
    # Build opponent lookup from all available results
    all_matchups = []

    for df in [results_df, compact_results_df]:
        if df is None or len(df) == 0:
            continue
        winners = df[["Season", "WTeamID", "LTeamID"]].rename(
            columns={"WTeamID": "TeamID", "LTeamID": "OppTeamID"}
        )
        losers = df[["Season", "WTeamID", "LTeamID"]].rename(
            columns={"LTeamID": "TeamID", "WTeamID": "OppTeamID"}
        )
        all_matchups.append(winners)
        all_matchups.append(losers)

    matchups = pd.concat(all_matchups, ignore_index=True)

    # Get opponent win_pct from team_stats
    stats_reset = team_stats.reset_index()
    opp_win_pct = stats_reset[["Season", "TeamID", "win_pct"]].rename(
        columns={"TeamID": "OppTeamID", "win_pct": "opp_win_pct"}
    )

    matchups = matchups.merge(opp_win_pct, on=["Season", "OppTeamID"], how="left")

    # Average opponent win_pct per team-season = SOS
    sos = matchups.groupby(["Season", "TeamID"])["opp_win_pct"].mean().rename("sos")

    # Opponent-adjusted efficiency: efficiency_margin adjusted by avg opp efficiency
    result = team_stats.join(sos, how="left")

    if "efficiency_margin" in team_stats.columns:
        # Get opponent efficiency margins
        opp_eff = stats_reset[["Season", "TeamID"]].copy()
        if "efficiency_margin" in stats_reset.columns:
            opp_eff["opp_eff_margin"] = stats_reset["efficiency_margin"]
            opp_eff = opp_eff.rename(columns={"TeamID": "OppTeamID"})

            matchups_eff = matchups[["Season", "TeamID", "OppTeamID"]].merge(
                opp_eff, on=["Season", "OppTeamID"], how="left"
            )
            avg_opp_eff = (
                matchups_eff
                .groupby(["Season", "TeamID"])["opp_eff_margin"]
                .mean()
                .rename("avg_opp_eff_margin")
            )
            result = result.join(avg_opp_eff, how="left")
            result["opp_adj_efficiency"] = (
                result["efficiency_margin"] - result["avg_opp_eff_margin"]
            )
            result.drop(columns=["avg_opp_eff_margin"], inplace=True)

    return result


# ---------------------------------------------------------------------------
# 3. Seed features
# ---------------------------------------------------------------------------

def add_seed_features(
    team_stats: pd.DataFrame,
    seeds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge tournament seed info and extract the numeric seed (1-16).

    Parameters
    ----------
    team_stats : pd.DataFrame
        Indexed by (Season, TeamID).
    seeds_df : pd.DataFrame
        Must contain columns Season, TeamID, Seed (e.g. ``"W01"``).

    Returns
    -------
    pd.DataFrame
        ``team_stats`` with an additional ``seed_number`` column
        (NaN for teams not in the tournament).
    """
    seeds = seeds_df.copy()
    # Seed strings look like "W01", "X16a" -- extract the numeric part
    seeds["seed_number"] = (
        seeds["Seed"]
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )
    seeds = seeds[["Season", "TeamID", "seed_number"]]

    # Merge; team_stats index is (Season, TeamID)
    merged = team_stats.merge(
        seeds,
        how="left",
        left_index=True,
        right_on=["Season", "TeamID"],
    ).set_index(["Season", "TeamID"])

    return merged


# ---------------------------------------------------------------------------
# 4. Massey ordinals (ranking) features
# ---------------------------------------------------------------------------

def add_massey_features(
    team_stats: pd.DataFrame,
    ordinals_df: pd.DataFrame,
    day_cutoff: int = 133,
) -> pd.DataFrame:
    """Add aggregated ranking features from Massey ordinals.

    For each (Season, TeamID) pair the function selects the latest
    available ranking day on or before ``day_cutoff`` and then computes
    summary statistics across all ranking systems for that day.

    Parameters
    ----------
    team_stats : pd.DataFrame
        Indexed by (Season, TeamID).
    ordinals_df : pd.DataFrame
        Massey ordinals with columns Season, RankingDayNum, SystemName,
        TeamID, OrdinalRank.
    day_cutoff : int
        Maximum ``RankingDayNum`` to include (default 133, roughly the
        last day before the tournament).

    Returns
    -------
    pd.DataFrame
        ``team_stats`` with additional columns: ``rank_mean``,
        ``rank_median``, ``rank_best``, ``rank_std``.
    """
    # Filter to pre-tournament rankings only
    pre_tourney = ordinals_df.loc[
        ordinals_df["RankingDayNum"] <= day_cutoff
    ].copy()

    # For each (Season, TeamID, SystemName), keep only the latest day
    idx_latest = (
        pre_tourney
        .groupby(["Season", "TeamID", "SystemName"])["RankingDayNum"]
        .idxmax()
    )
    latest_ranks = pre_tourney.loc[idx_latest]

    # Aggregate across all ranking systems per team-season
    rank_stats = (
        latest_ranks
        .groupby(["Season", "TeamID"])["OrdinalRank"]
        .agg(
            rank_mean="mean",
            rank_median="median",
            rank_best="min",
            rank_std="std",
        )
    )

    merged = team_stats.merge(
        rank_stats,
        how="left",
        left_index=True,
        right_index=True,
    )

    return merged


def add_single_system_ranking(
    team_stats: pd.DataFrame,
    ordinals_df: pd.DataFrame,
    system_name: str = "POM",
    col_name: str = "kenpom_rank",
    day_cutoff: int = 133,
) -> pd.DataFrame:
    """Add a single ranking system's ordinal as a feature.

    Parameters
    ----------
    team_stats : pd.DataFrame
        Indexed by (Season, TeamID).
    ordinals_df : pd.DataFrame
        Massey ordinals with columns Season, RankingDayNum, SystemName,
        TeamID, OrdinalRank.
    system_name : str
        The ranking system to extract (default ``"POM"`` for KenPom).
    col_name : str
        Name for the new column (default ``"kenpom_rank"``).
    day_cutoff : int
        Maximum ``RankingDayNum`` to include (default 133).

    Returns
    -------
    pd.DataFrame
        ``team_stats`` with an additional column for the ranking.
    """
    sys_df = ordinals_df.loc[
        (ordinals_df["SystemName"] == system_name)
        & (ordinals_df["RankingDayNum"] <= day_cutoff)
    ].copy()

    # For each (Season, TeamID), keep only the latest available day
    idx_latest = sys_df.groupby(["Season", "TeamID"])["RankingDayNum"].idxmax()
    latest = sys_df.loc[idx_latest, ["Season", "TeamID", "OrdinalRank"]]
    latest = latest.rename(columns={"OrdinalRank": col_name})
    latest = latest.set_index(["Season", "TeamID"])

    return team_stats.merge(latest, how="left", left_index=True, right_index=True)


# ---------------------------------------------------------------------------
# 4b. SOS-adjusted efficiency features
# ---------------------------------------------------------------------------

def add_sos_adjusted_features(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Add SOS-adjusted efficiency features to team stats.

    Multiplies efficiency metrics by SOS so that identical raw efficiency
    numbers from a weak schedule are discounted relative to a strong schedule.

    Parameters
    ----------
    team_stats : pd.DataFrame
        Must contain ``sos`` and optionally ``efficiency_margin``,
        ``off_efficiency``, ``def_efficiency``.

    Returns
    -------
    pd.DataFrame
        ``team_stats`` with additional ``sos_adj_*`` columns.
    """
    result = team_stats.copy()

    if "efficiency_margin" in result.columns and "sos" in result.columns:
        result["sos_adj_eff_margin"] = result["efficiency_margin"] * result["sos"]

    return result


# ---------------------------------------------------------------------------
# 5. Single-matchup feature vector
# ---------------------------------------------------------------------------

def build_matchup_features(
    team_stats: pd.DataFrame,
    team1_id: int,
    team2_id: int,
    season: int,
) -> pd.Series:
    """Compute feature differences (team1 - team2) for a single matchup.

    Parameters
    ----------
    team_stats : pd.DataFrame
        Indexed by (Season, TeamID) with numeric stat columns.
    team1_id, team2_id : int
        The two team identifiers.
    season : int
        The season to look up.

    Returns
    -------
    pd.Series
        Difference of every numeric stat column (team1 minus team2).

    Raises
    ------
    KeyError
        If either team is not found in the given season.
    """
    stats1 = team_stats.loc[(season, team1_id)]
    stats2 = team_stats.loc[(season, team2_id)]
    numeric_cols = stats1.index[stats1.apply(lambda x: np.issubdtype(type(x), np.number))]
    return stats1[numeric_cols] - stats2[numeric_cols]


# ---------------------------------------------------------------------------
# 6. Tournament training data
# ---------------------------------------------------------------------------

def build_tournament_training_data(
    team_stats: pd.DataFrame,
    tourney_results: pd.DataFrame,
) -> pd.DataFrame:
    """Build a labelled training set from historical tournament games.

    For each game the lower TeamID is designated **TeamA** and the higher
    TeamID is **TeamB** (matching the Kaggle submission format). Features
    are the stat differences (TeamA - TeamB) and the target is 1 when
    TeamA won, 0 otherwise.

    Parameters
    ----------
    team_stats : pd.DataFrame
        Indexed by (Season, TeamID) with numeric stat columns.
    tourney_results : pd.DataFrame
        Tournament results with at least Season, WTeamID, LTeamID.

    Returns
    -------
    pd.DataFrame
        Feature-difference columns, a ``target`` column, and ``Season``.
    """
    games = tourney_results[["Season", "WTeamID", "LTeamID"]].copy()

    # Determine TeamA (lower ID) and TeamB (higher ID)
    games["TeamA"] = np.minimum(games["WTeamID"], games["LTeamID"])
    games["TeamB"] = np.maximum(games["WTeamID"], games["LTeamID"])
    games["target"] = (games["WTeamID"] == games["TeamA"]).astype(int)

    # Select only numeric columns from team_stats for differencing
    numeric_stats = team_stats.select_dtypes(include=[np.number])

    # Lookup stats for TeamA and TeamB via merge (vectorised)
    stats_a = (
        games[["Season", "TeamA"]]
        .merge(
            numeric_stats.reset_index(),
            how="left",
            left_on=["Season", "TeamA"],
            right_on=["Season", "TeamID"],
        )
        .drop(columns=["TeamID"])
    )

    stats_b = (
        games[["Season", "TeamB"]]
        .merge(
            numeric_stats.reset_index(),
            how="left",
            left_on=["Season", "TeamB"],
            right_on=["Season", "TeamID"],
        )
        .drop(columns=["TeamID"])
    )

    # Feature columns are everything except the join keys
    feat_cols = [c for c in stats_a.columns if c not in ("Season", "TeamA")]

    diff_df = pd.DataFrame(
        stats_a[feat_cols].values - stats_b[feat_cols].values,
        columns=feat_cols,
        index=games.index,
    )

    diff_df["target"] = games["target"].values
    diff_df["Season"] = games["Season"].values

    return diff_df


# ---------------------------------------------------------------------------
# 7. Rolling window features
# ---------------------------------------------------------------------------

# Stats computed over each rolling window
_ROLLING_GAME_METRICS = [
    "Win", "PointDiff", "Score", "OppScore",
]

_ROLLING_ADVANCED_METRICS = [
    "eFG_pct", "ts_pct", "to_rate", "ast_to_ratio",
    "off_reb_pct", "def_reb_pct", "game_off_eff", "game_def_eff",
]


def _prepare_team_game_rows(
    results_df: pd.DataFrame,
    compact_results_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Produce a unified, sorted team-game DataFrame with per-game metrics.

    Returns one row per (Season, TeamID, DayNum, game_index) sorted
    chronologically within each team-season. Includes both basic and
    advanced per-game stats when box-score data is available.
    """
    has_detail = "WFGM" in results_df.columns

    if has_detail:
        team_games = _unpivot_results(results_df)
    else:
        team_games = _compact_to_team_games(results_df)

    # Merge compact-only seasons
    if compact_results_df is not None:
        detail_seasons = set(results_df["Season"].unique())
        compact_only = compact_results_df[
            ~compact_results_df["Season"].isin(detail_seasons)
        ]
        if len(compact_only) > 0:
            team_games = pd.concat(
                [team_games, _compact_to_team_games(compact_only)],
                ignore_index=True,
            )

    # Derived per-game columns
    team_games["PointDiff"] = team_games["Score"] - team_games["OppScore"]

    if has_detail:
        team_games["Poss"] = _estimate_possessions(
            team_games["FGA"], team_games["OR"], team_games["TO"], team_games["FTA"]
        )
        team_games["OppPoss"] = _estimate_possessions(
            team_games["OppFGA"], team_games["OppOR"], team_games["OppTO"],
            team_games["OppFTA"],
        )
        team_games["eFG_pct"] = (
            (team_games["FGM"] + 0.5 * team_games["FGM3"]) / team_games["FGA"]
        )
        team_games["ts_pct"] = team_games["Score"] / (
            2.0 * (team_games["FGA"] + 0.475 * team_games["FTA"])
        )
        team_games["ast_to_ratio"] = (
            team_games["Ast"] / team_games["TO"].replace(0, np.nan)
        )
        team_games["off_reb_pct"] = team_games["OR"] / (
            team_games["OR"] + team_games["OppDR"]
        )
        team_games["def_reb_pct"] = team_games["DR"] / (
            team_games["DR"] + team_games["OppOR"]
        )
        team_games["to_rate"] = team_games["TO"] / team_games["Poss"]
        team_games["game_off_eff"] = (
            team_games["Score"] / team_games["Poss"] * 100
        )
        team_games["game_def_eff"] = (
            team_games["OppScore"] / team_games["OppPoss"] * 100
        )

    # Sort chronologically within each team-season
    team_games = team_games.sort_values(
        ["Season", "TeamID", "DayNum"]
    ).reset_index(drop=True)

    return team_games


def compute_rolling_window_stats(
    results_df: pd.DataFrame,
    compact_results_df: pd.DataFrame | None = None,
    windows: tuple[int, ...] = (5, 7, 10),
) -> pd.DataFrame:
    """Compute rolling window aggregates for every team-game row.

    For each game, computes features using *only* the preceding N games
    within the same season (no future leakage). The first N games of a
    season will have NaN for that window size.

    Parameters
    ----------
    results_df : pd.DataFrame
        Detailed or compact game results.
    compact_results_df : pd.DataFrame, optional
        Additional compact results.
    windows : tuple of int
        Rolling window sizes (default: 5, 7, 10).

    Returns
    -------
    pd.DataFrame
        Same rows as input team-game data, with additional columns for
        each (metric, window) combination, e.g. ``Win_roll5``,
        ``PointDiff_roll7``, ``eFG_pct_roll10``.
        Also includes ``eff_margin_rollN`` (off_eff - def_eff per window).
    """
    team_games = _prepare_team_game_rows(results_df, compact_results_df)

    has_detail = "eFG_pct" in team_games.columns
    metrics = list(_ROLLING_GAME_METRICS)
    if has_detail:
        metrics += _ROLLING_ADVANCED_METRICS

    # Group by team-season and compute rolling means (shifted to exclude current game)
    grouped = team_games.groupby(["Season", "TeamID"], sort=False)

    for w in windows:
        for metric in metrics:
            col_name = f"{metric}_roll{w}"
            team_games[col_name] = grouped[metric].transform(
                lambda s: s.shift(1).rolling(window=w, min_periods=w).mean()
            )

        # Scoring consistency (std of PointDiff) per window
        team_games[f"scoring_consistency_roll{w}"] = grouped["PointDiff"].transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=w).std()
        )

        # Efficiency margin per window
        if has_detail:
            team_games[f"eff_margin_roll{w}"] = (
                team_games[f"game_off_eff_roll{w}"]
                - team_games[f"game_def_eff_roll{w}"]
            )

    return team_games


def build_rolling_training_data(
    results_df: pd.DataFrame,
    compact_results_df: pd.DataFrame | None = None,
    team_stats: pd.DataFrame | None = None,
    windows: tuple[int, ...] = (5, 7, 10),
    min_window: int | None = None,
) -> pd.DataFrame:
    """Build matchup-level training data from ALL games using rolling window features.

    Each game becomes one training row. The lower TeamID is TeamA, higher is
    TeamB (matching Kaggle format). Features are the rolling-window stat
    differences (TeamA - TeamB) plus optional season-level static features
    (seed, SOS, Massey ranks) from ``team_stats``.

    Parameters
    ----------
    results_df : pd.DataFrame
        Detailed or compact game results (regular season + tournament).
    compact_results_df : pd.DataFrame, optional
        Additional compact results for extra seasons.
    team_stats : pd.DataFrame, optional
        Season-level static features indexed by (Season, TeamID).
        Columns like ``seed_number``, ``sos``, ``rank_mean``, etc.
        These are added as-is (not differenced) for both teams.
    windows : tuple of int
        Rolling window sizes.
    min_window : int, optional
        Drop rows where the largest window has NaN (i.e. team hasn't
        played enough games yet). Defaults to max(windows).

    Returns
    -------
    pd.DataFrame
        Columns: rolling feature diffs, static feature diffs, ``target``,
        ``Season``, ``DayNum``, ``is_tourney``.
    """
    if min_window is None:
        min_window = max(windows)

    # Get rolling stats per team-game
    team_games = compute_rolling_window_stats(
        results_df, compact_results_df, windows
    )

    # Identify rolling feature columns
    roll_cols = [c for c in team_games.columns if "_roll" in c]

    # Reconstruct game-level pairs from the team_games rows
    # Each original game produces two rows (one per team). We need to pair them.
    # Use (Season, DayNum, sorted team pair) as join key.

    # Add game identifier: for each row, the game is between TeamID and OppTeamID
    team_games["GameTeamA"] = np.minimum(
        team_games["TeamID"], team_games["OppTeamID"]
    )
    team_games["GameTeamB"] = np.maximum(
        team_games["TeamID"], team_games["OppTeamID"]
    )

    # Split into TeamA perspective and TeamB perspective
    is_team_a = team_games["TeamID"] == team_games["GameTeamA"]

    team_a_rows = team_games[is_team_a].copy()
    team_b_rows = team_games[~is_team_a].copy()

    # Join on game identifier
    merge_keys = ["Season", "DayNum", "GameTeamA", "GameTeamB"]
    suffix_a = "_A"
    suffix_b = "_B"

    # Keep only columns we need for the merge
    keep_cols = merge_keys + roll_cols + ["Win", "TeamID"]

    merged = team_a_rows[keep_cols].merge(
        team_b_rows[keep_cols],
        on=merge_keys,
        suffixes=(suffix_a, suffix_b),
    )

    # Target: 1 if TeamA (lower ID) won
    merged["target"] = merged[f"Win{suffix_a}"].astype(int)
    merged["Season"] = merged["Season"]
    merged["DayNum"] = merged["DayNum"]

    # Compute differences: TeamA - TeamB for rolling features
    diff_data = {}
    for col in roll_cols:
        diff_data[col] = merged[f"{col}{suffix_a}"].values - merged[f"{col}{suffix_b}"].values

    diff_df = pd.DataFrame(diff_data, index=merged.index)
    diff_df["target"] = merged["target"].values
    diff_df["Season"] = merged["Season"].values
    diff_df["DayNum"] = merged["DayNum"].values
    diff_df["TeamA"] = merged["GameTeamA"].values
    diff_df["TeamB"] = merged["GameTeamB"].values

    # Mark tournament games (DayNum >= 134 is typically tournament)
    diff_df["is_tourney"] = diff_df["DayNum"] >= 134

    # Add static season-level features as differences if provided
    if team_stats is not None:
        static_numeric = team_stats.select_dtypes(include=[np.number])
        static_reset = static_numeric.reset_index()

        # Merge static features for TeamA
        static_a = diff_df[["Season", "TeamA"]].merge(
            static_reset,
            how="left",
            left_on=["Season", "TeamA"],
            right_on=["Season", "TeamID"],
        )
        # Merge static features for TeamB
        static_b = diff_df[["Season", "TeamB"]].merge(
            static_reset,
            how="left",
            left_on=["Season", "TeamB"],
            right_on=["Season", "TeamID"],
        )

        static_cols = [c for c in static_a.columns
                       if c not in ("Season", "TeamID", "TeamA", "TeamB")]

        for col in static_cols:
            diff_df[f"static_{col}"] = (
                static_a[col].values - static_b[col].values
            )

    # Drop rows where the largest window hasn't filled yet
    largest_window_col = f"Win_roll{min_window}"
    if largest_window_col in diff_df.columns:
        # Both teams need valid rolling stats
        both_valid = diff_df[largest_window_col].notna()
        diff_df = diff_df[both_valid].reset_index(drop=True)

    return diff_df


def get_team_rolling_snapshot(
    results_df: pd.DataFrame,
    compact_results_df: pd.DataFrame | None = None,
    season: int = 2025,
    windows: tuple[int, ...] = (5, 7, 10),
) -> pd.DataFrame:
    """Get the latest rolling window stats for each team in a given season.

    Used for generating predictions: takes each team's most recent N-game
    rolling stats as their "current form" snapshot.

    Parameters
    ----------
    results_df : pd.DataFrame
        Game results including the target season.
    compact_results_df : pd.DataFrame, optional
        Additional compact results.
    season : int
        Season to extract snapshots for.
    windows : tuple of int
        Rolling window sizes.

    Returns
    -------
    pd.DataFrame
        One row per TeamID with rolling window features, indexed by TeamID.
    """
    team_games = compute_rolling_window_stats(
        results_df, compact_results_df, windows
    )

    # Filter to target season
    season_games = team_games[team_games["Season"] == season].copy()

    # For each team, take the last game's rolling stats
    roll_cols = [c for c in season_games.columns if "_roll" in c]
    keep_cols = ["Season", "TeamID", "DayNum"] + roll_cols

    latest = (
        season_games[keep_cols]
        .sort_values("DayNum")
        .groupby("TeamID")
        .last()
    )

    return latest


# ---------------------------------------------------------------------------
# 8. Elo rating system
# ---------------------------------------------------------------------------

def compute_elo_ratings(
    results_df: pd.DataFrame,
    k_factor: float = 20.0,
    home_advantage: float = 0.0,
    season_carryover: float = 0.75,
    initial_elo: float = 1500.0,
    margin_factor: bool = True,
) -> pd.DataFrame:
    """Compute Elo ratings with margin-of-victory adjustment.

    Uses a log-based MOV multiplier with autocorrelation dampening:
        MOV_mult = log(MOV + 1) * (2.2 / (2.2 + 0.001 * elo_diff))

    The log dampens blowout effects. The denominator reduces updates
    when a large Elo favorite wins big (prevents runaway ratings).

    Parameters
    ----------
    results_df : DataFrame with columns Season, DayNum, WTeamID, LTeamID, WScore, LScore
        Game results in chronological order (compact or detailed format).
    k_factor : float
        Base K-factor. Multiplied by MOV multiplier per game.
    home_advantage : float
        Elo points added for home team. 0 for neutral-site predictions.
    season_carryover : float
        Between 0 and 1. How much of prior season Elo carries over.
        0 = full reset to initial_elo, 1 = no regression.
    initial_elo : float
        Starting Elo for new teams.
    margin_factor : bool
        If True, scale K by margin of victory. If False, flat K.

    Returns
    -------
    DataFrame with columns: Season, DayNum, TeamID, Elo
        One row per team per game, with the Elo BEFORE that game was played.
        (shift-by-one to prevent leakage, matching the rolling window approach)
    """
    # Sort games chronologically
    games = results_df[["Season", "DayNum", "WTeamID", "LTeamID",
                         "WScore", "LScore"]].copy()
    if "WLoc" in results_df.columns:
        games["WLoc"] = results_df["WLoc"].values
    else:
        games["WLoc"] = "N"
    games = games.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    # Current Elo ratings -- carried across seasons
    elo: dict[int, float] = {}
    # Records: pre-game Elo for each team in each game
    records: list[dict] = []

    prev_season = None

    for _, row in games.iterrows():
        season = int(row["Season"])
        daynum = int(row["DayNum"])
        w_id = int(row["WTeamID"])
        l_id = int(row["LTeamID"])
        w_score = float(row["WScore"])
        l_score = float(row["LScore"])
        w_loc = row["WLoc"]

        # Season transition: regress all ratings toward mean
        if prev_season is not None and season != prev_season:
            for tid in list(elo.keys()):
                elo[tid] = initial_elo + season_carryover * (elo[tid] - initial_elo)
        prev_season = season

        # Initialize new teams
        if w_id not in elo:
            elo[w_id] = initial_elo
        if l_id not in elo:
            elo[l_id] = initial_elo

        # Record PRE-GAME Elo (before update)
        w_elo_pre = elo[w_id]
        l_elo_pre = elo[l_id]

        records.append({
            "Season": season, "DayNum": daynum,
            "TeamID": w_id, "Elo": w_elo_pre,
        })
        records.append({
            "Season": season, "DayNum": daynum,
            "TeamID": l_id, "Elo": l_elo_pre,
        })

        # Elo difference (winner's perspective, with home court)
        elo_diff_w = w_elo_pre - l_elo_pre
        if w_loc == "H":
            elo_diff_w += home_advantage
        elif w_loc == "A":
            elo_diff_w -= home_advantage
        # Neutral: no adjustment

        # Expected win probability (standard logistic)
        expected_w = 1.0 / (1.0 + 10.0 ** (-elo_diff_w / 400.0))
        expected_l = 1.0 - expected_w

        # Margin of victory multiplier
        score_diff = abs(w_score - l_score)
        elo_diff_abs = abs(elo_diff_w)
        if margin_factor:
            mov_mult = np.log(score_diff + 1) * (2.2 / (2.2 + 0.001 * elo_diff_abs))
        else:
            mov_mult = 1.0

        # Update ratings
        k_adj = k_factor * mov_mult
        elo[w_id] = w_elo_pre + k_adj * (1.0 - expected_w)
        elo[l_id] = l_elo_pre + k_adj * (0.0 - expected_l)

    return pd.DataFrame(records)


def get_team_elo_snapshot(elo_df: pd.DataFrame, season: int) -> pd.Series:
    """Get each team's final Elo for a given season (last recorded Elo).

    Parameters
    ----------
    elo_df : DataFrame
        Output of ``compute_elo_ratings`` with columns Season, DayNum, TeamID, Elo.
    season : int
        Season to extract final Elos for.

    Returns
    -------
    Series indexed by TeamID with Elo values.
    """
    season_data = elo_df[elo_df["Season"] == season]
    if len(season_data) == 0:
        return pd.Series(dtype=float, name="Elo")
    # Take the last recorded Elo per team (highest DayNum)
    latest = (
        season_data
        .sort_values("DayNum")
        .groupby("TeamID")["Elo"]
        .last()
    )
    return latest
