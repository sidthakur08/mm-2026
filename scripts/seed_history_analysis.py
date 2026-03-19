"""
Seed History Analysis for March Madness 2026
=============================================
Compares historical tournament seed performance (2003-2025) with our 2026 model
predictions to identify smart upset picks and build an optimized bracket.

Sections:
  1. Historical Seed Win Rates by Round
  2. Historical Seed-vs-Seed Matchup Win Rates (R64)
  3. Upset Frequency by Round
  4. Model vs History Comparison Table (2026 R64)
  5. Smart Upset Recommendations
  6. Full 63-Game Bracket Recommendation
"""

import pandas as pd
import numpy as np
import json
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:.1f}" if abs(x) > 1 else f"{x:.3f}")

BASE = Path("D:/Code/mm-2026")

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def parse_seed(seed_str: str) -> tuple:
    """Parse seed string like 'W01', 'X16a' -> (region, seed_number)."""
    m = re.match(r"([WXYZ])(\d{2})", seed_str)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def load_data():
    """Load all required data files."""
    results = pd.read_csv(BASE / "data" / "MNCAATourneyCompactResults.csv")
    seeds = pd.read_csv(BASE / "data" / "MNCAATourneySeeds.csv")

    with open(BASE / "website" / "public" / "data" / "predictions_men.json") as f:
        predictions_raw = json.load(f)

    with open(BASE / "website" / "public" / "data" / "seeds.json") as f:
        seeds_2026_raw = json.load(f)["men"]

    with open(BASE / "website" / "public" / "data" / "teams.json") as f:
        teams_raw = json.load(f)["men"]

    # Parse seeds: keep only the numeric seed (strip play-in letters)
    seeds["Region"] = seeds["Seed"].apply(lambda s: s[0])
    seeds["SeedNum"] = seeds["Seed"].apply(lambda s: int(re.search(r"\d+", s).group()))

    # Build team name lookup
    team_names = {int(k): v for k, v in teams_raw.items()}

    # Build 2026 seed lookup: team_id -> {region, seed, seedStr}
    seeds_2026 = {}
    for tid, info in seeds_2026_raw.items():
        seeds_2026[int(tid)] = info

    return results, seeds, predictions_raw, seeds_2026, team_names


# ---------------------------------------------------------------------------
# Round Assignment
# ---------------------------------------------------------------------------

# DayNum mapping (post-2011 format with First Four):
#   134-135: Play-in / First Four
#   136-137: Round of 64
#   138-139: Round of 32
#   143-144: Sweet 16
#   145-146: Elite 8
#   152:     Final Four
#   154:     Championship
ROUND_MAP = {
    136: "R64", 137: "R64",
    138: "R32", 139: "R32",
    143: "S16", 144: "S16",
    145: "E8", 146: "E8",
    152: "F4",
    154: "Championship",
}
PLAY_IN_DAYS = {134, 135}


def assign_rounds(results: pd.DataFrame) -> pd.DataFrame:
    """Add a Round column based on DayNum."""
    results = results.copy()
    results["Round"] = results["DayNum"].map(ROUND_MAP)
    # Mark play-in games
    results.loc[results["DayNum"].isin(PLAY_IN_DAYS), "Round"] = "PlayIn"
    return results


# ---------------------------------------------------------------------------
# Section 1: Historical Seed Win Rates by Round
# ---------------------------------------------------------------------------

ROUND_ORDER = ["R64", "R32", "S16", "E8", "F4", "Championship"]


def compute_seed_round_win_rates(results: pd.DataFrame, seeds: pd.DataFrame,
                                  start_season=2003, end_season=2025):
    """
    For each seed 1-16, compute win rates in each round.
    Win rate in R64 = games won / games played in R64 by that seed.
    Win rate in R32 = fraction of teams with that seed that REACHED and WON in R32, etc.
    """
    # Filter seasons
    res = results[(results.Season >= start_season) & (results.Season <= end_season)].copy()
    res = assign_rounds(res)
    res = res[res.Round.isin(ROUND_ORDER)]

    sd = seeds[(seeds.Season >= start_season) & (seeds.Season <= end_season)].copy()

    # Merge seeds for winner and loser
    res = res.merge(sd[["Season", "TeamID", "SeedNum"]],
                    left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"],
                    how="left").rename(columns={"SeedNum": "WSeed"}).drop(columns=["TeamID"])
    res = res.merge(sd[["Season", "TeamID", "SeedNum"]],
                    left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"],
                    how="left").rename(columns={"SeedNum": "LSeed"}).drop(columns=["TeamID"])

    # Drop rows where seeds are missing (play-in teams that lost before R64 etc.)
    res = res.dropna(subset=["WSeed", "LSeed"])
    res["WSeed"] = res["WSeed"].astype(int)
    res["LSeed"] = res["LSeed"].astype(int)

    # Count wins and losses by seed & round
    wins = res.groupby(["Round", "WSeed"]).size().reset_index(name="Wins")
    wins.rename(columns={"WSeed": "Seed"}, inplace=True)
    losses = res.groupby(["Round", "LSeed"]).size().reset_index(name="Losses")
    losses.rename(columns={"LSeed": "Seed"}, inplace=True)

    # Merge
    stats = wins.merge(losses, on=["Round", "Seed"], how="outer").fillna(0)
    stats["Games"] = stats["Wins"] + stats["Losses"]
    stats["WinPct"] = stats["Wins"] / stats["Games"]

    # Pivot
    pivot = stats.pivot_table(index="Seed", columns="Round", values="WinPct").reindex(
        columns=ROUND_ORDER)

    # Also compute games played per round for context
    pivot_games = stats.pivot_table(index="Seed", columns="Round", values="Games").reindex(
        columns=ROUND_ORDER)

    return pivot, pivot_games


# ---------------------------------------------------------------------------
# Section 2: Seed-vs-Seed Matchup Win Rates (R64)
# ---------------------------------------------------------------------------

R64_MATCHUPS = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]


def compute_r64_matchup_rates(results: pd.DataFrame, seeds: pd.DataFrame,
                               start_season=2003, end_season=2025):
    """Historical win rates for standard R64 matchups."""
    res = results[(results.Season >= start_season) & (results.Season <= end_season)].copy()
    res = assign_rounds(res)
    res = res[res.Round == "R64"]

    sd = seeds[(seeds.Season >= start_season) & (seeds.Season <= end_season)]

    # Merge seeds
    res = res.merge(sd[["Season", "TeamID", "SeedNum"]],
                    left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"],
                    how="left").rename(columns={"SeedNum": "WSeed"}).drop(columns=["TeamID"])
    res = res.merge(sd[["Season", "TeamID", "SeedNum"]],
                    left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"],
                    how="left").rename(columns={"SeedNum": "LSeed"}).drop(columns=["TeamID"])
    res = res.dropna(subset=["WSeed", "LSeed"])
    res["WSeed"] = res["WSeed"].astype(int)
    res["LSeed"] = res["LSeed"].astype(int)

    # For each game, identify higher seed (lower number) and lower seed
    res["HighSeed"] = res[["WSeed", "LSeed"]].min(axis=1)
    res["LowSeed"] = res[["WSeed", "LSeed"]].max(axis=1)
    res["HighSeedWon"] = (res["WSeed"] == res["HighSeed"]).astype(int)

    # Group by matchup
    matchup_stats = []
    for hi, lo in R64_MATCHUPS:
        mask = (res["HighSeed"] == hi) & (res["LowSeed"] == lo)
        subset = res[mask]
        games = len(subset)
        hi_wins = subset["HighSeedWon"].sum()
        matchup_stats.append({
            "Matchup": f"{hi} vs {lo}",
            "HighSeed": hi,
            "LowSeed": lo,
            "Games": games,
            "HighSeedWins": hi_wins,
            "HighSeedWinPct": hi_wins / games if games > 0 else np.nan,
        })

    return pd.DataFrame(matchup_stats)


# ---------------------------------------------------------------------------
# Section 3: Upset Frequency by Round
# ---------------------------------------------------------------------------

def compute_upset_frequency(results: pd.DataFrame, seeds: pd.DataFrame,
                             start_season=2003, end_season=2025):
    """How often does the lower-seeded team win, per round?"""
    res = results[(results.Season >= start_season) & (results.Season <= end_season)].copy()
    res = assign_rounds(res)
    res = res[res.Round.isin(ROUND_ORDER)]

    sd = seeds[(seeds.Season >= start_season) & (seeds.Season <= end_season)]

    res = res.merge(sd[["Season", "TeamID", "SeedNum"]],
                    left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"],
                    how="left").rename(columns={"SeedNum": "WSeed"}).drop(columns=["TeamID"])
    res = res.merge(sd[["Season", "TeamID", "SeedNum"]],
                    left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"],
                    how="left").rename(columns={"SeedNum": "LSeed"}).drop(columns=["TeamID"])
    res = res.dropna(subset=["WSeed", "LSeed"])
    res["WSeed"] = res["WSeed"].astype(int)
    res["LSeed"] = res["LSeed"].astype(int)

    # An upset = winner has a HIGHER seed number (worse seed) than loser
    # Exclude games where seeds are equal
    res = res[res["WSeed"] != res["LSeed"]]
    res["Upset"] = (res["WSeed"] > res["LSeed"]).astype(int)

    upset_by_round = res.groupby("Round").agg(
        Games=("Upset", "count"),
        Upsets=("Upset", "sum")
    ).reindex(ROUND_ORDER)
    upset_by_round["UpsetPct"] = upset_by_round["Upsets"] / upset_by_round["Games"]

    return upset_by_round


# ---------------------------------------------------------------------------
# Section 4 & 5: Model vs History for 2026 R64 + Smart Upsets
# ---------------------------------------------------------------------------

def get_2026_r64_matchups(seeds_2026: dict, predictions_raw: dict, team_names: dict):
    """
    Build the 32 R64 matchups for 2026 using bracket structure.
    R64 matchups follow the standard: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
    within each region.
    """
    # Organize seeds by region
    regions = {}
    for tid, info in seeds_2026.items():
        region = info["region"]
        seed = info["seed"]
        play_in = info.get("playIn", "")
        if region not in regions:
            regions[region] = {}
        # For play-in teams, store both under the same seed (we pick the first or handle later)
        if seed not in regions[region]:
            regions[region][seed] = []
        regions[region][seed].append(tid)

    # Standard R64 pairings (seed vs seed)
    r64_pairings = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

    matchups = []
    for region in ["W", "X", "Y", "Z"]:
        for hi_seed, lo_seed in r64_pairings:
            hi_teams = regions[region].get(hi_seed, [])
            lo_teams = regions[region].get(lo_seed, [])

            # For play-in seeds, we may have 2 teams -- use the first (play-in winner unknown)
            # We'll look up both combinations and average if needed
            if len(hi_teams) == 0 or len(lo_teams) == 0:
                continue

            # Get model prediction for each possible pairing and average
            probs = []
            for ht in hi_teams:
                for lt in lo_teams:
                    key_a = f"{min(ht, lt)}_{max(ht, lt)}"
                    if key_a in predictions_raw:
                        p = predictions_raw[key_a]
                        # p = probability that the smaller ID wins
                        if ht < lt:
                            probs.append(p)  # prob higher seed wins
                        else:
                            probs.append(1 - p)  # prob higher seed wins
                    else:
                        pass

            model_prob_hi = np.mean(probs) if probs else np.nan

            # Use first team for display purposes
            hi_tid = hi_teams[0]
            lo_tid = lo_teams[0]

            matchups.append({
                "Region": region,
                "HighSeed": hi_seed,
                "LowSeed": lo_seed,
                "HighTeamID": hi_tid,
                "LowTeamID": lo_tid,
                "HighTeamName": team_names.get(hi_tid, str(hi_tid)),
                "LowTeamName": team_names.get(lo_tid, str(lo_tid)),
                "ModelProbHighSeed": model_prob_hi,
                "Matchup": f"{hi_seed} vs {lo_seed}",
                "IsPlayIn": len(hi_teams) > 1 or len(lo_teams) > 1,
            })

    return pd.DataFrame(matchups)


def build_model_vs_history(matchups_2026: pd.DataFrame, historical_rates: pd.DataFrame):
    """Add historical base rates to each 2026 matchup and compute delta."""
    hist_map = historical_rates.set_index("Matchup")["HighSeedWinPct"].to_dict()

    matchups_2026 = matchups_2026.copy()
    matchups_2026["HistoricalWinPct"] = matchups_2026["Matchup"].map(hist_map)
    matchups_2026["Delta"] = matchups_2026["ModelProbHighSeed"] - matchups_2026["HistoricalWinPct"]

    # Blended: 70% model + 30% historical
    matchups_2026["BlendedProb"] = (
        0.7 * matchups_2026["ModelProbHighSeed"] + 0.3 * matchups_2026["HistoricalWinPct"]
    )

    return matchups_2026


# ---------------------------------------------------------------------------
# Section 6: Full Bracket Simulation
# ---------------------------------------------------------------------------

REGION_NAMES = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}

# Standard bracket structure within a region (seed matchups per round)
# R64 bracket order determines the R32/S16/E8 tree:
#   R64: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
#   R32: W(1v16) vs W(8v9), W(5v12) vs W(4v13), W(6v11) vs W(3v14), W(7v10) vs W(2v15)
#   S16: W(top_half) vs W(bottom_of_top), W(other_half) vs W(other)
#   E8:  final two in region


def get_prediction(tid_a: int, tid_b: int, predictions_raw: dict) -> float:
    """Get model probability that tid_a beats tid_b."""
    lo, hi = min(tid_a, tid_b), max(tid_a, tid_b)
    key = f"{lo}_{hi}"
    if key in predictions_raw:
        p_lo_wins = predictions_raw[key]
        return p_lo_wins if tid_a == lo else (1 - p_lo_wins)
    return 0.5  # fallback


def simulate_full_bracket(seeds_2026: dict, predictions_raw: dict, team_names: dict,
                           historical_rates: pd.DataFrame, hist_upset_by_round: pd.DataFrame):
    """
    Simulate a full 63-game bracket using blended probabilities for R64,
    and expected-value optimization for later rounds.
    """
    # Build region brackets
    hist_map = historical_rates.set_index("Matchup")["HighSeedWinPct"].to_dict()

    # Organize teams by region and seed
    regions = {}
    for tid, info in seeds_2026.items():
        region = info["region"]
        seed = info["seed"]
        play_in = info.get("playIn", "")
        if region not in regions:
            regions[region] = {}
        if seed not in regions[region]:
            regions[region][seed] = []
        regions[region][seed].append(tid)

    # For play-in slots, pick the team with higher model strength (average prediction vs field)
    for region in regions:
        for seed in regions[region]:
            if len(regions[region][seed]) > 1:
                teams = regions[region][seed]
                # Pick team with better average prediction against the other seed in the matchup
                # Simple heuristic: pick team with higher average model probability vs all others
                best_tid = teams[0]
                best_avg = -1
                for tid in teams:
                    probs_vs_others = []
                    for other_region in regions:
                        for other_seed in regions[other_region]:
                            for other_tid in regions[other_region][other_seed]:
                                if other_tid != tid:
                                    p = get_prediction(tid, other_tid, predictions_raw)
                                    probs_vs_others.append(p)
                    avg = np.mean(probs_vs_others) if probs_vs_others else 0
                    if avg > best_avg:
                        best_avg = avg
                        best_tid = tid
                regions[region][seed] = [best_tid]

    # Standard R64 bracket order within a region
    r64_order = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

    # F4 bracket: W vs X, Y vs Z  (standard)
    f4_matchups_regions = [("W", "X"), ("Y", "Z")]

    bracket_results = []  # list of dicts for each game
    region_winners = {}

    for region in ["W", "X", "Y", "Z"]:
        # --- R64 ---
        r64_winners = []
        for hi_seed, lo_seed in r64_order:
            hi_tid = regions[region][hi_seed][0]
            lo_tid = regions[region][lo_seed][0]

            # Model probability that higher seed wins
            model_p = get_prediction(hi_tid, lo_tid, predictions_raw)

            # Historical base rate
            matchup_key = f"{hi_seed} vs {lo_seed}"
            hist_p = hist_map.get(matchup_key, model_p)

            # Blended probability (70% model, 30% history)
            blended_p = 0.7 * model_p + 0.3 * hist_p

            # Pick winner: the team with > 50% blended probability
            if blended_p >= 0.5:
                winner_tid = hi_tid
                winner_seed = hi_seed
                win_prob = blended_p
            else:
                winner_tid = lo_tid
                winner_seed = lo_seed
                win_prob = 1 - blended_p

            is_upset = (winner_seed == lo_seed)
            r64_winners.append((winner_tid, winner_seed, win_prob))
            bracket_results.append({
                "Round": "R64",
                "Region": region,
                "Team1Seed": hi_seed,
                "Team1Name": team_names.get(hi_tid, str(hi_tid)),
                "Team1ID": hi_tid,
                "Team2Seed": lo_seed,
                "Team2Name": team_names.get(lo_tid, str(lo_tid)),
                "Team2ID": lo_tid,
                "WinnerSeed": winner_seed,
                "WinnerName": team_names.get(winner_tid, str(winner_tid)),
                "WinnerID": winner_tid,
                "WinProb": win_prob,
                "Upset": is_upset,
            })

        # --- R32 ---
        # Pairs: (0,1), (2,3), (4,5), (6,7)
        r32_winners = []
        for i in range(0, 8, 2):
            t1_tid, t1_seed, _ = r64_winners[i]
            t2_tid, t2_seed, _ = r64_winners[i + 1]

            model_p = get_prediction(t1_tid, t2_tid, predictions_raw)

            # Determine higher seed
            if t1_seed <= t2_seed:
                hi_tid, hi_seed, lo_tid, lo_seed = t1_tid, t1_seed, t2_tid, t2_seed
                p_hi = model_p
            else:
                hi_tid, hi_seed, lo_tid, lo_seed = t2_tid, t2_seed, t1_tid, t1_seed
                p_hi = 1 - model_p

            # For later rounds, use pure model (historical matchup rates are less meaningful)
            if p_hi >= 0.5:
                winner_tid, winner_seed, win_prob = hi_tid, hi_seed, p_hi
            else:
                winner_tid, winner_seed, win_prob = lo_tid, lo_seed, 1 - p_hi

            is_upset = (winner_seed == lo_seed) and (hi_seed != lo_seed)
            r32_winners.append((winner_tid, winner_seed, win_prob))
            bracket_results.append({
                "Round": "R32",
                "Region": region,
                "Team1Seed": t1_seed,
                "Team1Name": team_names.get(t1_tid, str(t1_tid)),
                "Team1ID": t1_tid,
                "Team2Seed": t2_seed,
                "Team2Name": team_names.get(t2_tid, str(t2_tid)),
                "Team2ID": t2_tid,
                "WinnerSeed": winner_seed,
                "WinnerName": team_names.get(winner_tid, str(winner_tid)),
                "WinnerID": winner_tid,
                "WinProb": win_prob,
                "Upset": is_upset,
            })

        # --- S16 ---
        s16_winners = []
        for i in range(0, 4, 2):
            t1_tid, t1_seed, _ = r32_winners[i]
            t2_tid, t2_seed, _ = r32_winners[i + 1]

            model_p = get_prediction(t1_tid, t2_tid, predictions_raw)

            if t1_seed <= t2_seed:
                hi_tid, hi_seed, lo_tid, lo_seed = t1_tid, t1_seed, t2_tid, t2_seed
                p_hi = model_p
            else:
                hi_tid, hi_seed, lo_tid, lo_seed = t2_tid, t2_seed, t1_tid, t1_seed
                p_hi = 1 - model_p

            if p_hi >= 0.5:
                winner_tid, winner_seed, win_prob = hi_tid, hi_seed, p_hi
            else:
                winner_tid, winner_seed, win_prob = lo_tid, lo_seed, 1 - p_hi

            is_upset = (winner_seed == lo_seed) and (hi_seed != lo_seed)
            s16_winners.append((winner_tid, winner_seed, win_prob))
            bracket_results.append({
                "Round": "S16",
                "Region": region,
                "Team1Seed": t1_seed,
                "Team1Name": team_names.get(t1_tid, str(t1_tid)),
                "Team1ID": t1_tid,
                "Team2Seed": t2_seed,
                "Team2Name": team_names.get(t2_tid, str(t2_tid)),
                "Team2ID": t2_tid,
                "WinnerSeed": winner_seed,
                "WinnerName": team_names.get(winner_tid, str(winner_tid)),
                "WinnerID": winner_tid,
                "WinProb": win_prob,
                "Upset": is_upset,
            })

        # --- E8 ---
        t1_tid, t1_seed, _ = s16_winners[0]
        t2_tid, t2_seed, _ = s16_winners[1]

        model_p = get_prediction(t1_tid, t2_tid, predictions_raw)

        if t1_seed <= t2_seed:
            hi_tid, hi_seed, lo_tid, lo_seed = t1_tid, t1_seed, t2_tid, t2_seed
            p_hi = model_p
        else:
            hi_tid, hi_seed, lo_tid, lo_seed = t2_tid, t2_seed, t1_tid, t1_seed
            p_hi = 1 - model_p

        if p_hi >= 0.5:
            winner_tid, winner_seed, win_prob = hi_tid, hi_seed, p_hi
        else:
            winner_tid, winner_seed, win_prob = lo_tid, lo_seed, 1 - p_hi

        is_upset = (winner_seed == lo_seed) and (hi_seed != lo_seed)
        region_winners[region] = (winner_tid, winner_seed, win_prob)
        bracket_results.append({
            "Round": "E8",
            "Region": region,
            "Team1Seed": t1_seed,
            "Team1Name": team_names.get(t1_tid, str(t1_tid)),
            "Team1ID": t1_tid,
            "Team2Seed": t2_seed,
            "Team2Name": team_names.get(t2_tid, str(t2_tid)),
            "Team2ID": t2_tid,
            "WinnerSeed": winner_seed,
            "WinnerName": team_names.get(winner_tid, str(winner_tid)),
            "WinnerID": winner_tid,
            "WinProb": win_prob,
            "Upset": is_upset,
        })

    # --- Final Four ---
    f4_winners = []
    for r1, r2 in f4_matchups_regions:
        t1_tid, t1_seed, _ = region_winners[r1]
        t2_tid, t2_seed, _ = region_winners[r2]

        model_p = get_prediction(t1_tid, t2_tid, predictions_raw)

        if t1_seed <= t2_seed:
            hi_tid, hi_seed, lo_tid, lo_seed = t1_tid, t1_seed, t2_tid, t2_seed
            hi_region, lo_region = r1, r2
            p_hi = model_p
        else:
            hi_tid, hi_seed, lo_tid, lo_seed = t2_tid, t2_seed, t1_tid, t1_seed
            hi_region, lo_region = r2, r1
            p_hi = 1 - model_p

        if p_hi >= 0.5:
            winner_tid, winner_seed, win_prob = hi_tid, hi_seed, p_hi
            winner_region = hi_region
        else:
            winner_tid, winner_seed, win_prob = lo_tid, lo_seed, 1 - p_hi
            winner_region = lo_region

        is_upset = (winner_seed == lo_seed) and (hi_seed != lo_seed)
        f4_winners.append((winner_tid, winner_seed, win_prob, winner_region))
        bracket_results.append({
            "Round": "F4",
            "Region": f"{r1} vs {r2}",
            "Team1Seed": t1_seed,
            "Team1Name": team_names.get(t1_tid, str(t1_tid)),
            "Team1ID": t1_tid,
            "Team2Seed": t2_seed,
            "Team2Name": team_names.get(t2_tid, str(t2_tid)),
            "Team2ID": t2_tid,
            "WinnerSeed": winner_seed,
            "WinnerName": team_names.get(winner_tid, str(winner_tid)),
            "WinnerID": winner_tid,
            "WinProb": win_prob,
            "Upset": is_upset,
        })

    # --- Championship ---
    t1_tid, t1_seed, _, t1_region = f4_winners[0]
    t2_tid, t2_seed, _, t2_region = f4_winners[1]

    model_p = get_prediction(t1_tid, t2_tid, predictions_raw)

    if t1_seed <= t2_seed:
        hi_tid, hi_seed, lo_tid, lo_seed = t1_tid, t1_seed, t2_tid, t2_seed
        p_hi = model_p
    else:
        hi_tid, hi_seed, lo_tid, lo_seed = t2_tid, t2_seed, t1_tid, t1_seed
        p_hi = 1 - model_p

    if p_hi >= 0.5:
        winner_tid, winner_seed, win_prob = hi_tid, hi_seed, p_hi
    else:
        winner_tid, winner_seed, win_prob = lo_tid, lo_seed, 1 - p_hi

    is_upset = (winner_seed == lo_seed) and (hi_seed != lo_seed)
    bracket_results.append({
        "Round": "Championship",
        "Region": "National",
        "Team1Seed": t1_seed,
        "Team1Name": team_names.get(t1_tid, str(t1_tid)),
        "Team1ID": t1_tid,
        "Team2Seed": t2_seed,
        "Team2Name": team_names.get(t2_tid, str(t2_tid)),
        "Team2ID": t2_tid,
        "WinnerSeed": winner_seed,
        "WinnerName": team_names.get(winner_tid, str(winner_tid)),
        "WinnerID": winner_tid,
        "WinProb": win_prob,
        "Upset": is_upset,
    })

    return pd.DataFrame(bracket_results)


# ---------------------------------------------------------------------------
# Main Output
# ---------------------------------------------------------------------------

def print_section(title: str, number: int):
    """Print a formatted section header."""
    border = "=" * 90
    print(f"\n{border}")
    print(f"  SECTION {number}: {title}")
    print(f"{border}\n")


def main():
    print("\n" + "#" * 90)
    print("#" + " " * 28 + "MARCH MADNESS 2026" + " " * 28 + "#")
    print("#" + " " * 18 + "SEED HISTORY ANALYSIS & BRACKET BUILDER" + " " * 17 + "#")
    print("#" * 90)

    # Load data
    results, seeds, predictions_raw, seeds_2026, team_names = load_data()
    num_seasons = len(range(2003, 2026))  # 23 seasons (2020 excluded in data but handled)

    # ===================================================================
    # SECTION 1: Historical Seed Win Rates by Round
    # ===================================================================
    print_section("HISTORICAL SEED WIN RATES BY ROUND (2003-2025)", 1)

    win_pct, games_played = compute_seed_round_win_rates(results, seeds)

    # Format as percentage
    display = win_pct.copy()
    for col in display.columns:
        display[col] = display[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")

    print("Win percentage when playing in each round (i.e., of teams that REACHED that round):\n")
    print(display.to_string())

    print("\n\nGames played per seed per round:\n")
    games_display = games_played.fillna(0).astype(int)
    print(games_display.to_string())

    # ===================================================================
    # SECTION 2: Historical Seed-vs-Seed Matchup Win Rates (R64)
    # ===================================================================
    print_section("HISTORICAL R64 SEED-vs-SEED MATCHUP WIN RATES (2003-2025)", 2)

    hist_matchups = compute_r64_matchup_rates(results, seeds)

    # Get 2026 model predictions for same matchup types
    matchups_2026 = get_2026_r64_matchups(seeds_2026, predictions_raw, team_names)

    # Average model prob by matchup type
    model_avg_by_matchup = matchups_2026.groupby("Matchup")["ModelProbHighSeed"].mean()

    hist_matchups["Model2026AvgProb"] = hist_matchups["Matchup"].map(model_avg_by_matchup)
    hist_matchups["Delta"] = hist_matchups["Model2026AvgProb"] - hist_matchups["HighSeedWinPct"]

    display2 = hist_matchups[["Matchup", "Games", "HighSeedWins", "HighSeedWinPct",
                               "Model2026AvgProb", "Delta"]].copy()
    display2.columns = ["Matchup", "Games", "Fav Wins", "Hist Win%", "Model Avg%", "Delta"]
    display2["Hist Win%"] = display2["Hist Win%"].apply(lambda x: f"{x*100:.1f}%")
    display2["Model Avg%"] = display2["Model Avg%"].apply(lambda x: f"{x*100:.1f}%")
    display2["Delta"] = display2["Delta"].apply(lambda x: f"{x*100:+.1f}pp")

    print(display2.to_string(index=False))

    print("\n  * Delta > 0: Model is MORE confident in the favorite than history suggests")
    print("  * Delta < 0: Model is LESS confident in the favorite than history suggests")

    # ===================================================================
    # SECTION 3: Upset Frequency by Round
    # ===================================================================
    print_section("UPSET FREQUENCY BY ROUND (2003-2025)", 3)

    hist_upsets = compute_upset_frequency(results, seeds)

    print("How often does the lower-seeded (worse) team win in each round?\n")
    upset_display = hist_upsets.copy()
    upset_display["UpsetPct"] = upset_display["UpsetPct"].apply(lambda x: f"{x*100:.1f}%")
    print(upset_display.to_string())

    # Model's predicted upset rate per round (from bracket)
    # We'll compute this in section 6 and reference back

    # ===================================================================
    # SECTION 4: Model vs History Comparison (2026 R64)
    # ===================================================================
    print_section("MODEL vs HISTORY: 2026 R64 MATCHUPS", 4)

    full_matchups = build_model_vs_history(matchups_2026, hist_matchups)

    display4 = full_matchups[[
        "Region", "HighSeed", "LowSeed", "HighTeamName", "LowTeamName",
        "ModelProbHighSeed", "HistoricalWinPct", "Delta", "BlendedProb"
    ]].copy()
    display4.columns = ["Reg", "Seed", "vs", "Favorite", "Underdog",
                         "Model%", "Hist%", "Delta", "Blended%"]

    # Sort by delta (most disagreement first)
    display4 = display4.sort_values("Delta", key=abs, ascending=False)

    # Format
    for col in ["Model%", "Hist%", "Blended%"]:
        display4[col] = display4[col].apply(lambda x: f"{x*100:.1f}%")
    display4["Delta"] = display4["Delta"].apply(lambda x: f"{x*100:+.1f}pp")

    print("Sorted by |Delta| (biggest model-vs-history disagreements first):\n")
    print(display4.to_string(index=False))

    # Flag significant disagreements
    print("\n--- SIGNIFICANT DISAGREEMENTS (|Delta| > 10pp) ---\n")
    sig = full_matchups[abs(full_matchups["Delta"]) > 0.10].sort_values("Delta")
    if len(sig) == 0:
        print("  No matchups with >10pp disagreement between model and history.")
    else:
        for _, row in sig.iterrows():
            direction = "MORE" if row["Delta"] > 0 else "LESS"
            print(f"  {row['Region']} Region: ({row['HighSeed']}) {row['HighTeamName']} vs "
                  f"({row['LowSeed']}) {row['LowTeamName']}")
            print(f"    Model: {row['ModelProbHighSeed']*100:.1f}% | "
                  f"History: {row['HistoricalWinPct']*100:.1f}% | "
                  f"Delta: {row['Delta']*100:+.1f}pp")
            print(f"    -> Model is {direction} confident in the favorite than history.\n")

    # ===================================================================
    # SECTION 5: Smart Upset Recommendations
    # ===================================================================
    print_section("SMART UPSET RECOMMENDATIONS", 5)

    # Identify games where blended prob suggests an upset or near-upset
    upset_candidates = full_matchups.copy()
    upset_candidates["UpsetBlended"] = 1 - upset_candidates["BlendedProb"]
    upset_candidates = upset_candidates.sort_values("UpsetBlended", ascending=False)

    print("Blended probability = 70% Model + 30% Historical base rate\n")
    print("All R64 games ranked by upset potential (underdog blended win probability):\n")

    for _, row in upset_candidates.iterrows():
        fav = f"({row['HighSeed']}) {row['HighTeamName']}"
        dog = f"({row['LowSeed']}) {row['LowTeamName']}"
        upset_p = (1 - row["BlendedProb"]) * 100
        model_p = (1 - row["ModelProbHighSeed"]) * 100
        hist_p = (1 - row["HistoricalWinPct"]) * 100

        marker = ""
        if upset_p > 50:
            marker = " ** PICK THE UPSET **"
        elif upset_p > 35:
            marker = " * Strong upset candidate *"
        elif upset_p > 25:
            marker = " ~ Viable upset ~"

        print(f"  {row['Region']} | {fav:>30s} vs {dog:<30s} | "
              f"Upset: {upset_p:5.1f}% (model {model_p:.0f}%, hist {hist_p:.0f}%){marker}")

    # Detailed recommendations
    print("\n" + "-" * 90)
    print("DETAILED UPSET RECOMMENDATIONS:")
    print("-" * 90 + "\n")

    strong_upsets = upset_candidates[
        (1 - upset_candidates["BlendedProb"]) > 0.30
    ].sort_values("BlendedProb")

    if len(strong_upsets) == 0:
        print("  No games where blended upset probability exceeds 30%.")
    else:
        for i, (_, row) in enumerate(strong_upsets.iterrows(), 1):
            fav = f"({row['HighSeed']}) {row['HighTeamName']}"
            dog = f"({row['LowSeed']}) {row['LowTeamName']}"
            upset_p = (1 - row["BlendedProb"]) * 100
            model_upset = (1 - row["ModelProbHighSeed"]) * 100
            hist_upset = (1 - row["HistoricalWinPct"]) * 100

            print(f"  {i}. {row['Region']} Region: {dog} over {fav}")
            print(f"     Blended upset prob: {upset_p:.1f}%")
            print(f"     Model upset prob:   {model_upset:.1f}%")
            print(f"     Historical upset rate for {row['HighSeed']}v{row['LowSeed']}: {hist_upset:.1f}%")

            # Path analysis: can the upset winner advance?
            next_matchup_seed = row["LowSeed"]
            if next_matchup_seed >= 9:
                print(f"     Path forward: As a {next_matchup_seed}-seed, would likely face "
                      f"a 1-seed or similar in R32 -- limited further advancement.")
            elif next_matchup_seed >= 5:
                print(f"     Path forward: As a {next_matchup_seed}-seed upset winner, "
                      f"has mid-tier path -- could reach Sweet 16 with favorable draws.")
            else:
                print(f"     Path forward: As a {next_matchup_seed}-seed, "
                      f"has a reasonable bracket path if performing well.")

            # Recommendation
            if upset_p > 50:
                print(f"     VERDICT: STRONGLY RECOMMEND picking this upset.")
            elif upset_p > 40:
                print(f"     VERDICT: RECOMMENDED upset pick for most brackets.")
            elif upset_p > 30:
                print(f"     VERDICT: Consider for contrarian/upset-heavy brackets.")
            print()

    # ===================================================================
    # SECTION 6: Full Bracket Recommendation
    # ===================================================================
    print_section("FULL 63-GAME BRACKET RECOMMENDATION", 6)

    bracket_df = simulate_full_bracket(seeds_2026, predictions_raw, team_names,
                                        hist_matchups, hist_upsets)

    # Display by round
    all_rounds = ["R64", "R32", "S16", "E8", "F4", "Championship"]

    for rnd in all_rounds:
        rnd_games = bracket_df[bracket_df.Round == rnd]
        print(f"\n{'-' * 90}")
        print(f"  {rnd} ({len(rnd_games)} games)")
        print(f"{'-' * 90}")

        for _, g in rnd_games.iterrows():
            t1 = f"({g['Team1Seed']:>2d}) {g['Team1Name']}"
            t2 = f"({g['Team2Seed']:>2d}) {g['Team2Name']}"
            winner = f"({g['WinnerSeed']:>2d}) {g['WinnerName']}"
            region = g["Region"]
            prob = g["WinProb"] * 100
            upset_marker = " [UPSET]" if g["Upset"] else ""

            print(f"  {region:>6s} | {t1:>30s}  vs  {t2:<30s} -> {winner:<28s} "
                  f"({prob:5.1f}%){upset_marker}")

    # Summary statistics
    print(f"\n{'=' * 90}")
    print("  BRACKET SUMMARY")
    print(f"{'=' * 90}\n")

    for rnd in all_rounds:
        rnd_games = bracket_df[bracket_df.Round == rnd]
        upsets = rnd_games["Upset"].sum()
        total = len(rnd_games)
        print(f"  {rnd:>14s}: {total:2d} games, {upsets:2d} upsets "
              f"({upsets/total*100:.0f}% upset rate)")

    total_upsets = bracket_df["Upset"].sum()
    total_games = len(bracket_df)
    print(f"\n  {'TOTAL':>14s}: {total_games:2d} games, {total_upsets:2d} upsets "
          f"({total_upsets/total_games*100:.0f}% upset rate)")

    # Final Four and Champion
    print(f"\n{'-' * 90}")
    print("  FINAL FOUR:")
    print(f"{'-' * 90}")
    f4_games = bracket_df[bracket_df.Round == "F4"]
    for _, g in f4_games.iterrows():
        print(f"    ({g['Team1Seed']}) {g['Team1Name']}  vs  "
              f"({g['Team2Seed']}) {g['Team2Name']}")
        print(f"      Winner: ({g['WinnerSeed']}) {g['WinnerName']} "
              f"(prob: {g['WinProb']*100:.1f}%)")

    champ = bracket_df[bracket_df.Round == "Championship"].iloc[0]
    print(f"\n{'-' * 90}")
    print("  CHAMPIONSHIP GAME:")
    print(f"{'-' * 90}")
    print(f"    ({champ['Team1Seed']}) {champ['Team1Name']}  vs  "
          f"({champ['Team2Seed']}) {champ['Team2Name']}")

    print(f"\n  *** PREDICTED CHAMPION: ({champ['WinnerSeed']}) {champ['WinnerName']} ***")
    print(f"  *** Championship win probability: {champ['WinProb']*100:.1f}% ***")
    print()

    # Compare model upset rates to historical
    print(f"\n{'=' * 90}")
    print("  MODEL vs HISTORICAL UPSET RATES BY ROUND")
    print(f"{'=' * 90}\n")

    hist_upset_data = compute_upset_frequency(results, seeds)
    print(f"  {'Round':>14s} | {'Hist Upset%':>12s} | {'Bracket Upsets':>15s} | {'Bracket Upset%':>15s}")
    print(f"  {'-'*14}-+-{'-'*12}-+-{'-'*15}-+-{'-'*15}")
    for rnd in all_rounds:
        rnd_games = bracket_df[bracket_df.Round == rnd]
        bracket_upsets = int(rnd_games["Upset"].sum())
        bracket_total = len(rnd_games)
        bracket_pct = bracket_upsets / bracket_total * 100 if bracket_total > 0 else 0

        if rnd in hist_upset_data.index:
            hist_pct = hist_upset_data.loc[rnd, "UpsetPct"] * 100
        else:
            hist_pct = 0

        print(f"  {rnd:>14s} | {hist_pct:>11.1f}% | {bracket_upsets:>7d}/{bracket_total:<7d} | "
              f"{bracket_pct:>14.1f}%")

    print()


if __name__ == "__main__":
    main()
