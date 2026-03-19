"""
Bracket Optimizer for March Madness 2026.

Generates multiple bracket-filling strategies that maximize expected scoring
in bracket pools, rather than simply picking the most likely winner of each game.

Strategies:
  1. Chalk bracket (always pick the favorite)
  2. Expected value bracket (maximize expected ESPN scoring points)
  3. Contrarian bracket (maximize leverage against estimated public picks)
  4. Upset-friendly bracket (force realistic upsets where model supports them)

Uses Monte Carlo simulation to compute true advancement probabilities
that account for the full tournament path, not just head-to-head matchups.
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "website" / "public" / "data"

NUM_SIMULATIONS = 50_000  # Monte Carlo bracket simulations
RANDOM_SEED = 42

# ESPN standard bracket scoring: points per correct pick per round
# R64=10, R32=20, S16=40, E8=80, F4=160, Championship=320
SCORING = [10, 20, 40, 80, 160, 320]

# Standard bracket matchup order per region (seed pairs for R64)
# 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
R64_SEED_PAIRS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

# Final Four pairing: W vs X, Y vs Z
FF_REGION_PAIRS = [("W", "X"), ("Y", "Z")]

# Regions in bracket order
REGIONS = ["W", "X", "Y", "Z"]

# Estimated public pick rates by seed matchup (higher seed's pick %)
# Format: (higher_seed, lower_seed) -> public pick % for higher seed
PUBLIC_PICK_RATES = {
    (1, 16): 0.97,
    (2, 15): 0.94,
    (3, 14): 0.90,
    (4, 13): 0.82,
    (5, 12): 0.65,
    (6, 11): 0.72,
    (7, 10): 0.62,
    (8, 9): 0.55,
}

# Public advancement rates for later rounds (by seed, approximate)
PUBLIC_ADVANCE_RATE = {
    1: {2: 0.93, 3: 0.78, 4: 0.55, 5: 0.38, 6: 0.22},
    2: {2: 0.88, 3: 0.65, 4: 0.40, 5: 0.25, 6: 0.15},
    3: {2: 0.80, 3: 0.48, 4: 0.22, 5: 0.12, 6: 0.06},
    4: {2: 0.72, 3: 0.38, 4: 0.16, 5: 0.08, 6: 0.03},
    5: {2: 0.52, 3: 0.22, 4: 0.08, 5: 0.03, 6: 0.01},
    6: {2: 0.58, 3: 0.25, 4: 0.10, 5: 0.04, 6: 0.015},
    7: {2: 0.48, 3: 0.18, 4: 0.06, 5: 0.02, 6: 0.007},
    8: {2: 0.42, 3: 0.12, 4: 0.04, 5: 0.015, 6: 0.005},
    9: {2: 0.38, 3: 0.10, 4: 0.03, 5: 0.012, 6: 0.004},
    10: {2: 0.32, 3: 0.10, 4: 0.03, 5: 0.01, 6: 0.003},
    11: {2: 0.25, 3: 0.08, 4: 0.025, 5: 0.008, 6: 0.002},
    12: {2: 0.28, 3: 0.10, 4: 0.03, 5: 0.01, 6: 0.003},
    13: {2: 0.12, 3: 0.03, 4: 0.008, 5: 0.002, 6: 0.0005},
    14: {2: 0.07, 3: 0.015, 4: 0.004, 5: 0.001, 6: 0.0002},
    15: {2: 0.04, 3: 0.008, 4: 0.002, 5: 0.0004, 6: 0.0001},
    16: {2: 0.02, 3: 0.003, 4: 0.0005, 5: 0.0001, 6: 0.00002},
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data():
    """Load predictions, seeds, and team names."""
    with open(DATA_DIR / "predictions_men.json") as f:
        predictions = json.load(f)

    with open(DATA_DIR / "seeds.json") as f:
        seeds_all = json.load(f)
    seeds = seeds_all["men"]

    with open(DATA_DIR / "teams.json") as f:
        teams_all = json.load(f)
    teams = teams_all["men"]

    return predictions, seeds, teams


def get_win_prob(predictions, team_a, team_b):
    """Get win probability for team_a against team_b.

    The predictions dict uses the key format 'lowerID_higherID' and stores
    the probability that the lower-ID team wins.
    """
    id_a, id_b = int(team_a), int(team_b)
    if id_a < id_b:
        key = f"{id_a}_{id_b}"
        return predictions.get(key, 0.5)
    else:
        key = f"{id_b}_{id_a}"
        return 1.0 - predictions.get(key, 0.5)


# ---------------------------------------------------------------------------
# Build Bracket Structure
# ---------------------------------------------------------------------------
def build_bracket_structure(seeds, teams, predictions):
    """Build the tournament bracket structure including play-in resolutions.

    Returns a dict of regions, each containing a list of 8 R64 matchup pairs
    in standard bracket order: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15.
    """
    # Group teams by region and seed
    region_seeds = defaultdict(lambda: defaultdict(list))
    for team_id, info in seeds.items():
        region_seeds[info["region"]][info["seed"]].append({
            "id": team_id,
            "seed": info["seed"],
            "region": info["region"],
            "playIn": info.get("playIn", ""),
            "name": teams.get(team_id, f"Team {team_id}"),
        })

    # Resolve play-in games: pick the team with higher win probability
    # (for bracket structure purposes, we resolve play-ins deterministically
    # by picking the favorite; the Monte Carlo sim will handle uncertainty)
    resolved_regions = {}
    playin_games = []

    for region in REGIONS:
        resolved_regions[region] = {}
        for seed_num in range(1, 17):
            team_list = region_seeds[region][seed_num]
            if len(team_list) == 1:
                resolved_regions[region][seed_num] = team_list[0]
            elif len(team_list) == 2:
                # Play-in game
                t_a, t_b = team_list[0], team_list[1]
                prob_a = get_win_prob(predictions, t_a["id"], t_b["id"])
                playin_games.append({
                    "region": region,
                    "seed": seed_num,
                    "team_a": t_a,
                    "team_b": t_b,
                    "prob_a": prob_a,
                })
                # For the main bracket, use whichever team the model favors
                if prob_a >= 0.5:
                    resolved_regions[region][seed_num] = t_a
                else:
                    resolved_regions[region][seed_num] = t_b

    # Build R64 matchups in standard order
    bracket = {}
    for region in REGIONS:
        matchups = []
        for high_seed, low_seed in R64_SEED_PAIRS:
            matchups.append((
                resolved_regions[region][high_seed],
                resolved_regions[region][low_seed],
            ))
        bracket[region] = matchups

    return bracket, playin_games, region_seeds


# ---------------------------------------------------------------------------
# Monte Carlo Tournament Simulation
# ---------------------------------------------------------------------------
def simulate_tournament(bracket, predictions, num_sims=NUM_SIMULATIONS, rng=None):
    """Simulate the full tournament num_sims times.

    Returns advancement_counts: dict[team_id][round_idx] = count of simulations
    where the team reached that round. round_idx: 0=R64 (they are in the bracket),
    1=R32 (won R64), 2=S16, 3=E8, 4=F4, 5=Championship, 6=Won it all.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    # Collect all teams in the bracket
    all_teams = set()
    for region in REGIONS:
        for t_a, t_b in bracket[region]:
            all_teams.add(t_a["id"])
            all_teams.add(t_b["id"])

    # Initialize advancement counts
    # Round labels: 0=In bracket (R64), 1=Won R64 (in R32), 2=Won R32 (in S16),
    #               3=Won S16 (in E8), 4=Won E8 (in F4), 5=Won F4 (in Champ),
    #               6=Won Championship
    advancement_counts = {tid: np.zeros(7, dtype=np.int32) for tid in all_teams}
    for tid in all_teams:
        advancement_counts[tid][0] = num_sims  # everyone starts in R64

    # Pre-compute win probability matrix for all possible matchups
    team_list = sorted(all_teams)
    team_idx = {tid: i for i, tid in enumerate(team_list)}
    n_teams = len(team_list)
    prob_matrix = np.full((n_teams, n_teams), 0.5)

    for i, t_a in enumerate(team_list):
        for j, t_b in enumerate(team_list):
            if i < j:
                p = get_win_prob(predictions, t_a, t_b)
                prob_matrix[i][j] = p
                prob_matrix[j][i] = 1.0 - p

    # Vectorized simulation: simulate all tournaments at once
    # For each region, simulate R64 -> R32 -> S16 -> E8 to get regional champions
    # Then simulate F4 and Championship

    # Build initial matchup arrays per region
    # Shape: (num_matchups, 2) where value is team index
    region_r64 = {}
    for region in REGIONS:
        matchups = []
        for t_a, t_b in bracket[region]:
            matchups.append((team_idx[t_a["id"]], team_idx[t_b["id"]]))
        region_r64[region] = np.array(matchups)  # shape (8, 2)

    def sim_round(matchup_pairs, rng):
        """Simulate one round for all sims at once.

        matchup_pairs: shape (num_sims, num_games, 2) - team indices for each game
        Returns: shape (num_sims, num_games) - winning team index for each game
        """
        n_s, n_g, _ = matchup_pairs.shape
        team_a = matchup_pairs[:, :, 0]  # (num_sims, num_games)
        team_b = matchup_pairs[:, :, 1]  # (num_sims, num_games)

        # Look up probabilities: prob that team_a wins
        probs = prob_matrix[team_a, team_b]  # (num_sims, num_games)

        # Sample outcomes
        randoms = rng.random((n_s, n_g))
        winners = np.where(randoms < probs, team_a, team_b)
        return winners

    # Simulate all regions
    region_champs = {}  # region -> shape (num_sims,) team indices

    for region in REGIONS:
        # R64: 8 games per region
        r64_pairs = np.tile(region_r64[region], (num_sims, 1, 1))  # (num_sims, 8, 2)
        r64_winners = sim_round(r64_pairs, rng)  # (num_sims, 8)

        # Record R64 winners (reached R32 = round_idx 1)
        for sim in range(num_sims):
            for g in range(8):
                w = team_list[r64_winners[sim, g]]
                advancement_counts[w][1] += 1

        # R32: 4 games (pair consecutive R64 winners)
        r32_pairs = np.stack([
            r64_winners[:, 0::2],  # winners of games 0,2,4,6
            r64_winners[:, 1::2],  # winners of games 1,3,5,7
        ], axis=-1)  # (num_sims, 4, 2)
        r32_winners = sim_round(r32_pairs, rng)  # (num_sims, 4)

        for sim in range(num_sims):
            for g in range(4):
                w = team_list[r32_winners[sim, g]]
                advancement_counts[w][2] += 1

        # S16: 2 games
        s16_pairs = np.stack([
            r32_winners[:, 0::2],
            r32_winners[:, 1::2],
        ], axis=-1)  # (num_sims, 2, 2)
        s16_winners = sim_round(s16_pairs, rng)  # (num_sims, 2)

        for sim in range(num_sims):
            for g in range(2):
                w = team_list[s16_winners[sim, g]]
                advancement_counts[w][3] += 1

        # E8: 1 game (regional final)
        e8_pairs = np.stack([
            s16_winners[:, 0:1],
            s16_winners[:, 1:2],
        ], axis=-1)  # (num_sims, 1, 2)
        e8_winners = sim_round(e8_pairs, rng)  # (num_sims, 1)

        for sim in range(num_sims):
            w = team_list[e8_winners[sim, 0]]
            advancement_counts[w][4] += 1

        region_champs[region] = e8_winners[:, 0]  # (num_sims,)

    # Final Four: W vs X, Y vs Z
    ff_pairs_list = []
    for r_a, r_b in FF_REGION_PAIRS:
        pair = np.stack([region_champs[r_a], region_champs[r_b]], axis=-1)
        ff_pairs_list.append(pair)
    ff_pairs = np.stack(ff_pairs_list, axis=1)  # (num_sims, 2, 2)
    ff_winners = sim_round(ff_pairs, rng)  # (num_sims, 2)

    for sim in range(num_sims):
        for g in range(2):
            w = team_list[ff_winners[sim, g]]
            advancement_counts[w][5] += 1

    # Championship
    champ_pairs = ff_winners[:, :, np.newaxis]  # need shape (num_sims, 1, 2)
    champ_pairs = np.stack([ff_winners[:, 0:1], ff_winners[:, 1:2]], axis=-1)
    champ_winners = sim_round(champ_pairs, rng)  # (num_sims, 1)

    for sim in range(num_sims):
        w = team_list[champ_winners[sim, 0]]
        advancement_counts[w][6] += 1

    return advancement_counts


def compute_advancement_probs(advancement_counts, num_sims):
    """Convert counts to probabilities."""
    probs = {}
    for tid, counts in advancement_counts.items():
        probs[tid] = counts.astype(np.float64) / num_sims
    return probs


# ---------------------------------------------------------------------------
# Bracket Filling Strategies
# ---------------------------------------------------------------------------
def fill_chalk_bracket(bracket, predictions, teams_dict):
    """Always pick the favorite in each game."""
    return _fill_bracket(bracket, predictions, teams_dict, strategy="chalk")


def fill_ev_bracket(bracket, predictions, teams_dict, adv_probs):
    """Pick the team that maximizes expected points at each slot.

    At each game, compare the two teams on CURRENT ROUND EV only:
      EV(team) = P(team reaches this round) * points_for_round
    This is a per-slot greedy optimization. Downstream value is captured
    implicitly because the team we advance becomes available in the next
    round where we again compare per-slot EVs.

    Difference from chalk: when advancement probabilities diverge from
    head-to-head probabilities (because of path effects), EV can pick
    differently.
    """
    return _fill_bracket(bracket, predictions, teams_dict, strategy="ev", adv_probs=adv_probs)


def fill_contrarian_bracket(bracket, predictions, teams_dict, adv_probs, seeds_data):
    """Build a bracket optimized for large-pool differentiation.

    Strategy: First select a non-chalk champion and Final Four that offers
    positive leverage (our model rates them higher than the public does).
    Then work backwards to ensure those teams advance through the bracket.
    For all other slots, use EV-based picks.
    """
    return _fill_contrarian_top_down(bracket, predictions, teams_dict, adv_probs, seeds_data)


def fill_upset_bracket(bracket, predictions, teams_dict, adv_probs, seeds_data):
    """Force upsets where underdogs have >30% win probability in R64,
    then use EV-based logic in later rounds."""
    return _fill_bracket(bracket, predictions, teams_dict, strategy="upset",
                         adv_probs=adv_probs, seeds_data=seeds_data)


def _fill_contrarian_top_down(bracket, predictions, teams_dict, adv_probs, seeds_data):
    """Top-down contrarian bracket: pick champion first, then build backward.

    1. Identify the best contrarian champion (highest leverage among viable teams).
    2. Identify contrarian Final Four picks (one per region).
    3. Build the bracket to support those picks, using EV for other slots.
    """
    all_picks = []
    round_names = ["R64", "R32", "S16", "E8", "F4", "Championship"]
    upset_counts = {rn: 0 for rn in round_names}

    # --- Step 1: Find best contrarian picks per region ---
    # For each region, find the team with the best leverage for reaching F4
    region_candidates = {}
    for region in REGIONS:
        candidates = []
        for t_a, t_b in bracket[region]:
            for t in [t_a, t_b]:
                if t["id"] not in adv_probs:
                    continue
                f4_prob = adv_probs[t["id"]][4]  # F4 probability
                if f4_prob < 0.005:  # filter out teams with <0.5% F4 chance
                    continue
                pub_f4 = _get_public_pick_rate(t["seed"], 4)
                leverage = f4_prob / max(pub_f4, 0.001)
                # Score: blend of raw probability and leverage
                # We want teams that are both viable AND undervalued
                score = f4_prob * (leverage ** 0.6)
                candidates.append({
                    "team": t,
                    "f4_prob": f4_prob,
                    "pub_f4": pub_f4,
                    "leverage": leverage,
                    "score": score,
                })
        candidates.sort(key=lambda x: -x["score"])
        region_candidates[region] = candidates

    # Pick the best contrarian F4 team per region
    # Require at least one non-1-seed in the F4 for differentiation
    contrarian_f4 = {}
    for region in REGIONS:
        cands = region_candidates[region]
        # Default: highest-scoring candidate
        contrarian_f4[region] = cands[0]["team"]

    # If all F4 picks are 1-seeds, force the weakest region to use a non-1-seed
    f4_seeds = [contrarian_f4[r]["seed"] for r in REGIONS]
    if all(s == 1 for s in f4_seeds):
        # Find the region where the best non-1-seed has the highest score
        best_swap_score = -1
        best_swap_region = None
        best_swap_team = None
        for region in REGIONS:
            for c in region_candidates[region]:
                if c["team"]["seed"] != 1 and c["f4_prob"] >= 0.02:
                    if c["score"] > best_swap_score:
                        best_swap_score = c["score"]
                        best_swap_region = region
                        best_swap_team = c["team"]
                    break  # only check best non-1-seed per region
        if best_swap_team is not None:
            contrarian_f4[best_swap_region] = best_swap_team

    # --- Step 2: Pick champion ---
    # Among F4 picks, find the one with best championship leverage.
    # Key insight for pool strategy: pick the 2nd-most-likely champion.
    # If the top champion is very popular (e.g., Duke at ~22%), picking
    # the 2nd-most-likely (e.g., Michigan at ~21%) gives almost the same
    # expected value but much better differentiation from the field.
    champ_candidates = []
    for r_a, r_b in FF_REGION_PAIRS:
        for r in [r_a, r_b]:
            t = contrarian_f4[r]
            champ_prob = adv_probs[t["id"]][6]
            pub_champ = _get_public_pick_rate(t["seed"], 6)
            lev = champ_prob / max(pub_champ, 0.0001)
            score = champ_prob * (lev ** 0.5)
            champ_candidates.append({"team": t, "region": r, "score": score,
                                      "champ_prob": champ_prob, "leverage": lev})
    champ_candidates.sort(key=lambda x: -x["champ_prob"])

    # If the top 2 candidates are close in championship probability (within 5%),
    # pick the 2nd-most-likely for differentiation from the field
    if (len(champ_candidates) >= 2 and
            champ_candidates[0]["champ_prob"] - champ_candidates[1]["champ_prob"] < 0.05):
        contrarian_champion = champ_candidates[1]["team"]
    else:
        contrarian_champion = champ_candidates[0]["team"]

    # --- Step 3: Build bracket forward, ensuring our F4 picks advance ---
    # For each game, if one team is our targeted F4 pick (or along its path),
    # pick that team. Otherwise, use a leverage-weighted EV pick.

    # Figure out which teams need to be "protected" (on the path to F4)
    # We will rebuild the bracket and tag games where our F4 pick must win
    target_ids = {r: contrarian_f4[r]["id"] for r in REGIONS}

    def ev_decide(t_a, t_b, round_idx):
        """EV-based decision with mild leverage bonus."""
        adv_a = adv_probs[t_a["id"]][round_idx + 1]
        adv_b = adv_probs[t_b["id"]][round_idx + 1]
        pts = SCORING[round_idx]

        pub_a = _get_public_pick_rate(t_a["seed"], round_idx + 1)
        pub_b = _get_public_pick_rate(t_b["seed"], round_idx + 1)

        if round_idx == 0:
            matchup_key = (min(t_a["seed"], t_b["seed"]), max(t_a["seed"], t_b["seed"]))
            pub_rate = PUBLIC_PICK_RATES.get(matchup_key, 0.5)
            if t_a["seed"] < t_b["seed"]:
                pub_a, pub_b = pub_rate, 1.0 - pub_rate
            else:
                pub_b, pub_a = pub_rate, 1.0 - pub_rate

        # Leverage-weighted EV with moderate alpha
        ALPHA = 0.4
        score_a = adv_a * pts * (adv_a / max(pub_a, 0.001)) ** ALPHA
        score_b = adv_b * pts * (adv_b / max(pub_b, 0.001)) ** ALPHA

        prob_a = get_win_prob(predictions, t_a["id"], t_b["id"])
        if score_a >= score_b:
            return t_a, prob_a
        else:
            return t_b, 1.0 - prob_a

    region_winners = {}

    for region in REGIONS:
        target = target_ids[region]

        # R64
        r64_winners = []
        for i, (t_a, t_b) in enumerate(bracket[region]):
            # If our target is in this game, pick them
            if t_a["id"] == target or t_b["id"] == target:
                winner = t_a if t_a["id"] == target else t_b
                prob = get_win_prob(predictions, t_a["id"], t_b["id"])
                if winner["id"] != t_a["id"]:
                    prob = 1.0 - prob
            else:
                winner, prob = ev_decide(t_a, t_b, 0)

            is_upset = winner["seed"] > min(t_a["seed"], t_b["seed"])
            if is_upset:
                upset_counts["R64"] += 1
            all_picks.append({
                "round": "R64", "region": region,
                "team_a": t_a, "team_b": t_b,
                "pick": winner, "win_prob": prob,
                "is_upset": is_upset,
            })
            r64_winners.append(winner)

        # R32
        r32_winners = []
        for i in range(0, 8, 2):
            t_a, t_b = r64_winners[i], r64_winners[i + 1]
            if t_a["id"] == target or t_b["id"] == target:
                winner = t_a if t_a["id"] == target else t_b
                prob = get_win_prob(predictions, t_a["id"], t_b["id"])
                if winner["id"] != t_a["id"]:
                    prob = 1.0 - prob
            else:
                winner, prob = ev_decide(t_a, t_b, 1)

            is_upset = winner["seed"] > min(t_a["seed"], t_b["seed"])
            if is_upset:
                upset_counts["R32"] += 1
            all_picks.append({
                "round": "R32", "region": region,
                "team_a": t_a, "team_b": t_b,
                "pick": winner, "win_prob": prob,
                "is_upset": is_upset,
            })
            r32_winners.append(winner)

        # S16
        s16_winners = []
        for i in range(0, 4, 2):
            t_a, t_b = r32_winners[i], r32_winners[i + 1]
            if t_a["id"] == target or t_b["id"] == target:
                winner = t_a if t_a["id"] == target else t_b
                prob = get_win_prob(predictions, t_a["id"], t_b["id"])
                if winner["id"] != t_a["id"]:
                    prob = 1.0 - prob
            else:
                winner, prob = ev_decide(t_a, t_b, 2)

            is_upset = winner["seed"] > min(t_a["seed"], t_b["seed"])
            if is_upset:
                upset_counts["S16"] += 1
            all_picks.append({
                "round": "S16", "region": region,
                "team_a": t_a, "team_b": t_b,
                "pick": winner, "win_prob": prob,
                "is_upset": is_upset,
            })
            s16_winners.append(winner)

        # E8 (regional final)
        t_a, t_b = s16_winners[0], s16_winners[1]
        if t_a["id"] == target or t_b["id"] == target:
            winner = t_a if t_a["id"] == target else t_b
            prob = get_win_prob(predictions, t_a["id"], t_b["id"])
            if winner["id"] != t_a["id"]:
                prob = 1.0 - prob
        else:
            winner, prob = ev_decide(t_a, t_b, 3)

        is_upset = winner["seed"] > min(t_a["seed"], t_b["seed"])
        if is_upset:
            upset_counts["E8"] += 1
        all_picks.append({
            "round": "E8", "region": region,
            "team_a": t_a, "team_b": t_b,
            "pick": winner, "win_prob": prob,
            "is_upset": is_upset,
        })
        region_winners[region] = winner

    # --- Final Four ---
    ff_winners = []
    for r_a, r_b in FF_REGION_PAIRS:
        t_a, t_b = region_winners[r_a], region_winners[r_b]
        # Pick whichever has better championship leverage
        champ_a = adv_probs[t_a["id"]][6]
        champ_b = adv_probs[t_b["id"]][6]
        pub_a = _get_public_pick_rate(t_a["seed"], 6)
        pub_b = _get_public_pick_rate(t_b["seed"], 6)
        score_a = champ_a * (champ_a / max(pub_a, 0.0001)) ** 0.4
        score_b = champ_b * (champ_b / max(pub_b, 0.0001)) ** 0.4

        prob_a = get_win_prob(predictions, t_a["id"], t_b["id"])
        if score_a >= score_b:
            winner = t_a
            prob = prob_a
        else:
            winner = t_b
            prob = 1.0 - prob_a

        is_upset = winner["seed"] > min(t_a["seed"], t_b["seed"])
        if is_upset:
            upset_counts["F4"] += 1
        all_picks.append({
            "round": "F4", "region": f"{r_a}v{r_b}",
            "team_a": t_a, "team_b": t_b,
            "pick": winner, "win_prob": prob,
            "is_upset": is_upset,
        })
        ff_winners.append(winner)

    # --- Championship ---
    t_a, t_b = ff_winners[0], ff_winners[1]
    # Pick our pre-selected champion if they are in the final, else pick by leverage
    if t_a["id"] == contrarian_champion["id"]:
        winner = t_a
    elif t_b["id"] == contrarian_champion["id"]:
        winner = t_b
    else:
        champ_a = adv_probs[t_a["id"]][6]
        champ_b = adv_probs[t_b["id"]][6]
        winner = t_a if champ_a >= champ_b else t_b

    prob_a = get_win_prob(predictions, t_a["id"], t_b["id"])
    if winner["id"] == t_a["id"]:
        prob = prob_a
    else:
        prob = 1.0 - prob_a

    is_upset = winner["seed"] > min(t_a["seed"], t_b["seed"])
    if is_upset:
        upset_counts["Championship"] += 1
    all_picks.append({
        "round": "Championship", "region": "Final",
        "team_a": t_a, "team_b": t_b,
        "pick": winner, "win_prob": prob,
        "is_upset": is_upset,
    })

    champion = winner
    final_four = [region_winners[r] for r in REGIONS]

    summary = {
        "champion": champion,
        "final_four": final_four,
        "region_winners": region_winners,
        "upset_counts": upset_counts,
        "total_upsets": sum(upset_counts.values()),
    }

    return all_picks, summary


def _get_public_pick_rate(seed, round_idx):
    """Estimate public pick rate for a team with given seed reaching round_idx.

    round_idx: 1=R32, 2=S16, 3=E8, 4=F4, 5=Championship, 6=Champion
    """
    if round_idx <= 1:
        # R64 public pick rate depends on the matchup, handled elsewhere
        return PUBLIC_ADVANCE_RATE.get(seed, {}).get(round_idx, 0.5)
    return PUBLIC_ADVANCE_RATE.get(seed, {}).get(round_idx, 0.01)


def _fill_bracket(bracket, predictions, teams_dict, strategy="chalk",
                   adv_probs=None, seeds_data=None):
    """Core bracket filling logic.

    Simulates the bracket round by round. In each game, decides which team
    to pick based on the chosen strategy.

    Returns:
        picks: list of dicts with pick info for all 63 games
        summary: dict with Final Four, champion, upset counts, etc.
    """
    all_picks = []
    region_winners = {}

    round_names = ["R64", "R32", "S16", "E8", "F4", "Championship"]
    upset_counts = {rn: 0 for rn in round_names}

    def decide_winner(t_a, t_b, round_idx, round_name):
        """Decide which team to pick for this game."""
        prob_a = get_win_prob(predictions, t_a["id"], t_b["id"])
        seed_a = t_a["seed"]
        seed_b = t_b["seed"]

        if strategy == "chalk":
            # Always pick the favorite
            if prob_a >= 0.5:
                return t_a, prob_a
            else:
                return t_b, 1.0 - prob_a

        elif strategy == "ev":
            # Per-slot EV maximization: pick the team with the highest
            # expected points for THIS specific round slot.
            #
            # EV(team, round) = P(team reaches this round) * points_for_round
            #
            # This is a PURE per-slot optimization. We intentionally do NOT
            # add downstream value because:
            # 1. Downstream rounds will be optimized separately when we get there
            # 2. Adding downstream EV always favors 1-seeds, making EV = chalk
            # 3. Per-slot EV correctly captures that a 5-seed with 52% R32 prob
            #    is better than a 4-seed with 41% R32 prob at that specific slot
            adv_a = adv_probs[t_a["id"]][round_idx + 1]
            adv_b = adv_probs[t_b["id"]][round_idx + 1]

            ev_a = adv_a * SCORING[round_idx]
            ev_b = adv_b * SCORING[round_idx]

            if ev_a >= ev_b:
                return t_a, prob_a
            else:
                return t_b, 1.0 - prob_a

        elif strategy == "contrarian":
            # Blended approach: EV * leverage_bonus
            # leverage_bonus = (model_prob / public_prob) ^ alpha
            # alpha = 0.5 to prevent over-weighting extreme leverage
            ALPHA = 0.5

            adv_a = adv_probs[t_a["id"]][round_idx + 1]
            adv_b = adv_probs[t_b["id"]][round_idx + 1]

            pub_a = _get_public_pick_rate(seed_a, round_idx + 1)
            pub_b = _get_public_pick_rate(seed_b, round_idx + 1)

            # For R64, use matchup-specific public picks
            if round_idx == 0:
                matchup_key = (min(seed_a, seed_b), max(seed_a, seed_b))
                pub_rate = PUBLIC_PICK_RATES.get(matchup_key, 0.5)
                if seed_a < seed_b:  # lower seed number = higher seed
                    pub_a = pub_rate
                    pub_b = 1.0 - pub_rate
                else:
                    pub_b = pub_rate
                    pub_a = 1.0 - pub_rate

            # Compute leverage-weighted EV for this round
            pts = SCORING[round_idx]
            lev_a = adv_a * pts * (adv_a / max(pub_a, 0.001)) ** ALPHA
            lev_b = adv_b * pts * (adv_b / max(pub_b, 0.001)) ** ALPHA

            # Add future rounds with same leverage weighting
            for fr in range(round_idx + 1, 6):
                f_pts = SCORING[fr]
                f_adv_a = adv_probs[t_a["id"]][fr + 1]
                f_adv_b = adv_probs[t_b["id"]][fr + 1]
                f_pub_a = _get_public_pick_rate(seed_a, fr + 1)
                f_pub_b = _get_public_pick_rate(seed_b, fr + 1)
                lev_a += f_adv_a * f_pts * (f_adv_a / max(f_pub_a, 0.001)) ** ALPHA
                lev_b += f_adv_b * f_pts * (f_adv_b / max(f_pub_b, 0.001)) ** ALPHA

            if lev_a >= lev_b:
                return t_a, prob_a
            else:
                return t_b, 1.0 - prob_a

        elif strategy == "upset":
            # Controlled upset strategy: pick upsets in R64 where the underdog
            # has >= 30% win probability. In later rounds, revert to EV-based
            # picks to keep the bracket competitive.
            if prob_a >= 0.5:
                favorite, underdog = t_a, t_b
                fav_prob = prob_a
            else:
                favorite, underdog = t_b, t_a
                fav_prob = 1.0 - prob_a

            upset_prob = 1.0 - fav_prob

            if round_idx == 0:
                # R64: pick upset if underdog has >= 30% and is lower-seeded
                if (upset_prob >= 0.30 and underdog["seed"] > favorite["seed"]):
                    return underdog, upset_prob
                else:
                    return favorite, fav_prob
            else:
                # Later rounds: use per-slot EV to keep bracket viable
                adv_a = adv_probs[t_a["id"]][round_idx + 1]
                adv_b = adv_probs[t_b["id"]][round_idx + 1]
                ev_a = adv_a * SCORING[round_idx]
                ev_b = adv_b * SCORING[round_idx]
                if ev_a >= ev_b:
                    return t_a, prob_a
                else:
                    return t_b, 1.0 - prob_a

    # --- Simulate region by region ---
    for region in REGIONS:
        # R64
        r64_winners = []
        for i, (t_a, t_b) in enumerate(bracket[region]):
            winner, prob = decide_winner(t_a, t_b, 0, "R64")
            is_upset = winner["seed"] > min(t_a["seed"], t_b["seed"])
            if is_upset:
                upset_counts["R64"] += 1
            all_picks.append({
                "round": "R64", "region": region,
                "team_a": t_a, "team_b": t_b,
                "pick": winner, "win_prob": prob,
                "is_upset": is_upset,
            })
            r64_winners.append(winner)

        # R32 (pair consecutive R64 winners)
        r32_winners = []
        for i in range(0, 8, 2):
            t_a, t_b = r64_winners[i], r64_winners[i + 1]
            winner, prob = decide_winner(t_a, t_b, 1, "R32")
            is_upset = winner["seed"] > min(t_a["seed"], t_b["seed"])
            if is_upset:
                upset_counts["R32"] += 1
            all_picks.append({
                "round": "R32", "region": region,
                "team_a": t_a, "team_b": t_b,
                "pick": winner, "win_prob": prob,
                "is_upset": is_upset,
            })
            r32_winners.append(winner)

        # S16
        s16_winners = []
        for i in range(0, 4, 2):
            t_a, t_b = r32_winners[i], r32_winners[i + 1]
            winner, prob = decide_winner(t_a, t_b, 2, "S16")
            is_upset = winner["seed"] > min(t_a["seed"], t_b["seed"])
            if is_upset:
                upset_counts["S16"] += 1
            all_picks.append({
                "round": "S16", "region": region,
                "team_a": t_a, "team_b": t_b,
                "pick": winner, "win_prob": prob,
                "is_upset": is_upset,
            })
            s16_winners.append(winner)

        # E8 (regional final)
        t_a, t_b = s16_winners[0], s16_winners[1]
        winner, prob = decide_winner(t_a, t_b, 3, "E8")
        is_upset = winner["seed"] > min(t_a["seed"], t_b["seed"])
        if is_upset:
            upset_counts["E8"] += 1
        all_picks.append({
            "round": "E8", "region": region,
            "team_a": t_a, "team_b": t_b,
            "pick": winner, "win_prob": prob,
            "is_upset": is_upset,
        })
        region_winners[region] = winner

    # --- Final Four ---
    ff_winners = []
    for r_a, r_b in FF_REGION_PAIRS:
        t_a, t_b = region_winners[r_a], region_winners[r_b]
        winner, prob = decide_winner(t_a, t_b, 4, "F4")
        is_upset = winner["seed"] > min(t_a["seed"], t_b["seed"])
        if is_upset:
            upset_counts["F4"] += 1
        all_picks.append({
            "round": "F4", "region": f"{r_a}v{r_b}",
            "team_a": t_a, "team_b": t_b,
            "pick": winner, "win_prob": prob,
            "is_upset": is_upset,
        })
        ff_winners.append(winner)

    # --- Championship ---
    t_a, t_b = ff_winners[0], ff_winners[1]
    winner, prob = decide_winner(t_a, t_b, 5, "Championship")
    is_upset = winner["seed"] > min(t_a["seed"], t_b["seed"])
    if is_upset:
        upset_counts["Championship"] += 1
    all_picks.append({
        "round": "Championship", "region": "Final",
        "team_a": t_a, "team_b": t_b,
        "pick": winner, "win_prob": prob,
        "is_upset": is_upset,
    })

    champion = winner
    final_four = [region_winners[r] for r in REGIONS]

    summary = {
        "champion": champion,
        "final_four": final_four,
        "region_winners": region_winners,
        "upset_counts": upset_counts,
        "total_upsets": sum(upset_counts.values()),
    }

    return all_picks, summary


def compute_expected_points(picks, adv_probs):
    """Compute expected total points for a filled bracket.

    For each pick, expected points = P(that team actually reaches that round) * round_points.
    """
    round_to_idx = {"R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "Championship": 6}
    total_ev = 0.0
    for pick in picks:
        round_name = pick["round"]
        round_idx = round_to_idx[round_name]
        pts = SCORING[round_idx - 1]
        team_id = pick["pick"]["id"]
        prob = adv_probs[team_id][round_idx]
        total_ev += prob * pts
    return total_ev


# ---------------------------------------------------------------------------
# Display Functions
# ---------------------------------------------------------------------------
def display_bracket(name, picks, summary, adv_probs):
    """Print a bracket in a readable format."""
    print("\n" + "=" * 80)
    print(f"  {name}")
    print("=" * 80)

    round_order = ["R64", "R32", "S16", "E8", "F4", "Championship"]
    round_to_adv_idx = {"R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "Championship": 6}

    for round_name in round_order:
        round_picks = [p for p in picks if p["round"] == round_name]
        if not round_picks:
            continue

        pts = SCORING[round_order.index(round_name)]
        print(f"\n  --- {round_name} ({pts} pts per correct pick) ---")

        for p in round_picks:
            region = p["region"]
            t_a = p["team_a"]
            t_b = p["team_b"]
            winner = p["pick"]
            upset_marker = " ** UPSET **" if p["is_upset"] else ""

            seed_a = t_a["seed"]
            seed_b = t_b["seed"]
            name_a = t_a["name"]
            name_b = t_b["name"]
            w_name = winner["name"]
            w_seed = winner["seed"]

            # Advancement probability for the picked team
            adv_idx = round_to_adv_idx[round_name]
            adv_p = adv_probs[winner["id"]][adv_idx]

            print(f"    [{region}] ({seed_a:>2}) {name_a:<20} vs ({seed_b:>2}) {name_b:<20} "
                  f"-> ({w_seed:>2}) {w_name:<20} [adv: {adv_p:.1%}]{upset_marker}")

    # Summary
    ev = compute_expected_points(picks, adv_probs)
    print(f"\n  --- Summary ---")
    print(f"  Final Four:")
    for t in summary["final_four"]:
        adv_f4 = adv_probs[t["id"]][4]
        print(f"    ({t['seed']}) {t['name']} [F4 prob: {adv_f4:.1%}]")
    print(f"  Champion: ({summary['champion']['seed']}) {summary['champion']['name']} "
          f"[win prob: {adv_probs[summary['champion']['id']][6]:.1%}]")
    print(f"  Total upsets: {summary['total_upsets']} "
          f"(R64: {summary['upset_counts']['R64']}, "
          f"R32: {summary['upset_counts']['R32']}, "
          f"S16: {summary['upset_counts']['S16']}, "
          f"E8: {summary['upset_counts']['E8']}, "
          f"F4: {summary['upset_counts']['F4']}, "
          f"Champ: {summary['upset_counts']['Championship']})")
    print(f"  Expected points (ESPN scoring): {ev:.1f}")


def display_advancement_table(adv_probs, seeds, teams_dict):
    """Display advancement probabilities for all tournament teams."""
    print("\n" + "=" * 80)
    print("  MONTE CARLO ADVANCEMENT PROBABILITIES")
    print(f"  ({NUM_SIMULATIONS:,} simulations)")
    print("=" * 80)

    round_labels = ["R64", "R32", "S16", "E8", "F4", "Champ Game", "Champion"]

    # Gather tournament teams with their info
    tourney_teams = []
    for team_id, info in seeds.items():
        if info.get("playIn", "") in ("", "a"):  # avoid double-counting play-in teams
            if team_id in adv_probs:
                tourney_teams.append({
                    "id": team_id,
                    "name": teams_dict.get(team_id, f"Team {team_id}"),
                    "seed": info["seed"],
                    "region": info["region"],
                    "probs": adv_probs[team_id],
                })

    # Sort by champion probability descending
    tourney_teams.sort(key=lambda x: -x["probs"][6])

    # Print header
    print(f"\n  {'Team':<25} {'Seed':>4} {'Rgn':>3}  "
          f"{'R32':>6} {'S16':>6} {'E8':>6} {'F4':>6} {'Chmp':>6} {'Win':>6}")
    print("  " + "-" * 75)

    for t in tourney_teams[:40]:  # Top 40 teams
        probs = t["probs"]
        print(f"  {t['name']:<25} {t['seed']:>4} {t['region']:>3}  "
              f"{probs[1]:>6.1%} {probs[2]:>6.1%} {probs[3]:>6.1%} "
              f"{probs[4]:>6.1%} {probs[5]:>6.1%} {probs[6]:>6.1%}")


def display_best_upset_candidates(adv_probs, bracket, predictions):
    """Show the best upset picks based on model probabilities."""
    print("\n" + "=" * 80)
    print("  BEST UPSET CANDIDATES (R64)")
    print("=" * 80)

    upsets = []
    for region in REGIONS:
        for t_a, t_b in bracket[region]:
            # Identify favorite and underdog
            prob_a = get_win_prob(predictions, t_a["id"], t_b["id"])
            if t_a["seed"] < t_b["seed"]:
                fav, dog = t_a, t_b
                dog_prob = 1.0 - prob_a
            elif t_b["seed"] < t_a["seed"]:
                fav, dog = t_b, t_a
                dog_prob = prob_a
            else:
                continue  # same seed, not really an upset

            if dog_prob >= 0.20:  # Only show meaningful upset chances
                upsets.append({
                    "region": region,
                    "favorite": fav,
                    "underdog": dog,
                    "upset_prob": dog_prob,
                    "adv_r32": adv_probs.get(dog["id"], np.zeros(7))[1],
                })

    upsets.sort(key=lambda x: -x["upset_prob"])

    print(f"\n  {'Matchup':<45} {'Upset Prob':>10} {'Dog R32':>8}")
    print("  " + "-" * 65)
    for u in upsets:
        matchup = (f"[{u['region']}] ({u['favorite']['seed']}) {u['favorite']['name']} vs "
                   f"({u['underdog']['seed']}) {u['underdog']['name']}")
        print(f"  {matchup:<45} {u['upset_prob']:>10.1%} {u['adv_r32']:>8.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("  MARCH MADNESS 2026 BRACKET OPTIMIZER")
    print("  Monte Carlo Simulation + Multi-Strategy Bracket Generation")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    predictions, seeds, teams_dict = load_data()
    print(f"  Loaded {len(predictions):,} matchup predictions")
    print(f"  Loaded {len(seeds)} team seeds")

    # Build bracket structure
    print("\nBuilding bracket structure...")
    bracket, playin_games, region_seeds = build_bracket_structure(seeds, teams_dict, predictions)

    if playin_games:
        print(f"\n  Play-in games resolved ({len(playin_games)} games):")
        for pg in playin_games:
            a, b = pg["team_a"], pg["team_b"]
            prob = pg["prob_a"]
            winner = a if prob >= 0.5 else b
            print(f"    [{pg['region']}] ({a['seed']}) {a['name']} vs "
                  f"({b['seed']}) {b['name']}: "
                  f"Winner -> {winner['name']} ({max(prob, 1-prob):.1%})")

    # Print bracket matchups
    print("\n  R64 Bracket Matchups:")
    for region in REGIONS:
        region_name = {"W": "West", "X": "South", "Y": "Midwest", "Z": "East"}[region]
        print(f"\n    {region_name} Region ({region}):")
        for t_a, t_b in bracket[region]:
            prob_a = get_win_prob(predictions, t_a["id"], t_b["id"])
            print(f"      ({t_a['seed']:>2}) {t_a['name']:<20} vs ({t_b['seed']:>2}) {t_b['name']:<20} "
                  f"[{prob_a:.1%} - {1-prob_a:.1%}]")

    # Run Monte Carlo simulation
    print(f"\nRunning Monte Carlo simulation ({NUM_SIMULATIONS:,} tournaments)...")
    rng = np.random.default_rng(RANDOM_SEED)
    advancement_counts = simulate_tournament(bracket, predictions, NUM_SIMULATIONS, rng)
    adv_probs = compute_advancement_probs(advancement_counts, NUM_SIMULATIONS)
    print("  Simulation complete.")

    # Display advancement probabilities
    display_advancement_table(adv_probs, seeds, teams_dict)

    # Display best upset candidates
    display_best_upset_candidates(adv_probs, bracket, predictions)

    # Generate brackets
    print("\n\nGenerating bracket strategies...")

    # 1. Chalk bracket
    chalk_picks, chalk_summary = fill_chalk_bracket(bracket, predictions, teams_dict)
    display_bracket("STRATEGY 1: CHALK BRACKET (Always Pick Favorite)", chalk_picks, chalk_summary, adv_probs)

    # 2. Expected Value bracket
    ev_picks, ev_summary = fill_ev_bracket(bracket, predictions, teams_dict, adv_probs)
    display_bracket("STRATEGY 2: EXPECTED VALUE BRACKET (Maximize ESPN Points)", ev_picks, ev_summary, adv_probs)

    # 3. Contrarian bracket
    contrarian_picks, contrarian_summary = fill_contrarian_bracket(
        bracket, predictions, teams_dict, adv_probs, seeds)
    display_bracket("STRATEGY 3: CONTRARIAN BRACKET (Maximize Leverage vs Public)",
                    contrarian_picks, contrarian_summary, adv_probs)

    # 4. Upset-friendly bracket
    upset_picks, upset_summary = fill_upset_bracket(
        bracket, predictions, teams_dict, adv_probs, seeds)
    display_bracket("STRATEGY 4: UPSET-FRIENDLY BRACKET (Realistic Upsets)",
                    upset_picks, upset_summary, adv_probs)

    # --- Comparison Summary ---
    print("\n" + "=" * 80)
    print("  STRATEGY COMPARISON")
    print("=" * 80)

    strategies = [
        ("Chalk", chalk_picks, chalk_summary),
        ("Expected Value", ev_picks, ev_summary),
        ("Contrarian", contrarian_picks, contrarian_summary),
        ("Upset-Friendly", upset_picks, upset_summary),
    ]

    print(f"\n  {'Strategy':<20} {'Champion':<25} {'Chmp Seed':>9} "
          f"{'Upsets':>7} {'Exp Pts':>8}")
    print("  " + "-" * 72)

    for name, picks, summary in strategies:
        ev = compute_expected_points(picks, adv_probs)
        champ = summary["champion"]
        print(f"  {name:<20} ({champ['seed']}) {champ['name']:<20} {champ['seed']:>9} "
              f"{summary['total_upsets']:>7} {ev:>8.1f}")

    print(f"\n  Final Four comparison:")
    print(f"  {'Strategy':<20} ", end="")
    for r in REGIONS:
        rname = {"W": "West", "X": "South", "Y": "Midwest", "Z": "East"}[r]
        print(f"{'  ' + rname:<20}", end="")
    print()
    print("  " + "-" * 82)

    for name, picks, summary in strategies:
        print(f"  {name:<20} ", end="")
        for r in REGIONS:
            t = summary["region_winners"][r]
            label = f"({t['seed']}) {t['name']}"
            print(f"{label:<20}", end="")
        print()

    print("\n" + "=" * 80)
    print("  RECOMMENDATION")
    print("=" * 80)
    print("""
  For a SMALL pool (< 25 entries):
    Use the EXPECTED VALUE bracket. It maximizes raw expected points by
    picking the teams most likely to advance through the full path.

  For a LARGE pool (25-100+ entries):
    Use the CONTRARIAN bracket. Differentiation matters more in large pools.
    You need to pick teams the public undervalues to gain an edge. The
    leverage-based approach finds spots where our model disagrees with
    consensus.

  For a FUN/SOCIAL pool:
    Use the UPSET-FRIENDLY bracket. It picks realistic upsets where the
    model gives the underdog a genuine chance (>30%), making the bracket
    exciting to follow while still being grounded in the probabilities.

  Avoid the CHALK bracket for pool play. While it maximizes game-by-game
  accuracy, it will look like everyone else's bracket, giving you no
  competitive advantage in a pool setting.
""")


if __name__ == "__main__":
    main()
