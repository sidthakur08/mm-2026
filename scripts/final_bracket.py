"""Final bracket picks combining model predictions, historical base rates, and Monte Carlo simulation."""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

preds = json.load(open(PROJECT / "website/public/data/predictions_men.json"))
seeds = json.load(open(PROJECT / "website/public/data/seeds.json"))
teams_data = json.load(open(PROJECT / "website/public/data/teams.json"))
hist_tourney = pd.read_csv(PROJECT / "data/MNCAATourneyCompactResults.csv")
hist_seeds = pd.read_csv(PROJECT / "data/MNCAATourneySeeds.csv")

men_seeds = seeds["men"]
men_teams = teams_data["men"]

def parse_seed(s):
    m = re.match(r"[WXYZ](\d{2})", s)
    return int(m.group(1)) if m else None

seed_lookup = {tid: info for tid, info in men_seeds.items()}

def get_prob(tid1, tid2):
    a, b = (tid1, tid2) if int(tid1) < int(tid2) else (tid2, tid1)
    p = preds.get(f"{a}_{b}", 0.5)
    return p if tid1 == a else 1 - p

# Historical seed-vs-seed win rates
hist_seeds_df = hist_seeds.copy()
hist_seeds_df["seed_num"] = hist_seeds_df["Seed"].apply(parse_seed)
seed_map = {}
for _, r in hist_seeds_df.iterrows():
    seed_map[(r["Season"], r["TeamID"])] = r["seed_num"]

hist_higher_seed_wins = {}
for _, g in hist_tourney[(hist_tourney["Season"] >= 2003) & (hist_tourney["Season"] <= 2025)].iterrows():
    ws = seed_map.get((g["Season"], g["WTeamID"]))
    ls = seed_map.get((g["Season"], g["LTeamID"]))
    if ws is None or ls is None:
        continue
    higher = min(ws, ls)
    key = (higher, max(ws, ls))
    if key not in hist_higher_seed_wins:
        hist_higher_seed_wins[key] = [0, 0]
    hist_higher_seed_wins[key][1] += 1
    if ws == higher:
        hist_higher_seed_wins[key][0] += 1

def get_hist_rate(seed1, seed2):
    higher, lower = min(seed1, seed2), max(seed1, seed2)
    key = (higher, lower)
    if key in hist_higher_seed_wins:
        w, t = hist_higher_seed_wins[key]
        return w / t if t > 0 else 0.5
    return 0.5

# Group teams by region
regions = {}
for tid, info in men_seeds.items():
    r = info["region"]
    if r not in regions:
        regions[r] = {}
    s = info["seed"]
    if s not in regions[r]:
        regions[r][s] = []
    regions[r][s].append(tid)

region_names = {"W": "East", "X": "South", "Y": "Midwest", "Z": "West"}

# First Four results
first_four = {
    ("Z", 11): "1400",  # Texas
    ("Y", 16): "1224",  # Howard
    ("X", 16): "1341",  # Prairie View
    ("Y", 11): "1275",  # Miami OH (model pick)
}

def resolve_playin(region, seed, candidates):
    key = (region, seed)
    if key in first_four:
        return first_four[key]
    if len(candidates) == 1:
        return candidates[0]
    p = get_prob(candidates[0], candidates[1])
    return candidates[0] if p >= 0.5 else candidates[1]

r64_pairs = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]

# Build initial matchups
region_matchups = {}
for rcode in ["W", "X", "Y", "Z"]:
    rteams = regions[rcode]
    matchups = []
    for top_s, bot_s in r64_pairs:
        top_tid = resolve_playin(rcode, top_s, rteams.get(top_s, []))
        bot_tid = resolve_playin(rcode, bot_s, rteams.get(bot_s, []))
        matchups.append((top_tid, bot_tid, top_s, bot_s))
    region_matchups[rcode] = matchups

# Monte Carlo simulation
N_SIMS = 50000
np.random.seed(42)
advance_counts = {}
champion_counts = {}

for sim in range(N_SIMS):
    region_champs = {}
    for rcode in ["W", "X", "Y", "Z"]:
        current = []
        for top_tid, bot_tid, ts, bs in region_matchups[rcode]:
            p = get_prob(top_tid, bot_tid)
            winner = top_tid if np.random.random() < p else bot_tid
            current.append(winner)
            if winner not in advance_counts:
                advance_counts[winner] = {r: 0 for r in ["R32","S16","E8","F4","CHAMP","WIN"]}
            advance_counts[winner]["R32"] += 1
        for rnd_name in ["S16", "E8", "F4"]:
            nxt = []
            for i in range(0, len(current), 2):
                p = get_prob(current[i], current[i+1])
                winner = current[i] if np.random.random() < p else current[i+1]
                nxt.append(winner)
                if winner not in advance_counts:
                    advance_counts[winner] = {r: 0 for r in ["R32","S16","E8","F4","CHAMP","WIN"]}
                advance_counts[winner][rnd_name] += 1
            current = nxt
        region_champs[rcode] = current[0]

    f4_pairs = [("W", "X"), ("Y", "Z")]
    f4w = []
    for r1, r2 in f4_pairs:
        p = get_prob(region_champs[r1], region_champs[r2])
        w = region_champs[r1] if np.random.random() < p else region_champs[r2]
        if w not in advance_counts:
            advance_counts[w] = {r: 0 for r in ["R32","S16","E8","F4","CHAMP","WIN"]}
        advance_counts[w]["CHAMP"] += 1
        f4w.append(w)
    p = get_prob(f4w[0], f4w[1])
    champ = f4w[0] if np.random.random() < p else f4w[1]
    if champ not in advance_counts:
        advance_counts[champ] = {r: 0 for r in ["R32","S16","E8","F4","CHAMP","WIN"]}
    advance_counts[champ]["WIN"] += 1
    champion_counts[champ] = champion_counts.get(champ, 0) + 1

def mc_pct(tid, rnd):
    return advance_counts.get(tid, {}).get(rnd, 0) / N_SIMS

def blended_prob(tid1, tid2, seed1, seed2):
    model_p = get_prob(tid1, tid2)
    hist_rate = get_hist_rate(seed1, seed2)
    hist_p = hist_rate if seed1 < seed2 else (1 - hist_rate)
    return 0.7 * model_p + 0.3 * hist_p

# Build final bracket
print("=" * 80)
print("FINAL BRACKET PICKS")
print("Model (70%) + Historical Base Rates (30%) + Monte Carlo EV")
print("=" * 80)

round_points = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}
region_champ_picks = {}
total_upsets_r64 = 0

for rcode in ["W", "X", "Z", "Y"]:  # East, South, West, Midwest
    rname = region_names[rcode]
    print(f"\n{'-' * 80}")
    print(f"  {rname} Region")
    print(f"{'-' * 80}")

    matchups = region_matchups[rcode]

    # R64
    print(f"\n  Round of 64:")
    r64_winners = []
    for top_tid, bot_tid, ts, bs in matchups:
        bp = blended_prob(top_tid, bot_tid, ts, bs)
        model_p = get_prob(top_tid, bot_tid)
        hist_rate = get_hist_rate(ts, bs)

        if bp >= 0.5:
            winner_tid, winner_seed = top_tid, ts
            loser_seed = bs
        else:
            winner_tid, winner_seed = bot_tid, bs
            loser_seed = ts

        is_upset = winner_seed > min(ts, bs)
        if is_upset:
            total_upsets_r64 += 1

        mc_adv = mc_pct(winner_tid, "R32") * 100
        upset_tag = " ** UPSET **" if is_upset else ""

        print(f"    ({ts:2d}) {men_teams[top_tid]:<18s} vs ({bs:2d}) {men_teams[bot_tid]:<18s}"
              f"  =>  ({winner_seed}) {men_teams[winner_tid]:<18s}"
              f" [model {model_p*100:.0f}%, hist {hist_rate*100:.0f}%, blend {bp*100:.0f}%, MC {mc_adv:.0f}%]{upset_tag}")

        r64_winners.append((winner_tid, winner_seed))

    # R32
    print(f"\n  Round of 32:")
    r32_winners = []
    for i in range(0, len(r64_winners), 2):
        t1, s1 = r64_winners[i]
        t2, s2 = r64_winners[i+1]

        ev1 = mc_pct(t1, "S16") * round_points[2]
        ev2 = mc_pct(t2, "S16") * round_points[2]
        model_p = get_prob(t1, t2) * 100

        if ev1 >= ev2:
            winner_tid, winner_seed = t1, s1
        else:
            winner_tid, winner_seed = t2, s2

        print(f"    ({s1:2d}) {men_teams[t1]:<18s} vs ({s2:2d}) {men_teams[t2]:<18s}"
              f"  =>  ({winner_seed}) {men_teams[winner_tid]:<18s}"
              f" [EV {ev1:.1f} vs {ev2:.1f}, model {model_p:.0f}%]")
        r32_winners.append((winner_tid, winner_seed))

    # S16
    print(f"\n  Sweet 16:")
    s16_winners = []
    for i in range(0, len(r32_winners), 2):
        t1, s1 = r32_winners[i]
        t2, s2 = r32_winners[i+1]

        ev1 = mc_pct(t1, "E8") * round_points[3]
        ev2 = mc_pct(t2, "E8") * round_points[3]

        if ev1 >= ev2:
            winner_tid, winner_seed = t1, s1
        else:
            winner_tid, winner_seed = t2, s2

        print(f"    ({s1:2d}) {men_teams[t1]:<18s} vs ({s2:2d}) {men_teams[t2]:<18s}"
              f"  =>  ({winner_seed}) {men_teams[winner_tid]:<18s}"
              f" [EV {ev1:.1f} vs {ev2:.1f}]")
        s16_winners.append((winner_tid, winner_seed))

    # E8
    print(f"\n  Elite 8:")
    t1, s1 = s16_winners[0]
    t2, s2 = s16_winners[1]
    ev1 = mc_pct(t1, "F4") * round_points[4]
    ev2 = mc_pct(t2, "F4") * round_points[4]

    if ev1 >= ev2:
        winner_tid, winner_seed = t1, s1
    else:
        winner_tid, winner_seed = t2, s2

    print(f"    ({s1:2d}) {men_teams[t1]:<18s} vs ({s2:2d}) {men_teams[t2]:<18s}"
          f"  =>  ({winner_seed}) {men_teams[winner_tid]:<18s}"
          f" [EV {ev1:.1f} vs {ev2:.1f}]")
    region_champ_picks[rcode] = (winner_tid, winner_seed)

# Final Four
print(f"\n{'=' * 80}")
print(f"  FINAL FOUR")
print(f"{'=' * 80}")
f4_pairs_list = [("W", "X"), ("Y", "Z")]
f4_winners = []
for r1, r2 in f4_pairs_list:
    t1, s1 = region_champ_picks[r1]
    t2, s2 = region_champ_picks[r2]
    ev1 = mc_pct(t1, "CHAMP") * round_points[5]
    ev2 = mc_pct(t2, "CHAMP") * round_points[5]

    if ev1 >= ev2:
        winner_tid, winner_seed = t1, s1
    else:
        winner_tid, winner_seed = t2, s2

    print(f"\n  ({s1}) {men_teams[t1]:<20s} vs ({s2}) {men_teams[t2]:<20s}"
          f"  =>  ({winner_seed}) {men_teams[winner_tid]}"
          f"  [EV {ev1:.1f} vs {ev2:.1f}]")
    f4_winners.append((winner_tid, winner_seed))

# Championship
print(f"\n{'=' * 80}")
print(f"  CHAMPIONSHIP")
print(f"{'=' * 80}")
t1, s1 = f4_winners[0]
t2, s2 = f4_winners[1]
mc1 = mc_pct(t1, "WIN") * 100
mc2 = mc_pct(t2, "WIN") * 100

if mc1 >= mc2:
    champ_tid, champ_seed = t1, s1
else:
    champ_tid, champ_seed = t2, s2

print(f"\n  ({s1}) {men_teams[t1]:<20s} vs ({s2}) {men_teams[t2]:<20s}")
print(f"  Title odds: {mc1:.1f}% vs {mc2:.1f}%")
print(f"\n  {'*' * 50}")
print(f"  ***  CHAMPION: ({champ_seed}) {men_teams[champ_tid]}  ***")
print(f"  {'*' * 50}")

print(f"\n  R64 upsets picked: {total_upsets_r64}")
print(f"  Historical avg R64 upsets: ~8 per tournament")

# Top 10 title contenders
print(f"\n{'=' * 80}")
print(f"  TOP 10 TITLE CONTENDERS (Monte Carlo)")
print(f"{'=' * 80}")
sorted_champs = sorted(champion_counts.items(), key=lambda x: -x[1])
for tid, count in sorted_champs[:10]:
    s = seed_lookup[tid]["seed"]
    r = region_names[seed_lookup[tid]["region"]]
    print(f"  ({s:2d}) {men_teams[tid]:<20s} {r:<10s} {count/N_SIMS*100:.1f}%")
