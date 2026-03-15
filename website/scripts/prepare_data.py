"""
Prepare static JSON data files for the March Madness website.
Reads from the mm-2026 data/outputs directories and generates
compact JSON files for the frontend.
"""
import csv
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
PUBLIC_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'public', 'data')


def load_teams():
    """Load team ID -> name mappings for men's and women's."""
    men = {}
    with open(os.path.join(DATA_DIR, 'MTeams.csv')) as f:
        for row in csv.DictReader(f):
            if int(row['LastD1Season']) >= 2026:
                men[row['TeamID']] = row['TeamName']

    women = {}
    with open(os.path.join(DATA_DIR, 'WTeams.csv')) as f:
        for row in csv.DictReader(f):
            women[row['TeamID']] = row['TeamName']

    return men, women


def load_predictions():
    """Load submission_stage2.csv and split into men's and women's predictions."""
    men_preds = {}
    women_preds = {}

    with open(os.path.join(OUTPUTS_DIR, 'submission_stage2.csv')) as f:
        for row in csv.DictReader(f):
            parts = row['ID'].split('_')
            team_a = parts[1]
            team_b = parts[2]
            prob = round(float(row['Pred']), 4)

            key = f"{team_a}_{team_b}"
            if team_a.startswith('1'):
                men_preds[key] = prob
            else:
                women_preds[key] = prob

    return men_preds, women_preds


def load_seeds(gender='M', season=2025):
    """Load tournament seeds for a given season."""
    filename = f"{'M' if gender == 'M' else 'W'}NCAATourneySeeds.csv"
    seeds = {}
    with open(os.path.join(DATA_DIR, filename)) as f:
        for row in csv.DictReader(f):
            if int(row['Season']) == season:
                seed_str = row['Seed']
                region = seed_str[0]
                seed_num = seed_str[1:3]
                play_in = seed_str[3:] if len(seed_str) > 3 else ''
                seeds[row['TeamID']] = {
                    'region': region,
                    'seed': int(seed_num),
                    'playIn': play_in,
                    'seedStr': seed_str,
                }
    return seeds


def load_model_info():
    """Load model metadata from feature_columns.json + PROJECT_SUMMARY stats."""
    with open(os.path.join(OUTPUTS_DIR, 'feature_columns.json')) as f:
        fc = json.load(f)

    return {
        'ensemble': {
            'men': {
                'models': fc['m_ensemble_models'],
                'weights': fc['m_ensemble_weights'],
                'features': fc['m_feature_columns'],
                'nFeatures': fc['m_n_features'],
            },
            'women': {
                'models': fc['w_ensemble_models'],
                'weights': fc['w_ensemble_weights'],
                'features': fc['w_feature_columns'],
                'nFeatures': fc['w_n_features'],
            },
        },
        'performance': {
            'men': {
                'testBrier': fc['m_test_brier'],
                'testLogLoss': fc['m_test_logloss'],
                'testAccuracy': fc['m_test_accuracy'],
                'valBrier': fc['m_val_brier'],
                'trainSeasons': fc['m_train_seasons'],
            },
            'women': {
                'testBrier': fc['w_test_brier'],
                'testLogLoss': fc['w_test_logloss'],
                'testAccuracy': fc['w_test_accuracy'],
                'valBrier': fc['w_val_brier'],
                'trainSeasons': fc['w_train_seasons'],
            },
        },
        'holdout2025': {
            'men': {'brier': 0.1754, 'accuracy': 0.731, 'logLoss': 0.5223, 'games': 67},
            'women': {'brier': 0.1450, 'accuracy': 0.821, 'logLoss': 0.4441, 'games': 67},
            'combined': {'brier': 0.1602, 'accuracy': 0.776, 'logLoss': 0.4832, 'games': 134},
        },
        'temporalCV': {
            'men': {
                '2015': 0.1869, '2016': 0.1977, '2017': 0.1782, '2018': 0.2007,
                '2019': 0.1897, '2021': 0.2132, '2022': 0.2248, '2023': 0.2230,
                '2024': 0.2051, '2025': 0.1699,
            },
            'women': {
                '2015': 0.1423, '2016': 0.1761, '2017': 0.1565, '2018': 0.1569,
                '2019': 0.1501, '2021': 0.1955, '2022': 0.1647, '2023': 0.1892,
                '2024': 0.1409, '2025': 0.1370,
            },
        },
        'featureImportance': {
            'men': [
                {'feature': 'Win %', 'importance': 0.40},
                {'feature': 'KenPom Rank', 'importance': 0.30},
                {'feature': 'Point Diff (10g)', 'importance': 0.05},
                {'feature': 'Strength of Schedule', 'importance': 0.03},
                {'feature': 'Win Rate (5g)', 'importance': 0.03},
                {'feature': 'Off. Efficiency (10g)', 'importance': 0.02},
                {'feature': 'Def. Efficiency (10g)', 'importance': 0.02},
                {'feature': 'True Shooting (10g)', 'importance': 0.02},
            ],
            'women': [
                {'feature': 'Point Diff (10g)', 'importance': 0.34},
                {'feature': 'Win %', 'importance': 0.30},
                {'feature': 'Off. Efficiency (10g)', 'importance': 0.11},
                {'feature': 'Strength of Schedule', 'importance': 0.05},
                {'feature': 'Def. Efficiency (10g)', 'importance': 0.02},
                {'feature': 'True Shooting (10g)', 'importance': 0.02},
                {'feature': 'Win Rate (5g)', 'importance': 0.02},
                {'feature': 'Ast/TO Ratio (10g)', 'importance': 0.01},
            ],
        },
        'productionTrainRange': fc['production_train_range'],
    }


def load_team_stats():
    """Load key team stats for the matchup explorer comparison view."""
    stats = {'men': {}, 'women': {}}

    for gender, prefix in [('men', 'm'), ('women', 'w')]:
        filepath = os.path.join(OUTPUTS_DIR, f'{prefix}_team_stats.csv')
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            for row in reader:
                if int(row.get('Season', 0)) != 2026:
                    continue
                team_id = row.get('TeamID', '')
                if not team_id:
                    continue
                col_map = {
                    'win_pct': 'winPct',
                    'sos': 'sos',
                    'kenpom_rank': 'kenpomRank',
                    'off_efficiency': 'offEfficiency',
                    'def_efficiency': 'defEfficiency',
                    'efficiency_margin': 'efficiencyMargin',
                    'point_diff_avg': 'pointDiff',
                    'ts_pct': 'trueShooting',
                    'to_rate': 'turnoverRate',
                    'ast_to_ratio': 'astToRatio',
                    'points_scored_avg': 'ppg',
                    'points_allowed_avg': 'oppPpg',
                    'scoring_consistency': 'consistency',
                }
                stat_entry = {}
                for csv_col, json_key in col_map.items():
                    if csv_col in row and row[csv_col]:
                        try:
                            stat_entry[json_key] = round(float(row[csv_col]), 4)
                        except ValueError:
                            pass
                if stat_entry:
                    stats[gender][team_id] = stat_entry

    return stats


def main():
    os.makedirs(PUBLIC_DATA_DIR, exist_ok=True)

    print("Loading teams...")
    men_teams, women_teams = load_teams()
    print(f"  Men: {len(men_teams)} teams, Women: {len(women_teams)} teams")

    print("Loading predictions...")
    men_preds, women_preds = load_predictions()
    print(f"  Men: {len(men_preds)} matchups, Women: {len(women_preds)} matchups")

    print("Loading seeds (2025 as demo)...")
    men_seeds = load_seeds('M', 2025)
    women_seeds = load_seeds('W', 2025)
    # Try 2026 seeds if available
    men_seeds_2026 = load_seeds('M', 2026)
    women_seeds_2026 = load_seeds('W', 2026)
    if men_seeds_2026:
        men_seeds = men_seeds_2026
        print("  Found 2026 men's seeds!")
    if women_seeds_2026:
        women_seeds = women_seeds_2026
        print("  Found 2026 women's seeds!")
    print(f"  Men: {len(men_seeds)} seeded teams, Women: {len(women_seeds)} seeded teams")

    print("Loading model info...")
    model_info = load_model_info()

    print("Loading team stats...")
    team_stats = load_team_stats()
    print(f"  Men: {len(team_stats['men'])} teams with stats, Women: {len(team_stats['women'])} teams with stats")

    # Write teams
    teams_data = {'men': men_teams, 'women': women_teams}
    with open(os.path.join(PUBLIC_DATA_DIR, 'teams.json'), 'w') as f:
        json.dump(teams_data, f, separators=(',', ':'))
    print(f"  teams.json: {os.path.getsize(os.path.join(PUBLIC_DATA_DIR, 'teams.json')) / 1024:.1f} KB")

    # Write predictions (separate files to allow lazy loading)
    with open(os.path.join(PUBLIC_DATA_DIR, 'predictions_men.json'), 'w') as f:
        json.dump(men_preds, f, separators=(',', ':'))
    print(f"  predictions_men.json: {os.path.getsize(os.path.join(PUBLIC_DATA_DIR, 'predictions_men.json')) / 1024:.1f} KB")

    with open(os.path.join(PUBLIC_DATA_DIR, 'predictions_women.json'), 'w') as f:
        json.dump(women_preds, f, separators=(',', ':'))
    print(f"  predictions_women.json: {os.path.getsize(os.path.join(PUBLIC_DATA_DIR, 'predictions_women.json')) / 1024:.1f} KB")

    # Write seeds
    seeds_data = {'men': men_seeds, 'women': women_seeds}
    with open(os.path.join(PUBLIC_DATA_DIR, 'seeds.json'), 'w') as f:
        json.dump(seeds_data, f, separators=(',', ':'))
    print(f"  seeds.json: {os.path.getsize(os.path.join(PUBLIC_DATA_DIR, 'seeds.json')) / 1024:.1f} KB")

    # Write model info
    with open(os.path.join(PUBLIC_DATA_DIR, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, separators=(',', ':'), indent=None)
    print(f"  model_info.json: {os.path.getsize(os.path.join(PUBLIC_DATA_DIR, 'model_info.json')) / 1024:.1f} KB")

    # Write team stats
    with open(os.path.join(PUBLIC_DATA_DIR, 'team_stats.json'), 'w') as f:
        json.dump(team_stats, f, separators=(',', ':'))
    print(f"  team_stats.json: {os.path.getsize(os.path.join(PUBLIC_DATA_DIR, 'team_stats.json')) / 1024:.1f} KB")

    print("\nDone! All data files written to public/data/")


if __name__ == '__main__':
    main()
