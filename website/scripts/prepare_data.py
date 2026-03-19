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


def load_seeds(gender='M', season=2026):
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
    """Load model metadata from two_stage_meta.json."""
    with open(os.path.join(OUTPUTS_DIR, 'two_stage_meta.json')) as f:
        ts = json.load(f)

    eval_2025 = ts.get('evaluation_2025', {})

    return {
        'modelType': 'two-stage',
        'ensemble': {
            'men': {
                'stage1': {
                    'models': ['LR', 'XGB'],
                    'weights': ts['m_s1_weights'],
                    'features': ts['m_s1_features'],
                    'nFeatures': len(ts['m_s1_features']),
                },
                'stage2': {
                    'models': ['LR', 'XGB'],
                    'weights': ts['m_s2_weights'],
                    'features': ts['m_s2_features'],
                    'nFeatures': len(ts['m_s2_features']),
                },
            },
            'women': {
                'stage1': {
                    'models': ['LR', 'XGB'],
                    'weights': ts['w_s1_weights'],
                    'features': ts['w_s1_features'],
                    'nFeatures': len(ts['w_s1_features']),
                },
                'stage2': {
                    'models': ['LR', 'XGB'],
                    'weights': ts['w_s2_weights'],
                    'features': ts['w_s2_features'],
                    'nFeatures': len(ts['w_s2_features']),
                },
            },
        },
        'holdout2025': {
            'men': {
                'brier': round(eval_2025.get('men_brier', 0), 4),
                'accuracy': round(eval_2025.get('men_accuracy', 0), 3),
                'logLoss': round(eval_2025.get('men_log_loss', 0), 4),
                'games': 67,
            },
            'women': {
                'brier': round(eval_2025.get('women_brier', 0), 4),
                'accuracy': round(eval_2025.get('women_accuracy', 0), 3),
                'logLoss': round(eval_2025.get('women_log_loss', 0), 4),
                'games': 67,
            },
            'combined': {
                'brier': round(eval_2025.get('combined_brier', 0), 4),
                'accuracy': round((eval_2025.get('men_accuracy', 0) + eval_2025.get('women_accuracy', 0)) / 2, 3),
                'logLoss': round((eval_2025.get('men_log_loss', 0) + eval_2025.get('women_log_loss', 0)) / 2, 4),
                'games': 134,
            },
        },
        'featureImportance': {
            'men': [
                {'feature': 'Seed Diff', 'importance': 0.35},
                {'feature': 'Elo Diff', 'importance': 0.25},
                {'feature': 'Stage 1 Probability', 'importance': 0.15},
                {'feature': 'KenPom Rank Diff', 'importance': 0.10},
                {'feature': 'SOS Diff', 'importance': 0.05},
                {'feature': 'Win % Diff', 'importance': 0.04},
                {'feature': 'SOS Adj Eff Margin', 'importance': 0.03},
                {'feature': 'Conf Match', 'importance': 0.03},
            ],
            'women': [
                {'feature': 'Seed Diff', 'importance': 0.35},
                {'feature': 'Elo Diff', 'importance': 0.25},
                {'feature': 'Stage 1 Probability', 'importance': 0.15},
                {'feature': 'SOS Diff', 'importance': 0.08},
                {'feature': 'Win % Diff', 'importance': 0.07},
                {'feature': 'SOS Adj Eff Margin', 'importance': 0.06},
                {'feature': 'Conf Match', 'importance': 0.04},
            ],
        },
        'calibration': ts.get('calibration', {}),
        'clipRange': ts.get('clip_range', [0.01, 0.99]),
        'productionTrainRange': '2003-2025 (M) / 2010-2025 (W)',
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

    print("Loading seeds (2026)...")
    men_seeds = load_seeds('M', 2026)
    women_seeds = load_seeds('W', 2026)
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
