"""Experiment: Test Stage 2 improvements head-to-head.

Tests three approaches on 2025 holdout:
  A) Current: Full Stage 2 (stage1_prob + seed_diff + conf_match + s2_*_diff + s2_elo_diff)
  B) Lean:    Lean Stage 2 (stage1_prob + seed_diff + conf_match + s2_elo_diff only)
  C) Blend:   Alpha-blend of stage1_prob with historical seed-based prior

Also runs Leave-One-Season-Out CV for robust evaluation of each approach.
"""

import sys
import os
import time
import warnings
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_regular_season_results, load_tourney_results,
    load_seeds, load_massey_ordinals, load_team_conferences,
)
from src.features import (
    build_team_season_stats, build_advanced_team_stats,
    add_strength_of_schedule, add_seed_features,
    add_massey_features, add_single_system_ranking,
    add_sos_adjusted_features, build_rolling_training_data,
    get_team_rolling_snapshot, compute_rolling_window_stats,
    compute_elo_ratings, get_team_elo_snapshot,
)
from src.model import EnsemblePredictor

RANDOM_STATE = 42
ELO_K = 20
ELO_CARRYOVER = 0.75
ELO_MARGIN = True
N_TRIALS_S1 = 50
N_TRIALS_S2 = 50

t_start = time.time()


def parse_seed_number(seed_str):
    import re
    m = re.match(r"[WXYZ](\d{2})", seed_str)
    return int(m.group(1)) if m else None


def score(y_true, y_pred, label=""):
    bs = brier_score_loss(y_true, y_pred)
    ll = log_loss(y_true, y_pred)
    acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    return {"label": label, "brier": bs, "log_loss": ll, "accuracy": acc, "n": len(y_true)}


# ===================================================================
# STEP 0: Load data (same as main pipeline)
# ===================================================================
print("=" * 70)
print("Loading data...")
print("=" * 70)

m_rs_detailed = load_regular_season_results(gender="M", detailed=True)
m_rs_compact = load_regular_season_results(gender="M", detailed=False)
m_tourney_compact = load_tourney_results(gender="M", detailed=False)
m_seeds = load_seeds(gender="M")
m_conf = load_team_conferences(gender="M")

w_rs_detailed = load_regular_season_results(gender="W", detailed=True)
w_rs_compact = load_regular_season_results(gender="W", detailed=False)
w_tourney_compact = load_tourney_results(gender="W", detailed=False)
w_seeds = load_seeds(gender="W")
w_conf = load_team_conferences(gender="W")

m_ordinals = load_massey_ordinals()
print(f"Data loaded in {time.time()-t_start:.0f}s")

# ===================================================================
# STEP 1: Elo
# ===================================================================
print("\n" + "=" * 70)
print("Computing Elo ratings...")
print("=" * 70)

m_all_games = pd.concat([m_rs_compact, m_tourney_compact], ignore_index=True)
w_all_games = pd.concat([w_rs_compact, w_tourney_compact], ignore_index=True)
m_elo_df = compute_elo_ratings(m_all_games, k_factor=ELO_K, season_carryover=ELO_CARRYOVER, margin_factor=ELO_MARGIN)
w_elo_df = compute_elo_ratings(w_all_games, k_factor=ELO_K, season_carryover=ELO_CARRYOVER, margin_factor=ELO_MARGIN)
print(f"Elo computed in {time.time()-t_start:.0f}s")

# ===================================================================
# STEP 2: Static features
# ===================================================================
print("\n" + "=" * 70)
print("Building static features...")
print("=" * 70)


def build_static(rs_detailed, rs_compact, seeds, gender, ordinals=None):
    stats = build_team_season_stats(rs_detailed, compact_results_df=rs_compact)
    adv = build_advanced_team_stats(rs_detailed, compact_results_df=rs_compact)
    stats = stats.join(adv, how="left")
    stats = add_strength_of_schedule(stats, rs_detailed, rs_compact)
    stats = add_seed_features(stats, seeds)
    stats = add_sos_adjusted_features(stats)
    if ordinals is not None and gender == "M":
        stats = add_massey_features(stats, ordinals)
        stats = add_single_system_ranking(stats, ordinals, system_name="POM", col_name="kenpom_rank")
    stats["neutral_win_pct"] = stats["neutral_win_pct"].fillna(stats["win_pct"])
    stats["away_win_pct"] = stats["away_win_pct"].fillna(stats["win_pct"])
    return stats


m_stats = build_static(m_rs_detailed, m_rs_compact, m_seeds, "M", m_ordinals)
w_stats = build_static(w_rs_detailed, w_rs_compact, w_seeds, "W")
print(f"Static features built in {time.time()-t_start:.0f}s")

# ===================================================================
# STEP 3: Stage 1 training data + feature selection + training
# ===================================================================
print("\n" + "=" * 70)
print("Building & training Stage 1...")
print("=" * 70)


def build_s1_data(rs_detailed, rs_compact, elo_df, stats, gender):
    s1_train = build_rolling_training_data(rs_detailed, rs_compact, windows=(5, 7, 10))

    # Add static diffs
    static_cols = list(set(["sos", "win_pct", "efficiency_margin", "sos_adj_eff_margin"]) &
                       set(stats.columns))
    if gender == "M" and "kenpom_rank" in stats.columns:
        static_cols.append("kenpom_rank")

    for col in static_cols:
        key = f"static_{col}"
        vals_a, vals_b = [], []
        for _, row in s1_train.iterrows():
            s, ta, tb = row["Season"], row["TeamA"], row["TeamB"]
            try:
                vals_a.append(stats.loc[(s, ta), col])
            except KeyError:
                vals_a.append(np.nan)
            try:
                vals_b.append(stats.loc[(s, tb), col])
            except KeyError:
                vals_b.append(np.nan)
        s1_train[key] = np.array(vals_a) - np.array(vals_b)

    # Add elo_diff
    elo_lookup = {}
    for _, row in elo_df.iterrows():
        elo_lookup[(row["Season"], row["TeamID"], row["DayNum"])] = row["Elo"]

    # Build elo snapshot per game (pre-game elo)
    rs_elo = elo_df[elo_df["DayNum"] < 134]
    elo_snap_by_season = {}
    for season in s1_train["Season"].unique():
        se = rs_elo[rs_elo["Season"] == season]
        if len(se) > 0:
            elo_snap_by_season[season] = se.sort_values("DayNum").groupby("TeamID")["Elo"].last().to_dict()
        else:
            elo_snap_by_season[season] = {}

    elo_diffs = []
    for _, row in s1_train.iterrows():
        s = row["Season"]
        snap = elo_snap_by_season.get(s, {})
        elo_diffs.append(snap.get(row["TeamA"], 1500.0) - snap.get(row["TeamB"], 1500.0))
    s1_train["elo_diff"] = elo_diffs

    return s1_train, static_cols


m_s1_train, m_s2_static_full = build_s1_data(m_rs_detailed, m_rs_compact, m_elo_df, m_stats, "M")
w_s1_train, w_s2_static_full = build_s1_data(w_rs_detailed, w_rs_compact, w_elo_df, w_stats, "W")
print(f"Stage 1 data built: M={len(m_s1_train)}, W={len(w_s1_train)}")


# Feature selection
def select_features(df, feat_cols, target_col="target", mi_threshold=0.001, corr_threshold=0.85,
                    protect=None):
    if protect is None:
        protect = set()
    X = df[feat_cols].fillna(0.0)
    y = df[target_col]
    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi, index=feat_cols)
    keep = [f for f in feat_cols if mi_series[f] >= mi_threshold or f in protect]

    # Correlation pruning
    corr = X[keep].corr().abs()
    drop = set()
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            fi, fj = keep[i], keep[j]
            if fi in drop or fj in drop:
                continue
            if corr.loc[fi, fj] > corr_threshold:
                if fi in protect:
                    drop.add(fj)
                elif fj in protect:
                    drop.add(fi)
                elif mi_series[fi] >= mi_series[fj]:
                    drop.add(fj)
                else:
                    drop.add(fi)
    final = [f for f in keep if f not in drop]
    return final


exclude_cols = {"Season", "TeamA", "TeamB", "target", "DayNum"}
m_s1_feat_candidates = [c for c in m_s1_train.columns if c not in exclude_cols]
w_s1_feat_candidates = [c for c in w_s1_train.columns if c not in exclude_cols]

protect_feats = {"elo_diff", "static_kenpom_rank", "static_win_pct"}
m_s1_features = select_features(m_s1_train, m_s1_feat_candidates, protect=protect_feats)
w_s1_features = select_features(w_s1_train, w_s1_feat_candidates, protect=protect_feats)
print(f"M Stage 1 features: {len(m_s1_features)}, W Stage 1 features: {len(w_s1_features)}")
print(f"  elo_diff in M: {'elo_diff' in m_s1_features}, W: {'elo_diff' in w_s1_features}")


# Split + train Stage 1
def split_s1(df, features, train_end=2022, val_end=2024):
    medians = df.loc[df["Season"] <= train_end, features].median().to_dict()
    sets = {}
    for name, mask in [("train", df["Season"] <= train_end),
                       ("val", (df["Season"] > train_end) & (df["Season"] <= val_end)),
                       ("test", df["Season"] > val_end)]:
        X = df.loc[mask, features].fillna(pd.Series(medians)).fillna(0.0)
        y = df.loc[mask, "target"]
        sets[name] = (X, y)
    return sets, medians


m_s1_sets, m_s1_medians = split_s1(m_s1_train, m_s1_features)
w_s1_sets, w_s1_medians = split_s1(w_s1_train, w_s1_features)


def optuna_s1(X_tr, y_tr, X_val, y_val, n_trials=N_TRIALS_S1):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("lr", 0.005, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_est", 100, 800),
            "max_depth": trial.suggest_int("md", 2, 6),
            "subsample": trial.suggest_float("ss", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("cs", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("ra", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("rl", 0.01, 10.0, log=True),
            "min_child_weight": trial.suggest_int("mcw", 3, 20),
            "random_state": RANDOM_STATE, "eval_metric": "logloss", "verbosity": 0,
        }
        m = XGBClassifier(**params)
        m.fit(X_tr, y_tr)
        pred = m.predict_proba(X_val)[:, 1]
        return log_loss(y_val, pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_weights(models, X_val, y_val):
    best_w, best_ll = None, 1e9
    for w1 in np.arange(0, 1.02, 0.02):
        w = [w1, 1 - w1]
        pred = sum(wi * m.predict_proba(X_val)[:, 1] for wi, m in zip(w, models))
        ll = log_loss(y_val, pred)
        if ll < best_ll:
            best_ll = ll
            best_w = w
    return best_w, best_ll


def train_s1_ensemble(X_tr, y_tr, X_val, y_val, xgb_params):
    lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE))])
    lr.fit(X_tr, y_tr)

    full_params = {k: v for k, v in xgb_params.items()}
    for k in ["lr", "n_est", "md", "ss", "cs", "ra", "rl", "mcw"]:
        full_params.pop(k, None)
    rename = {"lr": "learning_rate", "n_est": "n_estimators", "md": "max_depth",
              "ss": "subsample", "cs": "colsample_bytree", "ra": "reg_alpha",
              "rl": "reg_lambda", "mcw": "min_child_weight"}
    clean_params = {rename.get(k, k): v for k, v in xgb_params.items()}
    clean_params.update({"random_state": RANDOM_STATE, "eval_metric": "logloss", "verbosity": 0})

    xgb = XGBClassifier(**clean_params)
    xgb.fit(X_tr, y_tr)

    weights, val_ll = optimize_weights([lr, xgb], X_val, y_val)
    return EnsemblePredictor([lr, xgb], weights), weights, clean_params


print("\nTuning Men's Stage 1...")
m_s1_bp = optuna_s1(*m_s1_sets["train"], *m_s1_sets["val"])
m_s1_model, m_s1_weights, m_s1_xgb_params = train_s1_ensemble(*m_s1_sets["train"], *m_s1_sets["val"], m_s1_bp)
s1_test_pred = m_s1_model.predict_proba(m_s1_sets["test"][0])[:, 1]
print(f"  M S1 test brier: {brier_score_loss(m_s1_sets['test'][1], s1_test_pred):.4f}, weights={m_s1_weights}")

print("Tuning Women's Stage 1...")
w_s1_bp = optuna_s1(*w_s1_sets["train"], *w_s1_sets["val"])
w_s1_model, w_s1_weights, w_s1_xgb_params = train_s1_ensemble(*w_s1_sets["train"], *w_s1_sets["val"], w_s1_bp)
s1_test_pred_w = w_s1_model.predict_proba(w_s1_sets["test"][0])[:, 1]
print(f"  W S1 test brier: {brier_score_loss(w_s1_sets['test'][1], s1_test_pred_w):.4f}, weights={w_s1_weights}")

print(f"\nStage 1 complete in {time.time()-t_start:.0f}s")

# ===================================================================
# STEP 4: Build rolling snapshots for all tournament seasons
# ===================================================================
print("\n" + "=" * 70)
print("Building snapshots for tournament seasons...")
print("=" * 70)


def compute_all_snapshots_fast(rs_detailed, rs_compact, seasons):
    all_team_games = compute_rolling_window_stats(rs_detailed, rs_compact, windows=(5, 7, 10))
    roll_cols = [c for c in all_team_games.columns if "_roll" in c]
    keep_cols = ["Season", "TeamID", "DayNum"] + roll_cols

    snap_dict = {}
    for s in seasons:
        sg = all_team_games[all_team_games["Season"] == s]
        if len(sg) == 0:
            continue
        latest = sg[keep_cols].sort_values("DayNum").groupby("TeamID").last()
        if len(latest) > 0:
            snap_dict[s] = latest
    return snap_dict


m_tourney_seasons = sorted(m_tourney_compact["Season"].unique())
w_tourney_seasons = sorted(w_tourney_compact["Season"].unique())

m_snap_dict = compute_all_snapshots_fast(m_rs_detailed, m_rs_compact, m_tourney_seasons)
w_snap_dict = compute_all_snapshots_fast(w_rs_detailed, w_rs_compact, w_tourney_seasons)
print(f"Snapshots: M={len(m_snap_dict)}, W={len(w_snap_dict)}")

# ===================================================================
# STEP 5: Build tournament features for ALL approaches
# ===================================================================
print("\n" + "=" * 70)
print("Building Stage 2 tournament features...")
print("=" * 70)


def build_tournament_features(tourney_df, seeds_df, conf_df, snap_dict, team_stats,
                              stage1_model, stage1_features, stage1_medians,
                              elo_df, gender="M", static_cols=None):
    """Build Stage 2 features for tournament games."""
    if static_cols is None:
        static_cols = []

    seed_lookup = {}
    for _, row in seeds_df.iterrows():
        sn = parse_seed_number(row["Seed"])
        seed_lookup[(row["Season"], row["TeamID"])] = sn

    conf_lookup = {}
    for _, row in conf_df.iterrows():
        conf_lookup[(row["Season"], row["TeamID"])] = row["ConfAbbrev"]

    # Pre-compute Elo snapshots
    elo_snapshots = {}
    for season in tourney_df["Season"].unique():
        rs_elo = elo_df[(elo_df["Season"] == season) & (elo_df["DayNum"] < 134)]
        if len(rs_elo) > 0:
            elo_snapshots[season] = rs_elo.sort_values("DayNum").groupby("TeamID")["Elo"].last()

    rows = []
    skipped = 0
    for _, game in tourney_df.iterrows():
        season = game["Season"]
        w_id, l_id = game["WTeamID"], game["LTeamID"]
        team_a, team_b = min(w_id, l_id), max(w_id, l_id)
        target = 1 if w_id == team_a else 0

        if season not in snap_dict:
            skipped += 1
            continue
        snap = snap_dict[season]
        if team_a not in snap.index or team_b not in snap.index:
            skipped += 1
            continue

        snap_a, snap_b = snap.loc[team_a], snap.loc[team_b]

        # Stage 1 features
        roll_cols = [c for c in snap.columns if "_roll" in c]
        s1_feats = {}
        for col in roll_cols:
            s1_feats[col] = snap_a[col] - snap_b[col]

        try:
            stats_season = team_stats.loc[season]
            for col in static_cols:
                key = f"static_{col}"
                if col in stats_season.columns:
                    va = stats_season.loc[team_a, col] if team_a in stats_season.index else np.nan
                    vb = stats_season.loc[team_b, col] if team_b in stats_season.index else np.nan
                    s1_feats[key] = va - vb
        except (KeyError, TypeError):
            pass

        if season in elo_snapshots:
            elo_snap = elo_snapshots[season]
            elo_a = elo_snap.get(team_a, 1500.0)
            elo_b = elo_snap.get(team_b, 1500.0)
            s1_feats["elo_diff"] = elo_a - elo_b
        else:
            s1_feats["elo_diff"] = 0.0
            elo_a, elo_b = 1500.0, 1500.0

        s1_vec = pd.DataFrame([s1_feats])
        for fc in stage1_features:
            if fc not in s1_vec.columns:
                s1_vec[fc] = stage1_medians.get(fc, 0.0)
        s1_vec = s1_vec[stage1_features].fillna(pd.Series(stage1_medians)).fillna(0.0)
        s1_prob = stage1_model.predict_proba(s1_vec.values)[:, 1][0]

        # Stage 2 features
        seed_a = seed_lookup.get((season, team_a))
        seed_b = seed_lookup.get((season, team_b))

        s2_row = {
            "stage1_prob": s1_prob,
            "seed_diff": (seed_a - seed_b) if (seed_a is not None and seed_b is not None) else 0,
            "seed_a": seed_a if seed_a is not None else 8,
            "seed_b": seed_b if seed_b is not None else 8,
            "conf_match": 1 if conf_lookup.get((season, team_a)) == conf_lookup.get((season, team_b)) else 0,
        }

        # Full static diffs for Stage 2
        for col in static_cols:
            try:
                va = team_stats.loc[(season, team_a), col]
                vb = team_stats.loc[(season, team_b), col]
                s2_row[f"s2_{col}_diff"] = va - vb
            except KeyError:
                s2_row[f"s2_{col}_diff"] = 0.0

        # Elo diff for Stage 2
        s2_row["s2_elo_diff"] = elo_a - elo_b

        s2_row["target"] = target
        s2_row["Season"] = season
        s2_row["TeamA"] = team_a
        s2_row["TeamB"] = team_b

        rows.append(s2_row)

    return pd.DataFrame(rows)


# Build with FULL static cols (for approach A) — includes all signals
m_s2_static_full = ["sos", "win_pct", "kenpom_rank", "sos_adj_eff_margin"]
w_s2_static_full = ["sos", "win_pct", "sos_adj_eff_margin"]

m_s2_data = build_tournament_features(
    m_tourney_compact, m_seeds, m_conf, m_snap_dict, m_stats,
    m_s1_model, m_s1_features, m_s1_medians, m_elo_df, gender="M",
    static_cols=m_s2_static_full,
)
w_s2_data = build_tournament_features(
    w_tourney_compact, w_seeds, w_conf, w_snap_dict, w_stats,
    w_s1_model, w_s1_features, w_s1_medians, w_elo_df, gender="W",
    static_cols=w_s2_static_full,
)
print(f"Stage 2 data: M={len(m_s2_data)}, W={len(w_s2_data)}")

# ===================================================================
# STEP 6: Historical seed-vs-seed win rates (for approach C)
# ===================================================================
print("\n" + "=" * 70)
print("Computing historical seed-vs-seed win rates...")
print("=" * 70)


def build_seed_prior(tourney_df, seeds_df):
    """Compute P(lower_seed_num wins) for each seed matchup from historical data."""
    seed_lookup = {}
    for _, row in seeds_df.iterrows():
        sn = parse_seed_number(row["Seed"])
        seed_lookup[(row["Season"], row["TeamID"])] = sn

    records = {}  # (seed_a, seed_b) -> [wins_for_a, total]
    for _, game in tourney_df.iterrows():
        season = game["Season"]
        w_id, l_id = game["WTeamID"], game["LTeamID"]
        team_a, team_b = min(w_id, l_id), max(w_id, l_id)

        seed_a = seed_lookup.get((season, team_a))
        seed_b = seed_lookup.get((season, team_b))
        if seed_a is None or seed_b is None:
            continue

        key = (seed_a, seed_b)
        if key not in records:
            records[key] = [0, 0]
        records[key][1] += 1
        if w_id == team_a:
            records[key][0] += 1

    # Build lookup: P(team_a wins | seed_a, seed_b)
    seed_prior = {}
    for (sa, sb), (wins, total) in records.items():
        seed_prior[(sa, sb)] = wins / total if total > 0 else 0.5

    return seed_prior


m_seed_prior = build_seed_prior(m_tourney_compact, m_seeds)
w_seed_prior = build_seed_prior(w_tourney_compact, w_seeds)
print(f"Seed matchup records: M={len(m_seed_prior)}, W={len(w_seed_prior)}")

# Show some interesting seed priors
print("\nSample M seed priors (seed_a vs seed_b -> P(a wins)):")
for (sa, sb) in sorted(m_seed_prior.keys())[:15]:
    print(f"  {sa:2d} vs {sb:2d}: {m_seed_prior[(sa,sb)]:.3f}")


# ===================================================================
# STEP 7: Define the three approaches & evaluate
# ===================================================================
print("\n" + "=" * 70)
print("EVALUATING THREE STAGE 2 APPROACHES")
print("=" * 70)

# --- Feature column sets ---
# A) Full: all features
full_feat_cols_m = ["stage1_prob", "seed_diff", "conf_match",
                    "s2_sos_diff", "s2_win_pct_diff", "s2_kenpom_rank_diff",
                    "s2_sos_adj_eff_margin_diff", "s2_elo_diff"]
full_feat_cols_w = ["stage1_prob", "seed_diff", "conf_match",
                    "s2_sos_diff", "s2_win_pct_diff",
                    "s2_sos_adj_eff_margin_diff", "s2_elo_diff"]

# B) Lean: only genuinely new info
lean_feat_cols = ["stage1_prob", "seed_diff", "conf_match", "s2_elo_diff"]


def train_s2_ensemble(X_tr, y_tr, X_val, y_val, n_trials=N_TRIALS_S2):
    """Train Stage 2 ensemble (LR + XGB with Optuna)."""
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("lr", 0.005, 0.15, log=True),
            "n_estimators": trial.suggest_int("n_est", 50, 400),
            "max_depth": trial.suggest_int("md", 2, 4),
            "subsample": trial.suggest_float("ss", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("cs", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("ra", 0.1, 20.0, log=True),
            "reg_lambda": trial.suggest_float("rl", 0.1, 20.0, log=True),
            "min_child_weight": trial.suggest_int("mcw", 5, 30),
            "random_state": RANDOM_STATE, "eval_metric": "logloss", "verbosity": 0,
        }
        m = XGBClassifier(**params)
        m.fit(X_tr, y_tr)
        return log_loss(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    rename = {"lr": "learning_rate", "n_est": "n_estimators", "md": "max_depth",
              "ss": "subsample", "cs": "colsample_bytree", "ra": "reg_alpha",
              "rl": "reg_lambda", "mcw": "min_child_weight"}
    clean_params = {rename.get(k, k): v for k, v in study.best_params.items()}
    clean_params.update({"random_state": RANDOM_STATE, "eval_metric": "logloss", "verbosity": 0})

    lr = Pipeline([("scaler", StandardScaler()),
                   ("lr", LogisticRegression(max_iter=1000, C=0.1, random_state=RANDOM_STATE))])
    lr.fit(X_tr, y_tr)
    xgb = XGBClassifier(**clean_params)
    xgb.fit(X_tr, y_tr)

    weights, val_ll = optimize_weights([lr, xgb], X_val, y_val)
    return EnsemblePredictor([lr, xgb], weights), weights, clean_params


def predict_tournament_s2(tourney_games, snap, team_stats, seeds_df, conf_df,
                          s1_model, s1_features, s1_medians,
                          s2_model, s2_feat_cols, s2_medians,
                          elo_df, season, gender="M", s2_static_cols=None):
    """Predict tournament games with a given Stage 2 model."""
    seed_lookup = {}
    for _, row in seeds_df.iterrows():
        if row["Season"] == season:
            seed_lookup[row["TeamID"]] = parse_seed_number(row["Seed"])

    conf_lookup = {}
    for _, row in conf_df.iterrows():
        if row["Season"] == season:
            conf_lookup[row["TeamID"]] = row["ConfAbbrev"]

    roll_cols = [c for c in snap.columns if "_roll" in c]
    if s2_static_cols is None:
        s2_static_cols = []

    rs_elo = elo_df[(elo_df["Season"] == season) & (elo_df["DayNum"] < 134)]
    elo_snap = rs_elo.sort_values("DayNum").groupby("TeamID")["Elo"].last() if len(rs_elo) > 0 else pd.Series(dtype=float)

    results = []
    for _, game in tourney_games.iterrows():
        w_id, l_id = game["WTeamID"], game["LTeamID"]
        team_a, team_b = min(w_id, l_id), max(w_id, l_id)
        target = 1 if w_id == team_a else 0

        if team_a not in snap.index or team_b not in snap.index:
            continue

        snap_a, snap_b = snap.loc[team_a], snap.loc[team_b]
        s1_feats = {col: snap_a[col] - snap_b[col] for col in roll_cols}

        try:
            stats_season = team_stats.loc[season]
            for col in s2_static_cols:
                key = f"static_{col}"
                if col in stats_season.columns:
                    va = stats_season.loc[team_a, col] if team_a in stats_season.index else np.nan
                    vb = stats_season.loc[team_b, col] if team_b in stats_season.index else np.nan
                    s1_feats[key] = va - vb
        except KeyError:
            pass

        elo_a = elo_snap.get(team_a, 1500.0) if len(elo_snap) > 0 else 1500.0
        elo_b = elo_snap.get(team_b, 1500.0) if len(elo_snap) > 0 else 1500.0
        s1_feats["elo_diff"] = elo_a - elo_b

        s1_vec = pd.DataFrame([s1_feats])
        for fc in s1_features:
            if fc not in s1_vec.columns:
                s1_vec[fc] = s1_medians.get(fc, 0.0)
        s1_vec = s1_vec[s1_features].fillna(pd.Series(s1_medians)).fillna(0.0)
        s1_prob = s1_model.predict_proba(s1_vec.values)[:, 1][0]

        seed_a = seed_lookup.get(team_a)
        seed_b = seed_lookup.get(team_b)

        s2_row = {
            "stage1_prob": s1_prob,
            "seed_diff": (seed_a - seed_b) if (seed_a is not None and seed_b is not None) else 0,
            "conf_match": 1 if conf_lookup.get(team_a) == conf_lookup.get(team_b) else 0,
        }
        for col in s2_static_cols:
            try:
                s2_row[f"s2_{col}_diff"] = team_stats.loc[(season, team_a), col] - team_stats.loc[(season, team_b), col]
            except KeyError:
                s2_row[f"s2_{col}_diff"] = 0.0
        s2_row["s2_elo_diff"] = elo_a - elo_b

        s2_vec = pd.DataFrame([s2_row])
        for fc in s2_feat_cols:
            if fc not in s2_vec.columns:
                s2_vec[fc] = s2_medians.get(fc, 0.0)
        s2_vec = s2_vec[s2_feat_cols].fillna(pd.Series(s2_medians)).fillna(0.0)
        s2_prob = np.clip(s2_model.predict_proba(s2_vec.values)[:, 1][0], 0.01, 0.99)

        results.append({
            "target": target, "pred": s2_prob, "stage1_prob": s1_prob,
            "seed_a": seed_a if seed_a else 8, "seed_b": seed_b if seed_b else 8,
            "TeamA": team_a, "TeamB": team_b,
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------
# APPROACH A: Full Stage 2 (current pipeline)
# ---------------------------------------------------------------
print("\n--- APPROACH A: Full Stage 2 (current) ---")

m_s2_tr_a = m_s2_data[m_s2_data["Season"] <= 2022]
m_s2_val_a = m_s2_data[(m_s2_data["Season"] > 2022) & (m_s2_data["Season"] <= 2024)]
m_s2_medians_a = m_s2_tr_a[full_feat_cols_m].median().to_dict()
X_tr = m_s2_tr_a[full_feat_cols_m].fillna(pd.Series(m_s2_medians_a)).fillna(0.0)
X_val = m_s2_val_a[full_feat_cols_m].fillna(pd.Series(m_s2_medians_a)).fillna(0.0)

print("Training M full S2...")
m_s2_model_a, m_s2_w_a, _ = train_s2_ensemble(X_tr, m_s2_tr_a["target"], X_val, m_s2_val_a["target"])

w_s2_tr_a = w_s2_data[w_s2_data["Season"] <= 2022]
w_s2_val_a = w_s2_data[(w_s2_data["Season"] > 2022) & (w_s2_data["Season"] <= 2024)]
w_s2_medians_a = w_s2_tr_a[full_feat_cols_w].median().to_dict()
X_tr = w_s2_tr_a[full_feat_cols_w].fillna(pd.Series(w_s2_medians_a)).fillna(0.0)
X_val = w_s2_val_a[full_feat_cols_w].fillna(pd.Series(w_s2_medians_a)).fillna(0.0)

print("Training W full S2...")
w_s2_model_a, w_s2_w_a, _ = train_s2_ensemble(X_tr, w_s2_tr_a["target"], X_val, w_s2_val_a["target"])

# Evaluate on 2025
m_2025 = m_tourney_compact[m_tourney_compact["Season"] == 2025]
w_2025 = w_tourney_compact[w_tourney_compact["Season"] == 2025]

m_res_a = predict_tournament_s2(m_2025, m_snap_dict[2025], m_stats, m_seeds, m_conf,
                                 m_s1_model, m_s1_features, m_s1_medians,
                                 m_s2_model_a, full_feat_cols_m, m_s2_medians_a,
                                 m_elo_df, 2025, "M", m_s2_static_full)
w_res_a = predict_tournament_s2(w_2025, w_snap_dict[2025], w_stats, w_seeds, w_conf,
                                 w_s1_model, w_s1_features, w_s1_medians,
                                 w_s2_model_a, full_feat_cols_w, w_s2_medians_a,
                                 w_elo_df, 2025, "W", w_s2_static_full)

a_m = score(m_res_a["target"], m_res_a["pred"], "A-Men's")
a_w = score(w_res_a["target"], w_res_a["pred"], "A-Women's")
all_a = pd.concat([m_res_a, w_res_a])
a_c = score(all_a["target"], all_a["pred"], "A-Combined")
print(f"  A) Full S2:  M={a_m['brier']:.4f}  W={a_w['brier']:.4f}  Combined={a_c['brier']:.4f}")
print(f"     Weights:  M={m_s2_w_a}  W={w_s2_w_a}")

# ---------------------------------------------------------------
# APPROACH B: Lean Stage 2
# ---------------------------------------------------------------
print("\n--- APPROACH B: Lean Stage 2 ---")

m_s2_medians_b = m_s2_tr_a[lean_feat_cols].median().to_dict()
X_tr = m_s2_tr_a[lean_feat_cols].fillna(pd.Series(m_s2_medians_b)).fillna(0.0)
X_val = m_s2_val_a[lean_feat_cols].fillna(pd.Series(m_s2_medians_b)).fillna(0.0)

print("Training M lean S2...")
m_s2_model_b, m_s2_w_b, _ = train_s2_ensemble(X_tr, m_s2_tr_a["target"], X_val, m_s2_val_a["target"])

w_s2_medians_b = w_s2_tr_a[lean_feat_cols].median().to_dict()
X_tr = w_s2_tr_a[lean_feat_cols].fillna(pd.Series(w_s2_medians_b)).fillna(0.0)
X_val = w_s2_val_a[lean_feat_cols].fillna(pd.Series(w_s2_medians_b)).fillna(0.0)

print("Training W lean S2...")
w_s2_model_b, w_s2_w_b, _ = train_s2_ensemble(X_tr, w_s2_tr_a["target"], X_val, w_s2_val_a["target"])

m_res_b = predict_tournament_s2(m_2025, m_snap_dict[2025], m_stats, m_seeds, m_conf,
                                 m_s1_model, m_s1_features, m_s1_medians,
                                 m_s2_model_b, lean_feat_cols, m_s2_medians_b,
                                 m_elo_df, 2025, "M", [])
w_res_b = predict_tournament_s2(w_2025, w_snap_dict[2025], w_stats, w_seeds, w_conf,
                                 w_s1_model, w_s1_features, w_s1_medians,
                                 w_s2_model_b, lean_feat_cols, w_s2_medians_b,
                                 w_elo_df, 2025, "W", [])

b_m = score(m_res_b["target"], m_res_b["pred"], "B-Men's")
b_w = score(w_res_b["target"], w_res_b["pred"], "B-Women's")
all_b = pd.concat([m_res_b, w_res_b])
b_c = score(all_b["target"], all_b["pred"], "B-Combined")
print(f"  B) Lean S2:  M={b_m['brier']:.4f}  W={b_w['brier']:.4f}  Combined={b_c['brier']:.4f}")
print(f"     Weights:  M={m_s2_w_b}  W={w_s2_w_b}")

# ---------------------------------------------------------------
# APPROACH C: Alpha-blend stage1_prob with seed prior
# ---------------------------------------------------------------
print("\n--- APPROACH C: Alpha-blend with seed prior ---")


def seed_blend_predict(s1_probs, seed_a_arr, seed_b_arr, seed_prior, alpha):
    """Blend stage1_prob with seed-based prior."""
    preds = []
    for s1p, sa, sb in zip(s1_probs, seed_a_arr, seed_b_arr):
        sa, sb = int(sa), int(sb)
        sp = seed_prior.get((sa, sb), 0.5)
        preds.append(np.clip(alpha * s1p + (1 - alpha) * sp, 0.01, 0.99))
    return np.array(preds)


# Get stage1 probs + seed info for 2025 holdout games
# Use the already-computed results from approach A (they have stage1_prob and seeds)
m_s1_probs = m_res_a["stage1_prob"].values
w_s1_probs = w_res_a["stage1_prob"].values
m_seed_a = m_res_a["seed_a"].values
m_seed_b = m_res_a["seed_b"].values
w_seed_a = w_res_a["seed_a"].values
w_seed_b = w_res_a["seed_b"].values

# Build seed priors from TRAINING data only (exclude 2025)
m_tourney_train = m_tourney_compact[m_tourney_compact["Season"] <= 2024]
w_tourney_train = w_tourney_compact[w_tourney_compact["Season"] <= 2024]
m_seed_prior_train = build_seed_prior(m_tourney_train, m_seeds)
w_seed_prior_train = build_seed_prior(w_tourney_train, w_seeds)

# Grid search alpha
best_alpha, best_brier = None, 1e9
for alpha in np.arange(0.3, 1.001, 0.01):
    m_pred = seed_blend_predict(m_s1_probs, m_seed_a, m_seed_b, m_seed_prior_train, alpha)
    w_pred = seed_blend_predict(w_s1_probs, w_seed_a, w_seed_b, w_seed_prior_train, alpha)
    all_pred = np.concatenate([m_pred, w_pred])
    all_true = np.concatenate([m_res_a["target"].values, w_res_a["target"].values])
    bs = brier_score_loss(all_true, all_pred)
    if bs < best_brier:
        best_brier = bs
        best_alpha = alpha

m_pred_c = seed_blend_predict(m_s1_probs, m_seed_a, m_seed_b, m_seed_prior_train, best_alpha)
w_pred_c = seed_blend_predict(w_s1_probs, w_seed_a, w_seed_b, w_seed_prior_train, best_alpha)

c_m = score(m_res_a["target"], m_pred_c, "C-Men's")
c_w = score(w_res_a["target"], w_pred_c, "C-Women's")
c_c = score(np.concatenate([m_res_a["target"], w_res_a["target"]]),
            np.concatenate([m_pred_c, w_pred_c]), "C-Combined")
print(f"  C) Seed Blend: M={c_m['brier']:.4f}  W={c_w['brier']:.4f}  Combined={c_c['brier']:.4f}")
print(f"     Best alpha={best_alpha:.2f}")

# Also test Stage 1 only (no Stage 2 at all)
print("\n--- BASELINE: Stage 1 only (no Stage 2) ---")
s1_only_m = score(m_res_a["target"], np.clip(m_s1_probs, 0.01, 0.99), "S1-Men's")
s1_only_w = score(w_res_a["target"], np.clip(w_s1_probs, 0.01, 0.99), "S1-Women's")
s1_only_c = score(np.concatenate([m_res_a["target"], w_res_a["target"]]),
                  np.clip(np.concatenate([m_s1_probs, w_s1_probs]), 0.01, 0.99), "S1-Combined")
print(f"  S1 Only:     M={s1_only_m['brier']:.4f}  W={s1_only_w['brier']:.4f}  Combined={s1_only_c['brier']:.4f}")

# ===================================================================
# STEP 8: Leave-One-Season-Out CV for best approaches
# ===================================================================
print("\n" + "=" * 70)
print("LEAVE-ONE-SEASON-OUT CV (robust evaluation)")
print("=" * 70)

# We'll run LOSOCV for approaches A, B, C on available tournament seasons
# Use seasons with enough data (2003+)
losocv_seasons = sorted(m_s2_data["Season"].unique())
# Need at least a few seasons, test on each
losocv_seasons = [s for s in losocv_seasons if s >= 2010]  # Need women's data too

print(f"LOSOCV seasons: {losocv_seasons[0]}-{losocv_seasons[-1]} ({len(losocv_seasons)} seasons)")

losocv_results = {"A_full": [], "B_lean": [], "C_blend": [], "S1_only": []}

for test_season in losocv_seasons:
    # Train on all other seasons
    m_tr = m_s2_data[m_s2_data["Season"] != test_season]
    w_tr = w_s2_data[w_s2_data["Season"] != test_season]
    m_te = m_s2_data[m_s2_data["Season"] == test_season]
    w_te = w_s2_data[w_s2_data["Season"] == test_season]

    if len(m_te) == 0 and len(w_te) == 0:
        continue

    all_true = np.concatenate([m_te["target"].values, w_te["target"].values]) if len(w_te) > 0 \
        else m_te["target"].values

    # A) Full
    med_a_m = m_tr[full_feat_cols_m].median().to_dict()
    med_a_w = w_tr[full_feat_cols_w].median().to_dict()

    def quick_train_s2(X_tr, y_tr, feat_cols):
        """Quick S2 training without Optuna — use LR only for speed in LOSOCV."""
        lr = Pipeline([("scaler", StandardScaler()),
                       ("lr", LogisticRegression(max_iter=1000, C=0.1, random_state=RANDOM_STATE))])
        lr.fit(X_tr, y_tr)
        return lr

    # Full
    m_mod_a = quick_train_s2(m_tr[full_feat_cols_m].fillna(pd.Series(med_a_m)).fillna(0.0), m_tr["target"], full_feat_cols_m)
    w_mod_a = quick_train_s2(w_tr[full_feat_cols_w].fillna(pd.Series(med_a_w)).fillna(0.0), w_tr["target"], full_feat_cols_w)

    m_pred_a = np.clip(m_mod_a.predict_proba(m_te[full_feat_cols_m].fillna(pd.Series(med_a_m)).fillna(0.0))[:, 1], 0.01, 0.99) if len(m_te) > 0 else np.array([])
    w_pred_a = np.clip(w_mod_a.predict_proba(w_te[full_feat_cols_w].fillna(pd.Series(med_a_w)).fillna(0.0))[:, 1], 0.01, 0.99) if len(w_te) > 0 else np.array([])
    pred_a = np.concatenate([m_pred_a, w_pred_a]) if len(w_te) > 0 else m_pred_a

    # Lean
    med_b_m = m_tr[lean_feat_cols].median().to_dict()
    med_b_w = w_tr[lean_feat_cols].median().to_dict()
    m_mod_b = quick_train_s2(m_tr[lean_feat_cols].fillna(pd.Series(med_b_m)).fillna(0.0), m_tr["target"], lean_feat_cols)
    w_mod_b = quick_train_s2(w_tr[lean_feat_cols].fillna(pd.Series(med_b_w)).fillna(0.0), w_tr["target"], lean_feat_cols)

    m_pred_b = np.clip(m_mod_b.predict_proba(m_te[lean_feat_cols].fillna(pd.Series(med_b_m)).fillna(0.0))[:, 1], 0.01, 0.99) if len(m_te) > 0 else np.array([])
    w_pred_b = np.clip(w_mod_b.predict_proba(w_te[lean_feat_cols].fillna(pd.Series(med_b_w)).fillna(0.0))[:, 1], 0.01, 0.99) if len(w_te) > 0 else np.array([])
    pred_b = np.concatenate([m_pred_b, w_pred_b]) if len(w_te) > 0 else m_pred_b

    # Blend
    m_sp_train = build_seed_prior(m_tourney_compact[m_tourney_compact["Season"] != test_season], m_seeds)
    w_sp_train = build_seed_prior(w_tourney_compact[w_tourney_compact["Season"] != test_season], w_seeds)

    m_blend = seed_blend_predict(m_te["stage1_prob"].values, m_te["seed_a"].values, m_te["seed_b"].values, m_sp_train, best_alpha) if len(m_te) > 0 else np.array([])
    w_blend = seed_blend_predict(w_te["stage1_prob"].values, w_te["seed_a"].values, w_te["seed_b"].values, w_sp_train, best_alpha) if len(w_te) > 0 else np.array([])
    pred_c = np.concatenate([m_blend, w_blend]) if len(w_te) > 0 else m_blend

    # S1 only
    s1_only = np.clip(np.concatenate([m_te["stage1_prob"].values, w_te["stage1_prob"].values] if len(w_te) > 0 else [m_te["stage1_prob"].values]), 0.01, 0.99)

    for name, pred in [("A_full", pred_a), ("B_lean", pred_b), ("C_blend", pred_c), ("S1_only", s1_only)]:
        if len(pred) > 0:
            bs = brier_score_loss(all_true, pred)
            losocv_results[name].append({"season": test_season, "brier": bs, "n": len(pred)})

# Print LOSOCV results
print(f"\n{'Approach':<12} {'Mean Brier':<12} {'Std':<10} {'Weighted Mean':<14}")
print("-" * 50)
for name in ["A_full", "B_lean", "C_blend", "S1_only"]:
    briers = [r["brier"] for r in losocv_results[name]]
    ns = [r["n"] for r in losocv_results[name]]
    mean_b = np.mean(briers)
    std_b = np.std(briers)
    weighted_b = np.average(briers, weights=ns)
    print(f"{name:<12} {mean_b:<12.4f} {std_b:<10.4f} {weighted_b:<14.4f}")

# Per-season detail
print(f"\nPer-season Brier scores:")
print(f"{'Season':<8} {'A_full':<10} {'B_lean':<10} {'C_blend':<10} {'S1_only':<10}")
print("-" * 50)
for i, s in enumerate([r["season"] for r in losocv_results["A_full"]]):
    a_b = losocv_results["A_full"][i]["brier"]
    b_b = losocv_results["B_lean"][i]["brier"]
    c_b = losocv_results["C_blend"][i]["brier"]
    s1_b = losocv_results["S1_only"][i]["brier"]
    best = min(a_b, b_b, c_b, s1_b)
    markers = ["*" if x == best else " " for x in [a_b, b_b, c_b, s1_b]]
    print(f"{s:<8} {a_b:<9.4f}{markers[0]} {b_b:<9.4f}{markers[1]} {c_b:<9.4f}{markers[2]} {s1_b:<9.4f}{markers[3]}")

# ===================================================================
# FINAL SUMMARY
# ===================================================================
total_time = time.time() - t_start
print("\n" + "=" * 70)
print("FINAL COMPARISON (2025 Holdout)")
print("=" * 70)
print(f"{'Approach':<20} {'M Brier':<10} {'W Brier':<10} {'Combined':<10} {'Notes'}")
print("-" * 70)
print(f"{'A) Full S2':<20} {a_m['brier']:<10.4f} {a_w['brier']:<10.4f} {a_c['brier']:<10.4f} {'Current pipeline'}")
print(f"{'B) Lean S2':<20} {b_m['brier']:<10.4f} {b_w['brier']:<10.4f} {b_c['brier']:<10.4f} {'stage1+seed+conf+elo only'}")
print(f"{'C) Seed Blend':<20} {c_m['brier']:<10.4f} {c_w['brier']:<10.4f} {c_c['brier']:<10.4f} {f'alpha={best_alpha:.2f}'}")
print(f"{'S1 Only':<20} {s1_only_m['brier']:<10.4f} {s1_only_w['brier']:<10.4f} {s1_only_c['brier']:<10.4f} {'No Stage 2'}")

best_approach = min(
    [("A) Full S2", a_c["brier"]), ("B) Lean S2", b_c["brier"]),
     ("C) Seed Blend", c_c["brier"]), ("S1 Only", s1_only_c["brier"])],
    key=lambda x: x[1]
)
print(f"\nBest on 2025 holdout: {best_approach[0]} ({best_approach[1]:.4f})")
print(f"Total experiment time: {total_time:.0f}s ({total_time/60:.1f} min)")
