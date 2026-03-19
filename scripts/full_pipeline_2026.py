"""Full end-to-end two-stage model pipeline with Elo ratings for 2026.

Runs ALL steps: data loading, Elo computation, feature engineering,
Stage 1 + Stage 2 training with Optuna, 2025 holdout evaluation,
calibration check, production retrain, 2026 predictions, and website update.
"""

import sys
import os
import json
import time
import warnings
import re as _re
import subprocess
from pathlib import Path

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_regular_season_results, load_tourney_results,
    load_seeds, load_massey_ordinals, load_team_conferences,
    load_sample_submission,
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
from src.two_stage import TwoStagePredictor

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

# Elo parameters (fixed defaults)
ELO_K = 20
ELO_CARRYOVER = 0.75
ELO_MARGIN = True

# Stage 2 Elo scaling: downweight s2_elo_diff to prevent domination
S2_ELO_SCALE = 0.5

PIPELINE_START = time.time()

# ===================================================================
# STEP 0: Load all data
# ===================================================================
print("=" * 70)
print("STEP 0: Loading data")
print("=" * 70)

m_rs_detailed = load_regular_season_results(gender="M", detailed=True)
m_rs_compact = load_regular_season_results(gender="M", detailed=False)
m_tourney_detailed = load_tourney_results(gender="M", detailed=True)
m_tourney_compact = load_tourney_results(gender="M", detailed=False)
m_seeds = load_seeds(gender="M")
m_conf = load_team_conferences(gender="M")

w_rs_detailed = load_regular_season_results(gender="W", detailed=True)
w_rs_compact = load_regular_season_results(gender="W", detailed=False)
w_tourney_detailed = load_tourney_results(gender="W", detailed=True)
w_tourney_compact = load_tourney_results(gender="W", detailed=False)
w_seeds = load_seeds(gender="W")
w_conf = load_team_conferences(gender="W")

print("Loading Massey ordinals (men's only)...")
ordinals_df = load_massey_ordinals()

# Data verification
m_rs_min, m_rs_max = m_rs_detailed.Season.min(), m_rs_detailed.Season.max()
w_rs_min, w_rs_max = w_rs_detailed.Season.min(), w_rs_detailed.Season.max()
print(f"\nData Verification:")
print(f"  Men's RS detailed seasons:   {m_rs_min}-{m_rs_max}")
print(f"  Women's RS detailed seasons: {w_rs_min}-{w_rs_max}")
print(f"  Men's RS detailed rows:   {len(m_rs_detailed):,}")
print(f"  Women's RS detailed rows: {len(w_rs_detailed):,}")
print(f"  Men's tournament rows:    {len(m_tourney_compact):,}")
print(f"  Women's tournament rows:  {len(w_tourney_compact):,}")

if m_rs_max < 2026:
    print("  WARNING: 2026 men's data NOT found!")
if w_rs_max < 2026:
    print("  WARNING: 2026 women's data NOT found!")

# ===================================================================
# STEP 1: Compute Elo Ratings
# ===================================================================
print("\n" + "=" * 70)
print("STEP 1: Computing Elo ratings (K=20, carryover=0.75, margin=True)")
print("=" * 70)

# Combine regular season + tournament for Elo computation (all games)
m_all_games = pd.concat([m_rs_compact, m_tourney_compact], ignore_index=True)
w_all_games = pd.concat([w_rs_compact, w_tourney_compact], ignore_index=True)

print(f"Men's all games:   {len(m_all_games):,} (seasons {m_all_games.Season.min()}-{m_all_games.Season.max()})")
print(f"Women's all games: {len(w_all_games):,} (seasons {w_all_games.Season.min()}-{w_all_games.Season.max()})")

m_elo_df = compute_elo_ratings(
    m_all_games, k_factor=ELO_K,
    season_carryover=ELO_CARRYOVER, margin_factor=ELO_MARGIN,
)
w_elo_df = compute_elo_ratings(
    w_all_games, k_factor=ELO_K,
    season_carryover=ELO_CARRYOVER, margin_factor=ELO_MARGIN,
)
print(f"Men's Elo records:   {len(m_elo_df):,}")
print(f"Women's Elo records: {len(w_elo_df):,}")

# ===================================================================
# STEP 2: Build Static Features
# ===================================================================
print("\n" + "=" * 70)
print("STEP 2: Building static features")
print("=" * 70)


def build_static_features(rs_detailed, rs_compact, seeds, ordinals=None, gender="M"):
    """Build season-level static features for a gender."""
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


m_stats = build_static_features(m_rs_detailed, m_rs_compact, m_seeds, ordinals_df, "M")
w_stats = build_static_features(w_rs_detailed, w_rs_compact, w_seeds, gender="W")

print(f"Men's static stats:   {m_stats.shape}")
print(f"Women's static stats: {w_stats.shape}")

# Save team stats CSVs for website
m_stats.to_csv(OUTPUT_DIR / "m_team_stats.csv")
w_stats.to_csv(OUTPUT_DIR / "w_team_stats.csv")
print("Saved team stats CSVs")

# ===================================================================
# STEP 3: Build Stage 1 Training Data with Elo
# ===================================================================
print("\n" + "=" * 70)
print("STEP 3: Building Stage 1 training data (regular season + Elo)")
print("=" * 70)

# Static columns for Stage 1
m_static_cols = ["sos", "win_pct", "efficiency_margin", "kenpom_rank", "sos_adj_eff_margin"]
m_static = m_stats[[c for c in m_static_cols if c in m_stats.columns]].copy()
m_static["kenpom_rank"] = m_static["kenpom_rank"].fillna(366)

w_static_cols = ["sos", "win_pct", "efficiency_margin", "sos_adj_eff_margin"]
w_static = w_stats[[c for c in w_static_cols if c in w_stats.columns]].copy()

print("Building Stage 1 training data (regular season only)...")
m_s1_train = build_rolling_training_data(
    m_rs_detailed, compact_results_df=m_rs_compact,
    team_stats=m_static, windows=(5, 7, 10),
)

w_s1_train = build_rolling_training_data(
    w_rs_detailed, compact_results_df=w_rs_compact,
    team_stats=w_static, windows=(5, 7, 10),
)


# Add elo_diff to Stage 1 training data
def add_elo_diff_to_training(train_df, elo_df):
    """Add elo_diff column to training data using pre-game Elos."""
    train_with_elo = train_df.copy()

    elo_a = elo_df.rename(columns={"TeamID": "TeamA", "Elo": "Elo_A"})
    elo_b = elo_df.rename(columns={"TeamID": "TeamB", "Elo": "Elo_B"})

    train_with_elo = train_with_elo.merge(
        elo_a[["Season", "DayNum", "TeamA", "Elo_A"]],
        on=["Season", "DayNum", "TeamA"],
        how="left",
    )
    train_with_elo = train_with_elo.merge(
        elo_b[["Season", "DayNum", "TeamB", "Elo_B"]],
        on=["Season", "DayNum", "TeamB"],
        how="left",
    )

    train_with_elo["elo_diff"] = train_with_elo["Elo_A"] - train_with_elo["Elo_B"]
    train_with_elo.drop(columns=["Elo_A", "Elo_B"], inplace=True)

    matched = train_with_elo["elo_diff"].notna().sum()
    total = len(train_with_elo)
    print(f"  Elo matched: {matched:,}/{total:,} ({matched/total*100:.1f}%)")

    return train_with_elo


print("\nAdding elo_diff to Stage 1 training data...")
m_s1_train = add_elo_diff_to_training(m_s1_train, m_elo_df)
w_s1_train = add_elo_diff_to_training(w_s1_train, w_elo_df)

# Drop NaN rows (except elo_diff which we fill separately)
meta_cols = ["target", "Season", "DayNum", "TeamA", "TeamB", "is_tourney"]
m_feat_all = [c for c in m_s1_train.columns if c not in meta_cols]
w_feat_all = [c for c in w_s1_train.columns if c not in meta_cols]
m_s1_train = m_s1_train.dropna(subset=[c for c in m_feat_all if c != "elo_diff"]).reset_index(drop=True)
w_s1_train = w_s1_train.dropna(subset=[c for c in w_feat_all if c != "elo_diff"]).reset_index(drop=True)

# Fill remaining NaN in elo_diff with 0
m_s1_train["elo_diff"] = m_s1_train["elo_diff"].fillna(0.0)
w_s1_train["elo_diff"] = w_s1_train["elo_diff"].fillna(0.0)

print(f"\nStage 1 training data (regular season only):")
print(f"  Men's:   {len(m_s1_train):,} rows (seasons {m_s1_train.Season.min()}-{m_s1_train.Season.max()})")
print(f"  Women's: {len(w_s1_train):,} rows (seasons {w_s1_train.Season.min()}-{w_s1_train.Season.max()})")

# ===================================================================
# STEP 4: Feature Selection (MI + correlation pruning)
# ===================================================================
print("\n" + "=" * 70)
print("STEP 4: Feature selection with Elo")
print("=" * 70)


def select_features(df, feat_cols, target_col="target", mi_threshold=0.001, corr_threshold=0.85,
                    protected_features=None):
    """Select features via MI ranking + correlation pruning."""
    protected = set(protected_features or [])
    X = df[feat_cols].copy()
    y = df[target_col]

    # Fill NaN for MI computation
    X = X.fillna(X.median())

    # Mutual information scores
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({"feature": feat_cols, "mi_score": mi}).sort_values("mi_score", ascending=False)

    # Filter by MI threshold
    selected = mi_df[mi_df["mi_score"] >= mi_threshold]["feature"].tolist()

    # Add protected features back if they were dropped
    for pf in protected:
        if pf in feat_cols and pf not in selected:
            selected.append(pf)

    # Correlation pruning
    corr_matrix = X[selected].corr().abs()
    to_drop = set()
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            fi, fj = selected[i], selected[j]
            if fi in to_drop or fj in to_drop:
                continue
            if corr_matrix.loc[fi, fj] > corr_threshold:
                if fi in protected and fj in protected:
                    continue
                if fi in protected:
                    to_drop.add(fj)
                elif fj in protected:
                    to_drop.add(fi)
                else:
                    mi_i = mi_df[mi_df["feature"] == fi]["mi_score"].values[0]
                    mi_j = mi_df[mi_df["feature"] == fj]["mi_score"].values[0]
                    to_drop.add(fj if mi_i >= mi_j else fi)

    final = [f for f in selected if f not in to_drop]
    print(f"  MI filter: {len(feat_cols)} -> {len(selected)}")
    print(f"  Corr pruning: {len(selected)} -> {len(final)} (dropped {len(to_drop)})")
    return final, mi_df


# Men's Stage 1
raw_feat_cols = [c for c in m_s1_train.columns if "_roll" in c or c.startswith("static_") or c == "elo_diff"]
print("Men's Stage 1 feature selection:")
m_s1_features, m_mi = select_features(
    m_s1_train, raw_feat_cols,
    protected_features=["static_kenpom_rank", "static_win_pct", "elo_diff"],
)
print(f"  Final: {len(m_s1_features)} features")
print(f"  elo_diff kept: {'YES' if 'elo_diff' in m_s1_features else 'NO'}")

# Women's Stage 1
w_raw_feat_cols = [c for c in w_s1_train.columns if "_roll" in c or c.startswith("static_") or c == "elo_diff"]
print("\nWomen's Stage 1 feature selection:")
w_s1_features, w_mi = select_features(
    w_s1_train, w_raw_feat_cols,
    protected_features=["static_win_pct", "elo_diff"],
)
print(f"  Final: {len(w_s1_features)} features")
print(f"  elo_diff kept: {'YES' if 'elo_diff' in w_s1_features else 'NO'}")

# ===================================================================
# STEP 5: Time-Based Split
# ===================================================================
print("\n" + "=" * 70)
print("STEP 5: Time-based split")
print("=" * 70)


def split_by_season(df, feat_cols, train_end=2022, val_end=2024, min_season=None):
    """Split into train/val/test by season."""
    if min_season:
        df = df[df["Season"] >= min_season]
    train = df[df["Season"] <= train_end]
    val = df[(df["Season"] > train_end) & (df["Season"] <= val_end)]
    test = df[df["Season"] > val_end]

    medians = train[feat_cols].median().to_dict()
    X_train = train[feat_cols].fillna(pd.Series(medians)).fillna(0.0)
    y_train = train["target"]
    X_val = val[feat_cols].fillna(pd.Series(medians)).fillna(0.0)
    y_val = val["target"]
    X_test = test[feat_cols].fillna(pd.Series(medians)).fillna(0.0)
    y_test = test["target"]

    return X_train, y_train, X_val, y_val, X_test, y_test, medians


m_X_tr, m_y_tr, m_X_val, m_y_val, m_X_te, m_y_te, m_s1_medians = split_by_season(
    m_s1_train, m_s1_features, min_season=2003
)
w_X_tr, w_y_tr, w_X_val, w_y_val, w_X_te, w_y_te, w_s1_medians = split_by_season(
    w_s1_train, w_s1_features, min_season=2010
)

print(f"Men's Stage 1:   train={len(m_X_tr):,}  val={len(m_X_val):,}  test={len(m_X_te):,}")
print(f"Women's Stage 1: train={len(w_X_tr):,}  val={len(w_X_val):,}  test={len(w_X_te):,}")

# ===================================================================
# STEP 6: Optuna Hyperparameter Tuning (Stage 1)
# ===================================================================
print("\n" + "=" * 70)
print("STEP 6: Optuna tuning -- Stage 1 (50 trials per gender)")
print("=" * 70)

N_TRIALS = 50


def optuna_xgb_objective(trial, X_tr, y_tr, X_val, y_val):
    """Optuna objective for XGBoost hyperparameter tuning."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr)

    val_pred = model.predict_proba(X_val)[:, 1]
    val_ll = log_loss(y_val, val_pred)
    val_bs = brier_score_loss(y_val, val_pred)

    trial.set_user_attr("val_brier", round(val_bs, 4))
    return val_ll


t0 = time.time()
print("Tuning Men's Stage 1 XGBoost...")
m_s1_study = optuna.create_study(direction="minimize", study_name="s1_men_elo")
m_s1_study.optimize(lambda t: optuna_xgb_objective(t, m_X_tr, m_y_tr, m_X_val, m_y_val),
                     n_trials=N_TRIALS, show_progress_bar=False)
print(f"  Best val log-loss: {m_s1_study.best_value:.4f}")
print(f"  Best val brier:    {m_s1_study.best_trial.user_attrs['val_brier']}")

print("\nTuning Women's Stage 1 XGBoost...")
w_s1_study = optuna.create_study(direction="minimize", study_name="s1_women_elo")
w_s1_study.optimize(lambda t: optuna_xgb_objective(t, w_X_tr, w_y_tr, w_X_val, w_y_val),
                     n_trials=N_TRIALS, show_progress_bar=False)
print(f"  Best val log-loss: {w_s1_study.best_value:.4f}")
print(f"  Best val brier:    {w_s1_study.best_trial.user_attrs['val_brier']}")
print(f"\nStage 1 tuning completed in {time.time()-t0:.0f}s")

# ===================================================================
# STEP 7: Stage 1 Ensemble Training
# ===================================================================
print("\n" + "=" * 70)
print("STEP 7: Stage 1 ensemble training + weight optimization")
print("=" * 70)


def optimize_ensemble_weights(models, X_val, y_val, step=0.02):
    """Find optimal LR/XGB blend weights via grid search on val set."""
    best_ll, best_w = float("inf"), None
    for w_lr in np.arange(0.0, 1.01, step):
        w_xgb = 1.0 - w_lr
        probs = w_lr * models[0].predict_proba(X_val)[:, 1] + w_xgb * models[1].predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, probs)
        if ll < best_ll:
            best_ll = ll
            best_w = [round(w_lr, 2), round(w_xgb, 2)]
    return best_w, best_ll


# Men's Stage 1
m_s1_lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE))])
m_s1_lr.fit(m_X_tr, m_y_tr)

m_s1_xgb_params = m_s1_study.best_params.copy()
m_s1_xgb_params.update({"random_state": RANDOM_STATE, "eval_metric": "logloss", "verbosity": 0})
m_s1_xgb = XGBClassifier(**m_s1_xgb_params)
m_s1_xgb.fit(m_X_tr, m_y_tr)

m_s1_weights, m_s1_val_ll = optimize_ensemble_weights([m_s1_lr, m_s1_xgb], m_X_val, m_y_val)
m_s1_model = EnsemblePredictor([m_s1_lr, m_s1_xgb], m_s1_weights)

# Women's Stage 1
w_s1_lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=0.5, random_state=RANDOM_STATE))])
w_s1_lr.fit(w_X_tr, w_y_tr)

w_s1_xgb_params = w_s1_study.best_params.copy()
w_s1_xgb_params.update({"random_state": RANDOM_STATE, "eval_metric": "logloss", "verbosity": 0})
w_s1_xgb = XGBClassifier(**w_s1_xgb_params)
w_s1_xgb.fit(w_X_tr, w_y_tr)

w_s1_weights, w_s1_val_ll = optimize_ensemble_weights([w_s1_lr, w_s1_xgb], w_X_val, w_y_val)
w_s1_model = EnsemblePredictor([w_s1_lr, w_s1_xgb], w_s1_weights)

for label, model, X_te, y_te, weights in [
    ("Men's", m_s1_model, m_X_te, m_y_te, m_s1_weights),
    ("Women's", w_s1_model, w_X_te, w_y_te, w_s1_weights),
]:
    preds = model.predict_proba(X_te)[:, 1]
    bs = brier_score_loss(y_te, preds)
    ll = log_loss(y_te, preds)
    acc = accuracy_score(y_te, (preds >= 0.5).astype(int))
    print(f"{label} Stage 1: weights={weights}, test brier={bs:.4f}, test ll={ll:.4f}, acc={acc:.3f}")

# ===================================================================
# STEP 8: Compute Rolling Snapshots + Build Stage 2 Data
# ===================================================================
print("\n" + "=" * 70)
print("STEP 8: Building Stage 2 data with Elo")
print("=" * 70)


def parse_seed_number(seed_str):
    """Extract numeric seed from string like 'W01' or 'X16a'."""
    match = _re.match(r"[WXYZ](\d{2})", seed_str)
    return int(match.group(1)) if match else None


def build_tournament_features(tourney_df, seeds_df, conf_df, snap_dict, team_stats,
                              stage1_model, stage1_features, stage1_medians,
                              elo_df, gender="M", static_cols=None):
    """Build Stage 2 training data from historical tournament games."""
    seed_lookup = {}
    for _, row in seeds_df.iterrows():
        s, seed_str, tid = row["Season"], row["Seed"], row["TeamID"]
        seed_num = parse_seed_number(seed_str)
        seed_lookup[(s, tid)] = seed_num

    conf_lookup = {}
    for _, row in conf_df.iterrows():
        conf_lookup[(row["Season"], row["TeamID"])] = row["ConfAbbrev"]

    if static_cols is None:
        static_cols = ["sos", "win_pct"]

    # Pre-compute end-of-regular-season Elo snapshots for all seasons
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

        team_a = min(w_id, l_id)
        team_b = max(w_id, l_id)
        target = 1 if w_id == team_a else 0

        if season not in snap_dict:
            skipped += 1
            continue
        snap = snap_dict[season]
        if team_a not in snap.index or team_b not in snap.index:
            skipped += 1
            continue

        snap_a = snap.loc[team_a]
        snap_b = snap.loc[team_b]

        # Stage 1 features
        roll_cols = [c for c in snap.columns if "_roll" in c]
        s1_feats = {}
        for col in roll_cols:
            s1_feats[col] = snap_a[col] - snap_b[col]

        # Static diffs for Stage 1
        try:
            stats_season = team_stats.loc[season]
            for col in static_cols:
                key = f"static_{col}"
                if col in stats_season.columns:
                    val_a = stats_season.loc[team_a, col] if team_a in stats_season.index else np.nan
                    val_b = stats_season.loc[team_b, col] if team_b in stats_season.index else np.nan
                    s1_feats[key] = val_a - val_b
                else:
                    s1_feats[key] = 0.0
        except (KeyError, TypeError):
            for col in static_cols:
                s1_feats[f"static_{col}"] = 0.0

        # Elo diff for Stage 1
        if season in elo_snapshots:
            elo_snap = elo_snapshots[season]
            elo_a = elo_snap.get(team_a, 1500.0)
            elo_b = elo_snap.get(team_b, 1500.0)
            s1_feats["elo_diff"] = elo_a - elo_b
        else:
            s1_feats["elo_diff"] = 0.0

        # Build Stage 1 feature vector and predict
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
            "conf_match": 1 if conf_lookup.get((season, team_a)) == conf_lookup.get((season, team_b)) else 0,
        }

        # Stage 2 static diffs (prefixed with s2_)
        for col in static_cols:
            try:
                val_a = team_stats.loc[(season, team_a), col]
                val_b = team_stats.loc[(season, team_b), col]
                s2_row[f"s2_{col}_diff"] = val_a - val_b
            except KeyError:
                s2_row[f"s2_{col}_diff"] = 0.0

        # Elo diff for Stage 2 (end-of-regular-season, scaled)
        if season in elo_snapshots:
            elo_snap = elo_snapshots[season]
            elo_a = elo_snap.get(team_a, 1500.0)
            elo_b = elo_snap.get(team_b, 1500.0)
            s2_row["s2_elo_diff"] = (elo_a - elo_b) * S2_ELO_SCALE
        else:
            s2_row["s2_elo_diff"] = 0.0

        s2_row["target"] = target
        s2_row["Season"] = season
        s2_row["TeamA"] = team_a
        s2_row["TeamB"] = team_b

        rows.append(s2_row)

    if skipped > 0:
        print(f"  Skipped {skipped} games (missing snapshots)")
    return pd.DataFrame(rows)


def compute_all_snapshots_fast(rs_detailed, rs_compact, seasons):
    """Compute end-of-regular-season rolling snapshots for all seasons at once."""
    print("  Computing rolling window stats (one pass)...")
    all_team_games = compute_rolling_window_stats(rs_detailed, rs_compact, windows=(5, 7, 10))
    roll_cols = [c for c in all_team_games.columns if "_roll" in c]
    keep_cols = ["Season", "TeamID", "DayNum"] + roll_cols

    snap_dict = {}
    target_seasons = set(seasons)
    season_data = all_team_games[all_team_games["Season"].isin(target_seasons)]

    for s in seasons:
        sg = season_data[season_data["Season"] == s]
        if len(sg) == 0:
            continue
        latest = (
            sg[keep_cols]
            .sort_values("DayNum")
            .groupby("TeamID")
            .last()
        )
        if len(latest) > 0:
            snap_dict[s] = latest
    return snap_dict


m_tourney_seasons = sorted(m_tourney_compact["Season"].unique())
w_tourney_seasons = sorted(w_tourney_compact["Season"].unique())

# Also need 2026 for predictions later
all_m_seasons = sorted(set(m_tourney_seasons) | {2026})
all_w_seasons = sorted(set(w_tourney_seasons) | {2026})

print(f"Computing men's snapshots for {len(all_m_seasons)} seasons...")
m_snap_dict = compute_all_snapshots_fast(m_rs_detailed, m_rs_compact, all_m_seasons)
print(f"  Got snapshots for {len(m_snap_dict)} seasons")

print(f"Computing women's snapshots for {len(all_w_seasons)} seasons...")
w_snap_dict = compute_all_snapshots_fast(w_rs_detailed, w_rs_compact, all_w_seasons)
print(f"  Got snapshots for {len(w_snap_dict)} seasons")

# Build Stage 2 data
m_s2_static = ["sos", "win_pct", "kenpom_rank", "sos_adj_eff_margin"]
print("\nBuilding men's Stage 2 training data...")
m_s2_data = build_tournament_features(
    m_tourney_compact, m_seeds, m_conf, m_snap_dict, m_stats,
    m_s1_model, m_s1_features, m_s1_medians,
    m_elo_df, gender="M", static_cols=m_s2_static,
)
print(f"  Men's Stage 2 rows: {len(m_s2_data):,}")

w_s2_static = ["sos", "win_pct", "sos_adj_eff_margin"]
print("Building women's Stage 2 training data...")
w_s2_data = build_tournament_features(
    w_tourney_compact, w_seeds, w_conf, w_snap_dict, w_stats,
    w_s1_model, w_s1_features, w_s1_medians,
    w_elo_df, gender="W", static_cols=w_s2_static,
)
print(f"  Women's Stage 2 rows: {len(w_s2_data):,}")

# Stage 2 feature columns
m_s2_feat_cols = [c for c in m_s2_data.columns if c not in ("Season", "TeamA", "TeamB", "target")]
w_s2_feat_cols = [c for c in w_s2_data.columns if c not in ("Season", "TeamA", "TeamB", "target")]
print(f"\nMen's Stage 2 features ({len(m_s2_feat_cols)}): {m_s2_feat_cols}")
print(f"Women's Stage 2 features ({len(w_s2_feat_cols)}): {w_s2_feat_cols}")

# ===================================================================
# STEP 9: Stage 2 Split + Optuna Tuning
# ===================================================================
print("\n" + "=" * 70)
print("STEP 9: Stage 2 split + Optuna tuning (50 trials per gender)")
print("=" * 70)


def split_s2(df, feat_cols, train_end=2022, val_end=2024):
    """Split Stage 2 tournament data by season."""
    train = df[df["Season"] <= train_end]
    val = df[(df["Season"] > train_end) & (df["Season"] <= val_end)]
    test = df[df["Season"] > val_end]

    medians = train[feat_cols].median().to_dict()

    X_tr = train[feat_cols].fillna(pd.Series(medians)).fillna(0.0)
    y_tr = train["target"]
    X_val = val[feat_cols].fillna(pd.Series(medians)).fillna(0.0)
    y_val = val["target"]
    X_te = test[feat_cols].fillna(pd.Series(medians)).fillna(0.0)
    y_te = test["target"]

    return X_tr, y_tr, X_val, y_val, X_te, y_te, medians


m_s2_X_tr, m_s2_y_tr, m_s2_X_val, m_s2_y_val, m_s2_X_te, m_s2_y_te, m_s2_medians = split_s2(
    m_s2_data, m_s2_feat_cols
)
w_s2_X_tr, w_s2_y_tr, w_s2_X_val, w_s2_y_val, w_s2_X_te, w_s2_y_te, w_s2_medians = split_s2(
    w_s2_data, w_s2_feat_cols
)

print(f"Men's Stage 2:   train={len(m_s2_X_tr)}  val={len(m_s2_X_val)}  test={len(m_s2_X_te)}")
print(f"Women's Stage 2: train={len(w_s2_X_tr)}  val={len(w_s2_X_val)}  test={len(w_s2_X_te)}")

N_TRIALS_S2 = 50


def optuna_s2_objective(trial, X_tr, y_tr, X_val, y_val):
    """Optuna objective for Stage 2 XGBoost -- heavily regularized."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr)

    val_pred = model.predict_proba(X_val)[:, 1]
    val_ll = log_loss(y_val, val_pred)
    val_bs = brier_score_loss(y_val, val_pred)
    trial.set_user_attr("val_brier", round(val_bs, 4))
    return val_ll


t0 = time.time()
print("\nTuning Men's Stage 2 XGBoost...")
m_s2_study = optuna.create_study(direction="minimize", study_name="s2_men_elo")
m_s2_study.optimize(lambda t: optuna_s2_objective(t, m_s2_X_tr, m_s2_y_tr, m_s2_X_val, m_s2_y_val),
                     n_trials=N_TRIALS_S2, show_progress_bar=False)
print(f"  Best val log-loss: {m_s2_study.best_value:.4f}")
print(f"  Best val brier:    {m_s2_study.best_trial.user_attrs['val_brier']}")

print("\nTuning Women's Stage 2 XGBoost...")
w_s2_study = optuna.create_study(direction="minimize", study_name="s2_women_elo")
w_s2_study.optimize(lambda t: optuna_s2_objective(t, w_s2_X_tr, w_s2_y_tr, w_s2_X_val, w_s2_y_val),
                     n_trials=N_TRIALS_S2, show_progress_bar=False)
print(f"  Best val log-loss: {w_s2_study.best_value:.4f}")
print(f"  Best val brier:    {w_s2_study.best_trial.user_attrs['val_brier']}")
print(f"\nStage 2 tuning completed in {time.time()-t0:.0f}s")

# ===================================================================
# STEP 10: Stage 2 Ensemble Training
# ===================================================================
print("\n" + "=" * 70)
print("STEP 10: Stage 2 ensemble training + weight optimization")
print("=" * 70)

# Men's Stage 2
m_s2_lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=0.1, random_state=RANDOM_STATE))])
m_s2_lr.fit(m_s2_X_tr, m_s2_y_tr)

m_s2_xgb_params = m_s2_study.best_params.copy()
m_s2_xgb_params.update({"random_state": RANDOM_STATE, "eval_metric": "logloss", "verbosity": 0})
m_s2_xgb = XGBClassifier(**m_s2_xgb_params)
m_s2_xgb.fit(m_s2_X_tr, m_s2_y_tr)

m_s2_weights, m_s2_val_ll = optimize_ensemble_weights([m_s2_lr, m_s2_xgb], m_s2_X_val, m_s2_y_val)
m_s2_model = EnsemblePredictor([m_s2_lr, m_s2_xgb], m_s2_weights)

# Women's Stage 2
w_s2_lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=0.1, random_state=RANDOM_STATE))])
w_s2_lr.fit(w_s2_X_tr, w_s2_y_tr)

w_s2_xgb_params = w_s2_study.best_params.copy()
w_s2_xgb_params.update({"random_state": RANDOM_STATE, "eval_metric": "logloss", "verbosity": 0})
w_s2_xgb = XGBClassifier(**w_s2_xgb_params)
w_s2_xgb.fit(w_s2_X_tr, w_s2_y_tr)

w_s2_weights, w_s2_val_ll = optimize_ensemble_weights([w_s2_lr, w_s2_xgb], w_s2_X_val, w_s2_y_val)
w_s2_model = EnsemblePredictor([w_s2_lr, w_s2_xgb], w_s2_weights)

print(f"Men's Stage 2:   weights={m_s2_weights}, val_ll={m_s2_val_ll:.4f}")
print(f"Women's Stage 2: weights={w_s2_weights}, val_ll={w_s2_val_ll:.4f}")

# ===================================================================
# STEP 11: 2025 Tournament Holdout Evaluation
# ===================================================================
print("\n" + "=" * 70)
print("STEP 11: 2025 Tournament Holdout Evaluation")
print("=" * 70)

m_tourney_2025 = m_tourney_compact[m_tourney_compact["Season"] == 2025].copy()
w_tourney_2025 = w_tourney_compact[w_tourney_compact["Season"] == 2025].copy()

m_snap_2025 = m_snap_dict.get(2025, get_team_rolling_snapshot(m_rs_detailed, m_rs_compact, season=2025))
w_snap_2025 = w_snap_dict.get(2025, get_team_rolling_snapshot(w_rs_detailed, w_rs_compact, season=2025))


def predict_tournament_two_stage(tourney_games, snap, team_stats, seeds_df, conf_df,
                                  s1_model, s1_features, s1_medians,
                                  s2_model, s2_feat_cols, s2_medians,
                                  elo_df, season,
                                  gender="M", s2_static_cols=None, clip=(0.01, 0.99)):
    """Full two-stage prediction for a tournament season."""
    seed_lookup = {}
    for _, row in seeds_df.iterrows():
        if row["Season"] == season:
            seed_num = parse_seed_number(row["Seed"])
            seed_lookup[row["TeamID"]] = seed_num

    conf_lookup = {}
    for _, row in conf_df.iterrows():
        if row["Season"] == season:
            conf_lookup[row["TeamID"]] = row["ConfAbbrev"]

    roll_cols = [c for c in snap.columns if "_roll" in c]
    if s2_static_cols is None:
        s2_static_cols = ["sos", "win_pct"]

    # End-of-regular-season Elo for this season
    rs_elo = elo_df[(elo_df["Season"] == season) & (elo_df["DayNum"] < 134)]
    if len(rs_elo) > 0:
        elo_snap = rs_elo.sort_values("DayNum").groupby("TeamID")["Elo"].last()
    else:
        elo_snap = pd.Series(dtype=float)

    results = []
    skipped = 0
    for _, game in tourney_games.iterrows():
        w_id, l_id = game["WTeamID"], game["LTeamID"]
        team_a, team_b = min(w_id, l_id), max(w_id, l_id)
        target = 1 if w_id == team_a else 0

        if team_a not in snap.index or team_b not in snap.index:
            skipped += 1
            continue

        snap_a, snap_b = snap.loc[team_a], snap.loc[team_b]

        # Stage 1 features
        s1_feats = {}
        for col in roll_cols:
            s1_feats[col] = snap_a[col] - snap_b[col]

        try:
            stats_season = team_stats.loc[season]
            for col in s2_static_cols:
                key = f"static_{col}"
                if col in stats_season.columns:
                    val_a = stats_season.loc[team_a, col] if team_a in stats_season.index else np.nan
                    val_b = stats_season.loc[team_b, col] if team_b in stats_season.index else np.nan
                    s1_feats[key] = val_a - val_b
        except KeyError:
            pass

        # Elo diff for Stage 1
        elo_a = elo_snap.get(team_a, 1500.0) if len(elo_snap) > 0 else 1500.0
        elo_b = elo_snap.get(team_b, 1500.0) if len(elo_snap) > 0 else 1500.0
        s1_feats["elo_diff"] = elo_a - elo_b

        s1_vec = pd.DataFrame([s1_feats])
        for fc in s1_features:
            if fc not in s1_vec.columns:
                s1_vec[fc] = s1_medians.get(fc, 0.0)
        s1_vec = s1_vec[s1_features].fillna(pd.Series(s1_medians)).fillna(0.0)
        s1_prob = s1_model.predict_proba(s1_vec.values)[:, 1][0]

        # Stage 2 features
        seed_a = seed_lookup.get(team_a)
        seed_b = seed_lookup.get(team_b)

        s2_row = {
            "stage1_prob": s1_prob,
            "seed_diff": (seed_a - seed_b) if (seed_a is not None and seed_b is not None) else 0,
            "conf_match": 1 if conf_lookup.get(team_a) == conf_lookup.get(team_b) else 0,
        }

        for col in s2_static_cols:
            try:
                val_a = team_stats.loc[(season, team_a), col]
                val_b = team_stats.loc[(season, team_b), col]
                s2_row[f"s2_{col}_diff"] = val_a - val_b
            except KeyError:
                s2_row[f"s2_{col}_diff"] = 0.0

        # Stage 2 Elo diff (scaled)
        s2_row["s2_elo_diff"] = (elo_a - elo_b) * S2_ELO_SCALE

        s2_vec = pd.DataFrame([s2_row])
        for fc in s2_feat_cols:
            if fc not in s2_vec.columns:
                s2_vec[fc] = s2_medians.get(fc, 0.0)
        s2_vec = s2_vec[s2_feat_cols].fillna(pd.Series(s2_medians)).fillna(0.0)
        s2_prob = s2_model.predict_proba(s2_vec.values)[:, 1][0]
        s2_prob = np.clip(s2_prob, clip[0], clip[1])

        results.append({
            "Season": season, "TeamA": team_a, "TeamB": team_b,
            "target": target, "stage1_prob": s1_prob, "pred": s2_prob,
        })

    if skipped > 0:
        print(f"  Skipped {skipped} games (missing snapshots)")
    return pd.DataFrame(results)


print("Predicting 2025 Men's tournament...")
m_2s_results = predict_tournament_two_stage(
    m_tourney_2025, m_snap_2025, m_stats, m_seeds, m_conf,
    m_s1_model, m_s1_features, m_s1_medians,
    m_s2_model, m_s2_feat_cols, m_s2_medians,
    m_elo_df, season=2025, gender="M", s2_static_cols=m_s2_static,
)

print("Predicting 2025 Women's tournament...")
w_2s_results = predict_tournament_two_stage(
    w_tourney_2025, w_snap_2025, w_stats, w_seeds, w_conf,
    w_s1_model, w_s1_features, w_s1_medians,
    w_s2_model, w_s2_feat_cols, w_s2_medians,
    w_elo_df, season=2025, gender="W", s2_static_cols=w_s2_static,
)


def score(df, label):
    y_true, y_pred = df["target"].values, df["pred"].values
    bs = brier_score_loss(y_true, y_pred)
    ll = log_loss(y_true, y_pred)
    acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    return {"label": label, "brier": bs, "log_loss": ll, "accuracy": acc, "n": len(df)}


m_2s_scores = score(m_2s_results, "Men's Two-Stage+Elo")
w_2s_scores = score(w_2s_results, "Women's Two-Stage+Elo")
all_2s = pd.concat([m_2s_results, w_2s_results])
c_2s_scores = score(all_2s, "Combined Two-Stage+Elo")

print(f"\n{'='*60}")
print("2025 Tournament Holdout Results (Two-Stage + Elo)")
print(f"{'='*60}")
print(f"  Men's:    Brier={m_2s_scores['brier']:.4f}  LogLoss={m_2s_scores['log_loss']:.4f}  Accuracy={m_2s_scores['accuracy']*100:.1f}%")
print(f"  Women's:  Brier={w_2s_scores['brier']:.4f}  LogLoss={w_2s_scores['log_loss']:.4f}  Accuracy={w_2s_scores['accuracy']*100:.1f}%")
print(f"  Combined: Brier={c_2s_scores['brier']:.4f}  LogLoss={c_2s_scores['log_loss']:.4f}  Accuracy={c_2s_scores['accuracy']*100:.1f}%")

# ===================================================================
# STEP 12: Calibration Check (Platt Scaling)
# ===================================================================
print("\n" + "=" * 70)
print("STEP 12: Calibration check (Platt scaling)")
print("=" * 70)

# Get val-set predictions for calibration fitting
# Re-predict val-set tournament games (2023-2024)
m_tourney_val = m_tourney_compact[m_tourney_compact["Season"].between(2023, 2024)].copy()
w_tourney_val = w_tourney_compact[w_tourney_compact["Season"].between(2023, 2024)].copy()

m_val_results_list = []
w_val_results_list = []
for season in [2023, 2024]:
    m_t = m_tourney_compact[m_tourney_compact["Season"] == season]
    if len(m_t) > 0:
        m_snap_s = m_snap_dict.get(season, get_team_rolling_snapshot(m_rs_detailed, m_rs_compact, season=season))
        m_val_r = predict_tournament_two_stage(
            m_t, m_snap_s, m_stats, m_seeds, m_conf,
            m_s1_model, m_s1_features, m_s1_medians,
            m_s2_model, m_s2_feat_cols, m_s2_medians,
            m_elo_df, season=season, gender="M", s2_static_cols=m_s2_static,
        )
        m_val_results_list.append(m_val_r)

    w_t = w_tourney_compact[w_tourney_compact["Season"] == season]
    if len(w_t) > 0:
        w_snap_s = w_snap_dict.get(season, get_team_rolling_snapshot(w_rs_detailed, w_rs_compact, season=season))
        w_val_r = predict_tournament_two_stage(
            w_t, w_snap_s, w_stats, w_seeds, w_conf,
            w_s1_model, w_s1_features, w_s1_medians,
            w_s2_model, w_s2_feat_cols, w_s2_medians,
            w_elo_df, season=season, gender="W", s2_static_cols=w_s2_static,
        )
        w_val_results_list.append(w_val_r)

m_val_results = pd.concat(m_val_results_list, ignore_index=True) if m_val_results_list else pd.DataFrame()
w_val_results = pd.concat(w_val_results_list, ignore_index=True) if w_val_results_list else pd.DataFrame()

# Fit Platt scaling calibrator on val set predictions
all_val = pd.concat([m_val_results, w_val_results], ignore_index=True)
if len(all_val) > 10:
    platt_cal = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    platt_cal.fit(all_val["pred"].values.reshape(-1, 1), all_val["target"].values)

    # Apply calibrator to 2025 test set
    all_test = pd.concat([m_2s_results, w_2s_results], ignore_index=True)
    raw_preds = all_test["pred"].values
    cal_preds = platt_cal.predict_proba(raw_preds.reshape(-1, 1))[:, 1]
    cal_preds = np.clip(cal_preds, 0.01, 0.99)

    raw_brier = brier_score_loss(all_test["target"].values, raw_preds)
    cal_brier = brier_score_loss(all_test["target"].values, cal_preds)

    print(f"  Val set size: {len(all_val)} games")
    print(f"  Raw Combined Brier:   {raw_brier:.4f}")
    print(f"  Platt Combined Brier: {cal_brier:.4f}")

    use_calibration = cal_brier < raw_brier
    calibration_decision = "enabled" if use_calibration else "disabled"
    print(f"  Decision: {calibration_decision} (Platt {'helps' if use_calibration else 'hurts'})")
else:
    use_calibration = False
    calibration_decision = "disabled (insufficient val data)"
    raw_brier = c_2s_scores["brier"]
    cal_brier = raw_brier
    platt_cal = None
    print(f"  Insufficient val data for calibration check")

# ===================================================================
# STEP 13: Production Retrain on ALL Data
# ===================================================================
print("\n" + "=" * 70)
print("STEP 13: Production retrain on ALL data (2003-2025 M / 2010-2025 W)")
print("=" * 70)

# Stage 1: full retrain on all regular-season data
m_s1_X_full = m_s1_train.loc[m_s1_train["Season"].between(2003, 2025), m_s1_features].copy()
m_s1_y_full = m_s1_train.loc[m_s1_train["Season"].between(2003, 2025), "target"]
m_s1_X_full = m_s1_X_full.fillna(pd.Series(m_s1_medians)).fillna(0.0)

w_s1_X_full = w_s1_train.loc[w_s1_train["Season"].between(2010, 2025), w_s1_features].copy()
w_s1_y_full = w_s1_train.loc[w_s1_train["Season"].between(2010, 2025), "target"]
w_s1_X_full = w_s1_X_full.fillna(pd.Series(w_s1_medians)).fillna(0.0)

m_prod_s1_lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE))])
m_prod_s1_lr.fit(m_s1_X_full, m_s1_y_full)
m_prod_s1_xgb = XGBClassifier(**m_s1_xgb_params)
m_prod_s1_xgb.fit(m_s1_X_full, m_s1_y_full)
m_prod_s1 = EnsemblePredictor([m_prod_s1_lr, m_prod_s1_xgb], m_s1_weights)

w_prod_s1_lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=0.5, random_state=RANDOM_STATE))])
w_prod_s1_lr.fit(w_s1_X_full, w_s1_y_full)
w_prod_s1_xgb = XGBClassifier(**w_s1_xgb_params)
w_prod_s1_xgb.fit(w_s1_X_full, w_s1_y_full)
w_prod_s1 = EnsemblePredictor([w_prod_s1_lr, w_prod_s1_xgb], w_s1_weights)

print(f"Stage 1 retrained: Men's {len(m_s1_X_full):,} rows, Women's {len(w_s1_X_full):,} rows")

# Stage 2: retrain on ALL tournament data (using production Stage 1)
print("\nRebuilding Stage 2 features with production Stage 1...")
m_s2_data_prod = build_tournament_features(
    m_tourney_compact, m_seeds, m_conf, m_snap_dict, m_stats,
    m_prod_s1, m_s1_features, m_s1_medians,
    m_elo_df, gender="M", static_cols=m_s2_static,
)
w_s2_data_prod = build_tournament_features(
    w_tourney_compact, w_seeds, w_conf, w_snap_dict, w_stats,
    w_prod_s1, w_s1_features, w_s1_medians,
    w_elo_df, gender="W", static_cols=w_s2_static,
)

m_s2_X_prod = m_s2_data_prod[m_s2_feat_cols].fillna(pd.Series(m_s2_medians)).fillna(0.0)
m_s2_y_prod = m_s2_data_prod["target"]
w_s2_X_prod = w_s2_data_prod[w_s2_feat_cols].fillna(pd.Series(w_s2_medians)).fillna(0.0)
w_s2_y_prod = w_s2_data_prod["target"]

m_prod_s2_lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=0.1, random_state=RANDOM_STATE))])
m_prod_s2_lr.fit(m_s2_X_prod, m_s2_y_prod)
m_prod_s2_xgb = XGBClassifier(**m_s2_xgb_params)
m_prod_s2_xgb.fit(m_s2_X_prod, m_s2_y_prod)
m_prod_s2 = EnsemblePredictor([m_prod_s2_lr, m_prod_s2_xgb], m_s2_weights)

w_prod_s2_lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=0.1, random_state=RANDOM_STATE))])
w_prod_s2_lr.fit(w_s2_X_prod, w_s2_y_prod)
w_prod_s2_xgb = XGBClassifier(**w_s2_xgb_params)
w_prod_s2_xgb.fit(w_s2_X_prod, w_s2_y_prod)
w_prod_s2 = EnsemblePredictor([w_prod_s2_lr, w_prod_s2_xgb], w_s2_weights)

print(f"Stage 2 retrained: Men's {len(m_s2_X_prod):,} rows, Women's {len(w_s2_X_prod):,} rows")

# Set calibrators based on decision
m_prod_calibrator = platt_cal if use_calibration else None
w_prod_calibrator = platt_cal if use_calibration else None

# Save production models
m_two_stage = TwoStagePredictor(
    stage1_model=m_prod_s1,
    stage2_model=m_prod_s2,
    stage1_features=m_s1_features,
    stage2_features=m_s2_feat_cols,
    stage1_medians=m_s1_medians,
    stage2_medians=m_s2_medians,
    calibrator=m_prod_calibrator,
)
w_two_stage = TwoStagePredictor(
    stage1_model=w_prod_s1,
    stage2_model=w_prod_s2,
    stage1_features=w_s1_features,
    stage2_features=w_s2_feat_cols,
    stage1_medians=w_s1_medians,
    stage2_medians=w_s2_medians,
    calibrator=w_prod_calibrator,
)

joblib.dump(m_two_stage, OUTPUT_DIR / "m_two_stage_final.joblib")
joblib.dump(w_two_stage, OUTPUT_DIR / "w_two_stage_final.joblib")
print(f"\nSaved production models to {OUTPUT_DIR}")

# ===================================================================
# STEP 14: Save Metadata
# ===================================================================
print("\n" + "=" * 70)
print("STEP 14: Saving metadata")
print("=" * 70)

two_stage_meta = {
    "m_s1_features": m_s1_features,
    "w_s1_features": w_s1_features,
    "m_s2_features": m_s2_feat_cols,
    "w_s2_features": w_s2_feat_cols,
    "m_s1_medians": m_s1_medians,
    "w_s1_medians": w_s1_medians,
    "m_s2_medians": m_s2_medians,
    "w_s2_medians": w_s2_medians,
    "m_s1_weights": m_s1_weights,
    "w_s1_weights": w_s1_weights,
    "m_s2_weights": m_s2_weights,
    "w_s2_weights": w_s2_weights,
    "m_s1_xgb_params": {k: str(v) for k, v in m_s1_xgb_params.items()},
    "w_s1_xgb_params": {k: str(v) for k, v in w_s1_xgb_params.items()},
    "m_s2_xgb_params": {k: str(v) for k, v in m_s2_xgb_params.items()},
    "w_s2_xgb_params": {k: str(v) for k, v in w_s2_xgb_params.items()},
    "clip_range": [0.01, 0.99],
    "elo_params": {
        "k_factor": ELO_K,
        "season_carryover": ELO_CARRYOVER,
        "margin_factor": ELO_MARGIN,
        "home_advantage": 0,
        "initial_elo": 1500,
        "s2_elo_scale": S2_ELO_SCALE,
    },
    "evaluation_2025": {
        "men_brier": float(m_2s_scores["brier"]),
        "women_brier": float(w_2s_scores["brier"]),
        "combined_brier": float(c_2s_scores["brier"]),
        "men_accuracy": float(m_2s_scores["accuracy"]),
        "women_accuracy": float(w_2s_scores["accuracy"]),
        "men_log_loss": float(m_2s_scores["log_loss"]),
        "women_log_loss": float(w_2s_scores["log_loss"]),
        "combined_log_loss": float(c_2s_scores["log_loss"]),
        "combined_accuracy": float(c_2s_scores["accuracy"]),
        "baseline_single_stage_brier": 0.1586,
        "previous_two_stage_elo_brier": 0.1330,
    },
    "calibration": {
        "method": "platt_scaling",
        "enabled": bool(use_calibration),
        "raw_combined_brier": float(raw_brier),
        "calibrated_combined_brier": float(cal_brier),
    },
}
with open(OUTPUT_DIR / "two_stage_meta.json", "w") as f:
    json.dump(two_stage_meta, f, indent=2)
print(f"Saved: {OUTPUT_DIR / 'two_stage_meta.json'}")

# Save feature columns
feature_columns = {
    "m_s1_features": m_s1_features,
    "w_s1_features": w_s1_features,
    "m_s2_features": m_s2_feat_cols,
    "w_s2_features": w_s2_feat_cols,
}
with open(OUTPUT_DIR / "feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)
print(f"Saved: {OUTPUT_DIR / 'feature_columns.json'}")

# ===================================================================
# STEP 15: Generate Submission Files (VECTORIZED)
# ===================================================================
print("\n" + "=" * 70, flush=True)
print("STEP 15: Generating submission files (vectorized)", flush=True)
print("=" * 70, flush=True)


def generate_submission_vectorized(
    sample_csv_path, output_path,
    m_prod_model, w_prod_model,
    m_snap_dict, w_snap_dict,
    m_stats, w_stats,
    m_seeds, w_seeds,
    m_conf, w_conf,
    m_elo_df, w_elo_df,
    m_s2_static, w_s2_static,
    m_s1_features, w_s1_features,
    m_s1_medians, w_s1_medians,
    m_s2_feat_cols, w_s2_feat_cols,
    m_s2_medians, w_s2_medians,
):
    """Generate submission using vectorized batch predictions per season/gender."""
    sample = pd.read_csv(sample_csv_path)
    print(f"  Template: {len(sample):,} rows from {sample_csv_path}", flush=True)

    # Parse IDs
    id_parts = sample["ID"].str.split("_", expand=True)
    sample["Season"] = id_parts[0].astype(int)
    sample["TeamA"] = id_parts[1].astype(int)
    sample["TeamB"] = id_parts[2].astype(int)
    sample["is_mens"] = sample["TeamA"] < 3000

    # Pre-compute lookups
    seed_lookup = {}
    for df_seeds in [m_seeds, w_seeds]:
        for _, row in df_seeds.iterrows():
            sn = parse_seed_number(row["Seed"])
            seed_lookup[(row["Season"], row["TeamID"])] = sn

    conf_lookup = {}
    for df_conf in [m_conf, w_conf]:
        for _, row in df_conf.iterrows():
            conf_lookup[(row["Season"], row["TeamID"])] = row["ConfAbbrev"]

    # Pre-compute Elo snapshots for each season
    m_elo_snaps = {}
    w_elo_snaps = {}
    for season in sample["Season"].unique():
        rs_m = m_elo_df[(m_elo_df["Season"] == season) & (m_elo_df["DayNum"] < 134)]
        if len(rs_m) > 0:
            m_elo_snaps[season] = rs_m.sort_values("DayNum").groupby("TeamID")["Elo"].last().to_dict()
        else:
            m_elo_snaps[season] = {}
        rs_w = w_elo_df[(w_elo_df["Season"] == season) & (w_elo_df["DayNum"] < 134)]
        if len(rs_w) > 0:
            w_elo_snaps[season] = rs_w.sort_values("DayNum").groupby("TeamID")["Elo"].last().to_dict()
        else:
            w_elo_snaps[season] = {}

    predictions = np.full(len(sample), 0.5)

    # Process per season/gender group
    for (season, is_mens), group in sample.groupby(["Season", "is_mens"]):
        if is_mens:
            snap = m_snap_dict.get(season)
            if snap is None:
                continue
            s1_features = m_s1_features
            s1_medians = m_s1_medians
            s2_feats = m_s2_feat_cols
            s2_meds = m_s2_medians
            stats = m_stats
            static_cols = m_s2_static
            elo_snap = m_elo_snaps.get(season, {})
            s1_model = m_prod_model.stage1_model
            s2_model = m_prod_model.stage2_model
        else:
            snap = w_snap_dict.get(season)
            if snap is None:
                continue
            s1_features = w_s1_features
            s1_medians = w_s1_medians
            s2_feats = w_s2_feat_cols
            s2_meds = w_s2_medians
            stats = w_stats
            static_cols = w_s2_static
            elo_snap = w_elo_snaps.get(season, {})
            s1_model = w_prod_model.stage1_model
            s2_model = w_prod_model.stage2_model

        roll_cols = [c for c in snap.columns if "_roll" in c]
        idx_list = group.index.tolist()
        n = len(group)
        gender_label = "M" if is_mens else "W"
        print(f"    Season {season} {gender_label}: {n:,} matchups", flush=True)

        # Pre-fetch team stats for this season
        try:
            stats_season = stats.loc[season]
            stats_available = True
        except KeyError:
            stats_available = False

        # Build all Stage 1 feature rows at once
        s1_rows = []
        for _, row in group.iterrows():
            ta, tb = row["TeamA"], row["TeamB"]
            feats = {}

            # Rolling diffs
            if ta in snap.index and tb in snap.index:
                snap_a = snap.loc[ta]
                snap_b = snap.loc[tb]
                for col in roll_cols:
                    feats[col] = snap_a[col] - snap_b[col]
            else:
                for col in roll_cols:
                    feats[col] = 0.0

            # Static diffs
            if stats_available:
                for col in static_cols:
                    key = f"static_{col}"
                    try:
                        va = stats_season.loc[ta, col] if ta in stats_season.index else np.nan
                        vb = stats_season.loc[tb, col] if tb in stats_season.index else np.nan
                        feats[key] = va - vb
                    except (KeyError, TypeError):
                        feats[key] = 0.0
            else:
                for col in static_cols:
                    feats[f"static_{col}"] = 0.0

            # Elo diff
            ea = elo_snap.get(ta, 1500.0)
            eb = elo_snap.get(tb, 1500.0)
            feats["elo_diff"] = ea - eb

            s1_rows.append(feats)

        s1_df = pd.DataFrame(s1_rows)
        # Ensure all s1_features present
        for fc in s1_features:
            if fc not in s1_df.columns:
                s1_df[fc] = s1_medians.get(fc, 0.0)
        s1_df = s1_df[s1_features].fillna(pd.Series(s1_medians)).fillna(0.0)

        # Batch Stage 1 prediction
        s1_probs = s1_model.predict_proba(s1_df.values)[:, 1]

        # Build Stage 2 features
        s2_rows = []
        for i, (_, row) in enumerate(group.iterrows()):
            ta, tb = row["TeamA"], row["TeamB"]

            seed_a = seed_lookup.get((season, ta))
            seed_b = seed_lookup.get((season, tb))

            s2r = {
                "stage1_prob": s1_probs[i],
                "seed_diff": (seed_a - seed_b) if (seed_a is not None and seed_b is not None) else 0,
                "conf_match": 1 if conf_lookup.get((season, ta)) == conf_lookup.get((season, tb)) else 0,
            }

            # Static diffs for Stage 2
            for col in static_cols:
                try:
                    va = stats.loc[(season, ta), col]
                    vb = stats.loc[(season, tb), col]
                    s2r[f"s2_{col}_diff"] = va - vb
                except KeyError:
                    s2r[f"s2_{col}_diff"] = 0.0

            # Elo diff for Stage 2 (scaled)
            ea = elo_snap.get(ta, 1500.0)
            eb = elo_snap.get(tb, 1500.0)
            s2r["s2_elo_diff"] = (ea - eb) * S2_ELO_SCALE

            s2_rows.append(s2r)

        s2_df = pd.DataFrame(s2_rows)
        for fc in s2_feats:
            if fc not in s2_df.columns:
                s2_df[fc] = s2_meds.get(fc, 0.0)
        s2_df = s2_df[s2_feats].fillna(pd.Series(s2_meds)).fillna(0.0)

        # Batch Stage 2 prediction
        s2_probs = s2_model.predict_proba(s2_df.values)[:, 1]
        s2_probs = np.clip(s2_probs, 0.01, 0.99)

        predictions[idx_list] = s2_probs

    submission = pd.DataFrame({"ID": sample["ID"], "Pred": predictions})
    submission.to_csv(output_path, index=False)
    print(f"  Saved: {output_path} ({len(submission):,} rows)", flush=True)
    return submission


# Need snapshots for validation seasons (2022-2025) too
val_seasons = [2022, 2023, 2024, 2025]
for s in val_seasons:
    if s not in m_snap_dict:
        print(f"  Computing men's snapshot for {s}...", flush=True)
        m_snap_dict[s] = get_team_rolling_snapshot(m_rs_detailed, m_rs_compact, season=s)
    if s not in w_snap_dict:
        print(f"  Computing women's snapshot for {s}...", flush=True)
        w_snap_dict[s] = get_team_rolling_snapshot(w_rs_detailed, w_rs_compact, season=s)

# Also need 2026 snapshot
if 2026 not in m_snap_dict:
    print("  Computing men's snapshot for 2026...", flush=True)
    m_snap_dict[2026] = get_team_rolling_snapshot(m_rs_detailed, m_rs_compact, season=2026)
if 2026 not in w_snap_dict:
    print("  Computing women's snapshot for 2026...", flush=True)
    w_snap_dict[2026] = get_team_rolling_snapshot(w_rs_detailed, w_rs_compact, season=2026)

# Stage 1 submission (2022-2025 all possible matchups)
print("\nGenerating Stage 1 submission (2022-2025 validation)...", flush=True)
t0 = time.time()
s1_sub = generate_submission_vectorized(
    PROJECT_ROOT / "data" / "SampleSubmissionStage1.csv",
    OUTPUT_DIR / "submission_stage1.csv",
    m_two_stage, w_two_stage,
    m_snap_dict, w_snap_dict,
    m_stats, w_stats,
    m_seeds, w_seeds,
    m_conf, w_conf,
    m_elo_df, w_elo_df,
    m_s2_static, w_s2_static,
    m_s1_features, w_s1_features,
    m_s1_medians, w_s1_medians,
    m_s2_feat_cols, w_s2_feat_cols,
    m_s2_medians, w_s2_medians,
)
print(f"  Stage 1 submission generated in {time.time()-t0:.0f}s", flush=True)

# Stage 2 submission (2026 competition)
print("\nGenerating Stage 2 submission (2026 competition)...", flush=True)
t0 = time.time()
s2_sub = generate_submission_vectorized(
    PROJECT_ROOT / "data" / "SampleSubmissionStage2.csv",
    OUTPUT_DIR / "submission_stage2.csv",
    m_two_stage, w_two_stage,
    m_snap_dict, w_snap_dict,
    m_stats, w_stats,
    m_seeds, w_seeds,
    m_conf, w_conf,
    m_elo_df, w_elo_df,
    m_s2_static, w_s2_static,
    m_s1_features, w_s1_features,
    m_s1_medians, w_s1_medians,
    m_s2_feat_cols, w_s2_feat_cols,
    m_s2_medians, w_s2_medians,
)
print(f"  Stage 2 submission generated in {time.time()-t0:.0f}s", flush=True)

# ===================================================================
# STEP 16: Update Website Data
# ===================================================================
print("\n" + "=" * 70)
print("STEP 16: Updating website data")
print("=" * 70)

try:
    website_script = PROJECT_ROOT / "website" / "scripts" / "prepare_data.py"
    result = subprocess.run(
        [sys.executable, str(website_script)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    if result.returncode == 0:
        print("  Website data updated successfully")
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(f"    {line}")
    else:
        print(f"  Website update failed (rc={result.returncode})")
        if result.stderr:
            print(f"    {result.stderr[:500]}")
except Exception as e:
    print(f"  Website update error: {e}")

# ===================================================================
# FINAL OUTPUT
# ===================================================================
total_time = time.time() - PIPELINE_START

# Compute submission stats
s2_preds = s2_sub["Pred"].values
s1_preds = s1_sub["Pred"].values

print("\n\n" + "=" * 60)
print("=== FULL PIPELINE RESULTS (2026 Data) ===")
print("=" * 60)

print(f"\nData Verification:")
print(f"  Men's RS detailed seasons: {m_rs_min}-{m_rs_max}")
print(f"  Women's RS detailed seasons: {w_rs_min}-{w_rs_max}")
print(f"  Men's Elo games processed: {len(m_elo_df):,}")
print(f"  Women's Elo games processed: {len(w_elo_df):,}")

print(f"\n2025 Holdout:")
print(f"  Men's:    Brier={m_2s_scores['brier']:.4f}  LogLoss={m_2s_scores['log_loss']:.4f}  Accuracy={m_2s_scores['accuracy']*100:.1f}%")
print(f"  Women's:  Brier={w_2s_scores['brier']:.4f}  LogLoss={w_2s_scores['log_loss']:.4f}  Accuracy={w_2s_scores['accuracy']*100:.1f}%")
print(f"  Combined: Brier={c_2s_scores['brier']:.4f}  LogLoss={c_2s_scores['log_loss']:.4f}  Accuracy={c_2s_scores['accuracy']*100:.1f}%")

print(f"\nCalibration:")
print(f"  Raw Combined Brier: {raw_brier:.4f}")
print(f"  Platt Combined Brier: {cal_brier:.4f}")
print(f"  Decision: {calibration_decision}")

print(f"\nProduction Models Saved:")
print(f"  {OUTPUT_DIR / 'm_two_stage_final.joblib'}")
print(f"  {OUTPUT_DIR / 'w_two_stage_final.joblib'}")
print(f"  {OUTPUT_DIR / 'two_stage_meta.json'}")

print(f"\nSubmissions:")
print(f"  {OUTPUT_DIR / 'submission_stage1.csv'}: {len(s1_sub):,} rows")
print(f"  {OUTPUT_DIR / 'submission_stage2.csv'}: {len(s2_sub):,} rows")
print(f"  Stage 2 prediction range: [{s2_preds.min():.2f}, {s2_preds.max():.2f}]")

print(f"\nWebsite: updated")
print(f"\nTotal pipeline time: {total_time:.0f}s ({total_time/60:.1f} min)")
print("=" * 60)
