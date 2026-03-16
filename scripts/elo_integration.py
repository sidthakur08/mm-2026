"""Elo Integration into Two-Stage Model Pipeline.

Implements:
1. Elo parameter tuning (grid search)
2. Elo feature integration into Stage 1 (elo_diff) and Stage 2 (s2_elo_diff)
3. Feature selection with protected features
4. Optuna hyperparameter tuning (50 trials per stage per gender)
5. Ensemble weight optimization
6. 2025 holdout evaluation
7. Platt scaling calibration
8. Production model retrain and save (if improved)
9. Comprehensive comparison report
"""

import sys
import os
import json
import time
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

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

print(f"Men's regular season:   {m_rs_detailed.shape}")
print(f"Men's tournament:       {m_tourney_detailed.shape}")
print(f"Women's regular season: {w_rs_detailed.shape}")
print(f"Women's tournament:     {w_tourney_detailed.shape}")

# ===================================================================
# STEP 1: Elo Parameter Tuning (Grid Search)
# ===================================================================
print("\n" + "=" * 70)
print("STEP 1: Elo Parameter Tuning (Grid Search)")
print("=" * 70)

# Combine regular season + tournament for Elo computation (all games)
m_all_games = pd.concat([m_rs_compact, m_tourney_compact], ignore_index=True)
w_all_games = pd.concat([w_rs_compact, w_tourney_compact], ignore_index=True)

print(f"Men's all games:   {len(m_all_games):,} (seasons {m_all_games.Season.min()}-{m_all_games.Season.max()})")
print(f"Women's all games: {len(w_all_games):,} (seasons {w_all_games.Season.min()}-{w_all_games.Season.max()})")


def evaluate_elo_accuracy(all_games_df, k_factor, carryover, use_margin, test_seasons=range(2015, 2025)):
    """Evaluate Elo prediction accuracy on historical games."""
    elo_df = compute_elo_ratings(
        all_games_df,
        k_factor=k_factor,
        season_carryover=carryover,
        margin_factor=use_margin,
    )
    # Merge pre-game Elo back to game results
    # For each game, look up winner and loser pre-game Elo
    games = all_games_df[["Season", "DayNum", "WTeamID", "LTeamID"]].copy()
    games = games.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    # Build lookup: (Season, DayNum, TeamID) -> pre-game Elo
    # Since elo_df has exactly 2 rows per game (winner + loser), align by index
    elo_df_sorted = elo_df.copy()

    # Match records to games: records are in same order as games (2 per game)
    w_elos = elo_df_sorted.iloc[0::2]["Elo"].values  # even indices = winners
    l_elos = elo_df_sorted.iloc[1::2]["Elo"].values  # odd indices = losers

    games["w_elo"] = w_elos
    games["l_elo"] = l_elos

    # Filter to test seasons
    test_games = games[games["Season"].isin(test_seasons)]
    if len(test_games) == 0:
        return 0.5

    # Predict: higher Elo wins
    correct = (test_games["w_elo"] > test_games["l_elo"]).sum()
    return correct / len(test_games)


# Grid search parameters
k_factors = [16, 20, 24, 28, 32]
carryovers = [0.5, 0.6, 0.75, 0.85]
margin_options = [True, False]

print("\nRunning grid search on men's data...")
best_acc = 0
best_params = {}
grid_results = []

t0 = time.time()
for k, co, mf in product(k_factors, carryovers, margin_options):
    acc = evaluate_elo_accuracy(m_all_games, k, co, mf)
    grid_results.append({"k": k, "carryover": co, "margin": mf, "accuracy": acc})
    if acc > best_acc:
        best_acc = acc
        best_params = {"k_factor": k, "season_carryover": co, "margin_factor": mf}

grid_time = time.time() - t0
print(f"Grid search completed in {grid_time:.1f}s")
print(f"\nBest Elo parameters:")
print(f"  K-factor:  {best_params['k_factor']}")
print(f"  Carryover: {best_params['season_carryover']}")
print(f"  Margin:    {best_params['margin_factor']}")
print(f"  Accuracy:  {best_acc:.4f}")

# Show top 5
grid_df = pd.DataFrame(grid_results).sort_values("accuracy", ascending=False)
print("\nTop 5 parameter combinations:")
print(grid_df.head(5).to_string(index=False))

ELO_K = best_params["k_factor"]
ELO_CARRYOVER = best_params["season_carryover"]
ELO_MARGIN = best_params["margin_factor"]

# ===================================================================
# STEP 2: Compute Elo Ratings with Best Parameters
# ===================================================================
print("\n" + "=" * 70)
print("STEP 2: Computing Elo ratings with optimized parameters")
print("=" * 70)

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
# STEP 3: Build Static Features (same as notebook 06)
# ===================================================================
print("\n" + "=" * 70)
print("STEP 3: Building static features")
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

# ===================================================================
# STEP 4: Build Stage 1 Training Data with Elo
# ===================================================================
print("\n" + "=" * 70)
print("STEP 4: Building Stage 1 training data (regular season + Elo)")
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

# NOW ADD ELO_DIFF TO STAGE 1 TRAINING DATA
# For each game row (TeamA, TeamB, Season, DayNum), look up pre-game Elo
print("\nAdding elo_diff to Stage 1 training data...")


def add_elo_diff_to_training(train_df, elo_df):
    """Add elo_diff column to training data using pre-game Elos."""
    # Build a lookup: (Season, DayNum, TeamID) -> pre-game Elo
    # For each team-game, the Elo in elo_df is the pre-game rating
    elo_lookup = elo_df.set_index(["Season", "DayNum", "TeamID"])["Elo"]

    # For efficiency, merge instead of row-by-row lookup
    # Team A Elo
    train_with_elo = train_df.copy()

    # Merge Elo for TeamA: need the Elo for TeamA on this DayNum
    # But there may be multiple games on same DayNum for same team (unlikely but possible)
    # Use merge to get TeamA's Elo
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

    # Drop helper columns
    train_with_elo.drop(columns=["Elo_A", "Elo_B"], inplace=True)

    matched = train_with_elo["elo_diff"].notna().sum()
    total = len(train_with_elo)
    print(f"  Elo matched: {matched:,}/{total:,} ({matched/total*100:.1f}%)")

    return train_with_elo


# For regular season Elo, we need RS-only Elo (not including tournament games)
# because Stage 1 training uses only regular-season games.
# However, the Elo for those games was computed from all_games which includes tournament.
# The pre-game Elo for a regular-season game doesn't include future tournament games --
# it's computed chronologically, so this is fine.
# RS games have DayNum < 134 (before tournament)
m_s1_train = add_elo_diff_to_training(m_s1_train, m_elo_df)
w_s1_train = add_elo_diff_to_training(w_s1_train, w_elo_df)

# Drop NaN rows
meta_cols = ["target", "Season", "DayNum", "TeamA", "TeamB", "is_tourney"]
m_feat_all = [c for c in m_s1_train.columns if c not in meta_cols]
w_feat_all = [c for c in w_s1_train.columns if c not in meta_cols]
m_s1_train = m_s1_train.dropna(subset=[c for c in m_feat_all if c != "elo_diff"]).reset_index(drop=True)
w_s1_train = w_s1_train.dropna(subset=[c for c in w_feat_all if c != "elo_diff"]).reset_index(drop=True)

# Fill remaining NaN in elo_diff with 0 (teams not in Elo data)
m_s1_train["elo_diff"] = m_s1_train["elo_diff"].fillna(0.0)
w_s1_train["elo_diff"] = w_s1_train["elo_diff"].fillna(0.0)

print(f"\nStage 1 training data (regular season only):")
print(f"  Men's:   {len(m_s1_train):,} rows (seasons {m_s1_train.Season.min()}-{m_s1_train.Season.max()})")
print(f"  Women's: {len(w_s1_train):,} rows (seasons {w_s1_train.Season.min()}-{w_s1_train.Season.max()})")

# ===================================================================
# STEP 5: Feature Selection (MI + correlation pruning)
# ===================================================================
print("\n" + "=" * 70)
print("STEP 5: Feature selection with Elo")
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


# Men's Stage 1 -- include elo_diff in candidates
raw_feat_cols = [c for c in m_s1_train.columns if "_roll" in c or c.startswith("static_") or c == "elo_diff"]
print("Men's Stage 1 feature selection:")
m_s1_features, m_mi = select_features(
    m_s1_train, raw_feat_cols,
    protected_features=["static_kenpom_rank", "static_win_pct", "elo_diff"],
)
print(f"  Final: {len(m_s1_features)} features")
m_elo_in_s1_m = "elo_diff" in m_s1_features
print(f"  elo_diff kept: {'YES' if m_elo_in_s1_m else 'NO'}")

# Women's Stage 1
w_raw_feat_cols = [c for c in w_s1_train.columns if "_roll" in c or c.startswith("static_") or c == "elo_diff"]
print("\nWomen's Stage 1 feature selection:")
w_s1_features, w_mi = select_features(
    w_s1_train, w_raw_feat_cols,
    protected_features=["static_win_pct", "elo_diff"],
)
print(f"  Final: {len(w_s1_features)} features")
w_elo_in_s1_w = "elo_diff" in w_s1_features
print(f"  elo_diff kept: {'YES' if w_elo_in_s1_w else 'NO'}")

# ===================================================================
# STEP 6: Time-Based Split
# ===================================================================
print("\n" + "=" * 70)
print("STEP 6: Time-based split")
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
# STEP 7: Optuna Hyperparameter Tuning (Stage 1)
# ===================================================================
print("\n" + "=" * 70)
print("STEP 7: Optuna tuning -- Stage 1 (50 trials per gender)")
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
# STEP 8: Stage 1 Ensemble Training
# ===================================================================
print("\n" + "=" * 70)
print("STEP 8: Stage 1 ensemble training + weight optimization")
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
# STEP 9: Compute Rolling Snapshots + Build Stage 2 Data
# ===================================================================
print("\n" + "=" * 70)
print("STEP 9: Building Stage 2 data with Elo")
print("=" * 70)

import re as _re


def parse_seed_number(seed_str):
    """Extract numeric seed from string like 'W01' or 'X16a'."""
    match = _re.match(r"[WXYZ](\d{2})", seed_str)
    return int(match.group(1)) if match else None


def build_tournament_features(tourney_df, seeds_df, conf_df, snap_dict, team_stats,
                              stage1_model, stage1_features, stage1_medians,
                              elo_df, gender="M", static_cols=None):
    """Build Stage 2 training data from historical tournament games.

    Now includes s2_elo_diff: end-of-regular-season Elo difference.
    """
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
        # Get Elo before tournament (DayNum < 134 = regular season)
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

        # Elo diff for Stage 2 (end-of-regular-season)
        if season in elo_snapshots:
            elo_snap = elo_snapshots[season]
            elo_a = elo_snap.get(team_a, 1500.0)
            elo_b = elo_snap.get(team_b, 1500.0)
            s2_row["s2_elo_diff"] = elo_a - elo_b
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


# Compute rolling snapshots
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

print(f"Computing men's snapshots for {len(m_tourney_seasons)} seasons...")
m_snap_dict = compute_all_snapshots_fast(m_rs_detailed, m_rs_compact, m_tourney_seasons)
print(f"  Got snapshots for {len(m_snap_dict)} seasons")

print(f"Computing women's snapshots for {len(w_tourney_seasons)} seasons...")
w_snap_dict = compute_all_snapshots_fast(w_rs_detailed, w_rs_compact, w_tourney_seasons)
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
# STEP 10: Stage 2 Split + Optuna Tuning
# ===================================================================
print("\n" + "=" * 70)
print("STEP 10: Stage 2 split + Optuna tuning (50 trials per gender)")
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
# STEP 11: Stage 2 Ensemble Training
# ===================================================================
print("\n" + "=" * 70)
print("STEP 11: Stage 2 ensemble training + weight optimization")
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
# STEP 12: 2025 Tournament Holdout Evaluation
# ===================================================================
print("\n" + "=" * 70)
print("STEP 12: 2025 Tournament Holdout Evaluation")
print("=" * 70)

m_tourney_2025 = m_tourney_compact[m_tourney_compact["Season"] == 2025].copy()
w_tourney_2025 = w_tourney_compact[w_tourney_compact["Season"] == 2025].copy()

m_snap_2025 = m_snap_dict.get(2025, get_team_rolling_snapshot(m_rs_detailed, m_rs_compact, season=2025))
w_snap_2025 = w_snap_dict.get(2025, get_team_rolling_snapshot(w_rs_detailed, w_rs_compact, season=2025))


def predict_2025_two_stage(tourney_games, snap, team_stats, seeds_df, conf_df,
                            s1_model, s1_features, s1_medians,
                            s2_model, s2_feat_cols, s2_medians,
                            elo_df,
                            gender="M", s2_static_cols=None, clip=(0.01, 0.99)):
    """Full two-stage prediction for 2025 tournament."""
    seed_lookup = {}
    for _, row in seeds_df.iterrows():
        if row["Season"] == 2025:
            seed_num = parse_seed_number(row["Seed"])
            seed_lookup[row["TeamID"]] = seed_num

    conf_lookup = {}
    for _, row in conf_df.iterrows():
        if row["Season"] == 2025:
            conf_lookup[row["TeamID"]] = row["ConfAbbrev"]

    roll_cols = [c for c in snap.columns if "_roll" in c]
    if s2_static_cols is None:
        s2_static_cols = ["sos", "win_pct"]

    # End-of-regular-season Elo for 2025
    rs_elo = elo_df[(elo_df["Season"] == 2025) & (elo_df["DayNum"] < 134)]
    if len(rs_elo) > 0:
        elo_snap_2025 = rs_elo.sort_values("DayNum").groupby("TeamID")["Elo"].last()
    else:
        elo_snap_2025 = pd.Series(dtype=float)

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
            stats_2025 = team_stats.loc[2025]
            for col in s2_static_cols:
                key = f"static_{col}"
                if col in stats_2025.columns:
                    val_a = stats_2025.loc[team_a, col] if team_a in stats_2025.index else np.nan
                    val_b = stats_2025.loc[team_b, col] if team_b in stats_2025.index else np.nan
                    s1_feats[key] = val_a - val_b
        except KeyError:
            pass

        # Elo diff for Stage 1
        elo_a = elo_snap_2025.get(team_a, 1500.0) if len(elo_snap_2025) > 0 else 1500.0
        elo_b = elo_snap_2025.get(team_b, 1500.0) if len(elo_snap_2025) > 0 else 1500.0
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
                val_a = team_stats.loc[(2025, team_a), col]
                val_b = team_stats.loc[(2025, team_b), col]
                s2_row[f"s2_{col}_diff"] = val_a - val_b
            except KeyError:
                s2_row[f"s2_{col}_diff"] = 0.0

        # Stage 2 Elo diff
        s2_row["s2_elo_diff"] = elo_a - elo_b

        s2_vec = pd.DataFrame([s2_row])
        for fc in s2_feat_cols:
            if fc not in s2_vec.columns:
                s2_vec[fc] = s2_medians.get(fc, 0.0)
        s2_vec = s2_vec[s2_feat_cols].fillna(pd.Series(s2_medians)).fillna(0.0)
        s2_prob = s2_model.predict_proba(s2_vec.values)[:, 1][0]
        s2_prob = np.clip(s2_prob, clip[0], clip[1])

        results.append({
            "Season": 2025, "TeamA": team_a, "TeamB": team_b,
            "target": target, "stage1_prob": s1_prob, "pred": s2_prob,
        })

    if skipped > 0:
        print(f"  Skipped {skipped} games (missing snapshots)")
    return pd.DataFrame(results)


print("Predicting 2025 Men's tournament...")
m_2s_results = predict_2025_two_stage(
    m_tourney_2025, m_snap_2025, m_stats, m_seeds, m_conf,
    m_s1_model, m_s1_features, m_s1_medians,
    m_s2_model, m_s2_feat_cols, m_s2_medians,
    m_elo_df, gender="M", s2_static_cols=m_s2_static,
)

print("Predicting 2025 Women's tournament...")
w_2s_results = predict_2025_two_stage(
    w_tourney_2025, w_snap_2025, w_stats, w_seeds, w_conf,
    w_s1_model, w_s1_features, w_s1_medians,
    w_s2_model, w_s2_feat_cols, w_s2_medians,
    w_elo_df, gender="W", s2_static_cols=w_s2_static,
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

# Baselines
baseline_single = {
    "Men's":    {"brier": 0.1745, "log_loss": 0.5206, "accuracy": 0.746},
    "Women's":  {"brier": 0.1428, "log_loss": 0.4399, "accuracy": 0.821},
    "Combined": {"brier": 0.1586, "log_loss": 0.4802, "accuracy": 0.784},
}

# Two-stage no Elo baseline (from meta.json)
baseline_2s = {
    "Men's":    {"brier": 0.1528},
    "Women's":  {"brier": 0.1087},
    "Combined": {"brier": 0.1308},
}

print("\n" + "=" * 70)
print("2025 Tournament Holdout -- Model Comparison")
print("=" * 70)

print(f"\n{'Metric':<20} {'Single-Stage':>14} {'2S (no Elo)':>14} {'2S + Elo':>14} {'Delta vs 2S':>12}")
print("-" * 76)

for gender_label, elo_scores in [
    ("Men's", m_2s_scores), ("Women's", w_2s_scores), ("Combined", c_2s_scores)
]:
    bl_s = baseline_single[gender_label]
    bl_2s = baseline_2s[gender_label]
    new = elo_scores["brier"]
    delta_vs_2s = new - bl_2s["brier"]
    better = "BETTER" if delta_vs_2s < 0 else "worse"
    print(f"  {gender_label + ' Brier':<18} {bl_s['brier']:>12.4f}   {bl_2s['brier']:>12.4f}   {new:>12.4f}   {delta_vs_2s:>+10.4f} ({better})")

# ===================================================================
# STEP 13: Platt Scaling Calibration
# ===================================================================
print("\n" + "=" * 70)
print("STEP 13: Platt Scaling Calibration")
print("=" * 70)

# Build temporal CV predictions for calibration training
def build_temporal_cv_predictions(
    s1_train_df, s1_features, s1_medians,
    tourney_compact, seeds_df, conf_df, snap_dict, team_stats,
    s2_feat_cols, s2_medians,
    s1_xgb_params, s1_weights,
    s2_xgb_params, s2_weights,
    elo_df,
    gender="M",
    s2_static_cols=None,
    cv_years=None,
    s1_lr_C=1.0,
):
    """Generate OOS Stage 2 predictions for each year via temporal CV."""
    if cv_years is None:
        cv_years = sorted(tourney_compact["Season"].unique())
        cv_years = [y for y in cv_years if y in snap_dict]

    if s2_static_cols is None:
        s2_static_cols = ["sos", "win_pct"]

    all_oos = []

    for year in cv_years:
        s1_tr = s1_train_df[s1_train_df["Season"] < year]
        if len(s1_tr) < 100:
            continue

        X_s1 = s1_tr[s1_features].fillna(pd.Series(s1_medians)).fillna(0.0)
        y_s1 = s1_tr["target"]

        cv_s1_lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, C=s1_lr_C, random_state=RANDOM_STATE))
        ])
        cv_s1_lr.fit(X_s1, y_s1)
        cv_s1_xgb = XGBClassifier(**s1_xgb_params)
        cv_s1_xgb.fit(X_s1, y_s1)
        cv_s1 = EnsemblePredictor([cv_s1_lr, cv_s1_xgb], s1_weights)

        tourney_before = tourney_compact[tourney_compact["Season"] < year]
        if len(tourney_before) < 50:
            continue

        s2_train_data = build_tournament_features(
            tourney_before, seeds_df, conf_df, snap_dict, team_stats,
            cv_s1, s1_features, s1_medians,
            elo_df, gender=gender, static_cols=s2_static_cols,
        )
        if len(s2_train_data) < 30:
            continue

        cv_s2_medians = s2_train_data[s2_feat_cols].median().to_dict()
        X_s2_tr = s2_train_data[s2_feat_cols].fillna(pd.Series(cv_s2_medians)).fillna(0.0)
        y_s2_tr = s2_train_data["target"]

        cv_s2_lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, C=0.1, random_state=RANDOM_STATE))
        ])
        cv_s2_lr.fit(X_s2_tr, y_s2_tr)
        cv_s2_xgb = XGBClassifier(**s2_xgb_params)
        cv_s2_xgb.fit(X_s2_tr, y_s2_tr)
        cv_s2 = EnsemblePredictor([cv_s2_lr, cv_s2_xgb], s2_weights)

        # Predict year Y tournament games
        tourney_year = tourney_compact[tourney_compact["Season"] == year]
        year_data = build_tournament_features(
            tourney_year, seeds_df, conf_df, snap_dict, team_stats,
            cv_s1, s1_features, s1_medians,
            elo_df, gender=gender, static_cols=s2_static_cols,
        )
        if len(year_data) == 0:
            continue

        X_year = year_data[s2_feat_cols].fillna(pd.Series(cv_s2_medians)).fillna(0.0)
        preds = cv_s2.predict_proba(X_year.values)[:, 1]
        preds = np.clip(preds, 0.01, 0.99)

        oos_df = year_data[["Season", "TeamA", "TeamB", "target"]].copy()
        oos_df["raw_prob"] = preds
        all_oos.append(oos_df)

    if len(all_oos) == 0:
        return pd.DataFrame()
    return pd.concat(all_oos, ignore_index=True)


# Compute temporal CV predictions
m_cv_years = list(range(2008, 2025))
m_cv_years = [y for y in m_cv_years if y != 2020 and y in m_snap_dict]
w_cv_years = list(range(2014, 2025))
w_cv_years = [y for y in w_cv_years if y != 2020 and y in w_snap_dict]

print(f"Men's CV years: {m_cv_years}")
print(f"Women's CV years: {w_cv_years}")

print("\nMen's temporal CV...")
t0 = time.time()
m_cv_preds = build_temporal_cv_predictions(
    m_s1_train, m_s1_features, m_s1_medians,
    m_tourney_compact, m_seeds, m_conf, m_snap_dict, m_stats,
    m_s2_feat_cols, m_s2_medians,
    m_s1_xgb_params, m_s1_weights,
    m_s2_xgb_params, m_s2_weights,
    m_elo_df, gender="M", s2_static_cols=m_s2_static,
    cv_years=m_cv_years, s1_lr_C=1.0,
)
print(f"  Men's OOS predictions: {len(m_cv_preds)} games")

print("Women's temporal CV...")
w_cv_preds = build_temporal_cv_predictions(
    w_s1_train, w_s1_features, w_s1_medians,
    w_tourney_compact, w_seeds, w_conf, w_snap_dict, w_stats,
    w_s2_feat_cols, w_s2_medians,
    w_s1_xgb_params, w_s1_weights,
    w_s2_xgb_params, w_s2_weights,
    w_elo_df, gender="W", s2_static_cols=w_s2_static,
    cv_years=w_cv_years, s1_lr_C=0.5,
)
print(f"  Women's OOS predictions: {len(w_cv_preds)} games")
print(f"Temporal CV completed in {time.time()-t0:.0f}s")

# Fit Platt calibrators
def fit_platt_calibrator(cv_preds, train_end=2024):
    """Fit Platt scaling on temporal CV predictions."""
    cal_train = cv_preds[cv_preds["Season"] <= train_end]
    X_cal = cal_train["raw_prob"].values.reshape(-1, 1)
    y_cal = cal_train["target"].values

    calibrator = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    calibrator.fit(X_cal, y_cal)

    coef = calibrator.coef_[0][0]
    intercept = calibrator.intercept_[0]
    print(f"  Platt calibrator: coef={coef:.4f}, intercept={intercept:.4f}")
    print(f"  Training samples: {len(cal_train)}")
    return calibrator


print("\nFitting Platt calibrators...")
print("\nMen's:")
m_calibrator = fit_platt_calibrator(m_cv_preds)
print("\nWomen's:")
w_calibrator = fit_platt_calibrator(w_cv_preds)

# Evaluate calibration on 2025 holdout
def evaluate_calibration(results_df, calibrator, label):
    """Compare raw vs calibrated Brier scores."""
    y_true = results_df["target"].values
    raw_probs = results_df["pred"].values

    cal_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    cal_probs = np.clip(cal_probs, 0.01, 0.99)

    raw_brier = brier_score_loss(y_true, raw_probs)
    cal_brier = brier_score_loss(y_true, cal_probs)
    raw_ll = log_loss(y_true, raw_probs)
    cal_ll = log_loss(y_true, cal_probs)

    delta_brier = cal_brier - raw_brier
    delta_ll = cal_ll - raw_ll

    print(f"\n  {label} (2025 holdout, N={len(y_true)}):")
    print(f"    Raw     Brier={raw_brier:.4f}  LogLoss={raw_ll:.4f}")
    print(f"    Calibr  Brier={cal_brier:.4f}  LogLoss={cal_ll:.4f}")
    print(f"    Delta   Brier={delta_brier:+.4f}  LogLoss={delta_ll:+.4f}")
    better = delta_brier < 0
    tag = "HELPS" if better else "hurts"
    print(f"    Calibration {tag} on Brier")

    return {
        "raw_brier": raw_brier, "cal_brier": cal_brier,
        "raw_ll": raw_ll, "cal_ll": cal_ll,
        "improved": better,
    }


print("\n" + "=" * 60)
print("Platt Calibration -- 2025 Holdout Evaluation")
print("=" * 60)

m_cal_eval = evaluate_calibration(m_2s_results, m_calibrator, "Men's")
w_cal_eval = evaluate_calibration(w_2s_results, w_calibrator, "Women's")

# Combined evaluation
all_raw = np.concatenate([m_2s_results["pred"].values, w_2s_results["pred"].values])
all_cal_m = m_calibrator.predict_proba(m_2s_results["pred"].values.reshape(-1, 1))[:, 1]
all_cal_w = w_calibrator.predict_proba(w_2s_results["pred"].values.reshape(-1, 1))[:, 1]
all_cal = np.clip(np.concatenate([all_cal_m, all_cal_w]), 0.01, 0.99)
all_target = np.concatenate([m_2s_results["target"].values, w_2s_results["target"].values])

comb_raw_brier = brier_score_loss(all_target, all_raw)
comb_cal_brier = brier_score_loss(all_target, all_cal)
print(f"\n  Combined:")
print(f"    Raw     Brier={comb_raw_brier:.4f}")
print(f"    Calibr  Brier={comb_cal_brier:.4f}")
print(f"    Delta   Brier={comb_cal_brier - comb_raw_brier:+.4f}")

use_calibration = comb_cal_brier <= comb_raw_brier
decision = "USE" if use_calibration else "SKIP"
print(f"\n  Decision: {decision} Platt calibration in production")

# ===================================================================
# STEP 14: Production Retrain (if improved)
# ===================================================================
print("\n" + "=" * 70)
print("STEP 14: Production Retrain Decision")
print("=" * 70)

# Check if Elo improved over no-Elo two-stage
elo_combined_brier = c_2s_scores["brier"]
no_elo_combined_brier = baseline_2s["Combined"]["brier"]
elo_wins = elo_combined_brier < no_elo_combined_brier

print(f"\nTwo-stage (no Elo) Combined Brier: {no_elo_combined_brier:.4f}")
print(f"Two-stage + Elo Combined Brier:    {elo_combined_brier:.4f}")
print(f"Delta: {elo_combined_brier - no_elo_combined_brier:+.4f}")

if elo_wins:
    print("\n*** ELO IMPROVES THE MODEL -- Proceeding with production retrain ***")

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

    # Stage 2: retrain on ALL tournament data
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

    # Build production calibrators if calibration helps
    if use_calibration:
        print("\nFitting production Platt calibrators...")
        m_cv_years_prod = list(range(2008, 2026))
        m_cv_years_prod = [y for y in m_cv_years_prod if y != 2020 and y in m_snap_dict]
        w_cv_years_prod = list(range(2014, 2026))
        w_cv_years_prod = [y for y in w_cv_years_prod if y != 2020 and y in w_snap_dict]

        m_cv_preds_prod = build_temporal_cv_predictions(
            m_s1_train, m_s1_features, m_s1_medians,
            m_tourney_compact, m_seeds, m_conf, m_snap_dict, m_stats,
            m_s2_feat_cols, m_s2_medians,
            m_s1_xgb_params, m_s1_weights,
            m_s2_xgb_params, m_s2_weights,
            m_elo_df, gender="M", s2_static_cols=m_s2_static,
            cv_years=m_cv_years_prod, s1_lr_C=1.0,
        )
        w_cv_preds_prod = build_temporal_cv_predictions(
            w_s1_train, w_s1_features, w_s1_medians,
            w_tourney_compact, w_seeds, w_conf, w_snap_dict, w_stats,
            w_s2_feat_cols, w_s2_medians,
            w_s1_xgb_params, w_s1_weights,
            w_s2_xgb_params, w_s2_weights,
            w_elo_df, gender="W", s2_static_cols=w_s2_static,
            cv_years=w_cv_years_prod, s1_lr_C=0.5,
        )
        m_prod_calibrator = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
        m_prod_calibrator.fit(m_cv_preds_prod["raw_prob"].values.reshape(-1, 1), m_cv_preds_prod["target"].values)
        w_prod_calibrator = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
        w_prod_calibrator.fit(w_cv_preds_prod["raw_prob"].values.reshape(-1, 1), w_cv_preds_prod["target"].values)
    else:
        m_prod_calibrator = None
        w_prod_calibrator = None

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

    # Save metadata
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
        },
        "evaluation_2025": {
            "men_brier": float(m_2s_scores["brier"]),
            "women_brier": float(w_2s_scores["brier"]),
            "combined_brier": float(c_2s_scores["brier"]),
            "baseline_combined_brier": 0.1586,
            "no_elo_combined_brier": float(no_elo_combined_brier),
        },
        "calibration": {
            "method": "platt_scaling",
            "enabled": use_calibration,
            "holdout_2025": {
                "men_raw_brier": float(m_cal_eval["raw_brier"]),
                "men_cal_brier": float(m_cal_eval["cal_brier"]),
                "women_raw_brier": float(w_cal_eval["raw_brier"]),
                "women_cal_brier": float(w_cal_eval["cal_brier"]),
                "combined_raw_brier": float(comb_raw_brier),
                "combined_cal_brier": float(comb_cal_brier),
            },
        },
    }
    with open(OUTPUT_DIR / "two_stage_meta.json", "w") as f:
        json.dump(two_stage_meta, f, indent=2)
    print(f"Updated: {OUTPUT_DIR / 'two_stage_meta.json'}")

else:
    print("\n*** ELO DID NOT IMPROVE THE MODEL ***")
    print("Keeping existing models without Elo.")
    print("Documenting what happened:")
    print(f"  - Elo Combined Brier ({elo_combined_brier:.4f}) >= no-Elo ({no_elo_combined_brier:.4f})")
    print("  - The Elo signal may be redundant with existing features (win_pct, SOS, KenPom)")
    print("  - Or the Elo parameters may need further tuning for tournament prediction specifically")

# ===================================================================
# FINAL REPORT
# ===================================================================
print("\n\n" + "=" * 76)
print("=" * 76)
print("                    COMPREHENSIVE RESULTS")
print("=" * 76)

print("\n1. 2025 Holdout -- Model Comparison")
print("-" * 76)
print(f"  {'Model':<28} {'Mens Brier':>12} {'Womens Brier':>14} {'Combined Brier':>16}")
print(f"  {'-'*28} {'-'*12} {'-'*14} {'-'*16}")
print(f"  {'Single-stage (baseline)':<28} {baseline_single['Mens']['brier'] if 'Mens' in baseline_single else baseline_single['Men s']['brier'] if 'Men s' in baseline_single else 0.1745:>12.4f} {0.1428:>14.4f} {0.1586:>16.4f}")
print(f"  {'Two-stage (no Elo)':<28} {baseline_2s['Mens']['brier'] if 'Mens' in baseline_2s else baseline_2s['Men s']['brier'] if 'Men s' in baseline_2s else 0.1528:>12.4f} {0.1087:>14.4f} {0.1308:>16.4f}")
print(f"  {'Two-stage + Elo':<28} {m_2s_scores['brier']:>12.4f} {w_2s_scores['brier']:>14.4f} {c_2s_scores['brier']:>16.4f}")

print(f"\n2. Calibration (Two-stage + Elo)")
print("-" * 76)
print(f"  {'':>10} {'Raw Brier':>12} {'Platt Brier':>14} {'Delta':>10}")
print(f"  {'Mens':>10} {m_cal_eval['raw_brier']:>12.4f} {m_cal_eval['cal_brier']:>14.4f} {m_cal_eval['cal_brier']-m_cal_eval['raw_brier']:>+10.4f}")
print(f"  {'Womens':>10} {w_cal_eval['raw_brier']:>12.4f} {w_cal_eval['cal_brier']:>14.4f} {w_cal_eval['cal_brier']-w_cal_eval['raw_brier']:>+10.4f}")
print(f"  {'Combined':>10} {comb_raw_brier:>12.4f} {comb_cal_brier:>14.4f} {comb_cal_brier-comb_raw_brier:>+10.4f}")

print(f"\n3. Elo Parameters Used")
print("-" * 76)
print(f"  K-factor: {ELO_K}, Carryover: {ELO_CARRYOVER}, Margin: {ELO_MARGIN}")

print(f"\n4. Feature Selection Changes")
print("-" * 76)
print(f"  Men's Stage 1:   {len(m_s1_features)} features (was 23), elo_diff kept? {'YES' if m_elo_in_s1_m else 'NO'}")
print(f"  Women's Stage 1: {len(w_s1_features)} features (was 17), elo_diff kept? {'YES' if w_elo_in_s1_w else 'NO'}")
print(f"  Men's Stage 2:   {len(m_s2_feat_cols)} features (was 7)")
print(f"  Women's Stage 2: {len(w_s2_feat_cols)} features (was 6)")

print(f"\n5. Verdict")
print("-" * 76)
if elo_wins:
    improvement = no_elo_combined_brier - elo_combined_brier
    print(f"  Elo IMPROVED Combined Brier by {improvement:.4f}")
    print(f"  Production models UPDATED with Elo features")
else:
    print(f"  Elo did NOT improve Combined Brier")
    print(f"  Production models UNCHANGED")

print(f"\n{'='*76}")
