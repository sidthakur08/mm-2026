"""Full Two-Stage Pipeline with FiveThirtyEight Elo Integration.

Runs the complete pipeline:
1. Compute Elo ratings (K=20, Home=100, Carryover=0.75, MOV=FiveThirtyEight)
2. Build Stage 1 training data with elo_diff
3. Feature selection (MI + correlation pruning, protected features)
4. Optuna-tuned XGBoost + LR ensemble for Stage 1
5. Build Stage 2 tournament features with s2_elo_diff
6. Optuna-tuned XGBoost + LR ensemble for Stage 2
7. 2025 holdout evaluation
8. Platt scaling calibration comparison
9. Production retrain + submission generation (if improved)
"""

import sys
import os
import json
import re as _re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
N_TRIALS = 50
N_TRIALS_S2 = 50

# Fixed FiveThirtyEight Elo parameters
ELO_K = 20
ELO_HOME = 100
ELO_CARRYOVER = 0.75
ELO_MARGIN = True
ELO_REVERT = 1505

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
# STEP 1: Compute Elo Ratings (FiveThirtyEight parameters)
# ===================================================================
print("\n" + "=" * 70)
print("STEP 1: Computing Elo ratings (FiveThirtyEight parameters)")
print(f"  K={ELO_K}, Home={ELO_HOME}, Carryover={ELO_CARRYOVER}, MOV=538, Revert={ELO_REVERT}")
print("=" * 70)

# Combine regular season + tournament for Elo (all games, goes back to 1985/1998)
m_all_games = pd.concat([m_rs_compact, m_tourney_compact], ignore_index=True)
w_all_games = pd.concat([w_rs_compact, w_tourney_compact], ignore_index=True)

print(f"Men's all games:   {len(m_all_games):,} (seasons {m_all_games.Season.min()}-{m_all_games.Season.max()})")
print(f"Women's all games: {len(w_all_games):,} (seasons {w_all_games.Season.min()}-{w_all_games.Season.max()})")

t0 = time.time()
m_elo_df = compute_elo_ratings(
    m_all_games, k_factor=ELO_K,
    home_advantage=ELO_HOME,
    season_carryover=ELO_CARRYOVER,
    margin_factor=ELO_MARGIN,
)
w_elo_df = compute_elo_ratings(
    w_all_games, k_factor=ELO_K,
    home_advantage=ELO_HOME,
    season_carryover=ELO_CARRYOVER,
    margin_factor=ELO_MARGIN,
)
print(f"Elo computed in {time.time()-t0:.1f}s")
print(f"Men's Elo records:   {len(m_elo_df):,}")
print(f"Women's Elo records: {len(w_elo_df):,}")

# Quick validation: check prediction accuracy on recent seasons
for label, elo_df, all_games in [("Men's", m_elo_df, m_all_games), ("Women's", w_elo_df, w_all_games)]:
    games = all_games[["Season", "DayNum", "WTeamID", "LTeamID"]].sort_values(["Season", "DayNum"]).reset_index(drop=True)
    w_elos = elo_df.iloc[0::2]["Elo"].values
    l_elos = elo_df.iloc[1::2]["Elo"].values
    games["w_elo"] = w_elos
    games["l_elo"] = l_elos
    test_games = games[games["Season"].isin(range(2015, 2025))]
    correct = (test_games["w_elo"] > test_games["l_elo"]).sum()
    acc = correct / len(test_games) if len(test_games) > 0 else 0
    print(f"{label} Elo prediction accuracy (2015-2024): {acc:.4f}")

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

# Drop NaN rows (rolling windows not yet filled)
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


def select_features(df, feat_cols, target_col="target", mi_threshold=0.001,
                    corr_threshold=0.85, protected_features=None):
    """Select features via MI ranking + correlation pruning."""
    protected = set(protected_features or [])
    X = df[feat_cols].copy()
    y = df[target_col]
    X = X.fillna(X.median())

    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({"feature": feat_cols, "mi_score": mi}).sort_values("mi_score", ascending=False)

    selected = mi_df[mi_df["mi_score"] >= mi_threshold]["feature"].tolist()
    for pf in protected:
        if pf in feat_cols and pf not in selected:
            selected.append(pf)

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
print(f"STEP 6: Optuna tuning -- Stage 1 ({N_TRIALS} trials per gender)")
print("=" * 70)


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
m_s1_study = optuna.create_study(direction="minimize", study_name="s1_men_elo538")
m_s1_study.optimize(lambda t: optuna_xgb_objective(t, m_X_tr, m_y_tr, m_X_val, m_y_val),
                     n_trials=N_TRIALS, show_progress_bar=False)
print(f"  Best val log-loss: {m_s1_study.best_value:.4f}")
print(f"  Best val brier:    {m_s1_study.best_trial.user_attrs['val_brier']}")

print("\nTuning Women's Stage 1 XGBoost...")
w_s1_study = optuna.create_study(direction="minimize", study_name="s1_women_elo538")
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
    """Build Stage 2 training data from historical tournament games.

    Includes s2_elo_diff: end-of-regular-season Elo difference.
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
# STEP 9: Stage 2 Split + Optuna Tuning
# ===================================================================
print("\n" + "=" * 70)
print(f"STEP 9: Stage 2 split + Optuna tuning ({N_TRIALS_S2} trials per gender)")
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
m_s2_study = optuna.create_study(direction="minimize", study_name="s2_men_elo538")
m_s2_study.optimize(lambda t: optuna_s2_objective(t, m_s2_X_tr, m_s2_y_tr, m_s2_X_val, m_s2_y_val),
                     n_trials=N_TRIALS_S2, show_progress_bar=False)
print(f"  Best val log-loss: {m_s2_study.best_value:.4f}")
print(f"  Best val brier:    {m_s2_study.best_trial.user_attrs['val_brier']}")

print("\nTuning Women's Stage 2 XGBoost...")
w_s2_study = optuna.create_study(direction="minimize", study_name="s2_women_elo538")
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

# Stage 2 XGB feature importance
print("\nStage 2 XGB Feature Importance (gain):")
print("  Men's:")
m_s2_imp = dict(zip(m_s2_feat_cols, m_s2_xgb.feature_importances_))
m_s2_imp_sorted = sorted(m_s2_imp.items(), key=lambda x: x[1], reverse=True)
for feat, imp in m_s2_imp_sorted:
    print(f"    {feat}: {imp:.4f}")

print("  Women's:")
w_s2_imp = dict(zip(w_s2_feat_cols, w_s2_xgb.feature_importances_))
w_s2_imp_sorted = sorted(w_s2_imp.items(), key=lambda x: x[1], reverse=True)
for feat, imp in w_s2_imp_sorted:
    print(f"    {feat}: {imp:.4f}")

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

print(f"\nMen's 2025 holdout:   Brier={m_2s_scores['brier']:.4f}, Games={m_2s_scores['n']}")
print(f"Women's 2025 holdout: Brier={w_2s_scores['brier']:.4f}, Games={w_2s_scores['n']}")
print(f"Combined 2025 holdout: Brier={c_2s_scores['brier']:.4f}, Games={c_2s_scores['n']}")

# ===================================================================
# STEP 12: Platt Scaling Calibration
# ===================================================================
print("\n" + "=" * 70)
print("STEP 12: Platt Scaling Calibration")
print("=" * 70)

# Fit Platt calibrator on val set (2023-2024 tournament predictions)
# We fit on the val predictions from Stage 2
print("Fitting Platt calibrators on Stage 2 val set predictions...")

# Get val predictions from Stage 2
m_s2_val_preds = m_s2_model.predict_proba(m_s2_X_val.values)[:, 1]
w_s2_val_preds = w_s2_model.predict_proba(w_s2_X_val.values)[:, 1]

m_calibrator = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
m_calibrator.fit(m_s2_val_preds.reshape(-1, 1), m_s2_y_val.values)
print(f"  Men's Platt: coef={m_calibrator.coef_[0][0]:.4f}, intercept={m_calibrator.intercept_[0]:.4f}, samples={len(m_s2_y_val)}")

w_calibrator = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
w_calibrator.fit(w_s2_val_preds.reshape(-1, 1), w_s2_y_val.values)
print(f"  Women's Platt: coef={w_calibrator.coef_[0][0]:.4f}, intercept={w_calibrator.intercept_[0]:.4f}, samples={len(w_s2_y_val)}")

# Evaluate calibration on 2025 holdout
print("\nEvaluating calibration on 2025 holdout...")

m_raw_probs = m_2s_results["pred"].values
m_cal_probs = np.clip(m_calibrator.predict_proba(m_raw_probs.reshape(-1, 1))[:, 1], 0.01, 0.99)
m_raw_brier = brier_score_loss(m_2s_results["target"].values, m_raw_probs)
m_cal_brier = brier_score_loss(m_2s_results["target"].values, m_cal_probs)

w_raw_probs = w_2s_results["pred"].values
w_cal_probs = np.clip(w_calibrator.predict_proba(w_raw_probs.reshape(-1, 1))[:, 1], 0.01, 0.99)
w_raw_brier = brier_score_loss(w_2s_results["target"].values, w_raw_probs)
w_cal_brier = brier_score_loss(w_2s_results["target"].values, w_cal_probs)

all_target = np.concatenate([m_2s_results["target"].values, w_2s_results["target"].values])
all_raw = np.concatenate([m_raw_probs, w_raw_probs])
all_cal = np.concatenate([m_cal_probs, w_cal_probs])
comb_raw_brier = brier_score_loss(all_target, all_raw)
comb_cal_brier = brier_score_loss(all_target, all_cal)

use_calibration = comb_cal_brier <= comb_raw_brier
decision = "USE" if use_calibration else "SKIP"
print(f"  Men's:    Raw={m_raw_brier:.4f}  Platt={m_cal_brier:.4f}  Delta={m_cal_brier - m_raw_brier:+.4f}")
print(f"  Women's:  Raw={w_raw_brier:.4f}  Platt={w_cal_brier:.4f}  Delta={w_cal_brier - w_raw_brier:+.4f}")
print(f"  Combined: Raw={comb_raw_brier:.4f}  Platt={comb_cal_brier:.4f}  Delta={comb_cal_brier - comb_raw_brier:+.4f}")
print(f"  Decision: {decision} Platt calibration in production")

# ===================================================================
# STEP 13: Production Retrain + Submission Generation
# ===================================================================
print("\n" + "=" * 70)
print("STEP 13: Production Retrain Decision")
print("=" * 70)

# Check if Elo improved over no-Elo two-stage
no_elo_combined_brier = 0.1308
elo_combined_brier = c_2s_scores["brier"]
elo_wins = elo_combined_brier < no_elo_combined_brier

print(f"\nTwo-stage (no Elo) Combined Brier: {no_elo_combined_brier:.4f}")
print(f"Two-stage + 538 Elo Combined Brier: {elo_combined_brier:.4f}")
print(f"Delta: {elo_combined_brier - no_elo_combined_brier:+.4f}")

if elo_wins:
    print("\n*** ELO IMPROVES THE MODEL -- Proceeding with production retrain ***")

    # Stage 1: full retrain on all regular-season data (2003-2025 M, 2010-2025 W)
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

    # Calibrators for production
    if use_calibration:
        m_prod_calibrator = m_calibrator
        w_prod_calibrator = w_calibrator
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
            "home_advantage": ELO_HOME,
            "season_carryover": ELO_CARRYOVER,
            "margin_factor": ELO_MARGIN,
            "revert_target": ELO_REVERT,
            "formula": "FiveThirtyEight",
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
                "men_raw_brier": float(m_raw_brier),
                "men_cal_brier": float(m_cal_brier),
                "women_raw_brier": float(w_raw_brier),
                "women_cal_brier": float(w_cal_brier),
                "combined_raw_brier": float(comb_raw_brier),
                "combined_cal_brier": float(comb_cal_brier),
            },
        },
    }
    with open(OUTPUT_DIR / "two_stage_meta.json", "w") as f:
        json.dump(two_stage_meta, f, indent=2)
    print(f"Updated: {OUTPUT_DIR / 'two_stage_meta.json'}")

    # ===============================================================
    # Generate Submission Files
    # ===============================================================
    print("\n" + "=" * 70)
    print("Generating Submission Files")
    print("=" * 70)

    sub_stage1_raw = load_sample_submission(stage=1)
    sub_stage2_raw = load_sample_submission(stage=2)

    def parse_submission(sub_df):
        parsed = sub_df.copy()
        id_parts = parsed["ID"].str.split("_", expand=True).astype(int)
        parsed["Season"] = id_parts[0]
        parsed["TeamA"] = id_parts[1]
        parsed["TeamB"] = id_parts[2]
        parsed["Gender"] = np.where(parsed["TeamA"].between(1000, 1999), "M", "W")
        return parsed

    sub_stage1 = parse_submission(sub_stage1_raw)
    sub_stage2 = parse_submission(sub_stage2_raw)

    print(f"Stage 1: {len(sub_stage1):,} rows, Seasons {sorted(sub_stage1.Season.unique())}")
    print(f"Stage 2: {len(sub_stage2):,} rows, Seasons {sorted(sub_stage2.Season.unique())}")

    # Compute snapshots for all needed seasons
    needed_seasons = sorted(set(sub_stage1["Season"].unique()) | set(sub_stage2["Season"].unique()))
    print(f"Needed seasons: {needed_seasons}")

    m_sub_snaps = dict(m_snap_dict)
    w_sub_snaps = dict(w_snap_dict)
    for season in needed_seasons:
        if season not in m_sub_snaps:
            print(f"  Computing men's snapshot for {season}...")
            m_sub_snaps[season] = get_team_rolling_snapshot(m_rs_detailed, m_rs_compact, season=season)
        if season not in w_sub_snaps:
            print(f"  Computing women's snapshot for {season}...")
            w_sub_snaps[season] = get_team_rolling_snapshot(w_rs_detailed, w_rs_compact, season=season)

    # Compute Elo snapshots for all needed seasons
    m_elo_snaps = {}
    w_elo_snaps = {}
    for season in needed_seasons:
        rs_elo = m_elo_df[(m_elo_df["Season"] == season) & (m_elo_df["DayNum"] < 134)]
        if len(rs_elo) > 0:
            m_elo_snaps[season] = rs_elo.sort_values("DayNum").groupby("TeamID")["Elo"].last()
        rs_elo = w_elo_df[(w_elo_df["Season"] == season) & (w_elo_df["DayNum"] < 134)]
        if len(rs_elo) > 0:
            w_elo_snaps[season] = rs_elo.sort_values("DayNum").groupby("TeamID")["Elo"].last()

    def generate_two_stage_predictions(
        sub_df, m_snaps, w_snaps, m_stats_df, w_stats_df,
        m_model, w_model,
        m_seeds_df, w_seeds_df, m_conf_df, w_conf_df,
        m_s2_static_cols, w_s2_static_cols,
        m_elo_snaps_dict, w_elo_snaps_dict,
    ):
        """Generate predictions using TwoStagePredictor for submission."""
        result = sub_df.copy()
        result["Pred"] = 0.5

        m_seed_lookup, w_seed_lookup = {}, {}
        for _, row in m_seeds_df.iterrows():
            sn = parse_seed_number(row["Seed"])
            m_seed_lookup[(row["Season"], row["TeamID"])] = sn
        for _, row in w_seeds_df.iterrows():
            sn = parse_seed_number(row["Seed"])
            w_seed_lookup[(row["Season"], row["TeamID"])] = sn

        m_conf_lookup, w_conf_lookup = {}, {}
        for _, row in m_conf_df.iterrows():
            m_conf_lookup[(row["Season"], row["TeamID"])] = row["ConfAbbrev"]
        for _, row in w_conf_df.iterrows():
            w_conf_lookup[(row["Season"], row["TeamID"])] = row["ConfAbbrev"]

        for gender, snaps, stats, model, seed_lookup, conf_lookup, s2_static, elo_snaps_dict in [
            ("M", m_snaps, m_stats_df, m_model, m_seed_lookup, m_conf_lookup, m_s2_static_cols, m_elo_snaps_dict),
            ("W", w_snaps, w_stats_df, w_model, w_seed_lookup, w_conf_lookup, w_s2_static_cols, w_elo_snaps_dict),
        ]:
            mask = result["Gender"] == gender
            if mask.sum() == 0:
                continue

            gender_sub = result.loc[mask].copy()
            preds_out = pd.Series(0.5, index=gender_sub.index)

            for season in gender_sub["Season"].unique():
                season = int(season)
                season_mask = gender_sub["Season"] == season
                season_sub = gender_sub[season_mask]

                if season not in snaps:
                    print(f"  [{gender}] Season {season}: no snapshot, using 0.5")
                    continue

                snap = snaps[season]
                roll_cols = [c for c in snap.columns if "_roll" in c]

                try:
                    static_df = stats.loc[season].copy()
                    if "kenpom_rank" in static_df.columns:
                        static_df["kenpom_rank"] = static_df["kenpom_rank"].fillna(366)
                except KeyError:
                    static_df = pd.DataFrame()

                elo_snap = elo_snaps_dict.get(season, pd.Series(dtype=float))

                s1_features_list = []
                s2_features_list = []
                valid_indices = []
                n_missing = 0

                for idx, row in season_sub.iterrows():
                    team_a, team_b = int(row["TeamA"]), int(row["TeamB"])
                    if team_a not in snap.index or team_b not in snap.index:
                        n_missing += 1
                        continue

                    snap_a, snap_b = snap.loc[team_a], snap.loc[team_b]

                    # Stage 1 features
                    s1_feats = {}
                    for col in roll_cols:
                        s1_feats[col] = snap_a[col] - snap_b[col]
                    for col in s2_static:
                        key = f"static_{col}"
                        if len(static_df) > 0 and team_a in static_df.index and team_b in static_df.index:
                            if col in static_df.columns:
                                s1_feats[key] = static_df.loc[team_a, col] - static_df.loc[team_b, col]
                            else:
                                s1_feats[key] = 0.0
                        else:
                            s1_feats[key] = 0.0

                    # Elo diff for Stage 1
                    elo_a = elo_snap.get(team_a, 1500.0) if len(elo_snap) > 0 else 1500.0
                    elo_b = elo_snap.get(team_b, 1500.0) if len(elo_snap) > 0 else 1500.0
                    s1_feats["elo_diff"] = elo_a - elo_b

                    # Stage 2 extra features
                    seed_a = seed_lookup.get((season, team_a))
                    seed_b = seed_lookup.get((season, team_b))
                    s2_feats = {
                        "seed_diff": (seed_a - seed_b) if (seed_a is not None and seed_b is not None) else 0,
                        "conf_match": 1 if conf_lookup.get((season, team_a)) == conf_lookup.get((season, team_b)) else 0,
                    }
                    for col in s2_static:
                        key = f"s2_{col}_diff"
                        if len(static_df) > 0 and team_a in static_df.index and team_b in static_df.index:
                            if col in static_df.columns:
                                s2_feats[key] = static_df.loc[team_a, col] - static_df.loc[team_b, col]
                            else:
                                s2_feats[key] = 0.0
                        else:
                            s2_feats[key] = 0.0
                    # Elo diff for Stage 2
                    s2_feats["s2_elo_diff"] = elo_a - elo_b

                    s1_features_list.append(s1_feats)
                    s2_features_list.append(s2_feats)
                    valid_indices.append(idx)

                if not valid_indices:
                    print(f"  [{gender}] Season {season}: {n_missing} missing, 0 predicted")
                    continue

                # Build Stage 1 feature matrix
                s1_df = pd.DataFrame(s1_features_list, index=valid_indices)
                for fc in model.stage1_features:
                    if fc not in s1_df.columns:
                        s1_df[fc] = model.stage1_medians.get(fc, 0.0)
                if model.stage1_medians:
                    s1_df = s1_df[model.stage1_features].fillna(pd.Series(model.stage1_medians)).fillna(0.0)
                else:
                    s1_df = s1_df[model.stage1_features].fillna(0.0)

                # Build Stage 2 extra features
                s2_df = pd.DataFrame(s2_features_list, index=valid_indices)

                # Call two-stage predict (includes calibration + clipping)
                preds = model.predict_proba(s1_df.values, s2_df)
                preds_out.loc[valid_indices] = preds

                msg = f"  [{gender}] Season {season}: {len(valid_indices)} predicted"
                if n_missing > 0:
                    msg += f", {n_missing} missing -> 0.5"
                print(msg)

            result.loc[mask, "Pred"] = preds_out.values

        return result

    print("\nGenerating Stage 1 predictions...")
    sub_stage1 = generate_two_stage_predictions(
        sub_stage1, m_sub_snaps, w_sub_snaps, m_stats, w_stats,
        m_two_stage, w_two_stage,
        m_seeds, w_seeds, m_conf, w_conf,
        m_s2_static, w_s2_static,
        m_elo_snaps, w_elo_snaps,
    )

    print("\nGenerating Stage 2 predictions...")
    sub_stage2 = generate_two_stage_predictions(
        sub_stage2, m_sub_snaps, w_sub_snaps, m_stats, w_stats,
        m_two_stage, w_two_stage,
        m_seeds, w_seeds, m_conf, w_conf,
        m_s2_static, w_s2_static,
        m_elo_snaps, w_elo_snaps,
    )

    # Clip and save
    sub_stage1["Pred"] = sub_stage1["Pred"].clip(0.01, 0.99)
    sub_stage2["Pred"] = sub_stage2["Pred"].clip(0.01, 0.99)

    sub_stage1[["ID", "Pred"]].to_csv(OUTPUT_DIR / "submission_stage1.csv", index=False)
    sub_stage2[["ID", "Pred"]].to_csv(OUTPUT_DIR / "submission_stage2.csv", index=False)

    s1_size = (OUTPUT_DIR / "submission_stage1.csv").stat().st_size / 1024
    s2_size = (OUTPUT_DIR / "submission_stage2.csv").stat().st_size / 1024
    print(f"\nSaved: submission_stage1.csv ({s1_size:.1f} KB, {len(sub_stage1):,} rows)")
    print(f"Saved: submission_stage2.csv ({s2_size:.1f} KB, {len(sub_stage2):,} rows)")

    for label, sub in [("Stage 1", sub_stage1), ("Stage 2", sub_stage2)]:
        preds = sub["Pred"]
        print(f"  {label}: mean={preds.mean():.4f}, std={preds.std():.4f}, min={preds.min():.4f}, max={preds.max():.4f}")

else:
    print("\n*** ELO DID NOT IMPROVE THE MODEL ***")
    print("Keeping existing models without Elo.")

# ===================================================================
# FINAL COMPREHENSIVE RESULTS
# ===================================================================
print("\n\n")
print("=" * 76)
print("=== COMPREHENSIVE RESULTS (FiveThirtyEight Elo) ===")
print("=" * 76)

print(f"""
1. 2025 Holdout -- Model Comparison
   | Model                    | Men's Brier | Women's Brier | Combined Brier |
   |--------------------------|-------------|---------------|----------------|
   | Single-stage (baseline)  | 0.1745      | 0.1428        | 0.1586         |
   | Two-stage (no Elo)       | 0.1528      | 0.1087        | 0.1308         |
   | Two-stage + 538 Elo      | {m_2s_scores['brier']:.4f}      | {w_2s_scores['brier']:.4f}        | {c_2s_scores['brier']:.4f}         |

2. Calibration (Two-stage + 538 Elo)
   |          | Raw Brier | Platt Brier | Delta   |
   |----------|-----------|-------------|---------|
   | Men's    | {m_raw_brier:.4f}    | {m_cal_brier:.4f}      | {m_cal_brier - m_raw_brier:+.4f}  |
   | Women's  | {w_raw_brier:.4f}    | {w_cal_brier:.4f}      | {w_cal_brier - w_raw_brier:+.4f}  |
   | Combined | {comb_raw_brier:.4f}    | {comb_cal_brier:.4f}      | {comb_cal_brier - comb_raw_brier:+.4f}  |

3. Elo Parameters
   K={ELO_K}, Home={ELO_HOME}, Carryover={ELO_CARRYOVER}, MOV=FiveThirtyEight, Revert={ELO_REVERT}

4. Feature Selection
   Men's S1: {len(m_s1_features)} features, elo_diff kept: {'Y' if m_elo_in_s1_m else 'N'}
   Women's S1: {len(w_s1_features)} features, elo_diff kept: {'Y' if w_elo_in_s1_w else 'N'}
   Men's S2: {len(m_s2_feat_cols)} features ({', '.join(m_s2_feat_cols)})
   Women's S2: {len(w_s2_feat_cols)} features ({', '.join(w_s2_feat_cols)})

5. Stage 2 Feature Importance (XGB gain)
   Men's: {', '.join(f'{f}={v:.3f}' for f, v in m_s2_imp_sorted)}
   Women's: {', '.join(f'{f}={v:.3f}' for f, v in w_s2_imp_sorted)}""")

if elo_wins:
    print("""
6. Production models saved -- new submission generated""")

print(f"\n{'=' * 76}")
print("Pipeline complete.")
total_time = time.time()
