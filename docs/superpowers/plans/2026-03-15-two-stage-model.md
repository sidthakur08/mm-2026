# Two-Stage Tournament Prediction Model — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-stage model where Stage 1 captures regular-season team quality and Stage 2 learns how that quality translates to tournament performance, using tournament-specific features (seeds, conference strength, seed matchup dynamics).

**Architecture:** Stage 1 is a regular-season-only model (LR+XGB ensemble) trained on all regular-season games with rolling windows, producing calibrated team-quality probabilities. Stage 2 is a tournament-specific model trained on ~2,500 men's / ~1,650 women's historical tournament games, using Stage 1 probabilities as the primary input alongside tournament-specific features (seed diff, conference matchup, SOS gap, KenPom). The final output is Stage 2's probability, clipped to [0.01, 0.99].

**Tech Stack:** Python 3.14, pandas, scikit-learn, XGBoost, Optuna, numpy, matplotlib

---

## Chunk 1: Stage 1 — Regular-Season-Only Model

### Overview

Stage 1 answers: "Based on regular season play alone, what's the probability TeamA beats TeamB?" This model is trained exclusively on regular-season games (no tournament data), using the existing rolling window features. Its predictions become the primary input feature for Stage 2.

Key difference from the current model: Stage 1 trains on regular-season games ONLY, not all games. This is intentional — it isolates regular-season signal so Stage 2 can learn the tournament-specific adjustment.

### File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/two_stage.py` | Create | TwoStagePredictor class, Stage 1 builder, Stage 2 builder, prediction pipeline |
| `notebooks/06_two_stage_model.ipynb` | Create | End-to-end two-stage pipeline: feature engineering, Stage 1 training, Stage 2 feature construction, Stage 2 training, evaluation, comparison vs current model |
| `src/features.py` | Read only | Reuse existing `build_rolling_training_data`, `get_team_rolling_snapshot`, `build_team_season_stats` |
| `src/model.py` | Read only | Reuse existing `EnsemblePredictor` |
| `src/data_loader.py` | Read only | Reuse existing loaders |

---

### Task 1: Create `src/two_stage.py` — TwoStagePredictor class

**Files:**
- Create: `src/two_stage.py`

- [ ] **Step 1: Write the TwoStagePredictor class skeleton**

```python
"""Two-stage tournament prediction model.

Stage 1: Regular-season model — trained on regular-season games only,
         produces team-quality probabilities.
Stage 2: Tournament model — trained on historical tournament games,
         uses Stage 1 probabilities + tournament-specific features
         (seed diff, conference, SOS gap) to produce final predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.model import EnsemblePredictor


class TwoStagePredictor:
    """Two-stage tournament prediction model.

    Parameters
    ----------
    stage1_model : EnsemblePredictor
        Fitted Stage 1 model (regular-season trained).
    stage2_model : fitted sklearn-compatible classifier
        Fitted Stage 2 model (tournament trained).
    stage1_features : list[str]
        Feature column names for Stage 1.
    stage2_features : list[str]
        Feature column names for Stage 2 (includes 'stage1_prob').
    stage1_medians : dict
        Median fill values for Stage 1 features.
    stage2_medians : dict
        Median fill values for Stage 2 features.
    clip_range : tuple[float, float]
        Min/max probability bounds for final output.
    """

    def __init__(
        self,
        stage1_model,
        stage2_model,
        stage1_features: list[str],
        stage2_features: list[str],
        stage1_medians: dict | None = None,
        stage2_medians: dict | None = None,
        clip_range: tuple[float, float] = (0.01, 0.99),
    ):
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.stage1_features = stage1_features
        self.stage2_features = stage2_features
        self.stage1_medians = stage1_medians or {}
        self.stage2_medians = stage2_medians or {}
        self.clip_range = clip_range

    def predict_stage1(self, X: np.ndarray) -> np.ndarray:
        """Get Stage 1 probabilities (regular-season model)."""
        return self.stage1_model.predict_proba(X)[:, 1]

    def predict_proba(self, X_stage1: np.ndarray, X_stage2_extra: pd.DataFrame) -> np.ndarray:
        """Full two-stage prediction.

        Parameters
        ----------
        X_stage1 : array-like of shape (n_samples, n_stage1_features)
            Stage 1 features (rolling window + static diffs).
        X_stage2_extra : DataFrame of shape (n_samples, n_extra_features)
            Tournament-specific features (seed_diff, conf_match, etc.).
            Will be combined with Stage 1 probabilities.

        Returns
        -------
        probs : array of shape (n_samples,)
            Final clipped probabilities.
        """
        # Get Stage 1 probabilities
        s1_probs = self.predict_stage1(X_stage1)

        # Build Stage 2 feature matrix
        X_s2 = X_stage2_extra.copy()
        X_s2["stage1_prob"] = s1_probs

        # Ensure correct column order and fill NaN
        for col in self.stage2_features:
            if col not in X_s2.columns:
                X_s2[col] = self.stage2_medians.get(col, 0.0)
        X_s2 = X_s2[self.stage2_features]

        if self.stage2_medians:
            for col in self.stage2_features:
                if col in self.stage2_medians and X_s2[col].isna().any():
                    X_s2[col] = X_s2[col].fillna(self.stage2_medians[col])
        X_s2 = X_s2.fillna(0.0)

        # Stage 2 prediction
        raw_probs = self.stage2_model.predict_proba(X_s2.values)[:, 1]

        # Clip to [0.01, 0.99]
        return np.clip(raw_probs, self.clip_range[0], self.clip_range[1])

    def __repr__(self):
        return (
            f"TwoStagePredictor(stage1={self.stage1_model!r}, "
            f"stage2={type(self.stage2_model).__name__}, "
            f"clip={self.clip_range})"
        )
```

- [ ] **Step 2: Verify the module imports correctly**

Run: `cd /Users/sidthakur/Github/mm-2026 && python3 -c "from src.two_stage import TwoStagePredictor; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/two_stage.py
git commit -m "feat: add TwoStagePredictor class for two-stage tournament model"
```

---

### Task 2: Create notebook 06 — Setup & Stage 1 Training (regular-season only)

**Files:**
- Create: `notebooks/06_two_stage_model.ipynb`

The notebook will be built incrementally across Tasks 2-6. Each task adds cells to the same notebook.

- [ ] **Step 1: Create notebook with setup and data loading cells**

Cells to create:

**Cell 1 (markdown):**
```
# Notebook 06 — Two-Stage Tournament Prediction Model

**Philosophy:** Teams come into the tournament with their regular-season identity
(form, efficiency, shooting, etc.), then tournament-specific factors (seed matchups,
conference strength, bracket position) determine how that identity translates to
March outcomes.

**Stage 1:** Regular-season model — trained only on regular-season games with rolling
windows. Produces team-quality probabilities.

**Stage 2:** Tournament model — trained on historical tournament games (~2,500 men's,
~1,650 women's). Uses Stage 1 probabilities + seed diff + conference + SOS gap to
learn tournament-specific adjustments.
```

**Cell 2 (code) — Setup:**
```python
import sys, os, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(os.getcwd()).parent
if str(PROJECT_ROOT) not in sys.path:
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
    get_team_rolling_snapshot,
)
from src.model import EnsemblePredictor
from src.two_stage import TwoStagePredictor

OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42

plt.rcParams.update({"figure.figsize": (12, 6), "figure.dpi": 100})
sns.set_style("whitegrid")
print(f"Project root: {PROJECT_ROOT}")
```

**Cell 3 (code) — Load data:**
```python
# Load all data sources
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

print(f"Men's regular season:  {m_rs_detailed.shape}")
print(f"Men's tournament:      {m_tourney_detailed.shape}")
print(f"Women's regular season: {w_rs_detailed.shape}")
print(f"Women's tournament:     {w_tourney_detailed.shape}")
print(f"Conferences:            {m_conf.ConfAbbrev.nunique()} unique")
```

**Cell 4 (code) — Build static season-level features (reuse from notebook 02):**
```python
# Build static stats from regular-season data only
# (same process as notebook 02, but we keep it self-contained here)

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
```

**Cell 5 (markdown):**
```
## Stage 1: Regular-Season Model

Train on regular-season games ONLY. This model captures team quality from how teams
play during the regular season — rolling window efficiency, shooting, turnovers, etc.

The key difference from the current (single-stage) model: no tournament games in training.
Tournament games are reserved entirely for Stage 2.
```

**Cell 6 (code) — Build Stage 1 training data (regular-season only):**
```python
%%time
# Stage 1: Build rolling window training data from REGULAR SEASON ONLY
# This is intentionally different from notebook 02 which uses all games

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

# Drop NaN rows (teams without enough games for rolling windows)
meta_cols = ["target", "Season", "DayNum", "TeamA", "TeamB", "is_tourney"]
m_feat_all = [c for c in m_s1_train.columns if c not in meta_cols]
w_feat_all = [c for c in w_s1_train.columns if c not in meta_cols]
m_s1_train = m_s1_train.dropna(subset=m_feat_all).reset_index(drop=True)
w_s1_train = w_s1_train.dropna(subset=w_feat_all).reset_index(drop=True)

print(f"\nStage 1 training data (regular season only):")
print(f"  Men's:   {len(m_s1_train):,} rows (seasons {m_s1_train.Season.min()}-{m_s1_train.Season.max()})")
print(f"  Women's: {len(w_s1_train):,} rows (seasons {w_s1_train.Season.min()}-{w_s1_train.Season.max()})")
```

- [ ] **Step 2: Add Stage 1 feature selection and model training cells**

**Cell 7 (code) — Feature selection (reuse MI + correlation pruning from notebook 03):**
```python
# Feature selection via mutual information + correlation pruning
# Same methodology as notebook 03, applied to regular-season data

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

    # Correlation pruning: among highly correlated pairs, keep the one with higher MI
    corr_matrix = X[selected].corr().abs()
    to_drop = set()
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            fi, fj = selected[i], selected[j]
            if fi in to_drop or fj in to_drop:
                continue
            if corr_matrix.loc[fi, fj] > corr_threshold:
                # Don't drop protected features
                if fi in protected and fj in protected:
                    continue
                if fi in protected:
                    to_drop.add(fj)
                elif fj in protected:
                    to_drop.add(fi)
                else:
                    # Drop the one with lower MI
                    mi_i = mi_df[mi_df["feature"] == fi]["mi_score"].values[0]
                    mi_j = mi_df[mi_df["feature"] == fj]["mi_score"].values[0]
                    to_drop.add(fj if mi_i >= mi_j else fi)

    final = [f for f in selected if f not in to_drop]
    print(f"  MI filter: {len(feat_cols)} -> {len(selected)}")
    print(f"  Corr pruning: {len(selected)} -> {len(final)} (dropped {len(to_drop)})")
    return final, mi_df

# Men's Stage 1 feature selection
raw_feat_cols = [c for c in m_s1_train.columns if "_roll" in c or c.startswith("static_")]
print("Men's Stage 1 feature selection:")
m_s1_features, m_mi = select_features(
    m_s1_train, raw_feat_cols,
    protected_features=["static_kenpom_rank", "static_win_pct"],
)
print(f"  Final: {len(m_s1_features)} features")

# Women's Stage 1 feature selection
w_raw_feat_cols = [c for c in w_s1_train.columns if "_roll" in c or c.startswith("static_")]
print("\nWomen's Stage 1 feature selection:")
w_s1_features, w_mi = select_features(
    w_s1_train, w_raw_feat_cols,
    protected_features=["static_win_pct"],
)
print(f"  Final: {len(w_s1_features)} features")
```

**Cell 8 (code) — Train/val/test split + Stage 1 LR + Optuna-tuned XGB:**
```python
# Time-based split (same as notebook 03)
# Train: 2003-2022 (M) / 2010-2022 (W)
# Val: 2023-2024
# Test: 2025

def split_by_season(df, feat_cols, train_end=2022, val_end=2024, min_season=None):
    """Split into train/val/test by season."""
    if min_season:
        df = df[df["Season"] >= min_season]
    train = df[df["Season"] <= train_end]
    val = df[(df["Season"] > train_end) & (df["Season"] <= val_end)]
    test = df[df["Season"] > val_end]

    X_train = train[feat_cols].fillna(train[feat_cols].median())
    y_train = train["target"]
    X_val = val[feat_cols].fillna(train[feat_cols].median())
    y_val = val["target"]
    X_test = test[feat_cols].fillna(train[feat_cols].median())
    y_test = test["target"]
    medians = train[feat_cols].median().to_dict()

    return X_train, y_train, X_val, y_val, X_test, y_test, medians

m_X_tr, m_y_tr, m_X_val, m_y_val, m_X_te, m_y_te, m_s1_medians = split_by_season(
    m_s1_train, m_s1_features, min_season=2003
)
w_X_tr, w_y_tr, w_X_val, w_y_val, w_X_te, w_y_te, w_s1_medians = split_by_season(
    w_s1_train, w_s1_features, min_season=2010
)

print(f"Men's Stage 1:   train={len(m_X_tr):,}  val={len(m_X_val):,}  test={len(m_X_te):,}")
print(f"Women's Stage 1: train={len(w_X_tr):,}  val={len(w_X_val):,}  test={len(w_X_te):,}")
```

**Cell 9 (code) — Optuna tuning for Stage 1 XGBoost:**
```python
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

print("Tuning Men's Stage 1 XGBoost...")
m_s1_study = optuna.create_study(direction="minimize", study_name="s1_men")
m_s1_study.optimize(lambda t: optuna_xgb_objective(t, m_X_tr, m_y_tr, m_X_val, m_y_val),
                     n_trials=N_TRIALS, show_progress_bar=True)
print(f"  Best val log-loss: {m_s1_study.best_value:.4f}")
print(f"  Best val brier:    {m_s1_study.best_trial.user_attrs['val_brier']}")

print("\nTuning Women's Stage 1 XGBoost...")
w_s1_study = optuna.create_study(direction="minimize", study_name="s1_women")
w_s1_study.optimize(lambda t: optuna_xgb_objective(t, w_X_tr, w_y_tr, w_X_val, w_y_val),
                     n_trials=N_TRIALS, show_progress_bar=True)
print(f"  Best val log-loss: {w_s1_study.best_value:.4f}")
print(f"  Best val brier:    {w_s1_study.best_trial.user_attrs['val_brier']}")
```

**Cell 10 (code) — Build Stage 1 ensemble with weight optimization:**
```python
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

# Men's Stage 1 ensemble
m_s1_lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE))])
m_s1_lr.fit(m_X_tr, m_y_tr)

m_s1_xgb_params = m_s1_study.best_params.copy()
m_s1_xgb_params.update({"random_state": RANDOM_STATE, "eval_metric": "logloss", "verbosity": 0})
m_s1_xgb = XGBClassifier(**m_s1_xgb_params)
m_s1_xgb.fit(m_X_tr, m_y_tr)

m_s1_weights, m_s1_val_ll = optimize_ensemble_weights([m_s1_lr, m_s1_xgb], m_X_val, m_y_val)
m_s1_model = EnsemblePredictor([m_s1_lr, m_s1_xgb], m_s1_weights)

# Women's Stage 1 ensemble
w_s1_lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=0.5, random_state=RANDOM_STATE))])
w_s1_lr.fit(w_X_tr, w_y_tr)

w_s1_xgb_params = w_s1_study.best_params.copy()
w_s1_xgb_params.update({"random_state": RANDOM_STATE, "eval_metric": "logloss", "verbosity": 0})
w_s1_xgb = XGBClassifier(**w_s1_xgb_params)
w_s1_xgb.fit(w_X_tr, w_y_tr)

w_s1_weights, w_s1_val_ll = optimize_ensemble_weights([w_s1_lr, w_s1_xgb], w_X_val, w_y_val)
w_s1_model = EnsemblePredictor([w_s1_lr, w_s1_xgb], w_s1_weights)

# Evaluate on test set
for label, model, X_te, y_te, weights in [
    ("Men's", m_s1_model, m_X_te, m_y_te, m_s1_weights),
    ("Women's", w_s1_model, w_X_te, w_y_te, w_s1_weights),
]:
    preds = model.predict_proba(X_te)[:, 1]
    bs = brier_score_loss(y_te, preds)
    ll = log_loss(y_te, preds)
    acc = accuracy_score(y_te, (preds >= 0.5).astype(int))
    print(f"{label} Stage 1: weights={weights}, test brier={bs:.4f}, test ll={ll:.4f}, acc={acc:.3f}")
```

- [ ] **Step 3: Run notebook cells 1-10 and verify Stage 1 trains successfully**

Run: `cd /Users/sidthakur/Github/mm-2026 && jupyter execute notebooks/06_two_stage_model.ipynb --timeout=1800`
Expected: Stage 1 models train, metrics printed for both genders

- [ ] **Step 4: Commit**

```bash
git add notebooks/06_two_stage_model.ipynb src/two_stage.py
git commit -m "feat: add two-stage model — Stage 1 regular-season ensemble"
```

---

## Chunk 2: Stage 2 — Tournament Model

### Overview

Stage 2 answers: "Given what we know about these teams from regular season (Stage 1 prob) AND their tournament-specific context (seeds, conference, SOS), what happens when they meet in March?"

This model trains exclusively on historical tournament games (~2,500 men's, ~1,650 women's). The key features are:
- **stage1_prob**: Stage 1's probability (the primary signal)
- **seed_diff**: Seed number difference (TeamA seed - TeamB seed)
- **static_sos_diff**: Strength of schedule gap
- **static_kenpom_rank_diff**: KenPom ranking gap (men's only)
- **conf_match**: Binary — are both teams from the same conference?
- **static_win_pct_diff**: Win percentage gap

The training data is small (~2,500 games for men) so Stage 2 must be heavily regularized.

---

### Task 3: Build Stage 2 tournament training data

**Files:**
- Modify: `notebooks/06_two_stage_model.ipynb` (add cells)

- [ ] **Step 1: Add cells for Stage 2 feature construction**

**Cell 11 (markdown):**
```
## Stage 2: Tournament Model

Stage 2 learns how regular-season quality (Stage 1 prob) + tournament-specific
context (seeds, conference, SOS) translates to actual tournament outcomes.

Training data: ~2,500 historical men's tournament games, ~1,650 women's.
This is much smaller than Stage 1, so regularization is critical.
```

**Cell 12 (code) — Parse seeds and conferences into matchup features:**
```python
def parse_seed_number(seed_str):
    """Extract numeric seed from string like 'W01' or 'X16a'."""
    import re
    match = re.match(r"[WXYZ](\d{2})", seed_str)
    return int(match.group(1)) if match else None

def build_tournament_features(tourney_df, seeds_df, conf_df, snap_dict, team_stats,
                              stage1_model, stage1_features, stage1_medians,
                              gender="M", static_cols=None):
    """Build Stage 2 training data from historical tournament games.

    For each tournament game, computes:
    - stage1_prob: Stage 1 model's prediction for this matchup
    - seed_diff: TeamA seed - TeamB seed
    - conf_match: 1 if same conference, 0 otherwise
    - static feature diffs (SOS, win_pct, kenpom_rank for men's)
    """
    # Parse seeds into TeamID -> seed_number lookup per season
    seed_lookup = {}
    for _, row in seeds_df.iterrows():
        s, seed_str, tid = row["Season"], row["Seed"], row["TeamID"]
        seed_num = parse_seed_number(seed_str)
        seed_lookup[(s, tid)] = seed_num

    # Conference lookup: (season, team) -> conf
    conf_lookup = {}
    for _, row in conf_df.iterrows():
        conf_lookup[(row["Season"], row["TeamID"])] = row["ConfAbbrev"]

    if static_cols is None:
        static_cols = ["sos", "win_pct"]

    rows = []
    for _, game in tourney_df.iterrows():
        season = game["Season"]
        w_id, l_id = game["WTeamID"], game["LTeamID"]

        # Convention: TeamA = lower ID
        team_a = min(w_id, l_id)
        team_b = max(w_id, l_id)
        target = 1 if w_id == team_a else 0

        # Get rolling snapshots for this season
        if season not in snap_dict:
            continue
        snap = snap_dict[season]
        if team_a not in snap.index or team_b not in snap.index:
            continue

        snap_a = snap.loc[team_a]
        snap_b = snap.loc[team_b]

        # Stage 1 features (rolling + static diffs)
        roll_cols = [c for c in snap.columns if "_roll" in c]
        s1_feats = {}
        for col in roll_cols:
            s1_feats[col] = snap_a[col] - snap_b[col]

        # Static diffs for Stage 1
        if (season,) in team_stats.index.names or True:
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

        # Build Stage 1 feature vector and predict
        s1_vec = pd.DataFrame([s1_feats])
        for fc in stage1_features:
            if fc not in s1_vec.columns:
                s1_vec[fc] = stage1_medians.get(fc, 0.0)
        s1_vec = s1_vec[stage1_features].fillna(
            pd.Series(stage1_medians)
        ).fillna(0.0)

        stage1_prob = stage1_model.predict_proba(s1_vec.values)[:, 1][0]

        # Stage 2 features
        row = {
            "Season": season,
            "TeamA": team_a,
            "TeamB": team_b,
            "target": target,
            "stage1_prob": stage1_prob,
        }

        # Seed diff
        seed_a = seed_lookup.get((season, team_a))
        seed_b = seed_lookup.get((season, team_b))
        row["seed_diff"] = (seed_a - seed_b) if (seed_a and seed_b) else 0

        # Conference match
        conf_a = conf_lookup.get((season, team_a))
        conf_b = conf_lookup.get((season, team_b))
        row["conf_match"] = 1 if (conf_a and conf_b and conf_a == conf_b) else 0

        # Static feature diffs (for Stage 2 — SOS, win_pct, kenpom)
        for col in static_cols:
            key = f"s2_{col}_diff"
            try:
                stats_season = team_stats.loc[season]
                val_a = stats_season.loc[team_a, col] if team_a in stats_season.index else np.nan
                val_b = stats_season.loc[team_b, col] if team_b in stats_season.index else np.nan
                row[key] = val_a - val_b
            except (KeyError, TypeError):
                row[key] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)

print("Stage 2 feature builder ready.")
```

**Cell 13 (code) — Compute rolling snapshots per season for historical tournament games:**
```python
%%time
# Pre-compute rolling snapshots for each season we have tournament data
# Stage 1 uses regular-season-only rolling windows

def compute_all_snapshots(rs_detailed, rs_compact, seasons):
    """Compute end-of-regular-season rolling snapshots for each season."""
    snap_dict = {}
    for s in seasons:
        snap = get_team_rolling_snapshot(rs_detailed, rs_compact, season=s)
        if len(snap) > 0:
            snap_dict[s] = snap
    return snap_dict

m_tourney_seasons = sorted(m_tourney_compact["Season"].unique())
w_tourney_seasons = sorted(w_tourney_compact["Season"].unique())

print(f"Computing men's snapshots for {len(m_tourney_seasons)} seasons...")
m_snap_dict = compute_all_snapshots(m_rs_detailed, m_rs_compact, m_tourney_seasons)
print(f"  Got snapshots for {len(m_snap_dict)} seasons")

print(f"Computing women's snapshots for {len(w_tourney_seasons)} seasons...")
w_snap_dict = compute_all_snapshots(w_rs_detailed, w_rs_compact, w_tourney_seasons)
print(f"  Got snapshots for {len(w_snap_dict)} seasons")
```

**Cell 14 (code) — Build Stage 2 training data:**
```python
%%time
# Men's Stage 2 features
m_s2_static = ["sos", "win_pct", "kenpom_rank", "sos_adj_eff_margin"]
print("Building men's Stage 2 training data...")
m_s2_data = build_tournament_features(
    m_tourney_compact, m_seeds, m_conf, m_snap_dict, m_stats,
    m_s1_model, m_s1_features, m_s1_medians,
    gender="M", static_cols=m_s2_static,
)
print(f"  Men's Stage 2 rows: {len(m_s2_data):,}")

# Women's Stage 2 features (no kenpom)
w_s2_static = ["sos", "win_pct", "sos_adj_eff_margin"]
print("Building women's Stage 2 training data...")
w_s2_data = build_tournament_features(
    w_tourney_compact, w_seeds, w_conf, w_snap_dict, w_stats,
    w_s1_model, w_s1_features, w_s1_medians,
    gender="W", static_cols=w_s2_static,
)
print(f"  Women's Stage 2 rows: {len(w_s2_data):,}")

# Show feature columns
m_s2_feat_cols = [c for c in m_s2_data.columns if c not in ("Season", "TeamA", "TeamB", "target")]
w_s2_feat_cols = [c for c in w_s2_data.columns if c not in ("Season", "TeamA", "TeamB", "target")]
print(f"\nMen's Stage 2 features ({len(m_s2_feat_cols)}): {m_s2_feat_cols}")
print(f"Women's Stage 2 features ({len(w_s2_feat_cols)}): {w_s2_feat_cols}")
print(f"\nTarget balance:")
print(f"  Men's:   {m_s2_data['target'].mean():.3f}")
print(f"  Women's: {w_s2_data['target'].mean():.3f}")
```

- [ ] **Step 2: Run cells 11-14 and verify Stage 2 data construction**

Expected: ~2,500 men's rows, ~1,650 women's rows with stage1_prob, seed_diff, conf_match, and static diffs

- [ ] **Step 3: Commit**

```bash
git add notebooks/06_two_stage_model.ipynb
git commit -m "feat: two-stage model — Stage 2 tournament feature construction"
```

---

### Task 4: Train Stage 2 tournament model

**Files:**
- Modify: `notebooks/06_two_stage_model.ipynb` (add cells)

- [ ] **Step 1: Add Stage 2 training cells with Optuna tuning**

**Cell 15 (code) — Stage 2 train/val/test split + Optuna tuning:**
```python
# Stage 2 split: same temporal split
# Train: up to 2022, Val: 2023-2024, Test: 2025

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
```

**Cell 16 (code) — Stage 2 model options: LR + Optuna XGB + ensemble:**
```python
# Stage 2 Optuna tuning — more regularization due to small dataset
N_TRIALS_S2 = 50

# Note: Stage 2 has much less data, so we bias toward simpler models
def optuna_s2_objective(trial, X_tr, y_tr, X_val, y_val):
    """Optuna objective for Stage 2 XGBoost — heavily regularized."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 4),  # shallow trees
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

print("Tuning Men's Stage 2 XGBoost...")
m_s2_study = optuna.create_study(direction="minimize", study_name="s2_men")
m_s2_study.optimize(lambda t: optuna_s2_objective(t, m_s2_X_tr, m_s2_y_tr, m_s2_X_val, m_s2_y_val),
                     n_trials=N_TRIALS_S2, show_progress_bar=True)
print(f"  Best val log-loss: {m_s2_study.best_value:.4f}")
print(f"  Best val brier:    {m_s2_study.best_trial.user_attrs['val_brier']}")

print("\nTuning Women's Stage 2 XGBoost...")
w_s2_study = optuna.create_study(direction="minimize", study_name="s2_women")
w_s2_study.optimize(lambda t: optuna_s2_objective(t, w_s2_X_tr, w_s2_y_tr, w_s2_X_val, w_s2_y_val),
                     n_trials=N_TRIALS_S2, show_progress_bar=True)
print(f"  Best val log-loss: {w_s2_study.best_value:.4f}")
print(f"  Best val brier:    {w_s2_study.best_trial.user_attrs['val_brier']}")
```

**Cell 17 (code) — Build Stage 2 ensembles:**
```python
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

print(f"Men's Stage 2:   weights={m_s2_weights}")
print(f"Women's Stage 2: weights={w_s2_weights}")
```

- [ ] **Step 2: Run cells 15-17 and verify Stage 2 trains**

- [ ] **Step 3: Commit**

```bash
git add notebooks/06_two_stage_model.ipynb
git commit -m "feat: two-stage model — Stage 2 tournament model trained"
```

---

## Chunk 3: Evaluation & Comparison

### Task 5: 2025 Holdout Evaluation — Two-Stage vs Single-Stage

**Files:**
- Modify: `notebooks/06_two_stage_model.ipynb` (add cells)

- [ ] **Step 1: Add 2025 holdout evaluation cells**

**Cell 18 (markdown):**
```
## 2025 Tournament Holdout — Two-Stage vs Current Model

The ultimate test: predict the 2025 tournament using only data available before it started.

**Two-Stage approach:**
1. Compute 2025 regular-season rolling snapshots
2. Stage 1 predicts matchup probabilities from regular-season features
3. Stage 2 adjusts using seeds, conference, SOS + Stage 1 probability

**Comparison:** Current single-stage model baseline:
- Men's Brier: 0.1745, Women's: 0.1428, Combined: 0.1586
```

**Cell 19 (code) — Two-stage 2025 predictions:**
```python
# Load 2025 tournament results
m_tourney_2025 = m_tourney_compact[m_tourney_compact["Season"] == 2025].copy()
w_tourney_2025 = w_tourney_compact[w_tourney_compact["Season"] == 2025].copy()

# Get 2025 snapshots (from regular-season data only)
m_snap_2025 = m_snap_dict.get(2025, get_team_rolling_snapshot(m_rs_detailed, m_rs_compact, season=2025))
w_snap_2025 = w_snap_dict.get(2025, get_team_rolling_snapshot(w_rs_detailed, w_rs_compact, season=2025))

def predict_2025_two_stage(tourney_games, snap, team_stats, seeds_df, conf_df,
                            s1_model, s1_features, s1_medians,
                            s2_model, s2_feat_cols, s2_medians,
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

    results = []
    for _, game in tourney_games.iterrows():
        w_id, l_id = game["WTeamID"], game["LTeamID"]
        team_a, team_b = min(w_id, l_id), max(w_id, l_id)
        target = 1 if w_id == team_a else 0

        if team_a not in snap.index or team_b not in snap.index:
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
            "seed_diff": (seed_a - seed_b) if (seed_a and seed_b) else 0,
            "conf_match": 1 if conf_lookup.get(team_a) == conf_lookup.get(team_b) else 0,
        }

        for col in s2_static_cols:
            key = f"s2_{col}_diff"
            try:
                stats_2025 = team_stats.loc[2025]
                val_a = stats_2025.loc[team_a, col] if team_a in stats_2025.index else np.nan
                val_b = stats_2025.loc[team_b, col] if team_b in stats_2025.index else np.nan
                s2_row[key] = val_a - val_b
            except KeyError:
                s2_row[key] = 0.0

        s2_vec = pd.DataFrame([s2_row])
        for fc in s2_feat_cols:
            if fc not in s2_vec.columns:
                s2_vec[fc] = s2_medians.get(fc, 0.0)
        s2_vec = s2_vec[s2_feat_cols].fillna(pd.Series(s2_medians)).fillna(0.0)

        final_prob = np.clip(s2_model.predict_proba(s2_vec.values)[:, 1][0], clip[0], clip[1])

        results.append({
            "TeamA": team_a, "TeamB": team_b, "target": target,
            "stage1_prob": s1_prob, "pred": final_prob,
        })

    return pd.DataFrame(results)

print("Predicting 2025 tournament (two-stage)...\n")
m_2s_results = predict_2025_two_stage(
    m_tourney_2025, m_snap_2025, m_stats, m_seeds, m_conf,
    m_s1_model, m_s1_features, m_s1_medians,
    m_s2_model, m_s2_feat_cols, m_s2_medians,
    gender="M", s2_static_cols=m_s2_static,
)
w_2s_results = predict_2025_two_stage(
    w_tourney_2025, w_snap_2025, w_stats, w_seeds, w_conf,
    w_s1_model, w_s1_features, w_s1_medians,
    w_s2_model, w_s2_feat_cols, w_s2_medians,
    gender="W", s2_static_cols=w_s2_static,
)
print(f"Men's predictions:   {len(m_2s_results)}")
print(f"Women's predictions: {len(w_2s_results)}")
```

**Cell 20 (code) — Score and compare vs current single-stage model:**
```python
# Score two-stage predictions
def score(df, label):
    y_true, y_pred = df["target"].values, df["pred"].values
    bs = brier_score_loss(y_true, y_pred)
    ll = log_loss(y_true, y_pred)
    acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    return {"label": label, "brier": bs, "log_loss": ll, "accuracy": acc, "n": len(df)}

m_2s_scores = score(m_2s_results, "Men's Two-Stage")
w_2s_scores = score(w_2s_results, "Women's Two-Stage")
all_2s = pd.concat([m_2s_results, w_2s_results])
c_2s_scores = score(all_2s, "Combined Two-Stage")

# Current single-stage baseline (from holdout notebook 05)
baseline = {
    "Men's":    {"brier": 0.1745, "log_loss": 0.5206, "accuracy": 0.746},
    "Women's":  {"brier": 0.1428, "log_loss": 0.4399, "accuracy": 0.821},
    "Combined": {"brier": 0.1586, "log_loss": 0.4802, "accuracy": 0.784},
}

print("=" * 70)
print("2025 Tournament Holdout — Two-Stage vs Single-Stage")
print("=" * 70)
print(f"\n{'Metric':<20} {'Single-Stage':>14} {'Two-Stage':>14} {'Delta':>10}")
print("-" * 60)

for gender_label, two_stage_scores in [
    ("Men's", m_2s_scores), ("Women's", w_2s_scores), ("Combined", c_2s_scores)
]:
    bl = baseline[gender_label]
    print(f"\n  {gender_label}:")
    for metric in ["brier", "log_loss", "accuracy"]:
        old = bl[metric]
        new = two_stage_scores[metric]
        delta = new - old
        better = "better" if (delta < 0 and metric != "accuracy") or (delta > 0 and metric == "accuracy") else "worse"
        print(f"    {metric:<16} {old:>12.4f}   {new:>12.4f}   {delta:>+8.4f} ({better})")

print(f"\n{'=' * 70}")
```

**Cell 21 (code) — Stage 1 prob vs final prob scatter + reliability diagram:**
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Scatter: Stage 1 prob vs final (Stage 2) prob
for ax, results, label in [
    (axes[0, 0], m_2s_results, "Men's"),
    (axes[0, 1], w_2s_results, "Women's"),
]:
    colors = ["green" if t == 1 else "red" for t in results["target"]]
    ax.scatter(results["stage1_prob"], results["pred"], c=colors, alpha=0.6, edgecolor="black", s=40)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Stage 1 Probability")
    ax.set_ylabel("Stage 2 (Final) Probability")
    ax.set_title(f"{label}: Stage 1 vs Stage 2 Adjustments")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

# Reliability diagrams
for ax, results, label in [
    (axes[1, 0], m_2s_results, "Men's Two-Stage"),
    (axes[1, 1], w_2s_results, "Women's Two-Stage"),
]:
    y_true = results["target"].values
    y_pred = results["pred"].values
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_means = [], []
    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append(y_pred[mask].mean())
            bin_means.append(y_true[mask].mean())
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.scatter(bin_centers, bin_means, s=80, color="steelblue", edgecolor="black", zorder=5)
    ax.set_title(f"Reliability — {label}")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

plt.tight_layout()
plt.show()
```

**Cell 22 (code) — Feature importance for Stage 2:**
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, model, feat_cols, label in [
    (axes[0], m_s2_model.models[1], m_s2_feat_cols, "Men's Stage 2"),
    (axes[1], w_s2_model.models[1], w_s2_feat_cols, "Women's Stage 2"),
]:
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    ax.barh([feat_cols[i] for i in sorted_idx], importance[sorted_idx])
    ax.set_title(f"{label} — XGB Feature Importance (gain)")
    ax.set_xlabel("Importance")

plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Run cells 18-22 and verify evaluation completes**

Expected: Side-by-side comparison table with Brier/LogLoss/Accuracy for both approaches, visualization of Stage 1 vs Stage 2 adjustments

- [ ] **Step 3: Commit**

```bash
git add notebooks/06_two_stage_model.ipynb
git commit -m "feat: two-stage model — 2025 holdout evaluation vs single-stage"
```

---

### Task 6: Production retrain + save models

**Files:**
- Modify: `notebooks/06_two_stage_model.ipynb` (add cells)

- [ ] **Step 1: Add production retrain cells (only if two-stage beats single-stage)**

**Cell 23 (markdown):**
```
## Production Retrain (if two-stage improves over single-stage)

If the two-stage model shows improvement on the 2025 holdout, retrain both stages
on all available data:
- Stage 1: All regular-season games 2003-2025 (M) / 2010-2025 (W)
- Stage 2: All tournament games 2003-2025 (M) / 2010-2025 (W)

Then generate 2026 submission predictions.
```

**Cell 24 (code) — Production retrain:**
```python
# Retrain Stage 1 on all regular-season data (2003-2025)
print("=== Production Retrain ===\n")

# Stage 1: full retrain on all regular-season data
m_s1_X_full = m_s1_train.loc[m_s1_train["Season"].between(2003, 2025), m_s1_features].copy()
m_s1_y_full = m_s1_train.loc[m_s1_train["Season"].between(2003, 2025), "target"]
m_s1_X_full = m_s1_X_full.fillna(pd.Series(m_s1_medians)).fillna(0.0)

w_s1_X_full = w_s1_train.loc[w_s1_train["Season"].between(2010, 2025), w_s1_features].copy()
w_s1_y_full = w_s1_train.loc[w_s1_train["Season"].between(2010, 2025), "target"]
w_s1_X_full = w_s1_X_full.fillna(pd.Series(w_s1_medians)).fillna(0.0)

# Retrain Stage 1 LR + XGB
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

# Stage 2: retrain on ALL tournament data (2003-2025)
# Need to regenerate Stage 2 features using production Stage 1 model
# (This is because Stage 1 probabilities will be slightly different with more training data)
print("\nRebuilding Stage 2 features with production Stage 1...")
m_s2_data_prod = build_tournament_features(
    m_tourney_compact, m_seeds, m_conf, m_snap_dict, m_stats,
    m_prod_s1, m_s1_features, m_s1_medians,
    gender="M", static_cols=m_s2_static,
)
w_s2_data_prod = build_tournament_features(
    w_tourney_compact, w_seeds, w_conf, w_snap_dict, w_stats,
    w_prod_s1, w_s1_features, w_s1_medians,
    gender="W", static_cols=w_s2_static,
)

m_s2_X_prod = m_s2_data_prod[m_s2_feat_cols].fillna(pd.Series(m_s2_medians)).fillna(0.0)
m_s2_y_prod = m_s2_data_prod["target"]
w_s2_X_prod = w_s2_data_prod[w_s2_feat_cols].fillna(pd.Series(w_s2_medians)).fillna(0.0)
w_s2_y_prod = w_s2_data_prod["target"]

# Retrain Stage 2
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

# Build TwoStagePredictor objects
m_two_stage = TwoStagePredictor(
    stage1_model=m_prod_s1, stage2_model=m_prod_s2,
    stage1_features=m_s1_features, stage2_features=m_s2_feat_cols,
    stage1_medians=m_s1_medians, stage2_medians=m_s2_medians,
)
w_two_stage = TwoStagePredictor(
    stage1_model=w_prod_s1, stage2_model=w_prod_s2,
    stage1_features=w_s1_features, stage2_features=w_s2_feat_cols,
    stage1_medians=w_s1_medians, stage2_medians=w_s2_medians,
)

# Save
joblib.dump(m_two_stage, OUTPUT_DIR / "m_two_stage_final.joblib")
joblib.dump(w_two_stage, OUTPUT_DIR / "w_two_stage_final.joblib")

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
    "m_s1_xgb_params": {k: str(v) for k, v in m_s1_study.best_params.items()},
    "w_s1_xgb_params": {k: str(v) for k, v in w_s1_study.best_params.items()},
    "m_s2_xgb_params": {k: str(v) for k, v in m_s2_study.best_params.items()},
    "w_s2_xgb_params": {k: str(v) for k, v in w_s2_study.best_params.items()},
    "clip_range": [0.01, 0.99],
}
with open(OUTPUT_DIR / "two_stage_meta.json", "w") as f:
    json.dump(two_stage_meta, f, indent=2)

print("\nSaved production models:")
print(f"  {OUTPUT_DIR / 'm_two_stage_final.joblib'}")
print(f"  {OUTPUT_DIR / 'w_two_stage_final.joblib'}")
print(f"  {OUTPUT_DIR / 'two_stage_meta.json'}")
```

- [ ] **Step 2: Run cells 23-24 (only if two-stage beats single-stage on holdout)**

If two-stage is worse: Skip production retrain. Keep the single-stage model. Document the results and what we learned.

If two-stage is better: Run production retrain and save models.

- [ ] **Step 3: Commit**

```bash
git add notebooks/06_two_stage_model.ipynb src/two_stage.py outputs/two_stage_meta.json
git commit -m "feat: two-stage model — production retrain and saved models"
```

---

## Summary

| Task | What | Files | Estimated Cells |
|------|------|-------|-----------------|
| 1 | TwoStagePredictor class | `src/two_stage.py` | N/A |
| 2 | Stage 1 training (regular-season only) | `notebooks/06_two_stage_model.ipynb` cells 1-10 | 10 |
| 3 | Stage 2 feature construction | cells 11-14 | 4 |
| 4 | Stage 2 training + Optuna | cells 15-17 | 3 |
| 5 | 2025 holdout evaluation + comparison | cells 18-22 | 5 |
| 6 | Production retrain (conditional) | cells 23-24 | 2 |

**Key design decisions:**
1. Stage 1 trains on regular-season ONLY — isolating regular-season signal
2. Stage 2 has ~2,500 men's / ~1,650 women's games — must be heavily regularized (shallow trees, high alpha/lambda, low C for LR)
3. Both stages use LR+XGB ensembles with Optuna tuning
4. Final probabilities clipped to [0.01, 0.99] to avoid extreme log-loss penalties
5. Seeds used as Stage 2 feature (seed_diff), not as Stage 1 feature — this is where seeds belong (tournament context)
6. Conference match is a binary feature — captures intra-conference familiarity dynamics
