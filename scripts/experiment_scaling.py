"""Experiment: Test different scalings of seed_diff and s2_elo_diff in Stage 2.

Scales these features before training to control their influence.
Tests on 2025 holdout with Optuna-tuned XGB ensemble.
"""

import sys, warnings, time, re, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import optuna
import joblib

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import *
from src.features import (
    build_team_season_stats, build_advanced_team_stats,
    add_strength_of_schedule, add_seed_features,
    add_massey_features, add_single_system_ranking,
    add_sos_adjusted_features, compute_rolling_window_stats,
    compute_elo_ratings,
)
from src.model import EnsemblePredictor

RANDOM_STATE = 42
t0 = time.time()

# Load production models + meta
m_model = joblib.load("outputs/m_two_stage_final.joblib")
w_model = joblib.load("outputs/w_two_stage_final.joblib")
meta = json.load(open("outputs/two_stage_meta.json"))

# Load raw data
m_rs_d = load_regular_season_results("M", detailed=True)
m_rs_c = load_regular_season_results("M", detailed=False)
w_rs_d = load_regular_season_results("W", detailed=True)
w_rs_c = load_regular_season_results("W", detailed=False)
m_tc = load_tourney_results("M")
w_tc = load_tourney_results("W")
m_seeds = load_seeds("M")
w_seeds = load_seeds("W")
m_conf = load_team_conferences("M")
w_conf = load_team_conferences("W")
m_ordinals = load_massey_ordinals()

def build_static_features(rs_detailed, rs_compact, seeds, ordinals=None, gender="M"):
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

m_stats = build_static_features(m_rs_d, m_rs_c, m_seeds, m_ordinals, "M")
w_stats = build_static_features(w_rs_d, w_rs_c, w_seeds, gender="W")

# Elo
m_elo = compute_elo_ratings(pd.concat([m_rs_c, m_tc], ignore_index=True),
                            k_factor=20, season_carryover=0.75, margin_factor=True)
w_elo = compute_elo_ratings(pd.concat([w_rs_c, w_tc], ignore_index=True),
                            k_factor=20, season_carryover=0.75, margin_factor=True)

# Rolling snapshots
def compute_snaps(rs_d, rs_c, seasons):
    tg = compute_rolling_window_stats(rs_d, rs_c, windows=(5, 7, 10))
    rc = [c for c in tg.columns if "_roll" in c]
    kc = ["Season", "TeamID", "DayNum"] + rc
    sd = {}
    for s in seasons:
        sg = tg[tg["Season"] == s]
        if len(sg) == 0:
            continue
        sd[s] = sg[kc].sort_values("DayNum").groupby("TeamID").last()
    return sd

m_snap = compute_snaps(m_rs_d, m_rs_c, sorted(m_tc["Season"].unique()))
w_snap = compute_snaps(w_rs_d, w_rs_c, sorted(w_tc["Season"].unique()))

print(f"Data loaded in {time.time()-t0:.0f}s")


# Build Stage 2 data using production Stage 1 models
def parse_sn(s):
    m = re.match(r"[WXYZ](\d{2})", s)
    return int(m.group(1)) if m else None


def build_s2(tc, seeds_df, conf_df, snap_dict, stats, s1_model, s1_feats, s1_meds,
             elo_df, static_cols):
    seed_lk = {}
    for _, r in seeds_df.iterrows():
        seed_lk[(r["Season"], r["TeamID"])] = parse_sn(r["Seed"])
    conf_lk = {}
    for _, r in conf_df.iterrows():
        conf_lk[(r["Season"], r["TeamID"])] = r["ConfAbbrev"]

    elo_snaps = {}
    for s in tc["Season"].unique():
        re_ = elo_df[(elo_df["Season"] == s) & (elo_df["DayNum"] < 134)]
        if len(re_) > 0:
            elo_snaps[s] = re_.sort_values("DayNum").groupby("TeamID")["Elo"].last()

    rows = []
    for _, g in tc.iterrows():
        s = g["Season"]
        ta, tb = min(g["WTeamID"], g["LTeamID"]), max(g["WTeamID"], g["LTeamID"])
        target = 1 if g["WTeamID"] == ta else 0
        if s not in snap_dict:
            continue
        snap = snap_dict[s]
        if ta not in snap.index or tb not in snap.index:
            continue

        sa, sb = snap.loc[ta], snap.loc[tb]
        rc = [c for c in snap.columns if "_roll" in c]
        f1 = {c: sa[c] - sb[c] for c in rc}

        try:
            ss = stats.loc[s]
            for c in static_cols:
                k = f"static_{c}"
                va = ss.loc[ta, c] if ta in ss.index else np.nan
                vb = ss.loc[tb, c] if tb in ss.index else np.nan
                f1[k] = va - vb
        except Exception:
            pass

        es = elo_snaps.get(s, pd.Series(dtype=float))
        ea = es.get(ta, 1500.0) if len(es) > 0 else 1500.0
        eb = es.get(tb, 1500.0) if len(es) > 0 else 1500.0
        f1["elo_diff"] = ea - eb

        v1 = pd.DataFrame([f1])
        for fc in s1_feats:
            if fc not in v1.columns:
                v1[fc] = s1_meds.get(fc, 0.0)
        v1 = v1[s1_feats].fillna(pd.Series(s1_meds)).fillna(0.0)
        s1p = s1_model.predict_proba(v1.values)[:, 1][0]

        seed_a = seed_lk.get((s, ta))
        seed_b = seed_lk.get((s, tb))

        row = {
            "stage1_prob": s1p,
            "seed_diff": (seed_a - seed_b) if seed_a and seed_b else 0,
            "conf_match": 1 if conf_lk.get((s, ta)) == conf_lk.get((s, tb)) else 0,
            "s2_elo_diff": ea - eb,
            "target": target, "Season": s, "TeamA": ta, "TeamB": tb,
        }
        for c in static_cols:
            try:
                row[f"s2_{c}_diff"] = stats.loc[(s, ta), c] - stats.loc[(s, tb), c]
            except KeyError:
                row[f"s2_{c}_diff"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


m_s2_static = ["sos", "win_pct", "kenpom_rank", "sos_adj_eff_margin"]
w_s2_static = ["sos", "win_pct", "sos_adj_eff_margin"]

m_s2 = build_s2(m_tc, m_seeds, m_conf, m_snap, m_stats,
                m_model.stage1_model, meta["m_s1_features"], meta["m_s1_medians"],
                m_elo, m_s2_static)
w_s2 = build_s2(w_tc, w_seeds, w_conf, w_snap, w_stats,
                w_model.stage1_model, meta["w_s1_features"], meta["w_s1_medians"],
                w_elo, w_s2_static)

print(f"Stage 2 data built: M={len(m_s2)}, W={len(w_s2)} in {time.time()-t0:.0f}s")

m_feat_cols = ["stage1_prob", "seed_diff", "conf_match", "s2_sos_diff",
               "s2_win_pct_diff", "s2_kenpom_rank_diff", "s2_sos_adj_eff_margin_diff",
               "s2_elo_diff"]
w_feat_cols = ["stage1_prob", "seed_diff", "conf_match", "s2_sos_diff",
               "s2_win_pct_diff", "s2_sos_adj_eff_margin_diff", "s2_elo_diff"]


def run_variant(m_data, w_data, m_feats, w_feats, seed_scale=1.0, elo_scale=1.0):
    md = m_data.copy()
    wd = w_data.copy()
    md["seed_diff"] = md["seed_diff"] * seed_scale
    md["s2_elo_diff"] = md["s2_elo_diff"] * elo_scale
    wd["seed_diff"] = wd["seed_diff"] * seed_scale
    wd["s2_elo_diff"] = wd["s2_elo_diff"] * elo_scale

    m_tr = md[md["Season"] <= 2022]
    m_val = md[md["Season"].between(2023, 2024)]
    m_te = md[md["Season"] == 2025]
    w_tr = wd[wd["Season"] <= 2022]
    w_val = wd[wd["Season"].between(2023, 2024)]
    w_te = wd[wd["Season"] == 2025]

    m_med = m_tr[m_feats].median().to_dict()
    w_med = w_tr[w_feats].median().to_dict()

    def fit_ensemble(X_tr, y_tr, X_val, y_val):
        lr = Pipeline([("s", StandardScaler()),
                       ("lr", LogisticRegression(max_iter=1000, C=0.1, random_state=42))])
        lr.fit(X_tr, y_tr)

        def obj(trial):
            p = {
                "learning_rate": trial.suggest_float("lr", 0.01, 0.15, log=True),
                "n_estimators": trial.suggest_int("ne", 50, 300),
                "max_depth": trial.suggest_int("md", 2, 4),
                "subsample": trial.suggest_float("ss", 0.5, 0.9),
                "colsample_bytree": trial.suggest_float("cs", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("ra", 0.1, 20.0, log=True),
                "reg_lambda": trial.suggest_float("rl", 0.1, 20.0, log=True),
                "min_child_weight": trial.suggest_int("mcw", 5, 30),
                "random_state": 42, "eval_metric": "logloss", "verbosity": 0,
            }
            m = XGBClassifier(**p)
            m.fit(X_tr, y_tr)
            return log_loss(y_val, m.predict_proba(X_val)[:, 1])

        study = optuna.create_study(direction="minimize")
        study.optimize(obj, n_trials=25, show_progress_bar=False)
        bp = study.best_params
        rn = {"lr": "learning_rate", "ne": "n_estimators", "md": "max_depth",
              "ss": "subsample", "cs": "colsample_bytree", "ra": "reg_alpha",
              "rl": "reg_lambda", "mcw": "min_child_weight"}
        cp = {rn.get(k, k): v for k, v in bp.items()}
        cp.update({"random_state": 42, "eval_metric": "logloss", "verbosity": 0})
        xgb = XGBClassifier(**cp)
        xgb.fit(X_tr, y_tr)

        best_w, best_ll = None, 1e9
        for w1 in np.arange(0, 1.02, 0.02):
            w = [w1, 1 - w1]
            pred = w[0] * lr.predict_proba(X_val)[:, 1] + w[1] * xgb.predict_proba(X_val)[:, 1]
            ll = log_loss(y_val, pred)
            if ll < best_ll:
                best_ll = ll
                best_w = w
        return EnsemblePredictor([lr, xgb], best_w)

    m_mod = fit_ensemble(
        m_tr[m_feats].fillna(pd.Series(m_med)).fillna(0.0), m_tr["target"],
        m_val[m_feats].fillna(pd.Series(m_med)).fillna(0.0), m_val["target"])
    w_mod = fit_ensemble(
        w_tr[w_feats].fillna(pd.Series(w_med)).fillna(0.0), w_tr["target"],
        w_val[w_feats].fillna(pd.Series(w_med)).fillna(0.0), w_val["target"])

    m_pred = np.clip(m_mod.predict_proba(
        m_te[m_feats].fillna(pd.Series(m_med)).fillna(0.0))[:, 1], 0.01, 0.99)
    w_pred = np.clip(w_mod.predict_proba(
        w_te[w_feats].fillna(pd.Series(w_med)).fillna(0.0))[:, 1], 0.01, 0.99)

    m_bs = brier_score_loss(m_te["target"], m_pred)
    w_bs = brier_score_loss(w_te["target"], w_pred)
    c_bs = brier_score_loss(
        np.concatenate([m_te["target"], w_te["target"]]),
        np.concatenate([m_pred, w_pred]))

    return m_bs, w_bs, c_bs


# Test grid
configs = [
    ("Baseline (1.0x seed, 1.0x elo)",    1.0,  1.0),
    # Downweight seed only
    ("0.75x seed, 1.0x elo",              0.75, 1.0),
    ("0.5x seed, 1.0x elo",               0.5,  1.0),
    ("0.25x seed, 1.0x elo",              0.25, 1.0),
    ("0.1x seed, 1.0x elo",               0.1,  1.0),
    ("0x seed, 1.0x elo",                 0.0,  1.0),
    # Downweight elo only
    ("1.0x seed, 0.5x elo",               1.0,  0.5),
    ("1.0x seed, 0.25x elo",              1.0,  0.25),
    ("1.0x seed, 0x elo",                 1.0,  0.0),
    # Downweight both
    ("0.5x seed, 0.5x elo",               0.5,  0.5),
    ("0.25x seed, 0.25x elo",             0.25, 0.25),
    ("0.5x seed, 0.25x elo",              0.5,  0.25),
    ("0.25x seed, 0.5x elo",              0.25, 0.5),
    # Stronger elo, weaker seed
    ("0.25x seed, 1.5x elo",              0.25, 1.5),
    ("0.5x seed, 1.5x elo",               0.5,  1.5),
    ("0.1x seed, 1.5x elo",               0.1,  1.5),
    # Stronger seed, weaker elo
    ("1.5x seed, 0.25x elo",              1.5,  0.25),
    ("1.5x seed, 0.5x elo",               1.5,  0.5),
]

print(f"\n{'Label':<35s} {'M Brier':<10s} {'W Brier':<10s} {'Combined':<10s}")
print("-" * 65)

results = []
for label, ss, es in configs:
    m_b, w_b, c_b = run_variant(m_s2, w_s2, m_feat_cols, w_feat_cols,
                                 seed_scale=ss, elo_scale=es)
    results.append((label, m_b, w_b, c_b, ss, es))
    marker = " ***" if c_b < 0.1318 else ""
    print(f"{label:<35s} {m_b:<10.4f} {w_b:<10.4f} {c_b:<10.4f}{marker}")

print()
best = min(results, key=lambda x: x[3])
print(f"Best combined: {best[0]} -> {best[3]:.4f}")

# Top 5
print("\nTop 5:")
for i, r in enumerate(sorted(results, key=lambda x: x[3])[:5]):
    print(f"  {i+1}. {r[0]:<35s} Combined={r[3]:.4f}  (M={r[1]:.4f}, W={r[2]:.4f})")

print(f"\nTotal time: {time.time()-t0:.0f}s")
