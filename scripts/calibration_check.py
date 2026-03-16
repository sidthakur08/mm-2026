"""Calibration analysis for the two-stage tournament prediction model.

Reconstructs 2025 tournament predictions and Stage 2 validation predictions
(2023-2024), then computes:
  1. Expected Calibration Error (ECE) with 10 bins
  2. Reliability table (bin_center, observed_frequency, count)
  3. Platt scaling and isotonic regression calibration experiments
  4. Clear summary: are the predictions well-calibrated?

Usage:
    python scripts/calibration_check.py
"""

import sys
import os
import json
import re
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_regular_season_results,
    load_tourney_results,
    load_seeds,
    load_massey_ordinals,
    load_team_conferences,
)
from src.features import (
    build_team_season_stats,
    build_advanced_team_stats,
    add_strength_of_schedule,
    add_seed_features,
    add_massey_features,
    add_single_system_ranking,
    add_sos_adjusted_features,
    compute_rolling_window_stats,
)

import joblib

OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42


# ============================================================
# Helpers
# ============================================================

def parse_seed_number(seed_str):
    """Extract numeric seed from string like 'W01' or 'X16a'."""
    match = re.match(r"[WXYZ](\d{2})", seed_str)
    return int(match.group(1)) if match else None


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
        stats = add_single_system_ranking(
            stats, ordinals, system_name="POM", col_name="kenpom_rank"
        )

    stats["neutral_win_pct"] = stats["neutral_win_pct"].fillna(stats["win_pct"])
    stats["away_win_pct"] = stats["away_win_pct"].fillna(stats["win_pct"])
    return stats


def compute_all_snapshots_fast(rs_detailed, rs_compact, seasons):
    """Compute end-of-regular-season rolling snapshots for all seasons at once."""
    all_team_games = compute_rolling_window_stats(
        rs_detailed, rs_compact, windows=(5, 7, 10)
    )
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
            sg[keep_cols].sort_values("DayNum").groupby("TeamID").last()
        )
        if len(latest) > 0:
            snap_dict[s] = latest
    return snap_dict


def predict_tournament_games(
    tourney_df,
    snap_dict,
    team_stats,
    seeds_df,
    conf_df,
    two_stage_model,
    meta,
    gender="M",
):
    """Reconstruct two-stage predictions for tournament games.

    Returns a DataFrame with columns:
        Season, TeamA, TeamB, target, stage1_prob, pred
    """
    s1_features = meta[f"{gender.lower()}_s1_features"]
    s2_features = meta[f"{gender.lower()}_s2_features"]
    s1_medians = meta[f"{gender.lower()}_s1_medians"]
    s2_medians = meta[f"{gender.lower()}_s2_medians"]

    s1_model = two_stage_model.stage1_model
    s2_model = two_stage_model.stage2_model

    if gender == "M":
        s2_static_cols = ["sos", "win_pct", "kenpom_rank", "sos_adj_eff_margin"]
    else:
        s2_static_cols = ["sos", "win_pct", "sos_adj_eff_margin"]

    # Build lookups
    seed_lookup = {}
    for _, row in seeds_df.iterrows():
        seed_num = parse_seed_number(row["Seed"])
        seed_lookup[(row["Season"], row["TeamID"])] = seed_num

    conf_lookup = {}
    for _, row in conf_df.iterrows():
        conf_lookup[(row["Season"], row["TeamID"])] = row["ConfAbbrev"]

    results = []
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
        roll_cols = [c for c in snap.columns if "_roll" in c]

        # Stage 1 features (rolling diffs + static diffs)
        s1_feats = {}
        for col in roll_cols:
            s1_feats[col] = snap_a[col] - snap_b[col]

        try:
            stats_season = team_stats.loc[season]
            for col in s2_static_cols:
                key = f"static_{col}"
                if col in stats_season.columns:
                    val_a = (
                        stats_season.loc[team_a, col]
                        if team_a in stats_season.index
                        else np.nan
                    )
                    val_b = (
                        stats_season.loc[team_b, col]
                        if team_b in stats_season.index
                        else np.nan
                    )
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
        seed_a = seed_lookup.get((season, team_a))
        seed_b = seed_lookup.get((season, team_b))

        s2_row = {
            "stage1_prob": s1_prob,
            "seed_diff": (
                (seed_a - seed_b)
                if (seed_a is not None and seed_b is not None)
                else 0
            ),
            "conf_match": (
                1
                if conf_lookup.get((season, team_a))
                == conf_lookup.get((season, team_b))
                else 0
            ),
        }

        for col in s2_static_cols:
            key = f"s2_{col}_diff"
            try:
                stats_season = team_stats.loc[season]
                val_a = (
                    stats_season.loc[team_a, col]
                    if team_a in stats_season.index
                    else np.nan
                )
                val_b = (
                    stats_season.loc[team_b, col]
                    if team_b in stats_season.index
                    else np.nan
                )
                s2_row[key] = val_a - val_b
            except KeyError:
                s2_row[key] = 0.0

        s2_vec = pd.DataFrame([s2_row])
        for fc in s2_features:
            if fc not in s2_vec.columns:
                s2_vec[fc] = s2_medians.get(fc, 0.0)
        s2_vec = s2_vec[s2_features].fillna(pd.Series(s2_medians)).fillna(0.0)

        final_prob = np.clip(
            s2_model.predict_proba(s2_vec.values)[:, 1][0], 0.01, 0.99
        )

        results.append(
            {
                "Season": season,
                "TeamA": team_a,
                "TeamB": team_b,
                "target": target,
                "stage1_prob": s1_prob,
                "pred": final_prob,
            }
        )

    if skipped > 0:
        print(f"    (skipped {skipped} games with missing snapshots)")
    return pd.DataFrame(results)


# ============================================================
# Calibration Metrics
# ============================================================


def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error with equal-width bins.

    ECE = sum_b (n_b / N) * |acc_b - conf_b|

    Returns ECE and the per-bin reliability table.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    table_rows = []
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        count = mask.sum()
        if count == 0:
            table_rows.append(
                {
                    "bin_center": round((lo + hi) / 2, 2),
                    "observed_freq": None,
                    "mean_predicted": None,
                    "count": 0,
                    "gap": None,
                }
            )
            continue

        observed = y_true[mask].mean()
        predicted = y_prob[mask].mean()
        gap = abs(observed - predicted)
        ece += (count / total) * gap

        table_rows.append(
            {
                "bin_center": round((lo + hi) / 2, 2),
                "observed_freq": round(observed, 4),
                "mean_predicted": round(predicted, 4),
                "count": int(count),
                "gap": round(gap, 4),
            }
        )

    return ece, pd.DataFrame(table_rows)


def fit_platt(y_train, probs_train, probs_test):
    """Fit Platt scaling (logistic regression on predicted probs) and transform."""
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)
    lr.fit(probs_train.reshape(-1, 1), y_train)
    calibrated = lr.predict_proba(probs_test.reshape(-1, 1))[:, 1]
    return np.clip(calibrated, 0.01, 0.99)


def fit_isotonic(y_train, probs_train, probs_test):
    """Fit isotonic regression on predicted probs and transform."""
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(probs_train, y_train)
    calibrated = iso.predict(probs_test)
    return np.clip(calibrated, 0.01, 0.99)


# ============================================================
# Main
# ============================================================


def main():
    print("=" * 72)
    print("  CALIBRATION ANALYSIS -- Two-Stage Tournament Model")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading data and models...")

    m_two_stage = joblib.load(OUTPUT_DIR / "m_two_stage_final.joblib")
    w_two_stage = joblib.load(OUTPUT_DIR / "w_two_stage_final.joblib")

    with open(OUTPUT_DIR / "two_stage_meta.json") as f:
        meta = json.load(f)

    # Load raw datasets
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

    print("  Loading Massey ordinals (men's only)...")
    ordinals_df = load_massey_ordinals()

    # Build static features
    print("  Building static features...")
    m_stats = build_static_features(
        m_rs_detailed, m_rs_compact, m_seeds, ordinals_df, "M"
    )
    w_stats = build_static_features(
        w_rs_detailed, w_rs_compact, w_seeds, gender="W"
    )

    # Compute rolling snapshots for 2023-2025 (val + test seasons)
    needed_seasons = [2023, 2024, 2025]
    print(f"  Computing rolling snapshots for seasons {needed_seasons}...")

    print("    Men's snapshots...")
    m_snap_dict = compute_all_snapshots_fast(
        m_rs_detailed, m_rs_compact, needed_seasons
    )
    print(f"    Got snapshots for {len(m_snap_dict)} men's seasons")

    print("    Women's snapshots...")
    w_snap_dict = compute_all_snapshots_fast(
        w_rs_detailed, w_rs_compact, needed_seasons
    )
    print(f"    Got snapshots for {len(w_snap_dict)} women's seasons")

    # ------------------------------------------------------------------
    # 2. Reconstruct predictions: val (2023-2024) and test (2025)
    # ------------------------------------------------------------------
    print("\n[2/4] Reconstructing tournament predictions...")

    # Val set: 2023-2024 tournament games
    m_tourney_val = m_tourney_compact[
        m_tourney_compact["Season"].isin([2023, 2024])
    ]
    w_tourney_val = w_tourney_compact[
        w_tourney_compact["Season"].isin([2023, 2024])
    ]
    # Test set: 2025 tournament games
    m_tourney_test = m_tourney_compact[m_tourney_compact["Season"] == 2025]
    w_tourney_test = w_tourney_compact[w_tourney_compact["Season"] == 2025]

    print("  Men's validation (2023-2024)...")
    m_val_preds = predict_tournament_games(
        m_tourney_val, m_snap_dict, m_stats, m_seeds, m_conf,
        m_two_stage, meta, gender="M",
    )
    print(f"    {len(m_val_preds)} predictions")

    print("  Women's validation (2023-2024)...")
    w_val_preds = predict_tournament_games(
        w_tourney_val, w_snap_dict, w_stats, w_seeds, w_conf,
        w_two_stage, meta, gender="W",
    )
    print(f"    {len(w_val_preds)} predictions")

    print("  Men's test (2025)...")
    m_test_preds = predict_tournament_games(
        m_tourney_test, m_snap_dict, m_stats, m_seeds, m_conf,
        m_two_stage, meta, gender="M",
    )
    print(f"    {len(m_test_preds)} predictions")

    print("  Women's test (2025)...")
    w_test_preds = predict_tournament_games(
        w_tourney_test, w_snap_dict, w_stats, w_seeds, w_conf,
        w_two_stage, meta, gender="W",
    )
    print(f"    {len(w_test_preds)} predictions")

    # Combine genders
    val_preds = pd.concat([m_val_preds, w_val_preds], ignore_index=True)
    test_preds = pd.concat([m_test_preds, w_test_preds], ignore_index=True)
    all_preds = pd.concat([test_preds], ignore_index=True)  # 2025 only for main eval

    # ------------------------------------------------------------------
    # 3. ECE and Reliability Tables
    # ------------------------------------------------------------------
    print("\n[3/4] Computing calibration metrics...")

    datasets = [
        ("Men's 2025", m_test_preds),
        ("Women's 2025", w_test_preds),
        ("Combined 2025", test_preds),
        ("Men's Val (2023-24)", m_val_preds),
        ("Women's Val (2023-24)", w_val_preds),
        ("Combined Val (2023-24)", val_preds),
    ]

    print("\n" + "=" * 72)
    print("  EXPECTED CALIBRATION ERROR (ECE) -- 10 bins")
    print("=" * 72)

    ece_results = {}
    for label, df in datasets:
        if len(df) == 0:
            print(f"\n  {label}: no data")
            continue

        y_true = df["target"].values.astype(float)
        y_prob = df["pred"].values.astype(float)
        ece, table = compute_ece(y_true, y_prob, n_bins=10)
        ece_results[label] = ece

        bs = brier_score_loss(y_true, y_prob)
        ll = log_loss(y_true, y_prob)

        print(f"\n  {label}")
        print(f"  {'':->60}")
        print(f"  ECE = {ece:.4f}   |  Brier = {bs:.4f}  |  Log-Loss = {ll:.4f}")
        print(f"  N = {len(df)}")
        print(
            f"  {'Bin Center':>12} {'Obs Freq':>12} {'Mean Pred':>12} "
            f"{'Count':>8} {'|Gap|':>8}"
        )
        print(f"  {'-' * 56}")
        for _, row in table.iterrows():
            if row["count"] == 0:
                print(
                    f"  {row['bin_center']:>12.2f} {'--':>12} {'--':>12} "
                    f"{0:>8} {'--':>8}"
                )
            else:
                print(
                    f"  {row['bin_center']:>12.2f} {row['observed_freq']:>12.4f} "
                    f"{row['mean_predicted']:>12.4f} {row['count']:>8} "
                    f"{row['gap']:>8.4f}"
                )

    # ------------------------------------------------------------------
    # 4. Platt scaling and isotonic regression experiments
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 72)
    print("  CALIBRATION EXPERIMENTS -- Platt Scaling vs Isotonic Regression")
    print("=" * 72)
    print(
        "\n  Approach: fit calibrators on validation set (2023-2024 tournament),\n"
        "  evaluate on test set (2025 tournament).\n"
    )

    cal_experiments = [
        ("Men's", m_val_preds, m_test_preds),
        ("Women's", w_val_preds, w_test_preds),
        ("Combined", val_preds, test_preds),
    ]

    summary_rows = []

    for label, val_df, test_df in cal_experiments:
        if len(val_df) == 0 or len(test_df) == 0:
            print(f"  {label}: insufficient data, skipping.")
            continue

        y_val = val_df["target"].values.astype(float)
        p_val = val_df["pred"].values.astype(float)
        y_test = test_df["target"].values.astype(float)
        p_test = test_df["pred"].values.astype(float)

        # Uncalibrated baseline on test
        bs_raw = brier_score_loss(y_test, p_test)
        ll_raw = log_loss(y_test, p_test)
        ece_raw, _ = compute_ece(y_test, p_test, n_bins=10)

        # Platt scaling
        p_platt = fit_platt(y_val, p_val, p_test)
        bs_platt = brier_score_loss(y_test, p_platt)
        ll_platt = log_loss(y_test, p_platt)
        ece_platt, _ = compute_ece(y_test, p_platt, n_bins=10)

        # Isotonic regression
        p_iso = fit_isotonic(y_val, p_val, p_test)
        bs_iso = brier_score_loss(y_test, p_iso)
        ll_iso = log_loss(y_test, p_iso)
        ece_iso, _ = compute_ece(y_test, p_iso, n_bins=10)

        print(f"\n  {label} (val N={len(val_df)}, test N={len(test_df)})")
        print(f"  {'-' * 64}")
        print(
            f"  {'Method':<20} {'Brier':>10} {'Log-Loss':>10} {'ECE':>10} "
            f"{'Brier delta':>12}"
        )
        print(f"  {'-' * 64}")
        print(
            f"  {'Uncalibrated':<20} {bs_raw:>10.4f} {ll_raw:>10.4f} "
            f"{ece_raw:>10.4f} {'--':>12}"
        )
        print(
            f"  {'Platt scaling':<20} {bs_platt:>10.4f} {ll_platt:>10.4f} "
            f"{ece_platt:>10.4f} {bs_platt - bs_raw:>+12.4f}"
        )
        print(
            f"  {'Isotonic regr.':<20} {bs_iso:>10.4f} {ll_iso:>10.4f} "
            f"{ece_iso:>10.4f} {bs_iso - bs_raw:>+12.4f}"
        )

        summary_rows.append(
            {
                "dataset": label,
                "bs_raw": bs_raw,
                "bs_platt": bs_platt,
                "bs_isotonic": bs_iso,
                "ece_raw": ece_raw,
                "ece_platt": ece_platt,
                "ece_isotonic": ece_iso,
                "best_method": (
                    "Platt"
                    if bs_platt < bs_iso and bs_platt < bs_raw
                    else (
                        "Isotonic"
                        if bs_iso < bs_platt and bs_iso < bs_raw
                        else "Uncalibrated"
                    )
                ),
            }
        )

    # ------------------------------------------------------------------
    # 5. Summary & Recommendations
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 72)
    print("  SUMMARY & RECOMMENDATIONS")
    print("=" * 72)

    # Determine overall calibration quality
    combined_ece = ece_results.get("Combined 2025", None)
    combined_val_ece = ece_results.get("Combined Val (2023-24)", None)

    print("\n  Calibration Quality Assessment:")
    print(f"  {'-' * 60}")

    if combined_ece is not None:
        if combined_ece < 0.03:
            quality = "EXCELLENT"
            msg = "predictions are very well-calibrated"
        elif combined_ece < 0.06:
            quality = "GOOD"
            msg = "predictions are reasonably well-calibrated"
        elif combined_ece < 0.10:
            quality = "MODERATE"
            msg = "predictions show some miscalibration"
        else:
            quality = "POOR"
            msg = "predictions are poorly calibrated"

        print(f"  Combined 2025 ECE = {combined_ece:.4f}  -->  {quality}: {msg}")

    if combined_val_ece is not None:
        print(f"  Combined Val ECE  = {combined_val_ece:.4f}")

    print(f"\n  Calibration Improvement Potential:")
    print(f"  {'-' * 60}")

    if summary_rows:
        combined_row = next(
            (r for r in summary_rows if r["dataset"] == "Combined"), None
        )
        if combined_row:
            best = combined_row["best_method"]
            delta_platt = combined_row["bs_platt"] - combined_row["bs_raw"]
            delta_iso = combined_row["bs_isotonic"] - combined_row["bs_raw"]

            print(f"  Best method on combined 2025: {best}")
            print(f"    Platt Brier delta:    {delta_platt:+.4f}")
            print(f"    Isotonic Brier delta: {delta_iso:+.4f}")

            # Recommendation threshold: improvement must be > 0.002 to be worth it
            # with such small sample sizes (67 games per gender)
            THRESHOLD = 0.002
            best_delta = min(delta_platt, delta_iso)

            if best_delta < -THRESHOLD:
                rec_method = "Platt" if delta_platt < delta_iso else "Isotonic"
                print(
                    f"\n  RECOMMENDATION: Apply {rec_method} scaling."
                )
                print(
                    f"    Expected Brier improvement: {abs(best_delta):.4f}"
                )
                print(
                    "    Fit on all available tournament data (2003-2024 for men's,"
                )
                print("    2010-2024 for women's) and apply to 2026 predictions.")
            elif best_delta < 0:
                print(
                    f"\n  RECOMMENDATION: Calibration shows marginal improvement "
                    f"({abs(best_delta):.4f})."
                )
                print(
                    "    With only ~67 test games per gender, this is within noise."
                )
                print(
                    "    The model is already well-calibrated enough. "
                    "Skip post-hoc calibration."
                )
            else:
                print(
                    f"\n  RECOMMENDATION: Post-hoc calibration does NOT help."
                )
                print(
                    "    The two-stage model is already well-calibrated."
                )
                print(
                    "    Platt/isotonic actually hurt Brier score, likely due to"
                )
                print(
                    "    small validation set (2023-2024 = ~134 games per gender)."
                )

    # Warn about sample sizes
    print(f"\n  Caveats:")
    print(f"  {'-' * 60}")
    print(
        "  - Small sample sizes: 67 games per gender in 2025, ~134 in val."
    )
    print(
        "  - ECE estimates are noisy with <150 samples."
    )
    print(
        "  - Isotonic regression is especially prone to overfitting with"
    )
    print(
        "    small calibration sets (<200 samples)."
    )
    print(
        "  - The two-stage architecture already includes an LR/XGB ensemble"
    )
    print(
        "    with blending weights, which provides implicit calibration."
    )

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
