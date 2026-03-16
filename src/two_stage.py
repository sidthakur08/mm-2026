"""Two-stage tournament prediction model.

Stage 1: Regular-season model -- trained on regular-season games only,
         produces team-quality probabilities.
Stage 2: Tournament model -- trained on historical tournament games,
         uses Stage 1 probabilities + tournament-specific features
         (seed diff, conference, SOS gap) to produce final predictions.

Optional Platt scaling calibration is applied after Stage 2 and before
final clipping to improve probability calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    calibrator : fitted sklearn LogisticRegression, optional
        Platt scaling calibrator fitted on out-of-sample Stage 2
        predictions. Applied after Stage 2 and before clipping.
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
        calibrator=None,
        clip_range: tuple[float, float] = (0.01, 0.99),
    ):
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.stage1_features = stage1_features
        self.stage2_features = stage2_features
        self.stage1_medians = stage1_medians or {}
        self.stage2_medians = stage2_medians or {}
        self.calibrator = calibrator
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

        # Platt scaling calibration (if calibrator is set)
        if self.calibrator is not None:
            raw_probs = self.calibrator.predict_proba(
                raw_probs.reshape(-1, 1)
            )[:, 1]

        # Clip to configured range
        return np.clip(raw_probs, self.clip_range[0], self.clip_range[1])

    def __repr__(self):
        cal_str = type(self.calibrator).__name__ if self.calibrator else "None"
        return (
            f"TwoStagePredictor(stage1={self.stage1_model!r}, "
            f"stage2={type(self.stage2_model).__name__}, "
            f"calibrator={cal_str}, "
            f"clip={self.clip_range})"
        )
