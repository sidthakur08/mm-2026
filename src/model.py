"""Model utilities for March Madness prediction pipeline."""

import numpy as np


class EnsemblePredictor:
    """Weighted average of multiple classifiers with optional calibration.

    Exposes the same ``predict_proba`` interface as scikit-learn estimators,
    so it can be used as a drop-in replacement in the prediction pipeline.

    Parameters
    ----------
    models : list
        Fitted classifiers, each with a ``predict_proba`` method.
    weights : list[float]
        Blend weights (must sum to 1).
    calibrator : object, optional
        A fitted calibrator (IsotonicRegression or LogisticRegression).
        If provided, raw blended probabilities are passed through it.
    """

    def __init__(self, models, weights, calibrator=None):
        self.models = models
        self.weights = np.array(weights, dtype=float)
        self.calibrator = calibrator

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], 2))
        for model, w in zip(self.models, self.weights):
            probs += w * model.predict_proba(X)

        if self.calibrator is not None:
            p1 = probs[:, 1]
            if hasattr(self.calibrator, "predict_proba"):
                # Platt scaling (LogisticRegression)
                p1 = self.calibrator.predict_proba(p1.reshape(-1, 1))[:, 1]
            else:
                # Isotonic regression
                p1 = self.calibrator.predict(p1)
            probs = np.column_stack([1 - p1, p1])

        return probs

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __repr__(self):
        names = [type(m).__name__ for m in self.models]
        w_str = ", ".join(f"{w:.3f}" for w in self.weights)
        cal = type(self.calibrator).__name__ if self.calibrator else "None"
        return (
            f"EnsemblePredictor(models={names}, weights=[{w_str}], "
            f"calibrator={cal})"
        )
