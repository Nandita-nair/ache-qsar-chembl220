"""Model zoo: every model is wrapped in an sklearn Pipeline so that imputation, variance filtering and
scaling are fitted inside each cross-validation fold (no information leaks from validation/test data)."""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def classifier_specs(seed: int) -> dict:
    """name -> (estimator, hyper-parameter grid)."""
    return {
        "RandomForest": (
            RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=1, random_state=seed),
            {"model__max_depth": [None, 20], "model__min_samples_leaf": [1, 3], "model__max_features": ["sqrt", 0.3]},
        ),
        "SVM-RBF": (
            # SVC probabilities via explicit calibration (SVC(probability=True) is deprecated in scikit-learn >= 1.9)
            CalibratedClassifierCV(SVC(kernel="rbf", class_weight="balanced", random_state=seed), method="sigmoid", cv=3),
            {"model__estimator__C": [0.3, 1, 3, 10], "model__estimator__gamma": ["scale", "auto"]},
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=seed),
            {"model__learning_rate": [0.05, 0.1], "model__max_depth": [3, None], "model__l2_regularization": [0.0, 1.0]},
        ),
        "kNN": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [3, 5, 9, 15], "model__weights": ["uniform", "distance"]},
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=5000, class_weight="balanced"),
            {"model__C": [0.001, 0.01, 0.1, 1.0]},
        ),
    }


def regressor_specs(seed: int) -> dict:
    return {
        "RandomForest": (
            RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=seed),
            {"model__max_features": ["sqrt", 0.3], "model__min_samples_leaf": [1, 3]},
        ),
        "SVR-RBF": (
            SVR(kernel="rbf"),
            {"model__C": [1, 3, 10], "model__epsilon": [0.1, 0.3], "model__gamma": ["scale", "auto"]},
        ),
        "HistGradientBoosting": (
            HistGradientBoostingRegressor(max_iter=200, random_state=seed),
            {"model__learning_rate": [0.05, 0.1], "model__max_depth": [3, None], "model__l2_regularization": [0.0, 1.0]},
        ),
        "kNN": (
            KNeighborsRegressor(),
            {"model__n_neighbors": [3, 5, 9, 15], "model__weights": ["uniform", "distance"]},
        ),
        "Ridge": (
            Ridge(),
            {"model__alpha": [10, 100, 1000, 10000]},
        ),
    }


def baseline(task: str):
    """Trivial model (class prior / training mean) - every real model must beat this."""
    return DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor(strategy="mean")


def shrink_grid(grid: dict) -> dict:
    """Keep only the first value of each hyper-parameter (used by --fast smoke tests)."""
    return {k: v[:1] for k, v in grid.items()}
