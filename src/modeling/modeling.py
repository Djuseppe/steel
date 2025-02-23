import logging
from collections.abc import Iterable

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from src import BASE_DIR
from src.config import settings
from src.processor import Processor
from src.utils.utils import asymmetric_cost_metric

logger = logging.getLogger(__name__)


def get_xgb_params_long(trial: optuna.Trial) -> dict[str, int | float]:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
    }


def get_xgb_params_short(trial: optuna.Trial) -> dict[str, int | float]:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
    }


def get_xgb_params(trial: optuna.Trial, short: bool = True) -> dict[str, int | float]:
    if short:
        return get_xgb_params_short(trial)
    return get_xgb_params_long(trial)


def build_model(trial: optuna.Trial, short: bool = True) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(**get_xgb_params(trial, short=short))


def objective(
        trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.DataFrame, ids: Iterable,
        short: bool = True) -> np.floating:
    time_threshold = settings.avg_elapsed
    gkf = GroupKFold(n_splits=settings.n_splits, shuffle=False)
    fold_errors = []
    for train_idx, test_idx in gkf.split(X_train, y_train["end_t"], groups=ids):
        X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]
        model = build_model(trial, short=short)
        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_test_fold)
        weights = np.where(X_test_fold["elapsed"] >= time_threshold, 1, 0)
        fold_error = asymmetric_cost_metric(preds, weights=weights)
        fold_errors.append(fold_error)
    return np.mean(fold_errors)


def optimize_hp(data: pd.DataFrame):
    processor = Processor()
    X_train, y_train, X_test, y_test = processor.split_data(data)
    ids = X_train["heatid"]
    X_train.drop(columns=["heatid"], inplace=True)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(
            trial, X_train=X_train, y_train=y_train, ids=ids, short=True
        ),
        n_trials=settings.n_trials,
    )
    logger.info("Best hyperparameters:", study.best_params)
    logger.info("Best weighted asymmetric metric:", study.best_value)
    return study.best_trial


if __name__ == "__main__":
    import joblib

    output_model_path = BASE_DIR / "api/models/best_xgb.pkl"
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    processor = Processor()
    raw_data = processor.read_data()
    processed_data = processor.process(raw_data)

    X_train, y_train, _, _ = processor.split_data(processed_data)

    logger.info("Optimizing HP.")
    best_trial = optimize_hp(processed_data)
    logger.info("Training best model on full data.")
    X_train, y_train, _, _ = processor.split_data(processed_data)
    best_model = build_model(trial=best_trial)
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, output_model_path)
    logger.info(f"Best model saved to {output_model_path}")
