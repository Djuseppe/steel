import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd
from pydantic import ValidationError

from api.config import settings
from api.fetch_data import fetch_and_validate_data
from api.models import PredictionInput

logger = logging.getLogger(__name__)


def calculate_current_features(
    df: pd.DataFrame,
    gas_col: str = "gas1",
    n_lags: int = 5,
    n_prev_processes: int = 5,
    feature_order: Iterable | None = None,
) -> pd.DataFrame:
    """
    Computes and returns a DataFrame containing feature calculations for the last heat
    ID group in the input DataFrame. The calculation includes lag features, summary
    statistics for the specified gas column, elapsed time, and optionally reorders
    the columns based on the provided feature order.

    :param df: The input DataFrame containing at least the columns `datetime`, `heatid`,
               `end_t`, and the specified `gas_col` for feature computation.
    :type df: pd.DataFrame
    :param gas_col: The name of the column representing gas measurements to be used
                    for feature calculation.
    :type gas_col: str
    :param n_lags: The number of lagged features to compute for the specified gas column.
                   Default is 5.
    :type n_lags: int
    :param n_prev_processes: The number of previous process end times to include as features.
                             Default is 5.
    :type n_prev_processes: int
    :param feature_order: An optional iterable specifying the desired column order for
                          the output DataFrame.
    :type feature_order: Iterable | None
    :return: A DataFrame with feature calculations for the last heat ID group. The
             resulting DataFrame contains a single row with computed features.
    :rtype: pd.DataFrame
    """
    df = df.sort_values(["datetime"]).copy()
    features_per_heat_id = df.groupby("heatid").agg(end_t=("end_t", "last")).reset_index(drop=True)
    for i in range(1, n_prev_processes + 1):
        df[f"prev_end_t_{i}"] = features_per_heat_id["end_t"].shift(i).iloc[-1]
    for lag in range(1, n_lags + 1):
        df[f"{gas_col}_lag{lag}"] = df.groupby("heatid")[gas_col].shift(lag)
    current_heatid = df["heatid"].iloc[-1]
    df = df.loc[df["heatid"] == current_heatid, :].copy()
    df["heatid"] = current_heatid
    df[f"{gas_col}_mean"] = df[gas_col].mean()
    df[f"{gas_col}_std"] = df[gas_col].std()
    df[f"{gas_col}_max"] = df[gas_col].max()
    df[f"{gas_col}_min"] = df[gas_col].min()
    df[f"{gas_col}_first"] = df[gas_col].iloc[0]
    df[f"{gas_col}_last"] = df[gas_col].iloc[-1]
    df["end_t"] = df["end_t"].iloc[-1]
    df["elapsed"] = (df["datetime"].iloc[-1] - df["datetime"].iloc[0]).total_seconds()
    if feature_order is not None:
        df = df.loc[:, feature_order]
    return df.iloc[[-1]]


def inference(input_data: PredictionInput, model) -> float:
    """Perform inference using the fitted model."""
    try:
        db_data = fetch_and_validate_data(request_up_to=input_data.datetime.strftime(settings.dt_format))
        if db_data.empty:
            logger.warning("Database returned empty DF.")
            return np.nan
        db_data["datetime"] = db_data["datetime_corrected"]
        features = calculate_current_features(db_data, feature_order=settings.feature_order)
        prediction = model.predict(features)
        return prediction[-1]
    except ValidationError as exception:
        logger.error(f"Validation error in input data: {exception}")
        return np.nan
    except Exception as exception:
        logger.exception(f"Unexpected prediction error: {exception}")
        return np.nan
