import logging
from decimal import Decimal

import numpy as np
import pandas as pd
import psycopg2

from src.config import settings
from src.models import Measurement

logger = logging.getLogger(__name__)


def fetch_and_validate_data():
    """Reads data from PostgreSQL, validates it with Pydantic, and returns a Pandas DataFrame."""
    try:
        logger.info("Connecting to PostgreSQL database...")
        conn = psycopg2.connect(
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            host=settings.postgres_host,
            port=settings.postgres_port
        )
        cursor = conn.cursor()
        query = f"SELECT * FROM {settings.postgres_table};"
        cursor.execute(query)
        rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        validated_data = []
        for row in rows:
            row_dict = dict(zip(column_names, row))
            if isinstance(row_dict["end_t"], Decimal):
                row_dict["end_t"] = float(row_dict["end_t"])
            row_dict["datetime"] = row_dict["datetime"].isoformat()
            row_dict["datetime_corrected"] = row_dict["datetime_corrected"].isoformat()
            try:
                validated_record = Measurement(**row_dict)
                validated_data.append(validated_record.model_dump())
            except Exception as e:
                logger.warning(f"Validation failed for row {row_dict}: {e}")
        df = pd.DataFrame(validated_data)
        cursor.close()
        conn.close()
        logger.info("Data successfully retrieved and validated.")
        return df
    except Exception as e:
        logger.error(f"Error retrieving data: {e}")
        return pd.DataFrame()


class Processor:

    @staticmethod
    def read_data(file_path: None | str = None, source: str = "sql") -> pd.DataFrame:
        """Reads data from a CSV file or PostgreSQL database."""
        if source == "csv":
            return pd.read_csv(file_path)
        elif source == "sql":
            return fetch_and_validate_data()
        else:
            raise ValueError("Invalid source. Supported sources are 'csv' and 'sql'.")

    @staticmethod
    def _interpolate_zero_values(df: pd.DataFrame, col: str = "gas1", method: str = "linear") -> pd.DataFrame:
        """
        Interpolate zero values in the specified column of a DataFrame. This function
        replaces zero values in the specified column with NaN, then interpolates those
        values using the specified method. It is useful for cleaning data where zero
        values are considered invalid or should be replaced by interpolated values.

        :param data: The input DataFrame that contains the column to be interpolated.
        :type data: pd.DataFrame
        :param col: The column name in the DataFrame where zeros should be replaced and
            interpolated. Defaults to "gas1".
        :type col: str
        :param method: The method to use for interpolation (e.g., "linear", "spline",
            "polynomial", etc.). Defaults to "linear".
        :type method: str
        :return: A new DataFrame with zero values replaced and interpolated in the
            specified column.
        :rtype: pd.DataFrame
        """
        df = df.copy()
        df.loc[df[col] == 0, col] = np.nan
        df[col] = df[col].interpolate(method=method)
        return df

    @staticmethod
    def _calculate_historical_features(
            df: pd.DataFrame,
            gas_col: str = "gas1",
            temp_col: str = "end_t",
            n_lags: int = 5,
            n_prev_processes: int = 5) -> pd.DataFrame:
        """
        Calculates historical features for each unique "heatid" in the given dataframe. It computes statistical
        aggregations of the specified gas and temperature columns and generates lag-based features and
        elapsed time for each record. Furthermore, it includes features derived from previous processes
        based on `n_prev_processes`.
        :param df: Input pandas DataFrame containing the necessary data columns. The DataFrame must have
            at least "heatid" and "datetime" columns that are utilized for grouping and time-based computations.
        :param gas_col: Column name in the DataFrame representing the gas measurement. Default is "gas1".
        :param temp_col: Column name in the DataFrame representing the temperature measurement. Default is "end_t".
        :param n_lags: Number of lag-based features to generate for the gas measurement column.
        :param n_prev_processes: Number of previous process features to compute based on the temperature column.
        :return: A pandas DataFrame with the original data augmented by the computed historical features.
        :rtype: pd.DataFrame
        """
        df = df.sort_values(["heatid", "datetime"]).copy()
        features_per_heat_id = df.groupby("heatid").agg(
            gas1_mean=(gas_col, "mean"),
            gas1_std=(gas_col, "std"),
            gas1_max=(gas_col, "max"),
            gas1_min=(gas_col, "min"),
            gas1_last=(gas_col, "last"),
            gas1_first=(gas_col, "first"),
            end_t=(temp_col, "first"),
            heatid=("heatid", "first"),
        ).reset_index(drop=True)
        for lag in range(1, n_lags + 1):
            df[f"{gas_col}_lag{lag}"] = df.groupby("heatid")[gas_col].shift(lag)
        for i in range(1, n_prev_processes):
            features_per_heat_id[f"prev_end_t_{i}"] = features_per_heat_id["end_t"].shift(i).iloc[-1]
        features = pd.merge(left=df, right=features_per_heat_id.drop(columns=["end_t"]), on="heatid", how="left")
        features["elapsed"] = df.groupby("heatid")["datetime"].transform(lambda x: (x - x.min()).dt.total_seconds())
        return features

    @staticmethod
    def _add_process_status(
            df: pd.DataFrame,
            temp_threshold: int | float = settings.t_threshold,
            temp_band: int | float = settings.t_band,
    ) -> pd.DataFrame:
        """
        This function adds a process status column to the given DataFrame based on the temperature
        threshold and temperature band provided. It evaluates the "end_t" column of the DataFrame to
        determine if the value is higher, lower, or equal relative to the threshold and the band. NaN
        values in the "end_t" column will also result in NaN for the "status" column.

        :param df: Input pandas DataFrame that must include "heatid", "datetime", and "end_t" columns.
        :type df: pd.DataFrame
        :param temp_threshold: Temperature threshold used to evaluate the status of "end_t".
        :type temp_threshold: int | float
        :param temp_band: Temperature band to allow for a tolerance around the threshold value.
        :type temp_band: int | float
        :return: A pandas DataFrame with an additional "status" column indicating the relative process status.
        :rtype: pd.DataFrame
        """
        df = df.sort_values(["heatid", "datetime"]).copy()
        df["status"] = np.where(
            df["end_t"] > (temp_threshold + temp_band), "higher",
            np.where(df["end_t"] < (temp_threshold - temp_band), "lower", "equal")
        )
        df.loc[df["end_t"].isna(), "status"] = np.nan
        return df

    @staticmethod
    def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
        """Splits the data into training and testing sets."""
        X_train = data.loc[data["datetime"] < settings.train_split].drop(
            columns=["end_t", "status", "datetime"]
        )
        y_train = data.loc[data["datetime"] < settings.train_split, ["end_t"]]
        X_test = data.loc[data["datetime"] >= settings.train_split, ["gas1", "datetime", "status"]]
        y_test = data.loc[data["datetime"] >= settings.train_split, ["end_t"]]
        return X_train, y_train, X_test, y_test

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._interpolate_zero_values(df)
        df = self._calculate_historical_features(df)
        df = self._add_process_status(df)
        df["end_t"] = df.groupby("heatid")["end_t"].bfill()
        df["status"] = df.groupby("heatid")["status"].bfill()
        df["datetime"] = df["datetime_corrected"]
        df.drop(columns=["datetime_corrected"], inplace=True)
        df.sort_values(by="datetime", inplace=True)
        return df


if __name__ == "__main__":
    data = Processor().read_data(source="sql")
    data = Processor().process(data)
    print(data)
