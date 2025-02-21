import logging

import pandas as pd
from pydantic import ValidationError
from sqlalchemy import create_engine

from src import BASE_DIR
from src.config import settings
from src.models import Measurement

logger = logging.getLogger(__name__)


def validate_data(df: pd.DataFrame) -> list[dict]:
    """Validates a dataframe against the Measurement."""
    valid_rows = []
    for _, row in df.iterrows():
        try:
            validated = Measurement(**row.to_dict())
            valid_rows.append(validated.model_dump())
        except ValidationError as e:
            logger.error(f"Skipping invalid row: {row.to_dict()} | Error: {e}")
    return valid_rows


def insert_data(file_path: None | str, df: pd.DataFrame = None):
    """Reads a CSV file and inserts heat data into PostgreSQL."""
    try:
        if df is not None and file_path is not None:
            df = pd.read_csv(file_path, parse_dates=["datetime"])
        df["datetime_corrected"] = df["datetime"] - pd.Timedelta(seconds=settings.measurement_delay)
        valid_rows = validate_data(df)
        if valid_rows:
            engine = create_engine(settings.db_url)
            valid_df = pd.DataFrame(valid_rows)
            with engine.connect() as conn:
                valid_df.to_sql(settings.postgres_table, con=conn, if_exists="append", index=False)
            logger.info(f"Inserted {len(valid_rows)} rows into {settings.postgres_table} table.")
        else:
            logger.warning("No valid rows to insert.")
    except Exception as e:
        logger.error(f"Failed to insert data: {e}")


if __name__ == "__main__":
    print(BASE_DIR)
    insert_data(BASE_DIR / f"data/{settings.input_file}")