import datetime as dt
import logging

import pandas as pd
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    input_file_path: str = "data/input/data.csv"

    postgres_user: str = "admin"
    postgres_password: str = "admin"
    postgres_db: str = "draslovka"
    postgres_table: str = "measurements"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    debug: bool = False
    num_threads: int = 8

    input_file: str = "input.csv"

    t_threshold: int = 1680
    t_band: int = 10
    cost_below: int = 100
    cost_above: int = 10

    measurement_delay: int = 5
    avg_elapsed: int = 819 - 20

    # HP
    n_splits: int = 2
    n_trials: int = 2
    train_split: dt.datetime | pd.Timestamp = pd.Timestamp("2023-01-29 00:00:00")
    test_split: dt.datetime | pd.Timestamp = pd.Timestamp("2023-01-30 00:00:00")

    @property
    def db_url(self) -> str:
        """PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


class Config:
    env_file = ".env"


def configure_logger() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


settings = Settings()
