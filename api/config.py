from collections.abc import Iterable
from pathlib import Path

from pydantic_settings import BaseSettings

from api import BASE_DIR


class Settings(BaseSettings):
    """Application settings."""

    model_path: str | Path = BASE_DIR / "fitted_models/best_xgb.pkl"

    postgres_user: str = "admin"
    postgres_password: str = "admin"
    postgres_db: str = "draslovka"
    postgres_table: str = "measurements"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    dt_format: str = "%Y-%m-%d %H:%M:%S"

    feature_order: Iterable = (
        "gas1",
        "gas1_lag1",
        "gas1_lag2",
        "gas1_lag3",
        "gas1_lag4",
        "gas1_lag5",
        "gas1_mean",
        "gas1_std",
        "gas1_max",
        "gas1_min",
        "gas1_last",
        "gas1_first",
        "prev_end_t_1",
        "prev_end_t_2",
        "prev_end_t_3",
        "prev_end_t_4",
        "elapsed",
    )

    def post_init(self) -> None:
        """Post-initialization hook to convert model_path to Path object."""
        self.model_path = Path(self.model_path)


settings = Settings()
