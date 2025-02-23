import logging
from decimal import Decimal

import pandas as pd
import psycopg2

from api.config import settings
from api.models import Measurement

logger = logging.getLogger(__name__)


def fetch_and_validate_data(request_up_to: str):
    """Reads data from PostgreSQL, validates it with Pydantic, and returns a Pandas DataFrame."""
    try:
        logger.info("Connecting to PostgreSQL database.")
        conn = psycopg2.connect(
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            host=settings.postgres_host,
            port=settings.postgres_port,
        )
        cursor = conn.cursor()
        query = f"SELECT * FROM {settings.postgres_table} WHERE datetime <= '{request_up_to}';"
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
