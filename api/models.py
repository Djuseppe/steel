import datetime as dt
from typing import Optional

from pydantic import BaseModel


class PredictionInput(BaseModel):
    """Request schema for inference."""

    gas1: float
    datetime: dt.datetime
    heatid: int


class Measurement(BaseModel):
    """Response schema for inference."""

    heatid: int
    datetime: dt.datetime
    datetime_corrected: dt.datetime
    end_t: Optional[float]
    gas1: float
