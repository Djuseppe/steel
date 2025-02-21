import datetime as dt
from typing import Optional

from pydantic import BaseModel


class Measurement(BaseModel):
    heatid: int
    datetime: dt.datetime
    datetime_corrected: dt.datetime
    end_t: Optional[float]
    gas1: float
