from pydantic import BaseModel, Field
from typing import Dict

"""
Example Input:
{
  "year": 2022,
  "target_label": "value for Meat/seafood",
  "features": {
    "MEHOINUSGAA672N": 75000,
    "GARETAILNQGSP": 205000,
    "Population": 10700000
  }
}
"""


class ForecastRequest(BaseModel):
    year: int
    target_label: str  # eg: Tons of Meat/Seafood

    # {'MEHOINUSGAA672N': 72.5, 'Population': 11.1, ..}
    features: Dict[str, float] = Field(
        ..., example={
            "MEHOINUSGAA672N": 75000,
            "GARETAILNQGSP": 205000,
            "Population": 10700000
        }
    )

    class Config:
        json_schema_extra = {
            "example": {
                "year": 2022,
                "target_label": "value for Meat/seafood",
                "features": {
                    "MEHOINUSGAA672N": 65000,
                    "GARETAILNQGSP": 180000,
                    "Population": 10800000
                }
            }
        }
