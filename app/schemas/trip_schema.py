from pydantic import BaseModel
from typing import List

class TripRequest(BaseModel):
    user_interests: List[str]  # Örn: ["history", "museum"]
    max_budget: float          # Örn: 1500.0
    places: List[dict]         # Şimdilik test için ham mekan listesi