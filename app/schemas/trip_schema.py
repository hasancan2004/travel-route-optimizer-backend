from pydantic import BaseModel
from typing import List

class TripRequest(BaseModel):
    user_interests: List[str]
    max_budget: float
    total_days: int
    max_walk_per_day: float
    places: List[dict]

class SaveTripRequest(BaseModel):
    user_id: str
    city: str
    max_budget: float
    itinerary: List[dict]

class ShareTripRequest(BaseModel):
    user_id: str
    author_name: str
    city: str
    title: str
    max_budget: float
    itinerary: List[dict]