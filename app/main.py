from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.algorithms.place_scoring import score_places

app = FastAPI(
    title="Travel Route Optimizer API",
    description="Kullanıcı tercihlerine göre optimize edilmiş seyahat rotaları oluşturan backend servisi.",
    version="1.0.0"
)

class TripRequest(BaseModel):
    user_interests: List[str]
    max_budget: float
    places: List[dict]

@app.get("/")
def read_root():
    return {
        "status": "success",
        "message": "Travel Route Optimizer API tıkır tıkır çalışıyor! 🚀"
    }

@app.post("/optimize-route")
def optimize_route(request: TripRequest):
    """
    Kullanıcı verilerini alır, kural tabanlı skorlama algoritmasından geçirip 
    en yüksek puanlı mekanları sıralı şekilde döner.
    """
    scored_result = score_places(
        user_interests=request.user_interests,
        max_budget=request.max_budget,
        places=request.places
    )
    
    return {
        "status": "success",
        "total_evaluated": len(scored_result),
        "ranked_places": scored_result
    }