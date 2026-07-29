import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.algorithms.place_scoring import score_places
from app.algorithms.route_optimizer import partition_into_days

app = FastAPI(
    title="Travel Route Optimizer API",
    description="Kullanıcı tercihlerine göre optimize edilmiş seyahat rotaları oluşturan backend servisi.",
    version="1.0.0"
)

class TripRequest(BaseModel):
    user_interests: List[str]
    max_budget: float
    total_days: int          # Kaç gün kalınacağı
    max_walk_per_day: float  # Günlük max yürüme sınırı (km)
    places: List[dict]       # İçinde "lat" ve "lng" bilgileri de olan mekan listesi

@app.get("/")
def read_root():
    return {
        "status": "success",
        "message": "Travel Route Optimizer API tıkır tıkır çalışıyor! 🚀"
    }

@app.get("/spots/{city}")
def get_city_spots(city: str):
    """
    Belirtilen şehre ait mekan listesini data/spots.json dosyasından okur ve döner.
    Türkçe karakter ve büyük/küçük harf duyarlılığını unify eder.
    """
    file_path = os.path.join("data", "spots.json")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Mekan verisi dosyası (data/spots.json) bulunamadı.")
    
    with open(file_path, "r", encoding="utf-8") as f:
        spots_data = json.load(f)
        
    # Türkçe karakterleri düzgün küçük harfe çevirme ve boşlukları temizleme
    city_key = city.replace("İ", "i").replace("I", "ı").lower().strip()
    
    if city_key not in spots_data:
        raise HTTPException(status_code=404, detail=f"'{city}' şehri için kayıtlı mekan bulunamadı.")
        
    return {
        "status": "success",
        "city": city_key,
        "total_spots": len(spots_data[city_key]),
        "spots": spots_data[city_key]
    }

@app.post("/optimize-route")
def optimize_route(request: TripRequest):
    """
    Kullanıcı verilerini alır, kural tabanlı skorlamadan geçirir 
    ve Haversine formülü ile günlere bölünmüş optimize seyahat rotasını döner.
    """
    scored_result = score_places(
        user_interests=request.user_interests,
        max_budget=request.max_budget,
        places=request.places
    )
    
    itinerary = partition_into_days(
        scored_places=scored_result,
        total_Days=request.total_days,
        max_walk_per_day=request.max_walk_per_day
    )
    
    return {
        "status": "success",
        "total_evaluated": len(scored_result),
        "itinerary": itinerary
    }