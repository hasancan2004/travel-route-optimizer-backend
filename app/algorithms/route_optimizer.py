import math
from typing import List, Dict

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    İki koordinat arasındaki kuş uçuşu mesafeyi kilometre cinsinden hesaplar (Haversine Formula).
    """
    R = 6371.0 # Dünya'nın yarıçapı (km)
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def partition_into_days(scored_places: List[Dict], total_Days: int, max_walk_per_day: float) -> List[Dict]:
    """
    Puanlanmış mekanları, günlük yürüme sınırına ve gün sayısına göre akıllıca günlere böler.
    """
    itinerary = []
    places_per_day = math.ceil(len(scored_places) / max(total_Days, 1))
    
    current_index = 0
    for day in range(1, total_Days + 1):
        day_places = scored_places[current_index:current_index + places_per_day]
        current_index += places_per_day
        
        # Günlük toplam mesafeyi ve süreyi hesapla
        day_distance = 0.0
        for i in range(len(day_places) - 1):
            p1 = day_places[i]
            p2 = day_places[i+1]
            day_distance += calculate_distance(p1["lat"], p1["lng"], p2["lat"], p2["lng"])
            
        itinerary.append({
            "day": day,
            "places": day_places,
            "estimated_walking_km": round(day_distance, 2)
        })
        
        if current_index >= len(scored_places):
            # Eğer mekanlar bittiyse döngüyü kır
            break
            
    return itinerary
