import math
from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# YENİ: max_places_per_day parametresini ekledik
def partition_into_days(scored_places: List[Dict], total_days: int, max_walk_per_day: float, max_places_per_day: int = 5) -> List[Dict]:
    if not scored_places:
        return [{"day": d, "places": [], "estimated_walking_km": 0.0} for d in range(1, total_days + 1)]

    num_clusters = min(total_days, len(scored_places))

    if num_clusters > 0:
        coords = np.array([[p.get("lat", 0.0), p.get("lng", 0.0)] for p in scored_places])
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)

        for i, place in enumerate(scored_places):
            place["assigned_day"] = int(labels[i]) + 1

    itinerary = []

    for day_num in range(1, total_days + 1):
        day_places = [p for p in scored_places if p.get("assigned_day") == day_num]

        day_places.sort(key=lambda x: (x.get("category", "") != "custom", -x.get("rating", 0.0)))

        capped_day_places = []
        category_counts = {"food": 0, "history": 0, "nature": 0, "shopping": 0}

        for p in day_places:
            # YENİ: Artık dışarıdan gelen dinamik limiti kullanıyor!
            if len(capped_day_places) >= max_places_per_day:
                break

            cat = p.get("category", "unknown").lower()

            if cat == "custom":
                capped_day_places.append(p)
                continue

            if cat == "food" and category_counts.get("food", 0) >= 2:
                continue
            if cat == "history" and category_counts.get("history", 0) >= 3:
                continue
            if cat == "nature" and category_counts.get("nature", 0) >= 3:
                continue
            if cat == "shopping" and category_counts.get("shopping", 0) >= 2:
                continue

            capped_day_places.append(p)
            category_counts[cat] = category_counts.get(cat, 0) + 1

        day_places = capped_day_places

        if not day_places:
            itinerary.append({
                "day": day_num,
                "places": [],
                "estimated_walking_km": 0.0
            })
            continue

        ordered_day_places = []
        current_place = day_places.pop(0)
        ordered_day_places.append(current_place)
        current_distance = 0.0

        while day_places:
            nearest_idx = 0
            min_dist = float('inf')

            for i, p in enumerate(day_places):
                dist = calculate_distance(
                    current_place.get("lat", 0.0), current_place.get("lng", 0.0),
                    p.get("lat", 0.0), p.get("lng", 0.0)
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i

            next_place = day_places.pop(nearest_idx)
            current_distance += min_dist
            ordered_day_places.append(next_place)
            current_place = next_place

        itinerary.append({
            "day": day_num,
            "places": ordered_day_places,
            "estimated_walking_km": round(current_distance, 2)
        })

    return itinerary