import math
from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # (Senin Haversine kodun aynı kalıyor, sadece uzaklık ölçmek için)
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def partition_into_days(scored_places: List[Dict], total_days: int, max_walk_per_day: float) -> List[Dict]:
    if not scored_places:
        return []

    # Eğer mekan sayısı gün sayısından azsa, gün sayısını mekan sayısına eşitle
    num_clusters = min(total_days, len(scored_places))

    # SADECE CUSTOM OLMAYANLARI KÜMELE
    # Kullanıcının eklediği (custom) mekanların koordinatları genelde alakasız olur,
    # onları modele bozmaması için önce ayırıyoruz.
    ml_places = [p for p in scored_places if p.get("category", "") != "custom"]
    custom_places = [p for p in scored_places if p.get("category", "") == "custom"]

    if len(ml_places) >= num_clusters:
        # YENİ: K-Means ile haritayı bölgelere ayırma (Makine Öğrenmesi)
        # Sadece enlem ve boylamları alıyoruz
        coords = np.array([[p["lat"], p["lng"]] for p in ml_places])

        # KMeans modelini oluştur (random_state hep aynı sonucu versin diye 42)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)

        # Mekanlara hangi güne (kümeye) ait olduklarını yazıyoruz
        for i, place in enumerate(ml_places):
            place["assigned_day"] = labels[i] + 1  # labels 0'dan başlar, gün 1'den
    else:
        # Mekan azsa dümdüz 1. güne koy gitsin
        for place in ml_places:
            place["assigned_day"] = 1

    # Custom mekanları da şimdilik 1. güne veya uygun bir güne ekleyebiliriz
    for custom in custom_places:
        custom["assigned_day"] = 1
        ml_places.append(custom)

    itinerary = []

    # Her bir küme (gün) için rotayı oluştur
    for day_num in range(1, total_days + 1):
        day_places = [p for p in ml_places if p.get("assigned_day") == day_num]

        if not day_places:
            continue

        # Günü kendi içinde, rotayı optimize etmek (birbirine yakınları sıraya dizmek) için bir nevi TSP (Gezgin Satıcı)
        # çözümü olarak en yakın komşuya göre sıralıyoruz.
        ordered_day_places = []
        current_place = day_places.pop(0)
        ordered_day_places.append(current_place)
        current_distance = 0.0

        while day_places:
            # Kalanlar arasından en yakın mekanı bul
            nearest_idx = 0
            min_dist = float('inf')

            for i, p in enumerate(day_places):
                dist = calculate_distance(
                    current_place["lat"], current_place["lng"],
                    p["lat"], p["lng"]
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i

            # En yakını al, gün sonuna ekle, mesafeyi topla
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