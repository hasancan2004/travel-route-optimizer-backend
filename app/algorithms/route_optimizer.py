import math
from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))  # DÜZELTİLDİ: Çarpım işareti eklendi

    return R * c


def partition_into_days(scored_places: List[Dict], total_days: int, max_walk_per_day: float) -> List[Dict]:
    # Eğer hiç mekan yoksa bile, kullanıcının talep ettiği gün sayısı kadar boş şablon döndür
    if not scored_places:
        return [{"day": d, "places": [], "estimated_walking_km": 0.0} for d in range(1, total_days + 1)]

    ml_places = [p for p in scored_places if p.get("category", "") != "custom"]
    custom_places = [p for p in scored_places if p.get("category", "") == "custom"]

    # KMeans'in çökmemesi için küme sayısını en fazla mekan sayısı kadar yapabiliriz
    num_clusters = min(total_days, len(ml_places))

    if num_clusters > 0:
        coords = np.array([[p["lat"], p["lng"]] for p in ml_places])
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)

        for i, place in enumerate(ml_places):
            place["assigned_day"] = labels[i] + 1

    for custom in custom_places:
        custom["assigned_day"] = 1
        ml_places.append(custom)

    itinerary = []

    # KESİNLİKLE kullanıcının istediği total_days kadar dön
    for day_num in range(1, total_days + 1):
        day_places = [p for p in ml_places if p.get("assigned_day") == day_num]

        # ==========================================
        # YENİ: GÜNLÜK KATEGORİ KOTASI SİSTEMİ (CAP)
        # ==========================================
        # Mekanları önce puana göre büyükten küçüğe sırala ki düşük puanlılar elensin
        day_places.sort(key=lambda x: x.get("rating", 0.0), reverse=True)

        capped_day_places = []
        category_counts = {"food": 0, "history": 0, "nature": 0, "shopping": 0}

        for p in day_places:
            cat = p.get("category", "unknown").lower()

            # Kullanıcının eklediği mekanlar kotaya takılmaz
            if cat == "custom":
                capped_day_places.append(p)
                continue

            # Limit Kontrolleri
            if cat == "food" and category_counts.get("food", 0) >= 2:
                continue  # Günde max 2 restoran (Öğle ve Akşam)
            if cat == "history" and category_counts.get("history", 0) >= 3:
                continue  # Günde max 3 tarihi yer
            if cat == "nature" and category_counts.get("nature", 0) >= 3:
                continue  # Günde max 3 doğa parkı/alanı
            if cat == "shopping" and category_counts.get("shopping", 0) >= 2:
                continue  # Günde max 2 alışveriş yeri

            # Limite takılmadıysa listeye al ve sayacı artır
            capped_day_places.append(p)
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Filtrelenmiş ve kotadan geçmiş temiz mekan listemizi güncelleyelim
        day_places = capped_day_places
        # ==========================================

        # Eğer filtrelemeden sonra bu güne mekan kalmadıysa, günü silmek yerine boş gün olarak ekle
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

        # TSP (En Yakın Komşu) Algoritması - Mesafeyi kısaltmak için mekanları mantıklı sırala
        while day_places:
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