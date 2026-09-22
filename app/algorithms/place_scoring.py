from typing import List, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors


def score_places(user_interests: List[str], max_budget: float, places: List[Dict]) -> List[Dict]:
    if not places:
        return []

    interests_lower = [i.lower() for i in user_interests] if user_interests else []

    features = []
    valid_places = []

    for place in places:
        category = place.get("category", "").lower()

        # 1. İlgi Alanı Uyuşması (Artık mekanları silmiyoruz, puanla ağırlıklandırıyoruz)
        # Kullanıcı spesifik bir şey belirtmediyse hepsi eşit (1.0).
        # Belirttiyse tam uyanlar 1.0, uymayanlar 0.3, custom mekanlar 0.5 alır.
        if not interests_lower:
            match_feature = 1.0
        elif category in interests_lower:
            match_feature = 1.0
        elif category == "custom":
            match_feature = 0.5
        else:
            match_feature = 0.3

        # 2. Rating Özelliği (0.0 - 1.0 arası)
        try:
            rating = float(place.get("rating", 3.0))
        except (ValueError, TypeError):
            rating = 3.0
        rating_feature = rating / 5.0

        # 3. Fiyat Özelliği (Ücretsizse 1.0, pahalılaştıkça düşer)
        try:
            fee = float(place.get("entry_fee", 0.0))
        except (ValueError, TypeError):
            fee = 0.0
        fee_feature = 1.0 if fee == 0 else max(0.1, 1.0 - (fee / 200.0))

        features.append([match_feature, rating_feature, fee_feature])
        valid_places.append(place)

    if not valid_places:
        return []

    # KNN (K-Nearest Neighbors) ile "İdeal Mekan" Arayışı
    ideal_place = np.array([[1.0, 1.0, 1.0]])
    X = np.array(features)

    n_neighbors = min(len(valid_places), 100)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean').fit(X)
    distances, indices = nbrs.kneighbors(ideal_place)

    scored_places = []
    for i, original_idx in enumerate(indices[0]):
        place = valid_places[original_idx]
        ml_score = round(max(0, 10.0 - (distances[0][i] * 5)), 2)
        place["calculated_score"] = ml_score
        scored_places.append(place)

    # Bütçe Filtresi
    final_places = []
    current_spent = 0.0

    for place in scored_places:
        fee = float(place.get("entry_fee", 0.0))
        if current_spent + fee <= max_budget:
            final_places.append(place)
            current_spent += fee

    # GÜVENLİK AĞI: Eğer bütçe aşırı kısıtlı kalıp listeyi sıfırladıysa, rotanın boş çıkmaması için en iyi mekanları ekle
    if not final_places and scored_places:
        final_places = scored_places[:6]

    return final_places