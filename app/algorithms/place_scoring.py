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

        # 1. İlgi Alanı Uyuşması
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

        # DÜZELTİLDİ: Operatör hatası giderildi (1.0 - (fee / 200.0))
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

        # DÜZELTİLDİ: Operatör hatası giderildi (10.0 - (distances...))
        ml_score = round(max(0, 10.0 - (distances[0][i] * 5)), 2)
        place["calculated_score"] = ml_score
        scored_places.append(place)

    # ==========================================
    # YENİ: KATEGORİ DENGELEME (CATEGORY BALANCING)
    # KNN'in sadece restoranları seçmesini engeller, her kategoriden en iyileri harmanlar
    # ==========================================
    category_buckets = {"food": [], "history": [], "nature": [], "shopping": [], "custom": [], "other": []}

    # Mekanları önce kategorilerine göre sepetlere ayır
    for place in scored_places:
        cat = place.get("category", "other").lower()
        if cat in category_buckets:
            category_buckets[cat].append(place)
        else:
            category_buckets["other"].append(place)

    balanced_places = []

    # Custom mekanları (kullanıcının kendi ekledikleri) direkt al
    balanced_places.extend(category_buckets["custom"])

    # Her bir kategoriden en yüksek puanlı (ML Score) mekanları alıp harmanlıyoruz.
    # Bu sayede KNN sadece restoranlara boğulmak yerine, şehrin en iyi müzelerini ve parklarını da listeye sokmak zorunda kalır.
    max_items_per_category = 15  # Havuza her kategoriden maksimum 15 tane gönderelim

    for cat in ["history", "nature", "shopping", "food", "other"]:
        # Sepeti ML Puanına göre sırala
        category_buckets[cat].sort(key=lambda x: x.get("calculated_score", 0), reverse=True)
        # Sadece en iyilerini alıp havuza at
        balanced_places.extend(category_buckets[cat][:max_items_per_category])

    # Havuzu tekrar ML puanına göre genel bir sıralamaya sok
    balanced_places.sort(key=lambda x: x.get("calculated_score", 0), reverse=True)
    # ==========================================

    # Bütçe Filtresi
    final_places = []
    current_spent = 0.0

    for place in balanced_places:  # Artık scored_places yerine balanced_places kullanıyoruz
        fee = float(place.get("entry_fee", 0.0))
        if current_spent + fee <= max_budget:
            final_places.append(place)
            current_spent += fee

    # GÜVENLİK AĞI: Eğer bütçe aşırı kısıtlı kalıp listeyi sıfırladıysa, rotanın boş çıkmaması için en iyi mekanları ekle
    if not final_places and balanced_places:
        final_places = balanced_places[:6]

    return final_places