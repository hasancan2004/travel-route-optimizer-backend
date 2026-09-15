from typing import List, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors


def score_places(user_interests: List[str], max_budget: float, places: List[Dict]) -> List[Dict]:
    if not places:
        return []

    interests_lower = [i.lower() for i in user_interests]

    # 1. Özellik Çıkarımı (Feature Engineering)
    # Makine öğrenmesi metin (tarih, doğa) anlamaz, sayı anlar.
    # Bu yüzden her mekan için bir vektör (array) oluşturuyoruz.
    # Örnek Vektör: [İlgi_Alanı_Uyuşması, Puan(Rating), Ücretsiz_Mi]

    features = []
    valid_places = []

    for place in places:
        category = place.get("category", "").lower()

        # İlgi alanında olmayanları hala eziyoruz
        if category not in interests_lower and category != "custom":
            continue

        valid_places.append(place)

        # Özellik 1: İlgi Alanı Uyuşması (1 veya 0)
        match_feature = 1.0 if category in interests_lower else 0.5  # Custom ise yarım puan

        # Özellik 2: Normalize Edilmiş Rating (0.0 - 1.0 arası)
        rating = float(place.get("rating", 0.0))
        rating_feature = rating / 5.0

        # Özellik 3: Fiyat Özelliği (Ucuzsa 1, pahalıysa 0'a yaklaşır)
        fee = float(place.get("entry_fee", 0.0))
        fee_feature = 1.0 if fee == 0 else max(0.1, 1.0 - (fee / 200.0))

        features.append([match_feature, rating_feature, fee_feature])

    if not valid_places:
        return []

    # 2. KNN (K-Nearest Neighbors) ile "Kusursuz Mekan" Arayışı
    # Hedefimiz (İdeal Mekan): İlgi alanıyla tam uyuşan(1.0), 5 yıldızlı(1.0) ve Ücretsiz(1.0) mekan.
    ideal_place = np.array([[1.0, 1.0, 1.0]])
    X = np.array(features)

    # Scikit-learn KNN modelini çalıştır
    # En iyi 100 mekanı (veya geçerli mekan sayısını) bizim ideal profilimize ne kadar yakın diye ölçer.
    n_neighbors = min(len(valid_places), 100)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean').fit(X)

    # İdeal mekana olan uzaklıklarını (distances) ve indekslerini (indices) al
    distances, indices = nbrs.kneighbors(ideal_place)

    # 3. Model Çıktısını Puanlamaya Çevir (Mesafe ne kadar KÜÇÜKSE mekan o kadar İYİDİR)
    scored_places = []

    # indices[0] bize en iyiden en kötüye sıralı listeyi verir
    for i, original_idx in enumerate(indices[0]):
        place = valid_places[original_idx]

        # Uzaklığı skora çevir (Ters orantı)
        # Euclidian distance genelde 0-2 arasıdır. Biz 10 üzerinden bir ML skoru üretiyoruz
        ml_score = round(max(0, 10.0 - (distances[0][i] * 5)), 2)
        place["calculated_score"] = ml_score
        scored_places.append(place)

    # 4. BÜTÇE FİLTRESİ: ML tarafından sıralanmış listeden bütçeye uyanları al
    final_places = []
    current_spent = 0.0

    for place in scored_places:
        entry_fee = float(place.get("entry_fee", 0.0))
        if current_spent + entry_fee <= max_budget:
            final_places.append(place)
            current_spent += entry_fee

    return final_places