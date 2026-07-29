from typing import List, Dict

def score_places(user_interests: List[str], max_budget: float, places: List[Dict]) -> List[Dict]:
    """
    Kullanıcının ilgi alanlarına ve kısıtlarına göre mekanları puanlar ve sıralar.
    - İlgi alanı eşleşmesi: +3 puan
    - Kullanıcı puanı (rating) yüksekse: +2 puan
    - Bütçeyi aşıyorsa: -3 puan
    - Yürüme sınırı/uzaklık faktörü: -2 puan (şimdilik mock veri üzerinden)
    """
    scored_places = []

    for place in places:
        score = 0.0
        
        # 1. İlgi Alanı Eşleşmesi (+3 puan)
        place_category = place.get("category", "").lower()
        if place_category in [interest.lower() for interest in user_interests]:
            score += 3.0
            
        # 2. Mekan Popülaritesi / Kalitesi (+2 puana kadar)
        # Örn: 5 üzerinden rating değerini normalize edip ekleyebiliriz
        rating = place.get("rating", 0.0) # 0 ile 5 arası
        if rating >= 4.0:
            score += 2.0
        elif rating >= 3.0:
            score += 1.0

        # 3. Bütçe Kontrolü (-3 puan)
        entry_fee = place.get("entry_fee", 0.0)
        # Ortalama mekan bütçe sınırını aşıyorsa cezalandır
        if entry_fee > (max_budget / max(len(places), 1) * 1.5):
            score -= 3.0

        # Hesaplanan skoru mekana ekleyelim
        place["calculated_score"] = max(score, 0.0) # Skor 0'ın altına düşmesin
        scored_places.append(place)

    # En yüksek skordan en düşüğe doğru sırala (Sorting)
    scored_places.sort(key=lambda x: x["calculated_score"], reverse=True)
    
    return scored_places