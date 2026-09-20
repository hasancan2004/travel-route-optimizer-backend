import os
from dotenv import load_dotenv
from supabase import create_client, Client

# .env dosyasındaki ortam değişkenlerini yükle
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("HATA: SUPABASE_URL veya SUPABASE_KEY bulunamadı. Lütfen .env dosyanızı kontrol edin.")

# Supabase istemcisini oluştur
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Supabase bağlantısı başarıyla kuruldu!")
except Exception as e:
    print(f"❌ Supabase bağlantı hatası: {e}")


# YENİ: Flutter'dan veya ML algoritmalarından gelen rotayı Supabase'e JSON olarak kaydeder
def save_itinerary_to_supabase(user_id: str, city: str, total_budget: float, itinerary_data: list):
    data = {
        "user_id": user_id,
        "city": city,
        "total_budget": total_budget,
        "route_json": itinerary_data  # PostgreSQL JSONB formatında tutulacak
    }

    # 'itineraries' tablosuna insert atıyoruz
    response = supabase.table("itineraries").insert(data).execute()
    return response.data

# 1. Rotayı Topluluk Havuzuna (public_itineraries) Paylaşma Fonksiyonu
def share_itinerary_to_supabase(user_id: str, author_name: str, city: str, title: str, total_budget: float, itinerary_data: list):
    data = {
        "user_id": user_id,
        "author_name": author_name,
        "city": city,
        "title": title,
        "total_budget": total_budget,
        "route_json": itinerary_data
    }

    response = supabase.table("public_itineraries").insert(data).execute()
    return response.data

# 2. Keşfet Ekranı İçin Paylaşılan Tüm Rotaları Çekme Fonksiyonu (Güvenli Hale Getirildi)
def get_public_itineraries_from_supabase():
    try:
        # Eğer 'created_at' sütunu tabloda yoksa sıralamayı kaldırıp direkt çekelim ki 500 patlamasın
        response = supabase.table("public_itineraries").select("*").execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"⚠️ Keşfet rotaları çekilirken uyarı/hata: {e}")
        # Tablo boşsa veya hata alırsak uygulamanın çökmemesi için boş liste dönüyoruz
        return []

# 3. YENİ: Kullanıcının kendi kaydettiği rotaları çekme fonksiyonu (Cloud-First Sync İçin)
def get_user_itineraries_from_supabase(user_id: str):
    try:
        response = supabase.table("itineraries").select("*").eq("user_id", user_id).execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"⚠️ Kullanıcı rotaları çekilirken hata: {e}")
        return []