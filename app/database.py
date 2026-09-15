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