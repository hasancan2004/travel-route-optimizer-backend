import os, uvicorn
import requests
import traceback
import json
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Algoritmalar ve veritabanı
from app.algorithms.place_scoring import score_places
from app.algorithms.route_optimizer import partition_into_days
from app.database import supabase, save_itinerary_to_supabase, share_itinerary_to_supabase, \
    get_public_itineraries_from_supabase, get_user_itineraries_from_supabase
from dotenv import load_dotenv

load_dotenv()

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="Travel Route Optimizer API",
    description="Kullanıcı tercihlerine göre optimize edilmiş seyahat rotaları oluşturan backend servisi.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TripRequest(BaseModel):
    user_interests: List[str]
    max_budget: float
    total_days: int
    max_walk_per_day: float
    places: List[dict]


class SaveTripRequest(BaseModel):
    user_id: str
    city: str
    max_budget: float
    itinerary: List[dict]


class ShareTripRequest(BaseModel):
    user_id: str
    author_name: str
    city: str
    title: str
    max_budget: float
    itinerary: List[dict]


# YENİ: Yapay Zeka için İstek Modeli
class AIPromptRequest(BaseModel):
    prompt: str


def map_google_type_to_category(types: List[str]) -> str:
    history_keywords = ["museum", "historic_site", "place_of_worship", "mosque", "church", "hindu_temple", "synagogue"]
    nature_keywords = ["park", "natural_feature", "zoo", "aquarium", "campground"]
    food_keywords = ["restaurant", "cafe", "bakery", "food", "bar"]
    shopping_keywords = ["shopping_mall", "store", "supermarket", "clothing_store"]

    for t in types:
        if t in history_keywords: return "history"
        if t in nature_keywords: return "nature"
        if t in food_keywords: return "food"
        if t in shopping_keywords: return "shopping"

    return "custom"


@app.get("/")
def read_root():
    return {
        "status": "success",
        "message": "Travel Route Optimizer API (Supabase, ML, AI & Google Places) tıkır tıkır çalışıyor! 🚀"
    }


# YENİ: Gemini AI NLP Asistan Endpoint'i (Dinamik Model Seçimli)
@app.post("/ai-analyze-prompt")
def analyze_prompt_with_ai(request: AIPromptRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key eksik veya okunamadı!")

    try:
        # 1. Google'a direkt soruyoruz: "Şu an hangi modeller hayatta ve metin üretebiliyor?"
        valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]

        if not valid_models:
            raise HTTPException(status_code=500, detail="API anahtarınızda aktif hiçbir model bulunamadı.")

        # 2. İçinde 'flash' geçen en hızlı modeli bul, yoksa listedeki ilk modeli al
        target_model = next((m for m in valid_models if 'flash' in m), valid_models[0])
        model_name = target_model.replace('models/', '')  # 'models/' önekini temizliyoruz

        print(f"🤖 Otomatik Seçilen AI Modeli: {model_name}")
        model = genai.GenerativeModel(model_name)

        system_instruction = """
        Sen akıllı bir seyahat asistanısın. Kullanıcının girdiği serbest metni analiz edip, rota algoritmasının anlayacağı parametreleri çıkaracaksın.
        SADECE VE SADECE JSON formatında çıktı ver. Hiçbir ekstra açıklama, selamlama veya markdown formatı kullanma. Çıktı doğrudan parse edilebilir saf JSON olmalı.
        Eğer kullanıcı bazı verileri (bütçe, gün) belirtmemişse mantıklı varsayılan değerler ata (Bütçe: 1500, Gün: 2).

        Format Kuralları:
        {
            "city": "Şehrin İngilizce karakterli, küçük harfli hali (örn: istanbul, konya, izmir)",
            "max_budget": Sayısal değer (sadece rakam),
            "total_days": Sayısal değer (sadece rakam),
            "user_interests": ["history", "nature", "food", "shopping"] listesinden metne uyanlar.
        }
        """

        full_prompt = f"{system_instruction}\n\nKullanıcının Mesajı: {request.prompt}"
        response = model.generate_content(full_prompt)

        # AI'ın markdown veya kod bloğu döndürme ihtimaline karşı temizlik yapıyoruz
        raw_text = response.text.replace('```json', '').replace('```', '').strip()
        parsed_data = json.loads(raw_text)

        return {
            "status": "success",
            "message": "AI metni başarıyla analiz etti 🧠",
            "data": parsed_data
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI JSON formatında geçerli bir yanıt veremedi.")
    except Exception as e:
        error_msg = traceback.format_exc()
        print("💥 AI HATASI:\n", error_msg)
        raise HTTPException(status_code=500, detail=f"AI Analiz Hatası: {str(e)}")


@app.get("/spots/{city}")
def get_city_spots(city: str):
    try:
        city_key = city.replace("İ", "i").replace("I", "ı").lower().strip()

        try:
            response = supabase.table("spots").select("*").eq("city", city_key).execute()
            spots_data = response.data
        except Exception as e:
            return {"status": "error", "detail": f"Supabase okuma hatası: {str(e)}"}

        if not spots_data:
            if not GOOGLE_PLACES_API_KEY:
                return {"status": "error", "detail": "Google API Key eksik veya okunamadı!"}

            print(f"[{city_key}] Google'dan çekiliyor...")
            search_query = f"top tourist attractions and restaurants in {city}"
            url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={search_query}&key={GOOGLE_PLACES_API_KEY}"

            g_resp = requests.get(url)
            if g_resp.status_code != 200:
                return {"status": "error", "detail": f"Google API Hatası: {g_resp.text}"}

            results = g_resp.json().get("results", [])
            if not results:
                return {"status": "error", "detail": "Google bu şehir için mekan bulamadı."}

            new_spots = []
            for r in results:
                price_level = r.get("price_level", 0)
                if price_level is None:
                    price_level = 0

                entry_fee = float(max(price_level * 100, 50.0))

                new_spot = {
                    "city": city_key,
                    "name": r.get("name", "Bilinmeyen Mekan"),
                    "category": map_google_type_to_category(r.get("types", [])),
                    "rating": float(r.get("rating", 3.0)),
                    "entry_fee": entry_fee,
                    "lat": float(r["geometry"]["location"]["lat"]),
                    "lng": float(r["geometry"]["location"]["lng"])
                }
                new_spots.append(new_spot)

            try:
                inserted = supabase.table("spots").insert(new_spots).execute()
                spots_data = inserted.data
                print(f"[{city_key}] için {len(spots_data)} adet mekan Supabase'e kaydedildi!")
            except Exception as e:
                return {"status": "error", "detail": f"Supabase yazma hatası: {str(e)}"}

        return {
            "status": "success",
            "city": city_key,
            "total_spots": len(spots_data),
            "spots": spots_data
        }

    except Exception as e:
        error_msg = traceback.format_exc()
        print("💥 DEVASA HATA YAKALANDI:\n", error_msg)
        return {
            "status": "FATAL_ERROR",
            "detail": str(e),
            "traceback": error_msg
        }


@app.post("/optimize-route")
def optimize_route(request: TripRequest):
    scored_result = score_places(
        user_interests=request.user_interests,
        max_budget=request.max_budget,
        places=request.places
    )

    itinerary = partition_into_days(
        scored_places=scored_result,
        total_days=request.total_days,
        max_walk_per_day=request.max_walk_per_day
    )

    return {
        "status": "success",
        "total_evaluated": len(scored_result),
        "itinerary": itinerary
    }


@app.post("/save-itinerary")
def save_itinerary(request: SaveTripRequest):
    try:
        result = save_itinerary_to_supabase(
            user_id=request.user_id,
            city=request.city,
            total_budget=request.max_budget,
            itinerary_data=request.itinerary
        )
        return {"status": "success", "message": "Rota başarıyla buluta kaydedildi!", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Veritabanı Kayıt Hatası: {str(e)}")


@app.post("/share-itinerary")
def share_itinerary(request: ShareTripRequest):
    try:
        result = share_itinerary_to_supabase(
            user_id=request.user_id,
            author_name=request.author_name,
            city=request.city,
            title=request.title,
            total_budget=request.max_budget,
            itinerary_data=request.itinerary
        )
        return {"status": "success", "message": "Rota başarıyla toplulukta paylaşıldı! 🌍", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Paylaşım Hatası: {str(e)}")


@app.get("/explore-itineraries")
def explore_itineraries():
    try:
        itineraries = get_public_itineraries_from_supabase()
        return {
            "status": "success",
            "total": len(itineraries),
            "itineraries": itineraries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rotalar getirilirken hata oluştu: {str(e)}")


@app.get("/radar")
def get_radar_spots(lat: float, lng: float, radius: int = 1500):
    if not GOOGLE_PLACES_API_KEY:
        raise HTTPException(status_code=500, detail="Google Places API Key eksik.")

    print(f"📡 Radar Taraması Başladı: {lat}, {lng} (Yarıçap: {radius}m)")

    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius={radius}&key={GOOGLE_PLACES_API_KEY}"

    g_resp = requests.get(url)
    if g_resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Google Radar API'ye ulaşılamadı.")

    results = g_resp.json().get("results", [])

    new_spots = []
    for r in results[:10]:
        price_level = r.get("price_level", 0)
        entry_fee = float(max(price_level * 100, 0.0))

        new_spot = {
            "city": "radar_live",
            "name": r.get("name"),
            "category": map_google_type_to_category(r.get("types", [])),
            "rating": float(r.get("rating", 3.0)),
            "entry_fee": entry_fee,
            "lat": float(r["geometry"]["location"]["lat"]),
            "lng": float(r["geometry"]["location"]["lng"])
        }
        new_spots.append(new_spot)

    return {
        "status": "success",
        "total_spots": len(new_spots),
        "spots": new_spots
    }


@app.get("/user-itineraries/{user_id}")
def get_user_itineraries(user_id: str):
    try:
        user_trips = get_user_itineraries_from_supabase(user_id)

        formatted_itineraries = []
        for trip in user_trips:
            formatted_itineraries.append({
                "itinerary": trip.get("route_json", [])
            })

        return {
            "status": "success",
            "total": len(formatted_itineraries),
            "itineraries": formatted_itineraries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kullanıcı rotaları getirilirken hata oluştu: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)