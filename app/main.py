import os, uvicorn
import requests
import traceback
import json
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

# Senin orijinal ve en doğru listen: 2026 güncel Gemini modelleri
GEMINI_ENDPOINTS = [
    ("v1beta", "gemini-3.5-flash-lite"),
    ("v1beta", "gemini-3.5-flash"),
    ("v1beta", "gemini-3.8-flash"),
]

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


# Yapay Zeka için İstek Modeli
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


# Render'ın HEAD sağlık kontrolü ve GET istekleri için ortak rota
@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {
        "status": "success",
        "message": "Travel Route Optimizer API (Supabase, ML, AI & Google Places) tıkır tıkır çalışıyor! 🚀"
    }


def call_gemini_rest(api_version: str, model_name: str, prompt: str) -> str:
    """Gemini REST API'yi doğrudan çağırır. SDK gerektirmez."""
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    body = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "maxOutputTokens": 512,
            # DİKKAT: Yeni nesil 3.5+ modellerde 'temperature' desteği bittiği için kaldırıldı!
        }
    }
    # Timeout süresi 60 saniye
    resp = requests.post(url, headers=headers, params=params, json=body, timeout=60)

    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code} [{api_version}/{model_name}]: {resp.text}")

    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


@app.post("/ai-analyze-prompt")
def analyze_prompt_with_ai(request: AIPromptRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key eksik veya okunamadı!")

    system_instruction = (
        "You are a strict travel data extraction engine. "
        "Your ONLY job is to parse the user's message and output a single JSON object. "
        "NEVER greet the user, NEVER ask questions, NEVER explain anything. "
        "If you cannot extract a field, use the default value shown below. "
        "Your response must start with '{' and end with '}' — nothing else.\n\n"
        "RULES:\n"
        "1. total_days: extract the number of days mentioned (e.g. '2 günlük' → 2, '3 day' → 3). Default: 2.\n"
        "2. city: extract the city name in lowercase ASCII (no Turkish characters). "
        "   Turkish char map: ş→s, ç→c, ğ→g, ü→u, ö→o, ı→i, İ→i, Ş→s, Ç→c, Ğ→g, Ü→u, Ö→o. "
        "   Examples: Antalya→antalya, İstanbul→istanbul, Şanlıurfa→sanliurfa, "
        "   Kaleiçi Antalya→antalya, Konya Selçuklu→konya.\n"
        "3. max_budget: extract number if mentioned (e.g. '500 lira' → 500). Default: 1000.\n"
        "4. user_interests: list from ONLY these values: history, nature, food, shopping. "
        "   If not mentioned, infer from context (e.g. 'ucuz' → food, 'tarihi' → history). "
        "   Default: [\"history\", \"nature\"].\n\n"
        "FEW-SHOT EXAMPLES:\n"
        "User: 'bana antalyada 2 gunluk rota ayarlar misin?'\n"
        '→ {"city": "antalya", "max_budget": 1000, "total_days": 2, "user_interests": ["history", "nature"]}\n\n'
        "User: '3 günlük ucuz bir İstanbul turu çiz'\n"
        '→ {"city": "istanbul", "max_budget": 500, "total_days": 3, "user_interests": ["food", "history"]}\n\n'
        "User: 'Konya için 1 günlük rota'\n"
        '→ {"city": "konya", "max_budget": 1000, "total_days": 1, "user_interests": ["history"]}\n\n'
        "User: '2 günlük ucuz bir Konya turu çiz'\n"
        '→ {"city": "konya", "max_budget": 500, "total_days": 2, "user_interests": ["food", "history"]}\n\n'
        "OUTPUT FORMAT (strictly this and nothing else):\n"
        '{"city": "string", "max_budget": number, "total_days": number, "user_interests": ["string"]}'
    )

    full_prompt = f"{system_instruction}\n\nUser message: {request.prompt}"

    last_error = None
    for api_version, model_name in GEMINI_ENDPOINTS:
        try:
            print(f"🤖 Denenen AI Modeli (REST): {api_version}/{model_name}")
            raw_text = call_gemini_rest(api_version, model_name, full_prompt)

            # Markdown veya kod bloğu temizliği
            raw_text = raw_text.strip().replace('```json', '').replace('```', '').strip()

            # JSON bloğunu bul ({ ... })
            start = raw_text.find('{')
            end = raw_text.rfind('}') + 1
            if start != -1 and end > start:
                raw_text = raw_text[start:end]

            parsed_data = json.loads(raw_text)

            # Zorunlu alanlar var mı kontrol et
            required_keys = {"city", "max_budget", "total_days", "user_interests"}
            if not required_keys.issubset(parsed_data.keys()):
                raise ValueError(f"Eksik JSON alanları: {required_keys - parsed_data.keys()}")

            # Normalize city: lowercase + remove Turkish characters
            city_raw = str(parsed_data.get("city", "")).strip()
            tr_map = str.maketrans("şçğüöıİŞÇĞÜÖ", "scguoiISCGUO")
            parsed_data["city"] = city_raw.lower().translate(tr_map)

            print(f"✅ AI başarılı ({model_name}): {parsed_data}")
            return {
                "status": "success",
                "message": "AI metni başarıyla analiz etti 🧠",
                "data": parsed_data
            }

        except (json.JSONDecodeError, ValueError) as e:
            last_error = f"Model {model_name} geçersiz JSON döndürdü: {str(e)}"
            print(f"⚠️ {last_error}")
            continue
        except Exception as e:
            last_error = str(e)
            print(f"💥 Model {model_name} HATASI: {last_error}")
            continue

    # Hiçbir model çalışmadı
    raise HTTPException(
        status_code=500,
        detail=f"AI servisi şu an yanıt veremiyor. Lütfen tekrar deneyin. Hata: {last_error}"
    )


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