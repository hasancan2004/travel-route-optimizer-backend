import os
import requests
import traceback
from fastapi import APIRouter, HTTPException
from typing import List
from app.database import supabase

router = APIRouter(tags=["Places & Radar"])
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")


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


@router.get("/spots/{city}")
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

                new_spot = {
                    "city": city_key,
                    "name": r.get("name", "Bilinmeyen Mekan"),
                    "category": map_google_type_to_category(r.get("types", [])),
                    "rating": float(r.get("rating", 3.0)),
                    "entry_fee": float(max(price_level * 100, 50.0)),
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

        return {"status": "success", "city": city_key, "total_spots": len(spots_data), "spots": spots_data}

    except Exception as e:
        error_msg = traceback.format_exc()
        print("💥 DEVASA HATA YAKALANDI:\n", error_msg)
        return {"status": "FATAL_ERROR", "detail": str(e), "traceback": error_msg}


@router.get("/radar")
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
        if price_level is None:
            price_level = 0

        new_spots.append({
            "city": "radar_live",
            "name": r.get("name"),
            "category": map_google_type_to_category(r.get("types", [])),
            "rating": float(r.get("rating", 3.0)),
            "entry_fee": float(max(price_level * 100, 0.0)),
            "lat": float(r["geometry"]["location"]["lat"]),
            "lng": float(r["geometry"]["location"]["lng"])
        })

    return {"status": "success", "total_spots": len(new_spots), "spots": new_spots}