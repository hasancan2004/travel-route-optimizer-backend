from fastapi import APIRouter, HTTPException
from app.schemas.trip_schema import TripRequest, SaveTripRequest, ShareTripRequest
from app.algorithms.place_scoring import score_places
from app.algorithms.route_optimizer import partition_into_days
from app.database import (
    save_itinerary_to_supabase,
    share_itinerary_to_supabase,
    get_public_itineraries_from_supabase,
    get_user_itineraries_from_supabase
)
import traceback
import numpy as np  # YENİ: Numpy tiplerini yakalamak için ekledik

router = APIRouter(tags=["Trip Optimizer & Database"])


# YENİ ZIRH: ML Algoritmalarından gelen "NumPy" veri tiplerini saf Python tiplerine çevirir.
def clean_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [clean_numpy(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: clean_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_numpy(x) for x in obj]
    return obj


@router.post("/optimize-route")
def optimize_route(request: TripRequest):
    try:
        if not request.places or len(request.places) == 0:
            return {
                "status": "error",
                "detail": "Seçilen şehir için Google'dan mekan bulunamadı. Lütfen şehir adını kontrol edin."
            }

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

        # PATLAYAN YERİN ÇÖZÜMÜ: Göndermeden önce NumPy verilerini temizliyoruz
        safe_itinerary = clean_numpy(itinerary)

        return {
            "status": "success",
            "total_evaluated": len(scored_result),
            "itinerary": safe_itinerary
        }
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"💥 OPTİMİZASYON HATASI:\n{error_details}")
        return {
            "status": "error",
            "detail": f"Rota optimize edilirken algoritmik bir hata oluştu: {str(e)}"
        }


@router.post("/save-itinerary")
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


@router.post("/share-itinerary")
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


@router.get("/explore-itineraries")
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


@router.get("/user-itineraries/{user_id}")
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