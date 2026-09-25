import os, uvicorn
import requests
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Rotaları (Router) içe aktarıyoruz (2. ve 3. adımda oluşturduğumuz dosyalar)
from app.api import places, trips

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Kapsamlı Model Listesi (Fallback Mantığı)
GEMINI_ENDPOINTS = [
    ("v1beta", "gemini-3.5-flash-lite"),
    ("v1beta", "gemini-3.5-flash"),
    ("v1beta", "gemini-3.8-flash"),
    ("v1beta", "gemini-1.5-flash"),  # Garanti çalışan stabil model
    ("v1beta", "gemini-1.5-pro"),  # Garanti çalışan stabil model
]

app = FastAPI(
    title="Travel Route Optimizer API",
    description="Supabase, ML ve Gemini AI entegreli, modüler seyahat rotası planlama servisi.",
    version="2.0.0"  # Clean Architecture Sürümü
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Harici dosyalara böldüğümüz mekan ve veritabanı rotalarını uygulamaya bağlıyoruz
app.include_router(places.router)
app.include_router(trips.router)


# --- YAPAY ZEKA (AI) BÖLÜMÜ --- #
class AIPromptRequest(BaseModel):
    prompt: str


def call_gemini_rest(api_version: str, model_name: str, prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 512}
    }
    resp = requests.post(url, headers=headers, params=params, json=body, timeout=60)
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code} [{api_version}/{model_name}]: {resp.text}")
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


@app.post("/ai-analyze-prompt", tags=["AI Integration"])
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
            raw_text = raw_text.strip().replace('```json', '').replace('```', '').strip()
            start = raw_text.find('{')
            end = raw_text.rfind('}') + 1
            if start != -1 and end > start:
                raw_text = raw_text[start:end]
            parsed_data = json.loads(raw_text)
            required_keys = {"city", "max_budget", "total_days", "user_interests"}
            if not required_keys.issubset(parsed_data.keys()):
                raise ValueError(f"Eksik JSON alanları: {required_keys - parsed_data.keys()}")
            city_raw = str(parsed_data.get("city", "")).strip()
            tr_map = str.maketrans("şçğüöıİŞÇĞÜÖ", "scguoiISCGUO")
            parsed_data["city"] = city_raw.lower().translate(tr_map)
            print(f"✅ AI başarılı ({model_name}): {parsed_data}")
            return {"status": "success", "message": "AI metni başarıyla analiz etti 🧠", "data": parsed_data}
        except (json.JSONDecodeError, ValueError) as e:
            last_error = f"Model {model_name} geçersiz JSON döndürdü: {str(e)}"
            print(f"⚠️ {last_error}")
            continue
        except Exception as e:
            last_error = str(e)
            print(f"💥 Model {model_name} HATASI: {last_error}")
            continue

    raise HTTPException(status_code=500,
                        detail=f"AI servisi şu an yanıt veremiyor. Lütfen tekrar deneyin. Hata: {last_error}")


@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {"status": "success", "message": "API tıkır tıkır çalışıyor! 🚀 (Modüler Mimari + AI)"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)