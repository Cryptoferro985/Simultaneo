import os
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq
import edge_tts
import base64
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

VOICES = {
    "it": "it-IT-ElsaNeural",
    "en": "en-US-JennyNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ar": "ar-SA-ZariyahNeural",
    "th": "th-TH-PremwadeeNeural",
    "ko": "ko-KR-SunHiNeural",
}

LANG_NAMES = {
    "it": "Italian", "en": "English", "es": "Spanish",
    "fr": "French", "de": "German", "pt": "Portuguese",
    "zh": "Chinese", "ja": "Japanese", "ru": "Russian",
    "ar": "Arabic", "th": "Thai", "ko": "Korean",
}


class TranslateRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str


@app.post("/translate-tts")
async def translate_tts(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    # 1. Translate with LLaMA
    tgt_name = LANG_NAMES.get(req.tgt_lang, req.tgt_lang)
    src_name = LANG_NAMES.get(req.src_lang, req.src_lang)

    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a professional simultaneous interpreter from {src_name} to {tgt_name}. "
                    f"Translate the input naturally, preserving tone, rhythm and spoken register. "
                    f"Output ONLY the translation. No explanations, no quotes, no punctuation changes."
                ),
            },
            {"role": "user", "content": req.text},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    translated = chat.choices[0].message.content.strip()

    # 2. Synthesize with Edge TTS
    voice = VOICES.get(req.tgt_lang, "en-US-JennyNeural")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp_path = tmp.name

    communicate = edge_tts.Communicate(translated, voice)
    await communicate.save(tmp_path)

    with open(tmp_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    os.unlink(tmp_path)

    return {
        "translated": translated,
        "audio_b64": audio_b64,
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
