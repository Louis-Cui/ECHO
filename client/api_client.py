# HTTP 客户端 → 后端 FastAPI

import base64
import requests
from config import Config

API = Config.API_BASE


def health():
    """健康检查"""
    try:
        return requests.get(f"{API}/health", timeout=5).json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def asr(audio_path):
    """
    语音识别
    返回示例: {"text": "你好", "emotion": {"emotion": "happy", "confidence": 0.85}}
    """
    with open(audio_path, "rb") as f:
        resp = requests.post(f"{API}/asr/upload", files={"file": f}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def chat(text, user_id, emotion=None):
    """
    对话 + TTS
    返回示例: {
        "text": "今天过得怎么样？",
        "emotion": "happy",
        "tts": {"audio_base64": "xxx", "format": "wav", "duration_ms": 3000}
    }
    """
    body = {"text": text, "user_id": user_id}
    if emotion:
        body["emotion"] = emotion
    resp = requests.post(f"{API}/chat", json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()


def tts_only(text, emotion="neutral"):
    """仅 TTS（不用完整对话流程时）"""
    resp = requests.post(
        f"{API}/tts",
        json={"text": text, "emotion": emotion},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def register_user():
    """注册用户（主动关怀用）"""
    try:
        resp = requests.post(
            f"{API}/user/register",
            params={"user_id": Config.USER_ID},
            timeout=5,
        )
        return resp.ok
    except:
        return False
