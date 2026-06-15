"""
FastAPI main entry — Digital Human Companion (数字人情感陪伴系统).

Routes:
  - /health          System health check
  - /asr             Speech-to-text (base64 audio)
  - /asr/upload      Speech-to-text (file upload)
  - /tts             Text-to-speech
  - /emotion/face    Facial emotion recognition
  - /emotion/fuse    Multi-modal emotion fusion
  - /chat            Full chat with agent
  - /memory/search   Search memories
  - /user/register   Register user for active care
  - /                Service root info
"""
from __future__ import annotations

import base64
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv

# ── Load .env from project root (before anything else) ─────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import (
    ASRInput,
    ASROutput,
    ChatInput,
    ChatOutput,
    EmotionLabel,
    EmotionOutput,
    TTSInput,
    TTSOutput,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("digital-companion")

# ═══════════════════════════════════════════════════════════════════
# Global module instances (initialised in lifespan)
# ═══════════════════════════════════════════════════════════════════
asr_engine: Optional["SenseVoiceASR"] = None  # noqa: F821
tts_engine: Optional["CosyVoiceTTS"] = None  # noqa: F821
fer_engine: Optional["FERClassifier"] = None  # noqa: F821
emotion_fusion: Optional["EmotionFusion"] = None  # noqa: F821
memory_rag: Optional["MemoryRAG"] = None  # noqa: F821
agent: Optional["CompanionAgent"] = None  # noqa: F821
care_scheduler: Optional["ActiveCareScheduler"] = None  # noqa: F821


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init on startup, shutdown on exit."""
    global asr_engine, tts_engine, fer_engine, emotion_fusion
    global memory_rag, agent, care_scheduler

    logger.info("=" * 50)
    logger.info("正在初始化数字人情感陪伴系统...")
    logger.info("=" * 50)

    # ── ASR (SenseVoiceSmall) ────────────────────────────────────
    try:
        from app.asr.sensevoice import SenseVoiceASR
        asr_engine = SenseVoiceASR(device="cpu")
        logger.info("[ASR] 模块初始化成功")
    except Exception as e:
        logger.warning("[ASR] 模块初始化失败（将在运行时重试）: %s", e)

    # ── TTS (CosyVoice) ──────────────────────────────────────────
    try:
        from app.tts.cosyvoice_tts import CosyVoiceTTS
        tts_engine = CosyVoiceTTS(device="cpu")
        logger.info("[TTS] 模块初始化成功")
    except Exception as e:
        logger.warning("[TTS] 模块初始化失败: %s", e)

    # ── FER (Facial Emotion) ─────────────────────────────────────
    try:
        from app.emotion.fer import FERClassifier
        fer_engine = FERClassifier()
        logger.info("[FER] 模块初始化成功")
    except Exception as e:
        logger.warning("[FER] 模块初始化失败: %s", e)

    # ── Emotion Fusion ───────────────────────────────────────────
    from app.emotion.fusion import EmotionFusion
    emotion_fusion = EmotionFusion()
    logger.info("[Fusion] 情绪融合模块初始化成功")

    # ── RAG Memory ───────────────────────────────────────────────
    from app.memory.rag import MemoryRAG
    memory_rag = MemoryRAG()
    logger.info("[Memory] RAG记忆模块初始化成功")

    # ── Agent ────────────────────────────────────────────────────
    from app.agent.workflow import CompanionAgent
    agent = CompanionAgent(memory=memory_rag, emotion_fusion=emotion_fusion)
    logger.info("[Agent] LangGraph智能体初始化成功")

    # ── Active Care Scheduler ────────────────────────────────────
    from app.schedule.care import ActiveCareScheduler
    care_scheduler = ActiveCareScheduler(agent=agent)
    care_scheduler.start()
    logger.info("[Schedule] 主动关怀调度器已启动")

    logger.info("=" * 50)
    logger.info("数字人情感陪伴系统启动完成 ✓")
    logger.info("=" * 50)

    yield  # ── Application runs here ──

    # ── Shutdown ─────────────────────────────────────────────────
    if care_scheduler:
        care_scheduler.stop()
        logger.info("[Schedule] 主动关怀调度器已停止")
    logger.info("系统关闭完成")


# ═══════════════════════════════════════════════════════════════════
# FastAPI Application
# ═══════════════════════════════════════════════════════════════════
app = FastAPI(
    title="数字人情感陪伴系统",
    description="基于 SenseVoiceSmall + CosyVoice + LangGraph 的情感陪伴数字人后端",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Service info."""
    return {
        "service": "数字人情感陪伴系统",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "/health",
    }


@app.get("/health")
async def health():
    """System health check — reports each module's availability."""
    return {
        "status": "ok",
        "modules": {
            "asr": asr_engine is not None,
            "tts": tts_engine is not None,
            "fer": fer_engine is not None,
            "fusion": emotion_fusion is not None,
            "memory": memory_rag is not None,
            "agent": agent is not None,
            "scheduler": care_scheduler is not None,
        },
    }


# ── OpenAI-compatible endpoint ──────────────────────────────────────
# 让 Human-Live2D 前端可以通过 OpenAI 协议调用我们的 Agent


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: dict):
    """OpenAI-compatible chat completions endpoint for Human-Live2D."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent模块未就绪")
    try:
        messages = request.get("messages", [])
        stream = request.get("stream", False)
        
        # Extract the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Call our agent
        result = agent.invoke(user_input=user_message, user_id="human-live2d")
        
        # Build OpenAI-compatible response
        response = {
            "id": f"chatcmpl-{id(result)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", "digital-companion"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        }
        return response
    except Exception as e:
        logger.exception("OpenAI endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


# ── ASR Endpoints ─────────────────────────────────────────────────

@app.post("/asr", response_model=ASROutput)
async def speech_to_text(input: ASRInput):
    """Speech-to-text from base64-encoded audio."""
    if not asr_engine:
        raise HTTPException(status_code=503, detail="ASR模块未就绪")
    try:
        audio_bytes = base64.b64decode(input.audio_base64)
        text, emotion = asr_engine.transcribe_bytes(audio_bytes, input.format)
        return ASROutput(text=text, emotion=emotion)
    except Exception as e:
        logger.exception("ASR endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/asr/upload", response_model=ASROutput)
async def speech_to_text_file(file: UploadFile = File(...)):
    """Speech-to-text from uploaded audio file."""
    if not asr_engine:
        raise HTTPException(status_code=503, detail="ASR模块未就绪")
    try:
        audio_bytes = await file.read()
        text, emotion = asr_engine.transcribe_bytes(audio_bytes)
        return ASROutput(text=text, emotion=emotion)
    except Exception as e:
        logger.exception("ASR upload endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


# ── TTS Endpoints ─────────────────────────────────────────────────

@app.post("/tts", response_model=TTSOutput)
async def text_to_speech(input: TTSInput):
    """Text-to-speech with optional emotion-based voice modulation."""
    if not tts_engine:
        raise HTTPException(status_code=503, detail="TTS模块未就绪")
    try:
        emotion = input.emotion or EmotionLabel.neutral
        audio_bytes, duration = tts_engine.synthesize(
            input.text, emotion, input.speaker
        )
        return TTSOutput(
            audio_base64=base64.b64encode(audio_bytes).decode(),
            format="wav",
            duration_ms=duration,
        )
    except Exception as e:
        logger.exception("TTS endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


# ── Emotion Endpoints ─────────────────────────────────────────────

@app.post("/emotion/face")
async def detect_face_emotion(file: UploadFile = File(...)):
    """Detect emotion from uploaded image."""
    if not fer_engine:
        raise HTTPException(status_code=503, detail="FER模块未就绪")
    try:
        image_bytes = await file.read()
        emotion = fer_engine.detect_emotion_from_bytes(image_bytes)
        return emotion
    except Exception as e:
        logger.exception("FER endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emotion/fuse", response_model=EmotionOutput)
async def fuse_emotions(
    text_emotion: Optional[EmotionOutput] = None,
    voice_emotion: Optional[EmotionOutput] = None,
    face_emotion: Optional[EmotionOutput] = None,
):
    """Fuse emotions from multiple modalities (text, voice, face)."""
    if not emotion_fusion:
        raise HTTPException(status_code=503, detail="Fusion模块未就绪")
    try:
        result = emotion_fusion.fuse(text_emotion, voice_emotion, face_emotion)
        return result
    except Exception as e:
        logger.exception("Fusion endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


# ── Chat Endpoint ─────────────────────────────────────────────────

@app.post("/chat", response_model=ChatOutput)
async def chat(input: ChatInput):
    """Full chat interaction — emotion analysis, memory, LLM response."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent模块未就绪")
    try:
        # Run the agent
        chat_result = agent.invoke(
            user_input=input.text,
            user_id=input.user_id,
            voice_emotion=input.emotion,
        )

        # Generate TTS if available
        tts_result = None
        if tts_engine:
            try:
                audio_bytes, duration = tts_engine.synthesize(
                    chat_result.text, chat_result.emotion
                )
                tts_result = TTSOutput(
                    audio_base64=base64.b64encode(audio_bytes).decode(),
                    format="wav",
                    duration_ms=duration,
                )
            except Exception as e:
                logger.warning("TTS generation in chat endpoint failed: %s", e)

        # Update care scheduler state
        if care_scheduler and input.emotion:
            care_scheduler.update_user_state(input.user_id, input.emotion.emotion)

        return ChatOutput(
            text=chat_result.text,
            emotion=chat_result.emotion,
            tts=tts_result,
        )
    except Exception as e:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


# ── Memory Endpoints ──────────────────────────────────────────────

@app.post("/memory/search")
async def search_memory(query: str, user_id: str, n_results: int = 5):
    """Semantic search over user memories."""
    if not memory_rag:
        raise HTTPException(status_code=503, detail="Memory模块未就绪")
    try:
        results = memory_rag.search(query, user_id, n_results)
        return {"results": results}
    except Exception as e:
        logger.exception("Memory search endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


# ── User Endpoints ────────────────────────────────────────────────

@app.post("/user/register")
async def register_user(user_id: str, name: Optional[str] = None):
    """Register a user for proactive care monitoring."""
    if not care_scheduler:
        raise HTTPException(status_code=503, detail="Scheduler模块未就绪")
    try:
        care_scheduler.register_user(user_id)
        return {"status": "ok", "user_id": user_id, "name": name}
    except Exception as e:
        logger.exception("User registration error")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info("Starting server on %s:%d", host, port)
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
