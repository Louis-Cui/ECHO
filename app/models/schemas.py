"""Pydantic models for the digital human companion system."""
from __future__ import annotations

import enum
import time
from typing import Optional, Dict

from pydantic import BaseModel, Field


class EmotionLabel(str, enum.Enum):
    """Emotion categories recognized by the system."""
    happy = "happy"
    sad = "sad"
    angry = "angry"
    surprised = "surprised"
    fearful = "fearful"
    disgusted = "disgusted"
    neutral = "neutral"
    anxious = "anxious"
    love = "love"
    upset = "upset"


class EmotionOutput(BaseModel):
    """Unified emotion prediction output from any modality."""
    emotion: EmotionLabel
    confidence: float = Field(ge=0.0, le=1.0)
    cause: Optional[str] = None
    sources: Optional[Dict[str, float]] = None  # modality -> confidence


class ASRInput(BaseModel):
    """Speech-to-text input."""
    audio_base64: str
    format: str = "wav"


class ASROutput(BaseModel):
    """Speech-to-text output with emotion."""
    text: str
    emotion: EmotionOutput


class TTSInput(BaseModel):
    """Text-to-speech input."""
    text: str
    emotion: Optional[EmotionLabel] = None
    speaker: str = "default"


class TTSOutput(BaseModel):
    """Text-to-speech output."""
    audio_base64: str
    format: str = "wav"
    duration_ms: int


class ChatInput(BaseModel):
    """Chat request."""
    text: str
    user_id: str
    emotion: Optional[EmotionOutput] = None


class ChatOutput(BaseModel):
    """Chat response."""
    text: str
    emotion: EmotionLabel
    tts: Optional[TTSOutput] = None


class MemoryItem(BaseModel):
    """A single memory entry."""
    id: str
    user_id: str
    content: str
    emotion: EmotionLabel
    timestamp: float = Field(default_factory=time.time)
    weight: float = 1.0


class CareEvent(BaseModel):
    """A proactive care event triggered by the scheduler."""
    event_type: str  # greeting / checkin / consolation
    trigger_reason: str
    message: str


class ScheduleConfig(BaseModel):
    """Configuration for the active care scheduler."""
    check_interval_seconds: int = 300
    low_emotion_threshold: float = 0.6
    consecutive_low_count: int = 3
