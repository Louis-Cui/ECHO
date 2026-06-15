"""
CosyVoice TTS wrapper with emotion-aware voice parameter adjustment.

Supports inference_sft (single speaker) and inference_zero_shot (clone).
Emotion is mapped to speed/pitch adjustments for natural delivery.
"""
from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
from typing import Dict, Optional, Tuple

import soundfile as sf

from app.models.schemas import EmotionLabel

# ── Add cloned CosyVoice repo to path ──────────────────────────
_cosyvoice_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "CosyVoice",
)
if os.path.exists(os.path.join(_cosyvoice_path, "cosyvoice", "cli")):
    sys.path.insert(0, _cosyvoice_path)
    logger = logging.getLogger("digital-companion.tts")
    logger.info("已添加 CosyVoice 仓库路径: %s", _cosyvoice_path)

logger = logging.getLogger("digital-companion.tts")

# ── Emotion → voice parameter mapping ─────────────────────────────
# Keys: EmotionLabel  Values: {speed, pitch_shift}
EMOTION_VOICE_MAP: Dict[EmotionLabel, Dict[str, float]] = {
    EmotionLabel.happy: {"speed": 1.10, "pitch_shift": 1.20},
    EmotionLabel.sad: {"speed": 0.85, "pitch_shift": 0.80},
    EmotionLabel.angry: {"speed": 1.15, "pitch_shift": 1.30},
    EmotionLabel.neutral: {"speed": 1.00, "pitch_shift": 1.00},
    EmotionLabel.love: {"speed": 0.90, "pitch_shift": 1.10},
    EmotionLabel.anxious: {"speed": 1.20, "pitch_shift": 1.15},
    EmotionLabel.surprised: {"speed": 1.30, "pitch_shift": 1.40},
    EmotionLabel.fearful: {"speed": 0.95, "pitch_shift": 0.85},
    EmotionLabel.disgusted: {"speed": 0.90, "pitch_shift": 0.90},
}


class CosyVoiceTTS:
    """CosyVoice Text-to-Speech wrapper with emotion control."""

    def __init__(
        self,
        model_dir: str = "pretrained_models/CosyVoice2-0.5B",
        device: str = "cpu",
    ):
        self.model_dir = model_dir
        self.device = device
        self.model = None
        self.sample_rate = 22050
        self._init_model()

    def _init_model(self) -> None:
        """Initialise CosyVoice model from local path or download."""
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
            import torch

            # Try CosyVoice2 first, fallback to CosyVoice
            try:
                logger.info(
                    "Loading CosyVoice from: %s (device=%s)",
                    self.model_dir, self.device,
                )
                self.model = CosyVoice2(self.model_dir)
                logger.info("CosyVoice2 loaded successfully")
            except Exception:
                logger.info("CosyVoice2 not available, trying CosyVoice...")
                self.model = CosyVoice(self.model_dir)
                logger.info("CosyVoice loaded successfully")

            # Move to device
            if hasattr(self.model, "to"):
                self.model.to(torch.device(self.device))

            # Get sample rate from model if available
            if hasattr(self.model, "sample_rate"):
                self.sample_rate = self.model.sample_rate

        except ImportError as e:
            logger.error(
                "CosyVoice not installed. Clone and pip install:\n"
                "  git clone https://github.com/FunAudioLLM/CosyVoice.git\n"
                "  cd CosyVoice && pip install -r requirements.txt\n"
                "  Install cosyvoice package. Details: %s", e
            )
            raise
        except Exception as e:
            logger.error("Failed to initialise CosyVoice: %s", e)
            raise

    def _adjust_audio(
        self, audio: bytes, emotion: EmotionLabel
    ) -> Tuple[bytes, int]:
        """Apply emotion-based speed/pitch adjustments to raw audio.

        For a production system this would use librosa / pyrubberband.
        For MVP we return the raw audio and note the target params.
        """
        # TODO: Apply librosa.effects.pitch_shift / time_stretch
        # For now this is a pass-through stub.
        return audio, self.sample_rate

    def synthesize(
        self,
        text: str,
        emotion: EmotionLabel = EmotionLabel.neutral,
        speaker: str = "default",
    ) -> Tuple[bytes, int]:
        """Synthesise speech from text with emotion-aware voice params.

        Args:
            text: Input text to speak.
            emotion: Target emotion label.
            speaker: Speaker ID name.

        Returns:
            Tuple of (wav_bytes, duration_ms).
        """
        logger.info(
            "TTS synthesize: text=%s emotion=%s speaker=%s",
            text[:80], emotion.value, speaker,
        )

        if self.model is None:
            raise RuntimeError("CosyVoice model not initialised")

        voice_params = EMOTION_VOICE_MAP.get(emotion, EMOTION_VOICE_MAP[EmotionLabel.neutral])
        logger.info("Voice params: %s", voice_params)

        try:
            # ── Generate audio ────────────────────────────────────
            # Try inference_sft (simplest, single speaker)
            audio_data = None

            # Attempt 1: inference_sft
            if hasattr(self.model, "inference_sft"):
                try:
                    result = self.model.inference_sft(text, spk_id=speaker)
                    # result is typically a generator yielding tensors
                    for chunk in result:
                        if hasattr(chunk, "numpy"):
                            audio_data = chunk.numpy().flatten()
                        elif isinstance(chunk, dict):
                            arr = chunk.get("audio", None)
                            if arr is not None and hasattr(arr, "numpy"):
                                audio_data = arr.numpy().flatten()
                        break  # Use first chunk only
                except Exception as e:
                    logger.warning("inference_sft failed: %s", e)

            # Attempt 2: inference_zero_shot (if prompt available)
            if audio_data is None and hasattr(self.model, "inference_zero_shot"):
                logger.info("Falling back to inference_zero_shot with neutral prompt")
                # Use a built-in neutral prompt as reference
                try:
                    result = self.model.inference_zero_shot(
                        prompt_text="你好，今天天气不错。",
                        prompt_audio=self._load_default_prompt(),
                        text=text,
                    )
                    for chunk in result:
                        if hasattr(chunk, "numpy"):
                            audio_data = chunk.numpy().flatten()
                        elif isinstance(chunk, dict):
                            arr = chunk.get("audio", None)
                            if arr is not None and hasattr(arr, "numpy"):
                                audio_data = arr.numpy().flatten()
                        break
                except Exception as e:
                    logger.error("inference_zero_shot also failed: %s", e)
                    raise RuntimeError(f"TTS generation failed: {e}") from e

            if audio_data is None:
                raise RuntimeError("No audio generated from CosyVoice")

            # ── Convert to WAV bytes ──────────────────────────────
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, self.sample_rate, format="wav")
            wav_bytes = wav_buffer.getvalue()

            # ── Duration ──────────────────────────────────────────
            duration_ms = int(len(audio_data) / self.sample_rate * 1000)

            logger.info(
                "TTS generated: %d bytes, %d ms",
                len(wav_bytes), duration_ms,
            )
            return wav_bytes, duration_ms

        except Exception as e:
            logger.exception("TTS synthesis failed: %s", e)
            raise

    def _load_default_prompt(self) -> bytes:
        """Load or generate a default neutral-prompt audio sample."""
        # In production this would load a pre-recorded file
        # For now, generate a short silence as placeholder
        import numpy as np
        silence = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, silence, self.sample_rate, format="wav")
        return buf.getvalue()
