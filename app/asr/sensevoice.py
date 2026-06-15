"""
SenseVoiceSmall ASR wrapper with Speech Emotion Recognition (SER).

Maps the model's raw emotion embeddings to EmotionLabel categories
using valence-arousal heuristics.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional, Tuple

import numpy as np

from app.models.schemas import EmotionLabel, EmotionOutput

logger = logging.getLogger("digital-companion.asr")

# ── valence-arousal → EmotionLabel mapping ──────────────────────────
# valence: [-1, +1]  negative↔positive
# arousal: [-1, +1]  calm↔excited
_VA_MAP: list[tuple[float, float, EmotionLabel]] = [
    (0.8, 0.8, EmotionLabel.happy),       # high valence, high arousal
    (0.6, 0.3, EmotionLabel.love),         # high valence, low-mid arousal
    (0.3, 0.0, EmotionLabel.neutral),      # mid valence, mid arousal
    (-0.8, 0.8, EmotionLabel.angry),       # low valence, high arousal
    (-0.6, -0.2, EmotionLabel.sad),        # low-mid valence, low arousal
    (-0.5, 0.6, EmotionLabel.fearful),     # low valence, high arousal
    (-0.3, 0.4, EmotionLabel.anxious),     # low-mid valence, mid-high arousal
    (0.5, 0.9, EmotionLabel.surprised),    # mid-high valence, high arousal
    (-0.7, 0.2, EmotionLabel.disgusted),   # low valence, low-mid arousal
]


def _embedding_to_emotion(embedding: np.ndarray) -> Tuple[EmotionLabel, float]:
    """Convert a raw emotion embedding vector to an EmotionLabel + confidence.

    When the embedding is 1D (size = number of SER classes), use softmax.
    When it's 2D (e.g. 2×192), we average and still map.
    """
    emb = np.asarray(embedding, dtype=np.float32)

    # ── Case A: 1D probability-like vector ───────────────────────
    if emb.ndim == 1 and emb.shape[0] in (7, 8, 9):
        # Treat as class logits/probabilities
        probs = np.exp(emb - emb.max()) / np.sum(np.exp(emb - emb.max()) + 1e-9)
        idx = int(np.argmax(probs))
        label_list = list(EmotionLabel)
        if idx < len(label_list):
            return label_list[idx], float(probs[idx])
        return EmotionLabel.neutral, 0.5

    # ── Case B: 1D vector (embedding) — find nearest VA point ────
    if emb.ndim == 1:
        # Assume embedding is normalised; use a simple norm-based classifier
        # Actually map based on expected dimensions — if small assume VA-like
        if emb.shape[0] >= 2:
            v, a = float(emb[0]), float(emb[1]) if emb.shape[0] > 1 else 0.0
            v = np.clip(v, -1.0, 1.0)
            a = np.clip(a, -1.0, 1.0)
            scores = []
            for ev, ea, label in _VA_MAP:
                dist = np.sqrt((v - ev) ** 2 + (a - ea) ** 2)
                confidence = 1.0 / (1.0 + dist)
                scores.append((label, confidence))
            best = max(scores, key=lambda x: x[1])
            return best
        return EmotionLabel.neutral, 0.5

    # ── Case C: 2D vector — average over frames ──────────────────
    if emb.ndim == 2:
        mean_emb = emb.mean(axis=0)
        return _embedding_to_emotion(mean_emb)

    return EmotionLabel.neutral, 0.5


class SenseVoiceASR:
    """Wrapper around FunASR SenseVoiceSmall for ASR + SER."""

    def __init__(
        self,
        model_dir: Optional[str] = "pretrained_models/SenseVoiceSmall",
        device: str = "cpu",
    ):
        self.model_dir = model_dir
        self.device = device
        self.model = None
        self._init_model()

    def _init_model(self) -> None:
        """Initialise the SenseVoice model (local or download)."""
        try:
            from funasr import AutoModel

            kwargs: dict = {
                "model": "iic/SenseVoiceSmall",
                "device": self.device,
            }

            # If model_dir exists, load from local path
            if self.model_dir and os.path.isdir(self.model_dir):
                kwargs["model"] = self.model_dir
                logger.info("Loading SenseVoiceSmall from local: %s", self.model_dir)
            else:
                # 在线下载时附上 VAD/标点模型
                kwargs["vad_model"] = "iic/speech_fsmn_vad_zh-cn"
                kwargs["punc_model"] = "iic/punc_ct-transformer_zh-cn-0.2.2"

            # If model_dir exists, load from local path
            if self.model_dir and os.path.isdir(self.model_dir):
                kwargs["model"] = self.model_dir
                logger.info("Loading SenseVoiceSmall from local: %s", self.model_dir)

            self.model = AutoModel(**kwargs)
            logger.info("SenseVoiceSmall ASR+SER model loaded successfully")
        except ImportError as e:
            logger.error(
                "funasr not installed. Run: pip install funasr\n  %s", e
            )
            raise
        except Exception as e:
            logger.error("Failed to initialise SenseVoiceSmall: %s", e)
            raise

    def transcribe(self, audio_path: str) -> Tuple[str, EmotionOutput]:
        """Transcribe audio file and return (text, EmotionOutput).

        Args:
            audio_path: Path to the audio file (wav / mp3 / …).

        Returns:
            Tuple of (transcribed_text, EmotionOutput).
        """
        logger.info("Transcribing: %s", audio_path)
        if self.model is None:
            return _fallback("ASR model not initialised")

        try:
            result = self.model.generate(audio_path)

            # ── Parse model output ────────────────────────────────
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            if isinstance(result, dict):
                text = result.get("text", "")
                if text is None:
                    text = ""

                # Emotion embedding parsing
                emotion_embedding = result.get("emotion_embedding")
                if emotion_embedding is not None:
                    emotion_label, confidence = _embedding_to_emotion(
                        emotion_embedding
                    )
                else:
                    # Fallback: check if 'emotion' key exists directly
                    emotion_label = EmotionLabel.neutral
                    confidence = 0.5

                emotion = EmotionOutput(
                    emotion=emotion_label,
                    confidence=confidence,
                    sources={"asr": confidence},
                )
                logger.info("ASR result: text=%s emotion=%s", text[:80], emotion_label)
                return text, emotion

            logger.warning("Unexpected ASR output format: %s", type(result))
            return _fallback("Unexpected output format")

        except Exception as e:
            logger.exception("ASR transcription failed: %s", e)
            return _fallback(str(e))

    def transcribe_file(self, file_path: str) -> Tuple[str, EmotionOutput]:
        """Wrapper that accepts a file path directly."""
        return self.transcribe(file_path)

    def transcribe_bytes(
        self, audio_bytes: bytes, format: str = "wav"
    ) -> Tuple[str, EmotionOutput]:
        """Transcribe raw byte audio.

        Saves bytes to a temp file, calls transcribe(), then cleans up.
        """
        suffix = f".{format.lstrip('.')}"
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            text, emotion = self.transcribe(tmp_path)
            return text, emotion
        except Exception as e:
            logger.exception("ASR transcribe_bytes failed: %s", e)
            return _fallback(str(e))
        finally:
            try:
                os.unlink(tmp_path)
            except (OSError, NameError, UnboundLocalError):
                pass


def _fallback(reason: str = "unknown") -> Tuple[str, EmotionOutput]:
    """Return a safe fallback result."""
    logger.warning("ASR returning fallback (reason=%s)", reason)
    return (
        "识别失败",
        EmotionOutput(emotion=EmotionLabel.neutral, confidence=0.5, cause=reason),
    )
