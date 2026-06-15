"""
Facial Emotion Recognition using the FER library.

Detects faces in images and classifies emotions using a pre-trained
deep learning model (MTCNN for detection + CNN for emotion).
"""
from __future__ import annotations

import io
import logging
import tempfile
from typing import Dict, Optional

import numpy as np

from app.models.schemas import EmotionLabel, EmotionOutput

logger = logging.getLogger("digital-companion.emotion.fer")

# Map FER library labels → our EmotionLabel
_FER_TO_EMOTION: Dict[str, EmotionLabel] = {
    "angry": EmotionLabel.angry,
    "disgust": EmotionLabel.disgusted,
    "fear": EmotionLabel.fearful,
    "happy": EmotionLabel.happy,
    "sad": EmotionLabel.sad,
    "surprise": EmotionLabel.surprised,
    "neutral": EmotionLabel.neutral,
}


class FERClassifier:
    """Facial emotion recognition using FER (PyPI)."""

    def __init__(self):
        self.detector = None
        self._init_detector()

    def _init_detector(self) -> None:
        """Initialise FER detector, falling back from MTCNN to OpenCV."""
        try:
            from fer.fer import FER

            # Attempt MTCNN (more accurate, slower)
            try:
                import tensorflow as tf  # noqa: F401 — required by MTCNN
                self.detector = FER(mtcnn=True)
                logger.info("FER MTCNN detector initialised")
            except (ImportError, Exception) as e:
                logger.warning(
                    "MTCNN not available (need tensorflow), falling back: %s", e
                )
                self.detector = FER(mtcnn=False)
                logger.info("FER OpenCV detector initialised")
        except ImportError as e:
            logger.error("FER library not installed: pip install fer\n  %s", e)
            raise
        except Exception as e:
            logger.error("Failed to initialise FER: %s", e)
            raise

    def detect_emotion(self, image_path: str) -> EmotionOutput:
        """Detect emotion from an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            EmotionOutput with the detected emotion and confidence.
        """
        logger.info("FER detecting emotion from: %s", image_path)
        if self.detector is None:
            return EmotionOutput(emotion=EmotionLabel.neutral, confidence=0.5)

        try:
            result = self.detector.detect_emotions(image_path)

            if not result:
                logger.info("No face detected, returning neutral")
                return EmotionOutput(emotion=EmotionLabel.neutral, confidence=0.3)

            # Use the first face's emotions
            emotions = result[0]["emotions"]
            emotion_name = max(emotions, key=emotions.get)

            mapped_label = _FER_TO_EMOTION.get(emotion_name, EmotionLabel.neutral)
            confidence = float(emotions[emotion_name])

            logger.info("FER result: emotion=%s (%.2f)", mapped_label.value, confidence)
            return EmotionOutput(
                emotion=mapped_label,
                confidence=confidence,
                sources={"face": confidence},
            )

        except Exception as e:
            logger.exception("FER detection failed: %s", e)
            return EmotionOutput(emotion=EmotionLabel.neutral, confidence=0.3, cause=str(e))

    def detect_emotion_from_bytes(self, image_bytes: bytes) -> EmotionOutput:
        """Detect emotion from raw image bytes."""
        suffix = ".jpg"
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name

            return self.detect_emotion(tmp_path)
        except Exception as e:
            logger.exception("FER bytes detection failed: %s", e)
            return EmotionOutput(emotion=EmotionLabel.neutral, confidence=0.3, cause=str(e))
        finally:
            try:
                import os
                os.unlink(tmp_path)
            except (OSError, NameError, UnboundLocalError):
                pass
