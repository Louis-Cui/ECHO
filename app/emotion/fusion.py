"""
Multi-modal emotion fusion module.

Combines text, voice, and facial emotion predictions using
weighted voting with veto rules for high-confidence signals.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from app.models.schemas import EmotionLabel, EmotionOutput

logger = logging.getLogger("digital-companion.emotion.fusion")

# Negative emotions that can trigger veto
_NEGATIVE_LABELS: List[EmotionLabel] = [
    EmotionLabel.sad,
    EmotionLabel.angry,
    EmotionLabel.fearful,
    EmotionLabel.disgusted,
    EmotionLabel.anxious,
]

# High-arousal labels
_HIGH_AROUSAL: List[EmotionLabel] = [
    EmotionLabel.angry,
    EmotionLabel.happy,
    EmotionLabel.surprised,
    EmotionLabel.anxious,
]

# Love-related keywords (Chinese)
_LOVE_KEYWORDS: List[str] = [
    "爱", "喜欢", "想念", "想你了", "爱你", "喜欢上",
    "love", "miss you", "like you",
]


class EmotionFusion:
    """Multi-modal emotion fusion with veto rules."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "text": 0.4,
            "voice": 0.35,
            "face": 0.25,
        }
        logger.info("EmotionFusion weights: %s", self.weights)

    def fuse(
        self,
        text_emotion: Optional[EmotionOutput] = None,
        voice_emotion: Optional[EmotionOutput] = None,
        face_emotion: Optional[EmotionOutput] = None,
    ) -> EmotionOutput:
        """Fuse emotions from multiple modalities.

        VETO rules are checked first. If none trigger, weighted voting is used.

        Args:
            text_emotion: Emotion from text/LLM analysis.
            voice_emotion: Emotion from voice/SER analysis.
            face_emotion: Emotion from facial/FER analysis.

        Returns:
            Fused EmotionOutput.
        """
        sources: Dict[str, EmotionOutput] = {}
        if text_emotion:
            sources["text"] = text_emotion
        if voice_emotion:
            sources["voice"] = voice_emotion
        if face_emotion:
            sources["face"] = face_emotion

        if not sources:
            logger.warning("No emotion sources provided, returning neutral")
            return EmotionOutput(emotion=EmotionLabel.neutral, confidence=0.5)

        # ── Check VETO rules (order matters) ──────────────────────
        veto_result = self._check_veto_rules(sources)
        if veto_result:
            logger.info("Veto triggered: %s", veto_result.emotion.value)
            return veto_result

        # ── Weighted voting ───────────────────────────────────────
        scores: Dict[EmotionLabel, float] = {}
        source_confidences: Dict[str, float] = {}

        for label in EmotionLabel:
            total = 0.0
            for source_name, output in sources.items():
                if output.emotion == label:
                    weight = self.weights.get(source_name, 0.0)
                    total += weight * output.confidence
            if total > 0:
                scores[label] = total

        if not scores:
            return EmotionOutput(emotion=EmotionLabel.neutral, confidence=0.5)

        # Pick the winner
        winner = max(scores, key=scores.get)  # type: EmotionLabel
        winner_score = scores[winner]

        # Build source contributions for the winner
        for source_name, output in sources.items():
            if output.emotion == winner:
                source_confidences[source_name] = output.confidence

        result = EmotionOutput(
            emotion=winner,
            confidence=min(winner_score, 1.0),
            sources=source_confidences if source_confidences else None,
        )

        logger.info(
            "Fusion result: %s (%.2f) from %d sources",
            winner.value, result.confidence, len(sources),
        )
        return result

    def _check_veto_rules(
        self, sources: Dict[str, EmotionOutput]
    ) -> Optional[EmotionOutput]:
        """Check all veto rules. Returns the overriding EmotionOutput or None."""
        text_out = sources.get("text")
        voice_out = sources.get("voice")

        # ── Rule 1: text_negative_veto ────────────────────────────
        if text_out and text_out.confidence > 0.7:
            if text_out.emotion in _NEGATIVE_LABELS:
                logger.info(
                    "VETO: text_negative_veto (%s, conf=%.2f)",
                    text_out.emotion.value, text_out.confidence,
                )
                text_out.cause = "text_negative_veto"
                return text_out

        # ── Rule 2: voice_high_arousal ────────────────────────────
        if voice_out and voice_out.emotion in _HIGH_AROUSAL:
            if voice_out.confidence > 0.8:
                logger.info(
                    "VETO: voice_high_arousal (%s, conf=%.2f)",
                    voice_out.emotion.value, voice_out.confidence,
                )
                voice_out.cause = "voice_high_arousal"
                return voice_out

        # ── Rule 3: text_love_veto ────────────────────────────────
        if text_out and text_out.emotion == EmotionLabel.love:
            # Check if the cause field contains love-related terms,
            # which should be set by the LLM emotion analysis.
            cause = (text_out.cause or "").lower()
            if any(kw in cause for kw in _LOVE_KEYWORDS):
                logger.info("VETO: text_love_veto")
                text_out.cause = "text_love_veto"
                return text_out

        return None
