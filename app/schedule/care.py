"""
Active care scheduler — proactively checks on users based on
emotion history and inactivity time.

Triggers greeting, check-in, or consolation events via the agent.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.models.schemas import (
    CareEvent,
    EmotionLabel,
    EmotionOutput,
    ScheduleConfig,
)
from app.agent.prompt import active_care_prompts

logger = logging.getLogger("digital-companion.schedule.care")


class ActiveCareScheduler:
    """Proactive care scheduler that periodically checks user states."""

    def __init__(
        self,
        agent: "CompanionAgent",  # type: ignore  # noqa: F821
        config: Optional[ScheduleConfig] = None,
    ):
        self.agent = agent
        self.config = config or ScheduleConfig()
        self.scheduler = AsyncIOScheduler()
        self.user_states: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "ActiveCareScheduler configured: interval=%ds threshold=%.2f count=%d",
            self.config.check_interval_seconds,
            self.config.low_emotion_threshold,
            self.config.consecutive_low_count,
        )

    def start(self) -> None:
        """Start the scheduler job."""
        self.scheduler.add_job(
            self._check_all_users,
            "interval",
            seconds=self.config.check_interval_seconds,
            id="active_care_check",
            replace_existing=True,
        )
        self.scheduler.start()
        logger.info("Active care scheduler started (interval=%ds)", self.config.check_interval_seconds)

    def stop(self) -> None:
        """Stop the scheduler."""
        self.scheduler.shutdown(wait=False)
        logger.info("Active care scheduler stopped")

    def register_user(self, user_id: str) -> None:
        """Register a user for active monitoring."""
        if user_id not in self.user_states:
            self.user_states[user_id] = {
                "consecutive_low": 0,
                "last_emotion": EmotionLabel.neutral,
                "last_active": time.time(),
                "last_care_time": 0,
                "last_care_type": None,
            }
            logger.info("User registered for active care: %s", user_id)

    def update_user_state(self, user_id: str, emotion: EmotionLabel) -> None:
        """Update tracked state for a user."""
        if user_id not in self.user_states:
            self.register_user(user_id)

        state = self.user_states[user_id]
        state["last_active"] = time.time()
        state["last_emotion"] = emotion

        # Track consecutive low emotion
        if emotion in (EmotionLabel.sad, EmotionLabel.angry, EmotionLabel.fearful,
                       EmotionLabel.anxious, EmotionLabel.disgusted):
            state["consecutive_low"] += 1
        else:
            state["consecutive_low"] = 0

        logger.debug(
            "User %s state updated: emotion=%s consecutive_low=%d",
            user_id, emotion.value, state["consecutive_low"],
        )

    def _check_all_users(self) -> None:
        """Periodic check of all registered users."""
        now = time.time()
        for user_id, state in self.user_states.items():
            try:
                self._check_user(user_id, state, now)
            except Exception as e:
                logger.exception("Error checking user %s: %s", user_id, e)

    def _check_user(
        self, user_id: str, state: Dict[str, Any], now: float
    ) -> None:
        """Evaluate whether to trigger a care event for a specific user."""
        inactive_duration = now - state["last_active"]
        hours_inactive = inactive_duration / 3600

        # Minimum cooldown between care events (30 min)
        min_cooldown = 1800
        if now - state.get("last_care_time", 0) < min_cooldown:
            return

        # ── Rule 1: Long inactivity → greeting ────────────────────
        if hours_inactive > 8:
            # Night check (23:00–06:00)
            current_hour = datetime.now().hour
            if 6 <= current_hour < 12:
                message = active_care_prompts["greeting_morning"]
            elif 12 <= current_hour < 18:
                message = active_care_prompts["greeting_afternoon"]
            elif 18 <= current_hour < 22:
                message = active_care_prompts["greeting_evening"]
            else:
                # Late night — don't disturb unless very important
                if state["consecutive_low"] < self.config.consecutive_low_count:
                    return
                message = active_care_prompts["greeting_night"]

            event = CareEvent(
                event_type="greeting",
                trigger_reason=f"inactive for {hours_inactive:.1f}h",
                message=message,
            )
            self._trigger_care(user_id, event)
            return

        # ── Rule 2: Consecutive low emotion → consolation ─────────
        if state["consecutive_low"] >= self.config.consecutive_low_count:
            event = CareEvent(
                event_type="consolation",
                trigger_reason=f"consecutive low emotion: {state['consecutive_low']}",
                message=active_care_prompts["consolation"],
            )
            self._trigger_care(user_id, event)
            # Reset counter after triggering
            state["consecutive_low"] = 0
            return

        # ── Rule 3: Moderate inactivity (4h+) → check-in ──────────
        if hours_inactive > 4:
            event = CareEvent(
                event_type="checkin",
                trigger_reason=f"inactive for {hours_inactive:.1f}h",
                message=active_care_prompts["checkin"],
            )
            self._trigger_care(user_id, event)
            return

    def _trigger_care(self, user_id: str, event: CareEvent) -> None:
        """Execute a care event by invoking the agent.

        In production, this would push the message to the user's device
        (WebSocket push, notification, etc.).
        """
        state = self.user_states.get(user_id)
        if state:
            state["last_care_time"] = time.time()
            state["last_care_type"] = event.event_type

        logger.info(
            "CARE EVENT [%s] user=%s reason=%s msg=%s",
            event.event_type, user_id, event.trigger_reason, event.message,
        )

        # Invoke agent to generate a personalised care message
        try:
            # Create a chat-style interaction with the care prompt
            response = self.agent.invoke(
                user_input=event.message,
                user_id=user_id,
            )
            logger.info(
                "Care response generated: %s", response.text[:100],
            )
        except Exception as e:
            logger.exception("Care agent invocation failed: %s", e)
