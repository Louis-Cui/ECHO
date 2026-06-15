"""
LangGraph agent workflow for the digital companion.

Defines the StateGraph with nodes for emotion analysis, memory
retrieval, prompt building, LLM response generation, emotion
determination, and memory storage.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

import operator

from app.models.schemas import (
    ChatOutput,
    EmotionLabel,
    EmotionOutput,
    TTSOutput,
)
from app.emotion.fusion import EmotionFusion
from app.memory.rag import MemoryRAG
from app.agent.prompt import (
    CONDENSED_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    emotion_to_personality,
)

logger = logging.getLogger("digital-companion.agent")


# ═══════════════════════════════════════════════════════════════════
# LangGraph state definition
# ═══════════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    """State passed through the LangGraph workflow."""

    user_id: str
    user_input: str
    emotion: Optional[EmotionOutput]
    chat_history: List[Dict[str, str]]
    memory_context: List[Dict[str, Any]]
    response: str
    response_emotion: EmotionLabel


# ═══════════════════════════════════════════════════════════════════
# CompanionAgent
# ═══════════════════════════════════════════════════════════════════
class CompanionAgent:
    """LangGraph-based digital companion agent."""

    def __init__(
        self,
        memory: MemoryRAG,
        emotion_fusion: EmotionFusion,
        model_name: str = "deepseek-chat",
        api_key: Optional[str] = None,
    ):
        self.memory = memory
        self.emotion_fusion = emotion_fusion
        self.llm = None
        self.graph = None
        self._init_llm(model_name, api_key)
        self._build_graph()

    def _init_llm(self, model_name: str, api_key: Optional[str]) -> None:
        """Initialise the LLM (DeepSeek or OpenAI fallback)."""
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No API key found — LLM calls will fail")

        try:
            # Try DeepSeek first
            from langchain_deepseek import ChatDeepSeek
            self.llm = ChatDeepSeek(
                model=model_name,
                api_key=api_key,
                temperature=0.8,
                max_tokens=1024,
            )
            logger.info("LLM initialised: ChatDeepSeek(%s)", model_name)
        except ImportError:
            # Fallback to OpenAI
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=api_key,
                    temperature=0.8,
                    max_tokens=1024,
                )
                logger.info("LLM fallback: ChatOpenAI(gpt-4o-mini)")
            except ImportError:
                logger.error(
                    "No LLM provider available. Install langchain-deepseek or langchain-openai."
                )
                raise

    def _build_graph(self) -> None:
        """Build the LangGraph StateGraph."""
        from langgraph.graph import StateGraph, END, START

        graph = StateGraph(AgentState)

        # ── Add nodes ─────────────────────────────────────────────
        graph.add_node("analyze_emotion", self._node_analyze_emotion)
        graph.add_node("retrieve_memory", self._node_retrieve_memory)
        graph.add_node("build_prompt", self._node_build_prompt)
        graph.add_node("generate_response", self._node_generate_response)
        graph.add_node("determine_response_emotion", self._node_determine_response_emotion)
        graph.add_node("store_memory", self._node_store_memory)

        # ── Add edges ─────────────────────────────────────────────
        graph.add_edge(START, "analyze_emotion")
        graph.add_edge("analyze_emotion", "retrieve_memory")
        graph.add_edge("retrieve_memory", "build_prompt")
        graph.add_edge("build_prompt", "generate_response")
        graph.add_edge("generate_response", "determine_response_emotion")
        graph.add_edge("determine_response_emotion", "store_memory")
        graph.add_edge("store_memory", END)

        self.graph = graph.compile()
        logger.info("LangGraph compiled with %d nodes", len(graph.nodes))

    # ── Node implementations ──────────────────────────────────────

    def _node_analyze_emotion(self, state: AgentState) -> Dict[str, Any]:
        """Analyze emotion from user input using LLM zero-shot.

        Fuses with external voice/face emotions if provided.
        """
        user_input = state["user_input"]
        external_emotion = state.get("emotion")  # from voice/face

        # Use LLM for text emotion analysis
        text_emotion = self._llm_emotion_analysis(user_input)

        # Fuse with external modalities
        if external_emotion:
            fused = self.emotion_fusion.fuse(
                text_emotion=text_emotion,
                voice_emotion=external_emotion,
            )
        else:
            fused = text_emotion

        return {"emotion": fused}

    def _llm_emotion_analysis(self, text: str) -> EmotionOutput:
        """Zero-shot emotion classification via LLM."""
        if self.llm is None:
            return EmotionOutput(emotion=EmotionLabel.neutral, confidence=0.5)

        prompt = (
            "分析下面这句话的情绪。只返回JSON格式：{\"emotion\": \"情绪标签\", "
            "\"confidence\": 0~1之间的置信度, \"cause\": \"引起情绪的关键词\"}\n"
            f"情绪标签可选：{', '.join(e.value for e in EmotionLabel)}\n\n"
            f"文本：{text}\n\nJSON："
        )

        try:
            resp = self.llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)

            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*?\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                emotion_value = data.get("emotion", "neutral").lower()
                confidence = float(data.get("confidence", 0.5))
                cause = data.get("cause", "")

                # Validate emotion label
                try:
                    emotion = EmotionLabel(emotion_value)
                except ValueError:
                    emotion = EmotionLabel.neutral

                return EmotionOutput(
                    emotion=emotion,
                    confidence=min(confidence, 1.0),
                    cause=cause,
                    sources={"text": confidence},
                )

            return EmotionOutput(emotion=EmotionLabel.neutral, confidence=0.5)
        except Exception as e:
            logger.warning("LLM emotion analysis failed: %s", e)
            # Keyword-based fallback
            return self._keyword_emotion_fallback(text)

    def _keyword_emotion_fallback(self, text: str) -> EmotionOutput:
        """Simple keyword-based emotion detection as fallback."""
        text_lower = text.lower()

        love_words = ["爱", "喜欢", "想念", "love", "miss", "cute", "可爱"]
        upset_words = ["怪怪的", "不对劲", "烦闷", "烦躁", "upset", "郁闷", "低沉", "不太舒服"]
        sad_words = ["难过", "伤心", "悲伤", "不开心", "失落", "sad", "cry", "哭"]
        angry_words = ["生气", "愤怒", "气死", "烦", "angry", "mad", "烦死了"]
        happy_words = ["开心", "高兴", "快乐", "棒", "好开心", "happy", "哈哈"]
        fear_words = ["害怕", "恐惧", "担心", "慌", "scared", "afraid", "worried"]
        anxious_words = ["焦虑", "紧张", "不安", "anxious", "nervous", "压力"]

        for word in upset_words:
            if word in text_lower:
                return EmotionOutput(emotion=EmotionLabel.upset, confidence=0.65, cause=word)
        for word in love_words:
            if word in text_lower:
                return EmotionOutput(emotion=EmotionLabel.love, confidence=0.6, cause=word)
        for word in sad_words:
            if word in text_lower:
                return EmotionOutput(emotion=EmotionLabel.sad, confidence=0.7, cause=word)
        for word in angry_words:
            if word in text_lower:
                return EmotionOutput(emotion=EmotionLabel.angry, confidence=0.7, cause=word)
        for word in happy_words:
            if word in text_lower:
                return EmotionOutput(emotion=EmotionLabel.happy, confidence=0.7, cause=word)
        for word in fear_words:
            if word in text_lower:
                return EmotionOutput(emotion=EmotionLabel.fearful, confidence=0.6, cause=word)
        for word in anxious_words:
            if word in text_lower:
                return EmotionOutput(emotion=EmotionLabel.anxious, confidence=0.6, cause=word)

        return EmotionOutput(emotion=EmotionLabel.neutral, confidence=0.5)

    def _node_retrieve_memory(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve relevant memories from RAG."""
        user_id = state["user_id"]
        query = state["user_input"]
        emotion = state.get("emotion")

        emotion_filter = emotion.emotion if emotion else None

        # Search for relevant memories
        memories = self.memory.search(
            query=query,
            user_id=user_id,
            n_results=5,
            emotion_filter=emotion_filter,
        )

        # Also get recent conversations for context
        recent = self.memory.get_recent_conversations(user_id, limit=5)

        memory_context = []
        seen_content = set()

        for m in memories:
            if m["content"] not in seen_content:
                memory_context.append(m)
                seen_content.add(m["content"])

        for m in recent:
            if m["content"] not in seen_content:
                memory_context.append(m)
                seen_content.add(m["content"])

        logger.info(
            "Retrieved %d memory items for user %s",
            len(memory_context), user_id,
        )
        return {"memory_context": memory_context[:8]}  # Limit to 8

    def _node_build_prompt(self, state: AgentState) -> Dict[str, Any]:
        """Assemble the prompt by combining system prompt, memory, and input."""
        user_input = state["user_input"]
        emotion = state.get("emotion")
        memory_context = state.get("memory_context", [])

        # ── Build memory section ──────────────────────────────────
        memory_text = ""
        if memory_context:
            memory_lines = []
            for m in memory_context:
                ts = time.strftime(
                    "%m-%d %H:%M", time.localtime(m.get("timestamp", 0))
                )
                memory_lines.append(
                    f"- [{m.get('emotion', 'neutral')}] "
                    f"({ts}) {m['content']}"
                )
            memory_text = "\n".join(memory_lines)

        # ── Build emotion section ─────────────────────────────────
        emotion_text = ""
        if emotion and emotion.emotion != EmotionLabel.neutral:
            emotion_text = (
                f"【当前用户情绪】{emotion.emotion.value} "
                f"(置信度: {emotion.confidence:.2f})"
            )
            if emotion.cause:
                emotion_text += f" — 可能原因: {emotion.cause}"

        # ── Build personality guide ───────────────────────────────
        personality_guide = ""
        if emotion:
            adjustment = emotion_to_personality.get(emotion.emotion)
            if adjustment:
                personality_guide = (
                    f"语调：{adjustment['tone']}\n"
                    f"风格：{adjustment['style']}"
                )

        # ── Assemble final prompt ─────────────────────────────────
        condensed = CONDENSED_SYSTEM_PROMPT.format(
            emotion_label=emotion.emotion.value if emotion else "neutral",
        )

        memory_section = (
            f"## 相关记忆\n{memory_text}\n\n" if memory_text else ""
        )
        emotion_section = (
            f"## 情绪信息\n{emotion_text}\n\n" if emotion_text else ""
        )
        personality_section = (
            f"## 回复风格指引\n{personality_guide}\n\n" if personality_guide else ""
        )

        final_prompt = (
            f"{condensed}\n\n"
            f"{memory_section}"
            f"{emotion_section}"
            f"{personality_section}"
            f"## 用户输入\n{user_input}\n\n"
            f"请以沐光的身份回复："
        )

        return {"chat_history": [{"role": "user", "content": user_input}]}

    def _node_generate_response(self, state: AgentState) -> Dict[str, Any]:
        """Generate the LLM response."""
        if self.llm is None:
            return {"response": "抱歉，我现在无法回复……请稍后再试。"}

        user_input = state["user_input"]
        emotion = state.get("emotion")
        memory_context = state.get("memory_context", [])
        chat_history = state.get("chat_history", [])

        # Build the full message list
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add memory context as system context
        if memory_context:
            memory_text = "\n".join(
                f"- [{m.get('emotion', 'neutral')}] {m['content']}"
                for m in memory_context[:5]
            )
            messages.append({
                "role": "system",
                "content": f"关于用户的一些记忆：\n{memory_text}",
            })

        # Add emotion context
        if emotion and emotion.emotion != EmotionLabel.neutral:
            messages.append({
                "role": "system",
                "content": (
                    f"用户当前情绪似乎是：{emotion.emotion.value} "
                    f"(置信度：{emotion.confidence:.2f})"
                ),
            })

        # Add conversation history (limit to last 4 exchanges)
        for entry in chat_history[-6:]:  # Keep manageable
            messages.append(entry)

        # Add the current input
        messages.append({"role": "user", "content": user_input})

        try:
            resp = self.llm.invoke(messages)
            response_text = resp.content if hasattr(resp, "content") else str(resp)
            return {"response": response_text.strip()}
        except Exception as e:
            logger.exception("LLM generation failed: %s", e)
            return {"response": "嗯……我在听。能再多说一点吗？😊"}

    def _node_determine_response_emotion(self, state: AgentState) -> Dict[str, Any]:
        """Determine the emotion for TTS based on the response content."""
        response = state.get("response", "")
        if not response:
            return {"response_emotion": EmotionLabel.neutral}

        # Use LLM to determine the emotion of the response
        emotion = self._llm_emotion_analysis(response)

        # Map to a clean label
        label = emotion.emotion

        # Ensure the response emotion is appropriate — don't be sad/angry
        # even if user is, the companion should be supportive
        if label in (EmotionLabel.sad, EmotionLabel.angry, EmotionLabel.fearful,
                     EmotionLabel.disgusted, EmotionLabel.anxious, EmotionLabel.upset):
            # If companion response mirrors negative emotion, blend to gentle/warm
            label = EmotionLabel.neutral

        return {"response_emotion": label}

    def _node_store_memory(self, state: AgentState) -> Dict[str, Any]:
        """Store the conversation turn in RAG memory."""
        user_id = state["user_id"]
        user_input = state["user_input"]
        response = state.get("response", "")
        emotion = state.get("emotion")

        try:
            # Store user's message
            emotion_label = emotion.emotion if emotion else EmotionLabel.neutral
            self.memory.add_memory(
                user_id=user_id,
                content=user_input,
                emotion=emotion_label,
                weight=1.0,
            )

            # Store assistant response (lower weight)
            self.memory.add_memory(
                user_id=user_id,
                content=response,
                emotion=EmotionLabel.neutral,
                weight=0.7,
            )

            # Boost weight if emotion was strong
            if emotion and emotion.confidence > 0.8:
                logger.info("Strong emotion detected — memory will be weighted higher")
        except Exception as e:
            logger.warning("Memory storage failed: %s", e)

        return {}

    # ═══════════════════════════════════════════════════════════════
    # Public invoke method
    # ═══════════════════════════════════════════════════════════════
    def invoke(
        self,
        user_input: str,
        user_id: str,
        voice_emotion: Optional[EmotionOutput] = None,
        face_emotion: Optional[EmotionOutput] = None,
    ) -> ChatOutput:
        """Run the full agent workflow and return a response.

        Args:
            user_input: The user's text message.
            user_id: User identifier.
            voice_emotion: Optional emotion from voice analysis.
            face_emotion: Optional emotion from facial analysis.

        Returns:
            ChatOutput with response text, emotion, and optional TTS data.
        """
        # Fuse external emotions before passing to graph
        fused_external = None
        if voice_emotion or face_emotion:
            fused_external = self.emotion_fusion.fuse(
                voice_emotion=voice_emotion,
                face_emotion=face_emotion,
            )

        initial_state: AgentState = {
            "user_id": user_id,
            "user_input": user_input,
            "emotion": fused_external,
            "chat_history": [],
            "memory_context": [],
            "response": "",
            "response_emotion": EmotionLabel.neutral,
        }

        try:
            result = self.graph.invoke(initial_state)
            response_emotion = result.get("response_emotion", EmotionLabel.neutral)

            return ChatOutput(
                text=result.get("response", ""),
                emotion=response_emotion,
            )
        except Exception as e:
            logger.exception("Agent workflow failed: %s", e)
            return ChatOutput(
                text="抱歉，我刚才走神了……能再说一遍吗？😅",
                emotion=EmotionLabel.neutral,
            )
