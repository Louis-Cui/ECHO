"""
RAG memory module using Chroma vector database for long-term
conversation memory with weighted recall and emotion filtering.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from app.models.schemas import EmotionLabel

logger = logging.getLogger("digital-companion.memory.rag")


class MemoryRAG:
    """Chroma-based vector memory for conversation history and facts.

    Each memory is stored with:
    - content (text)
    - embedding (for similarity search)
    - metadata: user_id, emotion, weight, timestamp
    """

    def __init__(
        self,
        persist_dir: str = "./data/memory",
        collection_name: str = "user_memories",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._embedding_fn = None
        self._init_chroma()

    def _init_chroma(self) -> None:
        """Initialise Chroma persistent client and collection."""
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Chroma ready: persist_dir=%s collection=%s",
                self.persist_dir, self.collection_name,
            )
        except ImportError as e:
            logger.error(
                "chromadb not installed: pip install chromadb\n  %s", e
            )
            raise

    def _get_embedding(self, text: str) -> List[float]:
        """Compute embedding for a text string.

        Uses sentence-transformers (all-MiniLM-L6-v2) by default.
        """
        if self._embedding_fn is not None:
            return self._embedding_fn(text)

        try:
            from sentence_transformers import SentenceTransformer
            # Cache the model
            model = SentenceTransformer("all-MiniLM-L6-v2")
            self._embedding_fn = lambda t: model.encode(t).tolist()
            return self._embedding_fn(text)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embeddings.\n"
                "  pip install sentence-transformers"
            )
        except Exception as e:
            logger.error("Embedding failed: %s", e)
            raise

    def add_memory(
        self,
        user_id: str,
        content: str,
        emotion: EmotionLabel,
        weight: float = 1.0,
    ) -> str:
        """Add a memory entry to the vector store.

        Args:
            user_id: User identifier.
            content: The text content to remember.
            emotion: Associated emotion label.
            weight: Importance weight (higher = more likely to recall).

        Returns:
            The unique memory ID string.
        """
        memory_id = str(uuid.uuid4())
        embedding = self._get_embedding(content)
        timestamp = time.time()

        metadata: Dict[str, Any] = {
            "user_id": user_id,
            "emotion": emotion.value,
            "weight": weight,
            "timestamp": timestamp,
        }

        try:
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[content],
            )
            logger.info(
                "Memory added: id=%s user=%s emotion=%s len=%d",
                memory_id[:8], user_id, emotion.value, len(content),
            )
        except Exception as e:
            logger.exception("Failed to add memory: %s", e)
            raise

        return memory_id

    def search(
        self,
        query: str,
        user_id: str,
        n_results: int = 5,
        emotion_filter: Optional[EmotionLabel] = None,
    ) -> List[Dict[str, Any]]:
        """Search memories by semantic similarity.

        Results are scored by:
        - Chroma's cosine similarity score (distance → similarity)
        - Time decay factor (recent memories get a boost)
        - Weight multiplier

        Args:
            query: Search query text.
            user_id: User to scope memories.
            n_results: Max number of results.
            emotion_filter: Optional emotion label to filter by.

        Returns:
            List of dicts with keys: content, emotion, weight,
            timestamp, similarity.
        """
        query_embedding = self._get_embedding(query)

        # Build Chroma where filter
        where_filter: Dict[str, Any] = {"user_id": user_id}
        if emotion_filter:
            where_filter["emotion"] = emotion_filter.value

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,  # Extra for post-filter
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.exception("Memory search failed: %s", e)
            return []

        # ── Parse and re-rank ─────────────────────────────────────
        now = time.time()
        parsed: List[Dict[str, Any]] = []

        ids_raw = results.get("ids", [[]])[0]
        docs_raw = results.get("documents", [[]])[0]
        meta_raw = results.get("metadatas", [[]])[0]
        dists_raw = results.get("distances", [[]])[0]

        for idx, doc in enumerate(docs_raw):
            if idx >= len(ids_raw):
                break
            meta = meta_raw[idx] if idx < len(meta_raw) else {}
            dist = dists_raw[idx] if idx < len(dists_raw) else 1.0

            # Cosine similarity = 1 - distance
            similarity = 1.0 - float(dist)

            timestamp = float(meta.get("timestamp", 0))
            weight = float(meta.get("weight", 1.0))

            # Time decay: memories > 7 days get reduced relevance
            age_hours = (now - timestamp) / 3600
            time_decay = max(0.3, 1.0 - (age_hours / (24 * 7)))

            # Combined score
            score = similarity * weight * time_decay

            parsed.append({
                "id": ids_raw[idx] if idx < len(ids_raw) else "",
                "content": doc,
                "emotion": meta.get("emotion", "neutral"),
                "weight": weight,
                "timestamp": timestamp,
                "similarity": round(score, 4),
            })

        # Sort by combined score descending, take top n_results
        parsed.sort(key=lambda x: x["similarity"], reverse=True)
        return parsed[:n_results]

    def update_weight(self, memory_id: str, delta: float) -> None:
        """Adjust the weight of a memory entry.

        Args:
            memory_id: The memory's unique ID.
            delta: Amount to add to the current weight.
        """
        try:
            # Chroma doesn't support read-modify-update directly,
            # so we must re-add with updated metadata.
            result = self.collection.get(ids=[memory_id])
            if not result["ids"]:
                logger.warning("Memory not found for weight update: %s", memory_id[:8])
                return

            meta = result["metadatas"][0]
            current_weight = float(meta.get("weight", 1.0))
            new_weight = max(0.1, current_weight + delta)

            meta["weight"] = new_weight

            self.collection.update(
                ids=[memory_id],
                metadatas=[meta],
            )
            logger.info(
                "Memory weight updated: id=%s %.2f → %.2f",
                memory_id[:8], current_weight, new_weight,
            )
        except Exception as e:
            logger.exception("Failed to update memory weight: %s", e)

    def forget(self, memory_id: str) -> None:
        """Delete a memory entry."""
        try:
            self.collection.delete(ids=[memory_id])
            logger.info("Memory deleted: id=%s", memory_id[:8])
        except Exception as e:
            logger.exception("Failed to delete memory: %s", e)

    def get_recent_conversations(
        self, user_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get most recent memories for a user.

        Uses an empty query to retrieve documents,
        sorted by timestamp descending.

        Args:
            user_id: User identifier.
            limit: Max number of conversations.

        Returns:
            List of memory dicts sorted by recency.
        """
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                limit=limit * 3,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.exception("Failed to get recent conversations: %s", e)
            return []

        parsed = []
        for idx, doc in enumerate(results.get("documents", [])):
            meta = results["metadatas"][idx] if idx < len(results["metadatas"]) else {}
            parsed.append({
                "id": results["ids"][idx] if idx < len(results["ids"]) else "",
                "content": doc,
                "emotion": meta.get("emotion", "neutral"),
                "weight": meta.get("weight", 1.0),
                "timestamp": meta.get("timestamp", 0),
            })

        parsed.sort(key=lambda x: x["timestamp"], reverse=True)
        return parsed[:limit]
