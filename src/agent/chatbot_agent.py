"""
Multi-turn chatbot agent for the NVR system.

Uses the local template-based LocalResponder (no LLM API required).
The ML intent classifier handles all routing; responses are built from
French templates enriched with live search / summary data.

API surface is unchanged — existing callers (demo.py, api.py) work as-is.
"""
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str      # "user" | "assistant"
    content: str


@dataclass
class AgentContext:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation: List[Message] = field(default_factory=list)
    last_events: List[Dict[str, Any]] = field(default_factory=list)


# ── agent ─────────────────────────────────────────────────────────────────────

class ChatbotAgent:
    def __init__(
        self,
        search_engine,
        llm_provider: str = "local",       # kept for API compat — ignored
        llm_model: Optional[str] = None,   # kept for API compat — ignored
        api_key: Optional[str] = None,     # kept for API compat — ignored
        max_history: int = 5,
        ml_model_dir: str = "src/ml/models",
        ml_confidence_threshold: float = 0.70,
    ):
        self.search_engine = search_engine
        self.max_history = max_history
        # Public base URL the frontend can fetch clips from (see GET /clip/{id}).
        # Override with CLIP_BASE_URL when the API is not on localhost:8000.
        self._clip_base_url = os.getenv("CLIP_BASE_URL", "http://localhost:8000").rstrip("/")

        from src.agent.local_responder import LocalResponder
        self._responder = LocalResponder(ml_model_dir)
        logger.info("ChatbotAgent ready — local template-based mode (no LLM API)")

    # ── public API ────────────────────────────────────────────────────────────

    def chat(self, user_message: str, ctx: AgentContext) -> dict:
        # Track conversation (last max_history turns = 2*max_history messages)
        ctx.conversation.append(Message(role="user", content=user_message))
        if len(ctx.conversation) > self.max_history * 2:
            ctx.conversation = ctx.conversation[-(self.max_history * 2):]

        # Step 1 — classify intent with local ML model
        clf = self._responder.predict(user_message)
        intent = clf["intent"]

        classification_info = {
            "ml": {
                "intent":     intent,
                "confidence": clf["confidence"],
                "all_scores": clf["all_scores"],
            },
            "source":     "ml",
            "llm_intent": None,
        }

        # Step 2 — route to handler
        if intent == "search":
            result = self._handle_search(user_message, ctx)
        elif intent == "summary":
            result = self._handle_summary(ctx)
        elif intent == "clip_request":
            result = self._handle_clip_request(user_message, ctx)
        else:
            # greeting, unknown, camera_info, …
            response = self._responder.build_response(intent, user_message=user_message)
            result = {"response": response}

        ctx.conversation.append(Message(role="assistant", content=result.get("response", "")))
        result["classification_info"] = classification_info
        return result

    # ── intent handlers ───────────────────────────────────────────────────────

    def _handle_search(self, user_message: str, ctx: AgentContext) -> dict:
        try:
            events = self.search_engine.search(
                query=user_message,
                top_k=3,
                extract_clips=False,
            )
            ctx.last_events = [e.to_dict() for e in events]
        except Exception as exc:
            logger.warning("Search failed: %s", exc)
            ctx.last_events = []
        response = self._responder.build_response("search", search_results=ctx.last_events)
        return {"response": response, "events": ctx.last_events}

    def _handle_summary(self, ctx: AgentContext) -> dict:
        try:
            summary = self.search_engine.get_summary()
        except Exception as exc:
            logger.warning("Summary failed: %s", exc)
            summary = {}
        response = self._responder.build_response("summary", summary=summary)
        return {"response": response, "summary": summary}

    def _handle_clip_request(self, user_message: str, ctx: AgentContext) -> dict:
        """Auto-run a search from the message, then return the top hit as a
        ready-to-play clip (URL + timestamp + detected objects)."""
        query = self._clip_query_from_message(user_message)
        try:
            events = self.search_engine.search(query=query, top_k=1, extract_clips=False)
            ctx.last_events = [e.to_dict() for e in events]
        except Exception as exc:
            logger.warning("Clip search failed: %s", exc)
            ctx.last_events = []

        if not ctx.last_events:
            return {
                "response": (
                    "Je n'ai trouvé aucun clip correspondant à votre demande. "
                    "Essayez de préciser ce que vous cherchez (par exemple « la moto » "
                    "ou « la personne »)."
                ),
                "events": [],
            }

        top       = ctx.last_events[0]
        event_id  = top["id"]
        timestamp = top["timestamp"]
        objects   = top.get("detected_objects") or []
        objs_str  = ", ".join(objects) if objects else "objet détecté"
        clip_url  = f"{self._clip_base_url}/clip/{event_id}"

        response = (
            f"J'ai trouvé le clip. Voici la séquence enregistrée à {timestamp} "
            f"(objets détectés : {objs_str})."
        )
        return {
            "response":         response,
            "clip_url":         clip_url,
            "event_id":         event_id,
            "timestamp":        timestamp,
            "detected_objects": objects,
            "events":           ctx.last_events,
        }

    @staticmethod
    def _clip_query_from_message(message: str) -> str:
        """Map French keywords in the message to an English CLIP query, which
        the image-text model scores far better than raw French."""
        m = message.lower()
        has_moto   = "moto" in m
        has_person = "personne" in m
        if has_moto and has_person:
            return "person riding motorcycle"
        if has_moto:
            return "motorcycle"
        if has_person:
            return "person"
        # No recognised keyword — fall back to the raw message so other
        # subjects (voiture, camion, …) still return their best match.
        return message
