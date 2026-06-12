"""
Multi-turn chatbot agent for the NVR system.

Classification strategy (hybrid):
  1. Local ML classifier (TF-IDF + NN) runs first — fast, no API cost.
  2. High-confidence greeting/unknown  → answer immediately, skip LLM call.
  3. High-confidence search/summary/clip → hint the LLM so it focuses on
     parameter extraction rather than re-classifying intent.
  4. Low-confidence or classifier unavailable → full LLM path unchanged.
"""
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── system prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Tu es un assistant IA intégré à un système NVR (Network Video Recorder) intelligent.
Tu comprends le français et l'anglais, et tu aides les agents de sécurité à retrouver
des événements vidéo, extraire des clips et obtenir des résumés d'activité.

Quand l'utilisateur pose une question, tu dois répondre UNIQUEMENT avec un objet JSON
valide (sans balises Markdown) respectant ce schéma :

{{
  "intent": "<search|clip_request|summary|camera_info|greeting|unknown>",
  "query": "<description en anglais de ce qu'on cherche dans les vidéos>",
  "params": {{
    "camera_id":    null | "<id>",
    "start_time":   null | "<YYYY-MM-DDTHH:MM:SS>",
    "end_time":     null | "<YYYY-MM-DDTHH:MM:SS>",
    "object_types": [] | ["person","car","motorcycle","bicycle","truck"],
    "event_id":     null | <int>
  }},
  "response": "<réponse en langage naturel à afficher>"
}}

Règles d'interprétation temporelle (date actuelle = {today}):
- "le mois dernier" -> start_time = premier jour du mois précédent, end_time = dernier jour
- "la semaine dernière" -> les 7 derniers jours
- "hier" -> date d'hier 00:00:00 à 23:59:59
- "ce matin" -> aujourd'hui 06:00 à 12:00
- "cette nuit" -> hier 22:00 à aujourd'hui 06:00

{ml_hint}
Exemples :
- "Quelqu'un a tagué mon mur le mois dernier" -> intent=search, query="graffiti tagging wall person"
- "Montre-moi le clip de l'événement 42" -> intent=clip_request, event_id=42
- "Résumé d'activité cette semaine caméra 2" -> intent=summary, camera_id="cam2"
- "Bonjour" -> intent=greeting
"""


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
        llm_provider: str = "anthropic",
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_history: int = 20,
        ml_model_dir: str = "src/ml/models",
        ml_confidence_threshold: float = 0.85,
    ):
        self.search_engine = search_engine
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.max_history = max_history
        self.ml_threshold = ml_confidence_threshold
        self._client = None

        if llm_provider == "anthropic":
            self.llm_model = llm_model or "claude-sonnet-4-6"
            self._init_anthropic()
        elif llm_provider == "openai":
            self.llm_model = llm_model or "gpt-4o-mini"
            self._init_openai()
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider!r}")

        self._ml_clf = self._load_ml_classifier(ml_model_dir)
        logger.info("ChatbotAgent ready: %s / %s  (ML classifier: %s)",
                    llm_provider, self.llm_model,
                    "loaded" if self._ml_clf else "unavailable")

    # ── LLM clients ──────────────────────────────────────────────────────────

    def _init_anthropic(self) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def _init_openai(self) -> None:
        import openai
        self._client = openai.OpenAI(api_key=self.api_key)

    # ── ML classifier ─────────────────────────────────────────────────────────

    def _load_ml_classifier(self, model_dir: str):
        try:
            from src.ml.intent_classifier import IntentClassifierInference
            return IntentClassifierInference(model_dir)
        except Exception as exc:
            logger.warning("ML classifier not loaded (%s) — LLM-only mode.", exc)
            return None

    def _ml_predict(self, text: str) -> Optional[Dict[str, Any]]:
        if self._ml_clf is None:
            return None
        try:
            return self._ml_clf.predict(text)
        except Exception as exc:
            logger.warning("ML predict failed: %s", exc)
            return None

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _system_prompt(self, ml_hint: str = "") -> str:
        hint_line = (
            f"NOTE: le classificateur local prédit intent='{ml_hint}' avec haute confiance. "
            f"Utilise cet intent sauf si le texte contredit clairement cette prédiction.\n"
            if ml_hint else ""
        )
        return _SYSTEM_PROMPT.format(
            today=datetime.now().strftime("%Y-%m-%d"),
            ml_hint=hint_line,
        )

    def _call_llm(self, messages: List[Message], ml_hint: str = "") -> str:
        msg_list = [{"role": m.role, "content": m.content} for m in messages]
        system = self._system_prompt(ml_hint)

        if self.llm_provider == "anthropic":
            resp = self._client.messages.create(
                model=self.llm_model,
                max_tokens=1024,
                system=system,
                messages=msg_list,
            )
            return resp.content[0].text

        resp = self._client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "system", "content": system}] + msg_list,
            max_tokens=1024,
        )
        return resp.choices[0].message.content

    # ── JSON parsing ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse(text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return {"intent": "unknown", "query": text, "params": {}, "response": text}

    # ── intent handlers ───────────────────────────────────────────────────────

    def _parse_times(self, params: dict):
        start = end = None
        for key, slot in [("start_time", "start"), ("end_time", "end")]:
            val = params.get(key)
            if val:
                try:
                    dt = datetime.fromisoformat(val)
                    if slot == "start":
                        start = dt
                    else:
                        end = dt
                except ValueError:
                    logger.warning("Could not parse time: %s", val)
        return start, end

    def _handle_search(self, parsed: dict, ctx: AgentContext) -> dict:
        p = parsed.get("params", {})
        start, end = self._parse_times(p)
        events = self.search_engine.search(
            query=parsed.get("query", ""),
            top_k=3,
            camera_id=p.get("camera_id"),
            start_time=start,
            end_time=end,
            object_types=p.get("object_types") or None,
            extract_clips=True,
        )
        ctx.last_events = [e.to_dict() for e in events]
        return {
            "response": parsed.get("response") or f"{len(events)} événement(s) trouvé(s).",
            "events":   ctx.last_events,
        }

    def _handle_summary(self, parsed: dict, ctx: AgentContext) -> dict:
        p = parsed.get("params", {})
        start, end = self._parse_times(p)
        summary = self.search_engine.get_summary(
            camera_id=p.get("camera_id"),
            start_time=start,
            end_time=end,
        )
        return {
            "response": parsed.get("response") or "Voici le résumé d'activité.",
            "summary":  summary,
        }

    def _handle_clip_request(self, parsed: dict, ctx: AgentContext) -> dict:
        event_id = (parsed.get("params") or {}).get("event_id")
        if event_id is not None:
            for e in ctx.last_events:
                if e["id"] == event_id:
                    return {
                        "response":  f"Clip disponible pour l'événement {event_id}.",
                        "clip_path": e.get("clip_path"),
                        "event":     e,
                    }
        return {
            "response":  "Événement introuvable. Lancez d'abord une recherche.",
            "clip_path": None,
        }

    # ── public chat API ───────────────────────────────────────────────────────

    def chat(self, user_message: str, ctx: AgentContext) -> dict:
        # ── Step 1: local ML pre-classification ──────────────────────────────
        ml_pred = self._ml_predict(user_message)
        confident = ml_pred is not None and ml_pred["confidence"] >= self.ml_threshold

        classification_info = {
            "ml":     ml_pred,
            "source": "llm",        # updated below
        }

        # ── Step 2: shortcut for simple intents (no LLM call needed) ─────────
        if confident and ml_pred["intent"] == "greeting":
            classification_info["source"] = "ml"
            classification_info["llm_intent"] = None
            return {
                "response":            "Bonjour ! Comment puis-je vous aider avec le système de surveillance ?",
                "classification_info": classification_info,
            }

        # ── Step 3: call LLM (with optional ML hint for complex intents) ─────
        ctx.conversation.append(Message(role="user", content=user_message))
        history = ctx.conversation[-self.max_history:]

        ml_hint = ml_pred["intent"] if confident else ""
        if ml_hint:
            classification_info["source"] = "ml+llm"

        raw = self._call_llm(history, ml_hint=ml_hint)
        ctx.conversation.append(Message(role="assistant", content=raw))

        parsed = self._parse(raw)
        intent = parsed.get("intent", "unknown")
        classification_info["llm_intent"] = intent

        # ── Step 4: route to handler ──────────────────────────────────────────
        if intent == "search":
            result = self._handle_search(parsed, ctx)
        elif intent == "summary":
            result = self._handle_summary(parsed, ctx)
        elif intent == "clip_request":
            result = self._handle_clip_request(parsed, ctx)
        elif intent == "greeting":
            result = {"response": parsed.get("response", "Bonjour ! Comment puis-je vous aider ?")}
        elif intent == "camera_info":
            result = {"response": parsed.get("response", "Interrogez /cameras pour la liste des caméras.")}
        else:
            result = {"response": parsed.get("response", "Je n'ai pas compris. Pouvez-vous reformuler ?")}

        result["classification_info"] = classification_info
        return result
