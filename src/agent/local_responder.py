"""
Local template-based response engine — no LLM API required.

Wraps IntentClassifierInference and maps each intent to a structured
French response, optionally incorporating search results or a summary.
"""
from typing import Any, Dict, List, Optional

from src.ml.intent_classifier import IntentClassifierInference


class LocalResponder:
    """Classify user messages and generate French responses from templates."""

    def __init__(self, model_dir: str = "src/ml/models"):
        self._clf = IntentClassifierInference(model_dir)

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, user_message: str) -> Dict[str, Any]:
        """Return {intent, confidence, all_scores} for *user_message*."""
        return self._clf.predict(user_message)

    def build_response(
        self,
        intent: str,
        search_results: Optional[List[Dict[str, Any]]] = None,
        summary: Optional[Dict[str, Any]] = None,
        user_message: str = "",
    ) -> str:
        """Build a French response string for *intent*, with optional data."""
        if intent == "greeting":
            return (
                "Bonjour ! Je suis votre assistant de surveillance IA. "
                "Je peux rechercher des événements, générer des clips .mp4 "
                "et créer des résumés d'activité. Comment puis-je vous aider ?"
            )
        if intent == "search":
            return self._format_search(search_results)
        if intent == "clip_request":
            return self._format_clips(search_results)
        if intent == "summary":
            return self._format_summary(summary)
        return (
            "Je peux vous aider à : rechercher des événements "
            "vidéo, télécharger des clips, ou générer des résumés d'activité."
        )

    def classify_and_respond(
        self,
        user_message: str,
        search_results: Optional[List[Dict[str, Any]]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Classify *user_message* and return intent + formatted response."""
        result = self._clf.predict(user_message)
        intent = result["intent"]
        response = self.build_response(intent, search_results, summary, user_message)
        return {
            "intent":     intent,
            "confidence": result["confidence"],
            "response":   response,
            "all_scores": result["all_scores"],
        }

    # ── response formatters ───────────────────────────────────────────────────

    def _format_search(self, results: Optional[List[Dict[str, Any]]]) -> str:
        if not results:
            return "Aucun événement trouvé correspondant à votre recherche."
        lines = [f"{len(results)} événement(s) trouvé(s) :"]
        for i, e in enumerate(results, 1):
            ts      = e.get("timestamp", "?")
            objects = ", ".join(e.get("detected_objects") or []) or "aucun objet"
            cam     = e.get("camera_id", "?")
            score   = e.get("score", 0.0)
            lines.append(
                f"  {i}. [{ts}]  cam={cam}  objets=[{objects}]  score={score:.3f}"
            )
        return "\n".join(lines)

    def _format_clips(self, results: Optional[List[Dict[str, Any]]]) -> str:
        if not results:
            return (
                "Aucun événement disponible pour l'extraction de clip. "
                "Lancez d'abord une recherche."
            )
        lines = ["Clips disponibles :"]
        for i, e in enumerate(results, 1):
            clip = e.get("clip_path") or "non extrait"
            ts   = e.get("timestamp", "?")
            lines.append(f"  {i}. Événement #{e.get('id', '?')}  [{ts}]  → {clip}")
        return "\n".join(lines)

    def _format_summary(self, summary: Optional[Dict[str, Any]]) -> str:
        if not summary:
            return "Résumé d'activité non disponible."
        total   = summary.get("total_events", 0)
        objects = summary.get("object_counts", {})
        hourly  = summary.get("hourly_distribution", {})
        lines   = [
            "Résumé d'activité :",
            f"  • Total d'événements : {total}",
        ]
        if objects:
            obj_str = ", ".join(
                f"{k}: {v}"
                for k, v in sorted(objects.items(), key=lambda x: -x[1])
            )
            lines.append(f"  • Objets détectés   : {obj_str}")
        if hourly:
            peak = max(hourly, key=hourly.get)
            lines.append(
                f"  • Heure de pointe   : {peak:02d}h00 ({hourly[peak]} événements)"
            )
        return "\n".join(lines)
