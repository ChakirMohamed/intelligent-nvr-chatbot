"""
Inference module for the trained TF-IDF + NN intent classifier.

Usage:
    from src.ml.intent_classifier import IntentClassifierInference

    clf = IntentClassifierInference()                   # loads from src/ml/models/
    result = clf.predict("montre-moi les événements d'hier")
    # {'intent': 'search', 'confidence': 0.97, 'all_scores': {...}}

    batch = clf.predict_batch(["bonjour", "envoie-moi le clip"])
"""
import os
import pickle
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


# ── architecture (must match train_intent_classifier.py) ─────────────────────
_HIDDEN1   = 128
_HIDDEN2   = 64
_DROPOUT1  = 0.3
_DROPOUT2  = 0.2


class _IntentNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, _HIDDEN1),
            nn.ReLU(),
            nn.Dropout(_DROPOUT1),
            nn.Linear(_HIDDEN1, _HIDDEN2),
            nn.ReLU(),
            nn.Dropout(_DROPOUT2),
            nn.Linear(_HIDDEN2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ── public inference class ────────────────────────────────────────────────────

class IntentClassifierInference:
    """
    Loads trained artifacts and runs intent classification inference.

    Args:
        model_dir: Directory containing intent_model.pt, tfidf_vectorizer.pkl,
                   and label_encoder.pkl.  Defaults to 'src/ml/models'.
    """

    def __init__(self, model_dir: str = "src/ml/models"):
        model_dir = os.path.abspath(model_dir)
        self._load_artifacts(model_dir)

    def _load_artifacts(self, model_dir: str) -> None:
        vec_path   = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        le_path    = os.path.join(model_dir, "label_encoder.pkl")
        model_path = os.path.join(model_dir, "intent_model.pt")

        for p in (vec_path, le_path, model_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Model artifact not found: {p}\n"
                    "Run  python src/ml/train_intent_classifier.py  first."
                )

        with open(vec_path, "rb") as fh:
            self._vectorizer = pickle.load(fh)

        with open(le_path, "rb") as fh:
            self._le = pickle.load(fh)

        self.classes: List[str] = list(self._le.classes_)
        input_dim   = len(self._vectorizer.get_feature_names_out())
        num_classes = len(self.classes)

        self._model = _IntentNet(input_dim, num_classes)
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            # weights_only not supported on older PyTorch
            state = torch.load(model_path, map_location="cpu")
        self._model.load_state_dict(state)
        self._model.eval()

    # ── inference helpers ─────────────────────────────────────────────────────

    def _vectorize(self, texts: List[str]) -> torch.Tensor:
        X = self._vectorizer.transform(texts).toarray().astype(np.float32)
        return torch.from_numpy(X)

    def _run(self, tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = self._model(tensor)
            probs  = torch.softmax(logits, dim=1).numpy()
        return probs

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, text: str) -> Dict:
        """
        Classify a single text.

        Returns:
            {
                "intent":     str,          # top predicted class
                "confidence": float,        # probability of top class
                "all_scores": dict          # {class_name: probability, ...}
            }
        """
        probs  = self._run(self._vectorize([text]))[0]   # shape (num_classes,)
        top    = int(np.argmax(probs))
        return {
            "intent":     self.classes[top],
            "confidence": float(probs[top]),
            "all_scores": {c: float(p) for c, p in zip(self.classes, probs)},
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Classify a list of texts in one forward pass.

        Returns:
            List of dicts, same structure as predict().
        """
        if not texts:
            return []
        probs_matrix = self._run(self._vectorize(texts))   # (n, num_classes)
        results = []
        for probs in probs_matrix:
            top = int(np.argmax(probs))
            results.append({
                "intent":     self.classes[top],
                "confidence": float(probs[top]),
                "all_scores": {c: float(p) for c, p in zip(self.classes, probs)},
            })
        return results
