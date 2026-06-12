from src.ml.intent_classifier import IntentClassifierInference

clf = IntentClassifierInference()

tests = [
    "quelqu'un a tague mon mur hier",
    "envoie moi la video",
    "fais un resume de la semaine",
    "bonjour",
    "quelle heure est il",
    "y a t-il eu une personne la nuit derniere",
    "telecharge le clip du premier resultat",
    "wesh",
    "je veux une clip d'un personne avex un chapeau rouge",
]

print("\n── Test Intent Classifier ──────────────────────")
for t in tests:
    r = clf.predict(t)
    bar = "█" * int(r["confidence"] * 20)
    print(f"  {r['intent']:15} {r['confidence']*100:5.1f}%  {bar}")
    print(f"  → '{t}'")
    print()