# 🎥 Intelligent NVR – Recherche Vidéo Sémantique par IA

> **Un agent conversationnel qui comprend vos questions en français et retrouve n'importe quel événement dans vos vidéos de surveillance – sans connaître la date, l'heure, ni la caméra.**

<div align="center">

| | |
|---|---|
| **Institution** | Université Mohammed V – Faculté des Sciences, Rabat |
| **Filière** | Master Informatique |
| **Module** | Machine Learning & Deep Learning |
| **Encadrant** | Pr. Abdelhak Mahmoudi |
| **Co-encadrants** | Saad Frihi & Yasine Lehmiani |
| **Auteurs** | CHAKIR Mohamed · EL ASRY Soufiane |
| **Année** | 2025 – 2026 |

</div>

---

## 🔍 Le problème qu'on résout

Les systèmes de vidéosurveillance classiques génèrent des quantités massives de données vidéo. Retrouver un événement précis impose une revue manuelle fastidieuse – souvent des heures pour une question de minutes.

**Exemple concret :**

> *"Quelqu'un a tagué mon mur le mois dernier – t'as quelque chose ?"*

Avec un NVR classique, l'agent doit parcourir manuellement des semaines d'enregistrements. Avec notre système, il tape cette phrase et obtient une réponse en **moins de 300 ms**, avec les timestamps exacts et un fichier `.mp4` téléchargeable.

---

## ✅ Ce que fait notre système

- 🗣️ **Comprend le langage naturel français** – pas besoin de dates ou de noms de caméra
- 🔍 **Recherche rétrospective** – fouille des mois d'enregistrements par description visuelle
- ⏱️ **Timestamps exacts** – date, heure, minute et seconde de chaque événement
- 🎥 **Export .mp4** – clip téléchargeable, utilisable comme preuve
- 🎬 **Chatbot multi-tours** – affine la recherche via une conversation naturelle
- 💡 **Compatible RTSP universel** – fonctionne avec n'importe quelle caméra IP du marché
- 🔒 **100 % local** – pas de cloud, pas de licence, déployable sur site

---

## 🧠 Contribution Machine Learning & Deep Learning

Ce projet s'inscrit dans le module ML & Deep Learning. Nous avons entraîné **deux modèles from scratch**, avec datasets construits à la main, courbes d'entraînement et métriques quantitatives.

### Modèle 1 – Classifieur d'intentions (NLP)

Le chatbot doit comprendre ce que l'utilisateur veut faire à partir d'une phrase libre en français. Plutôt que d'utiliser des règles à la main (`if "montre" in message`), nous avons entraîné un **réseau de neurones** pour cette tâche.

#### Pourquoi cette approche ?

Une règle manuelle casse facilement :
- `"montre-moi la vidéo"` contient à la fois `montre` (→ search ?) et `vidéo` (→ clip ?)
- `"j'aimerais voir ce qui s'est passé hier"` ne contient aucun mot-clé évident

Un modèle entraîné sur des exemples variés gère naturellement ces cas.

#### Construction du dataset

Nous avons annoté **300 phrases françaises** réparties en 5 classes :

| Classe | Exemples réels du dataset |
|---|---|
| `search` | *"quelqu'un a tagué mon mur le mois dernier"*, *"y'a eu quoi cette nuit ?"* |
| `clip_request` | *"envoie-moi la vidéo du 15 mai"*, *"passe-moi le clip stp"* |
| `summary` | *"fais-moi un résumé de la semaine"*, *"bilan des activités nocturnes"* |
| `greeting` | *"bonjour"*, *"salut"*, *"wesh"*, *"cc"* |
| `unknown` | *"quelle est la résolution des caméras ?"*, *"merci"*, *"à bientôt"* |

Les phrases incluent différents styles : formel, informel, argot (`stp`, `qqun`, `dl`), variantes avec fautes – pour que le modèle soit robuste aux vraies saisies utilisateur.

#### Architecture

```
Phrase utilisateur (texte brut)
        ↓
TF-IDF vectorizer (500 features, uni + bigrammes)
        ↓
Vecteur de 500 nombres (représentation numérique de la phrase)
        ↓
Linear(500 → 128) → ReLU → Dropout(0.3)
        ↓
Linear(128 → 64)  → ReLU → Dropout(0.2)
        ↓
Linear(64 → 5)    → score pour chaque classe
        ↓
Classe prédite + niveau de confiance
```

**~73 000 paramètres** – Adam (lr=0.001), CrossEntropyLoss, 100 epochs, batch 16.

#### Résultats

Nous avons réalisé **deux runs** pour démontrer l'effet de la taille du dataset :

| Run | Taille du dataset | Accuracy (test) | Observation |
|---|---|---|---|
| Run 1 | 125 phrases | **76,0 %** | Modèle fonctionnel mais limité |
| Run 2 | 300 phrases | **91,7 %** | +15,7 pts – plus de données = meilleur modèle |

Cette comparaison illustre directement le concept de **learning curve** : à architecture identique, agrandir le dataset améliore significativement les performances. Les courbes loss/accuracy montrent que le modèle généralise (la validation suit l'entraînement) sans mémorisation.

```
Accuracy finale sur 60 phrases jamais vues pendant l'entraînement : 91,7 %
```

#### Inférence

```python
from src.ml.intent_classifier import IntentClassifierInference

clf = IntentClassifierInference()
result = clf.predict("quelqu'un a tagué mon mur le mois dernier")
# → {"intent": "search", "confidence": 0.97}
```

---

### Modèle 2 – Fine-tuning YOLOv8 pour la surveillance

YOLOv8 est un modèle de détection d'objets state-of-the-art. Dans sa version originale, il est entraîné sur le dataset COCO – des photos prises au sol, avec des objets en gros plan. Or, une caméra de surveillance voit le monde **d'en haut**, avec des objets petits et denses.

**Preuve :** sur le dataset VisDrone (images de surveillance), YOLOv8 sans adaptation obtient **1,6 % de mAP50** – il détecte à peine un objet sur soixante.

#### Dataset : VisDrone

[VisDrone](https://github.com/VisDrone/VisDrone-Dataset) est un dataset académique public de l'Université de Tianjin contenant des images prises par des drones au-dessus de villes, annotées manuellement (10 classes : piéton, voiture, bus, moto, vélo...). La vue plongeante est proche des angles réels d'une caméra de surveillance.

| Paramètre | Valeur |
|---|---|
| Images d'entraînement | 6 471 |
| Images de validation | 548 |
| Classes | 10 (piéton, voiture, bus, moto, vélo, camion, tricycle, camionnette...) |
| Type de vue | Aérienne / plongeante |

#### Entraînement (Google Colab, GPU Tesla T4)

Notre PC (Intel Iris Xe intégré) ne dispose pas de GPU dédié – l'entraînement YOLO aurait pris ~10h en local. Nous avons utilisé **Google Colab** (GPU gratuit Tesla T4) pour un entraînement de **37 minutes**.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")         # modèle pré-entraîné (COCO)
model.train(
    data="VisDrone.yaml",          # notre domaine cible
    epochs=15,
    imgsz=640,
    batch=16,
)
```

#### Résultats

| Métrique | Avant fine-tuning | Après fine-tuning | Amélioration |
|---|---|---|---|
| mAP50 | 0,016 (1,6 %) | **0,256 (25,6 %)** | **×16** |
| mAP50-95 | 0,007 (0,7 %) | **0,142 (14,2 %)** | **×20** |

**Résultats par classe :**

| Classe | mAP50 | Note |
|---|---|---|
| voiture | **0,685** | Gros objet, bien visible de haut |
| bus | 0,350 | Bon |
| piéton | 0,263 | Correct |
| vélo | 0,039 | Petit objet vu de haut – limite connue de VisDrone |

La progression de la loss et du mAP sur 15 epochs est visible dans `src/ml/results/yolo_training_curves.png`.

---

## 🏗️ Architecture du système

```
────────────────────────────────────────────────────────────────────
│                        API REST – FastAPI                         │
│          /chat   /search   /clip   /events   /summary             │
──────────────────────────┬────────────────────────────────────────
                          │
         ─────────────────▼─────────────────
         │         Agent conversationnel    │
         │  Classifieur d'intentions (NN)   │
         │  → route vers la bonne fonction  │
         ─────────────────┬─────────────────
                          │
    ──────────────────────▼──────────────────────
    │           Recherche rétrospective           │
    │  Requête texte → CLIP → FAISS → SQLite      │
    │  → FFmpeg extrait le clip .mp4              │
    ──────────────┬─────────────┬────────────────
                  │             │
    ──────────────▼────  ───────▼───────────────
    │  FAISS IVF index │  │  SQLite metadata DB  │
    │  ~11M vecteurs   │  │  timestamps, objets  │
    ──────────────┬────  ───────┬───────────────
                  │             │
    ──────────────▼─────────────▼───────────────
    │               Frame Indexer                │
    │  CLIP ViT-B/32 (embedding 512 dimensions) │
    │  + YOLOv8 fine-tuné (détection d'objets)  │
    ────────────────────┬───────────────────────
                        │
    ────────────────────▼───────────────────────
    │              Ingestion RTSP                │
    │  OpenCV lit le flux → MOG2 filtre (-90 %) │
    │  → 1 frame par 2 secondes indexée          │
    ────────────────────────────────────────────
                        │
              Caméras IP / Flux RTSP
           (compatible toute marque)
```

---

## 💻 Matériel utilisé

| Composant | Spécification |
|---|---|
| CPU | Intel Core i5-1145G7 @ 2,6 GHz |
| RAM | 24 Go |
| GPU | Intel Iris Xe (intégré) – utilisé via OpenVINO |
| OS | Windows 11 |
| GPU entraînement | Tesla T4 (Google Colab) |

> **Aucune carte graphique NVIDIA requise.** La détection YOLO tourne via Intel OpenVINO sur Iris Xe (~8 fps). Les embeddings CLIP tournent sur CPU.

---

## 📁 Structure du projet

```
intelligent-nvr-chatbot/
├── src/
│   ├── ingestion/
│   │   └── rtsp_reader.py              # Lecture RTSP, filtrage MOG2
│   ├── detection/
│   │   └── yolo_detector.py            # YOLOv8 fine-tuné + OpenVINO
│   │   └── models/
│   │       └── yolov8n_visdrone_best.pt # Modèle fine-tuné (6 Mo)
│   ├── indexing/
│   │   └── frame_indexer.py            # CLIP embeddings → FAISS + SQLite
│   ├── search/
│   │   └── retrospective_search.py     # Texte → FAISS → SQLite → .mp4
│   ├── agent/
│   │   ├── chatbot_agent.py            # Agent multi-tours
│   │   └── local_responder.py          # Réponses sans API LLM
│   ├── api/
│   │   └── api.py                      # FastAPI – 7 endpoints
│   └── ml/
│       ├── dataset/
│       │   └── intent_dataset.csv      # 300 phrases annotées
│       ├── train_intent_classifier.py  # Script d'entraînement complet
│       ├── intent_classifier.py        # Module d'inférence
│       ├── models/                     # Modèles sauvegardés (.pt, .pkl)
│       ├── results/                    # Courbes + matrices de confusion
│       └── YOLO_FINETUNING.md         # Documentation fine-tuning YOLO
├── demo.py                             # Démo end-to-end (sans caméra réelle)
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Installation et lancement

### 1. Cloner le dépôt

```bash
git clone https://github.com/ChakirMohamed/intelligent-nvr-chatbot.git
cd intelligent-nvr-chatbot
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer la démo (sans caméra, sans clé API)

```bash
python demo.py
```

La démo génère une vidéo synthétique avec OpenCV, indexe ses frames, et exécute 4 requêtes chatbot pour valider l'ensemble du pipeline.

**Sortie attendue :**
```
STEP 1 – Génération vidéo synthétique          [OK]
STEP 2 – YOLO + CLIP → 30 frames indexées      [OK]
STEP 3 – 4 requêtes chatbot avec réponses      [OK]
Pipeline: YOLO ✓  CLIP ✓  FAISS ✓  SQLite ✓
          Intent Classifier ✓  Chatbot ✓
```

### 4. Lancer l'API

```bash
uvicorn src.api.api:app --host 0.0.0.0 --port 8000 --reload
```

Documentation interactive disponible sur `http://localhost:8000/docs`

### 5. Ré-entraîner le classifieur d'intentions (optionnel)

```bash
python src/ml/train_intent_classifier.py
# Génère : intent_model.pt, training_curves.png, confusion_matrix.png
```

---

## 📋 Référence API

| Méthode | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Dialogue multi-tours avec l'agent IA |
| `POST` | `/search` | Recherche sémantique par description |
| `GET` | `/clip/{event_id}` | Téléchargement du clip `.mp4` |
| `GET` | `/events` | Liste filtrée des événements détectés |
| `GET` | `/summary` | Résumé d'activité sur une période |
| `GET` | `/cameras` | État des caméras connectées |
| `GET` | `/health` | Santé du système |

**Exemple – recherche sémantique :**

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "homme avec veste rouge près de l entrée", "top_k": 3}'
```

```json
{
  "results": [
    {
      "event_id": "evt_20260515_143022_cam2",
      "camera_id": "cam_entrance_2",
      "timestamp": "2026-05-15T14:30:22",
      "detected_objects": ["person"],
      "similarity_score": 0.87,
      "clip_url": "/clip/evt_20260515_143022_cam2"
    }
  ],
  "search_time_ms": 287
}
```

---

## 🎬 Scénario de démonstration

> Un agent de sécurité suspecte un acte de vandalisme le mois dernier.
> Il ne connaît ni la date, ni l'heure, ni la caméra.

```
User : "Quelqu'un a tagué mon mur le mois dernier, t'as quelque chose ?"

→ Classifieur d'intentions : search (confiance 97 %)
→ CLIP encode la requête → vecteur 512 dimensions
→ FAISS recherche parmi ~11M frames → résultats en 287 ms
→ SQLite filtre par période

Bot  : "3 événements trouvés :
        📍 Cam 2 – 15 mai 2026 à 02:17:43 – personne près du mur sud
        📍 Cam 1 – 22 mai 2026 à 01:53:11 – silhouette détectée
        📍 Cam 3 – 28 mai 2026 à 03:05:22 – mouvement nocturne"

User : "Envoie-moi la vidéo du 15 mai"

→ FFmpeg extrait le segment ±30 s autour de 02:17:43
→ Fichier .mp4 disponible en téléchargement

Bot  : [Lien .mp4 → evt_20260515_021743_cam2]
```

---

## 📊 Performances mesurées

| Métrique | Valeur | Contexte |
|---|---|---|
| Détection YOLO | ~8 fps | Intel Iris Xe via OpenVINO |
| Latence recherche sémantique | ~300 ms | FAISS IVF sur ~11M vecteurs |
| Réduction index MOG2 | ~90 % | Filtrage frames statiques |
| Accuracy Intent Classifier | **91,7 %** | 60 phrases de test jamais vues |
| Amélioration mAP50 YOLO | **×16** | 1,6 % → 25,6 % après fine-tuning |

---

## 👥 Auteurs

**CHAKIR Mohamed**
Université Mohammed V – Faculté des Sciences Rabat

**EL ASRY Soufiane**
Université Mohammed V – Faculté des Sciences Rabat

---

## 📅 Soumission

- **Deadline :** 21 juin 2026
- **Formulaire :** [https://forms.gle/pDmMm6HW2BRRN9ZL6](https://forms.gle/pDmMm6HW2BRRN9ZL6)

**Livrables :**
1. Lien Google Drive vers la présentation PPTX
2. Ce dépôt GitHub (code + README)
3. Lien Google Drive vers la vidéo de démonstration (≤ 7 minutes)
