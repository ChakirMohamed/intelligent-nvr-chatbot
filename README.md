<p align="center">
  <img src="docs/banner.png" alt="Intelligent NVR — Recherche vidéo sémantique par IA" width="100%">
</p>

<!--h1 align="center">🎥 Intelligent NVR — Recherche Vidéo Sémantique par IA</h1-->

<p align="center">
  <em>Interrogez vos vidéos de surveillance en français naturel — sans date, sans caméra, en moins de 300 ms.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/YOLOv8-fine--tuned-00A896">
  <img src="https://img.shields.io/badge/CLIP-ViT--B%2F32-028090">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/React-TS-61DAFB?logo=react&logoColor=white">
  <img src="https://img.shields.io/badge/license-academic-lightgrey">
</p>

---

<div align="center">


|---|---|
| **Institution** | Université Mohammed V – Faculté des Sciences, Rabat |
| **Filière** | Master Informatique |
| **Module** | Machine Learning & Deep Learning |
| **Encadrant** | Pr. Abdelhak Mahmoudi |
| **Co-encadrants** | Saad Frihi & Yasine Lehmiani |
| **Auteurs** | CHAKIR Mohamed · EL ASRY Soufiane |
| **Année** | 2025 – 2026 |

</div>

> 🔗 **Frontend (React + TypeScript) :** https://github.com/ChakirMohamed/front-end-nvr-chatbot

---

## 🔍 Problème résolu

Les NVR classiques imposent une revue manuelle pour retrouver un événement. Notre système permet de simplement demander :

> *« Quelqu'un a tagué mon mur le mois dernier, t'as quelque chose ? »*

et reçoit en réponse les timestamps exacts + clips `.mp4` téléchargeables, en **moins de 300 ms**.

---

## 🧠 Contribution ML & Deep Learning

Deux modèles entraînés **from scratch** par nos soins :

### 1 — Classifieur d'intentions (PyTorch)

Comprend ce que l'utilisateur veut faire à partir d'une phrase libre en français.

| | |
|---|---|
| **Dataset** | 300 phrases françaises annotées à la main (5 classes) |
| **Architecture** | TF-IDF (500 features) → Linear(500→128) → ReLU → Dropout → Linear(128→64) → ReLU → Dropout → Linear(64→5) |
| **Paramètres** | ~73 000 |
| **Entraînement** | 100 epochs, Adam lr=0.001, batch 16 |

**Résultats — effet de la taille du dataset :**

| Run | Dataset | Accuracy |
|---|---|---|
| Run 1 | 125 phrases | 76,0 % |
| Run 2 | 300 phrases | **91,7 %** |

<p align="center">
  <img src="src/ml/results/training_curves.png" alt="Courbes d'entraînement du classifieur" width="48%">
  <img src="src/ml/results/confusion_matrix.png" alt="Matrice de confusion" width="38%">
</p>

| Classe | Exemples |
|---|---|
| `search` | *« y'a eu quoi cette nuit ? »*, *« quelqu'un a tagué mon mur »* |
| `clip_request` | *« envoie-moi le clip »*, *« passe-moi la vidéo stp »* |
| `summary` | *« fais un résumé de la semaine »*, *« bilan des activités »* |
| `greeting` | *« bonjour »*, *« salut »*, *« wesh »* |
| `unknown` | *« quelle est la résolution ? »*, *« merci »* |

---

### 2 — Fine-tuning YOLOv8 sur VisDrone (Google Colab)

YOLOv8 de base est entraîné sur COCO (photos au sol). Sur des images de surveillance (vue plongeante, objets petits), il ne détecte presque rien.

**Dataset :** [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) — 6 471 images aériennes annotées, 10 classes, Université de Tianjin.

| Métrique | Avant | Après | Gain |
|---|---|---|---|
| mAP50 | 1,6 % | **25,6 %** | **× 16** |
| mAP50-95 | 0,7 % | **14,2 %** | **× 20** |

Entraînement : 15 epochs, GPU Tesla T4 (Google Colab), **37 minutes**.

<p align="center">
  <img src="src/ml/results/yolo_training_curves.png" alt="Courbes d'entraînement YOLOv8" width="60%">
</p>

<p align="center">
  <img src="src/ml/results/val_batch0_pred.jpg" alt="Détections du modèle fine-tuné sur VisDrone" width="60%">
  <br><em>Exemple de détections du modèle fine-tuné (validation VisDrone)</em>
</p>

---

## 🏗️ Architecture

<!-- Optionnel : remplace par ton schéma généré (docs/architecture.png) -->
<!-- <p align="center"><img src="docs/architecture.png" width="90%"></p> -->

```
Caméras IP (RTSP)
    ↓
Ingestion — OpenCV + MOG2 (filtre les frames statiques, −90 %)
    ↓
Détection — YOLOv8 fine-tuné + OpenVINO (~8 fps sur Intel Iris Xe)
    ↓
Indexation — CLIP ViT-B/32 → vecteurs 512d → FAISS IVF + SQLite
    ↓
Recherche — requête texte → CLIP → FAISS → SQLite → FFmpeg → .mp4
    ↓
Chatbot — Intent Classifier → route → réponse française
    ↓
API REST — FastAPI (/chat /search /clip /events /summary /health)
```

---

## 🖥️ Interface (Frontend React)

> Dépôt frontend : https://github.com/ChakirMohamed/front-end-nvr-chatbot

<!-- ============ CAPTURES FRONTEND ============ -->
<!-- Ajoute tes captures dans docs/ puis garde les lignes ci-dessous -->
<p align="center">
  <img src="docs/screenshot_dashboard.png" alt="Tableau de bord" width="80%">
  <br><em>Tableau de bord — métriques ML en temps réel</em>
</p>

<p align="center">
  <img src="docs/screenshot_assistant.png" alt="Assistant IA" width="80%">
  <br><em>Assistant IA — visualisation des scores du classifieur en direct</em>
</p>

<p align="center">
  <img src="docs/screenshot_search.png" alt="Recherche sémantique" width="80%">
  <br><em>Recherche sémantique en langage naturel</em>
</p>

---

## 💻 Hardware

| | |
|---|---|
| CPU | Intel Core i5-1145G7 |
| RAM | 24 Go |
| GPU inférence | Intel Iris Xe (OpenVINO) |
| GPU entraînement | Tesla T4 (Google Colab) |

---

## ⚙️ Installation

```bash
git clone https://github.com/ChakirMohamed/intelligent-nvr-chatbot.git
cd intelligent-nvr-chatbot
pip install -r requirements.txt
```

### Démo (sans caméra, sans clé API)

```bash
python demo.py
```

```
STEP 1 — Vidéo synthétique générée          [OK]
STEP 2 — 30 frames indexées (YOLO + CLIP)   [OK]
STEP 3 — 4 requêtes chatbot                 [OK]
Pipeline: YOLO ✓  CLIP ✓  FAISS ✓  SQLite ✓  Intent Classifier ✓
```

### API

```bash
uvicorn src.api.api:app --port 8000
# Swagger UI → http://localhost:8000/docs
```

### Frontend

```bash
git clone https://github.com/ChakirMohamed/front-end-nvr-chatbot.git
cd front-end-nvr-chatbot
npm install
npm run dev
# → http://localhost:5173
```

### Ré-entraîner le classifieur

```bash
python src/ml/train_intent_classifier.py
# → training_curves.png, confusion_matrix.png, intent_model.pt
```

---

## 🌐 API

| Méthode | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Dialogue multi-tours |
| `POST` | `/search` | Recherche sémantique |
| `GET` | `/clip/{id}` | Téléchargement `.mp4` |
| `GET` | `/events` | Liste des événements |
| `GET` | `/summary` | Résumé d'activité |
| `GET` | `/health` | Santé du système |

---

## 📊 Performances

| Métrique | Valeur |
|---|---|
| Détection YOLO | ~8 fps (Intel Iris Xe, OpenVINO) |
| Latence recherche | ~300 ms (FAISS IVF, ~11M vecteurs) |
| Réduction index | −90 % (filtrage mouvement MOG2) |
| Intent Classifier | **91,7 %** accuracy |
| YOLOv8 fine-tuning | **× 16** mAP50 |

---

## 📁 Structure

```
intelligent-nvr-chatbot/
├── src/
│   ├── ingestion/rtsp_reader.py
│   ├── detection/
│   │   ├── yolo_detector.py
│   │   └── models/yolov8n_visdrone_best.pt   ← modèle fine-tuné
│   ├── indexing/frame_indexer.py
│   ├── search/retrospective_search.py
│   ├── agent/
│   │   ├── chatbot_agent.py
│   │   └── local_responder.py
│   ├── api/api.py
│   └── ml/
│       ├── dataset/intent_dataset.csv        ← 300 phrases annotées
│       ├── train_intent_classifier.py
│       ├── intent_classifier.py
│       ├── models/                           ← .pt, .pkl sauvegardés
│       ├── results/                          ← courbes, matrices
│       └── YOLO_FINETUNING.md
├── docs/                                     ← bannière + captures README
├── demo.py
├── requirements.txt
└── .env.example
```

---

## 👥 Auteurs

**CHAKIR Mohamed** · **EL ASRY Soufiane**
*Master Informatique — Université Mohammed V, Faculté des Sciences, Rabat — 2025/2026*

**Soumission :** [forms.gle/pDmMm6HW2BRRN9ZL6](https://forms.gle/pDmMm6HW2BRRN9ZL6) · Deadline : 21 juin 2026
