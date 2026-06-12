"""
TF-IDF + Feedforward Neural Network intent classifier.
Trains on src/ml/dataset/intent_dataset.csv and saves model artifacts +
result plots to src/ml/models/ and src/ml/results/.

Usage:
    python src/ml/train_intent_classifier.py
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")   # headless — save to file, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(_HERE, "dataset", "intent_dataset.csv")
MODEL_DIR   = os.path.join(_HERE, "models")
RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── hyper-parameters ──────────────────────────────────────────────────────────
MAX_FEATURES = 500
NGRAM_RANGE  = (1, 2)
HIDDEN1      = 128
HIDDEN2      = 64
DROPOUT1     = 0.3
DROPOUT2     = 0.2
EPOCHS       = 100
BATCH_SIZE   = 16
LR           = 1e-3
TEST_SIZE    = 0.20


# ── model ─────────────────────────────────────────────────────────────────────

class IntentNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN1),
            nn.ReLU(),
            nn.Dropout(DROPOUT1),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Dropout(DROPOUT2),
            nn.Linear(HIDDEN2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ── dataset ───────────────────────────────────────────────────────────────────

class IntentDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── helpers ───────────────────────────────────────────────────────────────────

def _evaluate(model: IntentNet, loader: DataLoader, criterion) -> tuple[float, float]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            n          += len(yb)
    return total_loss / n, correct / n


def _predict_all(model: IntentNet, loader: DataLoader) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.extend(model(xb).argmax(1).numpy())
    return np.array(preds)


# ── main training routine ─────────────────────────────────────────────────────

PREV_ACCURACY = 76.0  # Run 1 — 125 samples


def _archive_run1_results() -> None:
    """Rename existing result files with _run1_125samples suffix before overwriting."""
    import shutil
    for fname in ("training_curves.png", "confusion_matrix.png", "classification_report.txt"):
        src = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(src):
            stem, ext = os.path.splitext(fname)
            dst = os.path.join(RESULTS_DIR, f"{stem}_run1_125samples{ext}")
            shutil.copy2(src, dst)
            print(f"       [archived] {fname} -> {os.path.basename(dst)}")


def main() -> None:
    # 1 ── Load dataset ────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Intent Classifier - TF-IDF + Feedforward NN")
    print("=" * 60)
    _archive_run1_results()
    df = pd.read_csv(DATA_PATH)
    print(f"\n[1/6] Dataset loaded: {len(df)} samples")
    print(df["intent"].value_counts().rename("count").to_string())

    # 2 ── Feature extraction ──────────────────────────────────────────────────
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        strip_accents=None,    # keep French accents — they carry meaning
        analyzer="word",
    )
    X = vectorizer.fit_transform(df["text"]).toarray().astype(np.float32)
    actual_features = X.shape[1]
    print(f"\n[2/6] TF-IDF fitted -> {actual_features} features  (max={MAX_FEATURES}, ngrams={NGRAM_RANGE})")

    # 3 ── Label encoding + split ──────────────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(df["intent"])
    classes = list(le.classes_)
    print(f"\n[3/6] Classes ({len(classes)}): {classes}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"       Split -> {len(X_tr)} train  /  {len(X_te)} test  (stratified)")

    tr_loader = DataLoader(IntentDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    te_loader = DataLoader(IntentDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 4 ── Model + training ────────────────────────────────────────────────────
    model     = IntentNet(actual_features, len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[4/6] Model architecture:")
    print(f"       Input({actual_features}) -> Linear({HIDDEN1}) -> ReLU -> Dropout({DROPOUT1})")
    print(f"       -> Linear({HIDDEN2}) -> ReLU -> Dropout({DROPOUT2}) -> Linear({len(classes)})")
    print(f"       Total parameters: {total_params:,}")
    print(f"\n       Training for {EPOCHS} epochs  (batch={BATCH_SIZE}, lr={LR}) ...\n")

    tr_losses, te_losses = [], []
    tr_accs,   te_accs   = [], []

    for epoch in range(1, EPOCHS + 1):
        # ── train one epoch ──────────────────────────────────────────────────
        model.train()
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            ep_loss    += loss.item() * len(yb)
            ep_correct += (logits.argmax(1) == yb).sum().item()
            ep_n       += len(yb)

        tr_loss = ep_loss / ep_n
        tr_acc  = ep_correct / ep_n
        te_loss, te_acc = _evaluate(model, te_loader, criterion)

        tr_losses.append(tr_loss);  te_losses.append(te_loss)
        tr_accs.append(tr_acc);     te_accs.append(te_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}"
                f"  |  Train  loss={tr_loss:.4f}  acc={tr_acc:.4f}"
                f"  |  Test   loss={te_loss:.4f}  acc={te_acc:.4f}"
            )

    print(f"\n  [OK] Final test accuracy: {te_accs[-1]:.4f}  ({te_accs[-1]*100:.1f} %)")

    # 5 ── Save model artifacts ────────────────────────────────────────────────
    print(f"\n[5/6] Saving model artifacts to {MODEL_DIR} ...")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "intent_model.pt"))
    print("       [OK] intent_model.pt")

    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as fh:
        pickle.dump(vectorizer, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print("       [OK] tfidf_vectorizer.pkl")

    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as fh:
        pickle.dump(le, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print("       [OK] label_encoder.pkl")

    # 6 ── Result visualisations ───────────────────────────────────────────────
    print(f"\n[6/6] Saving result plots to {RESULTS_DIR} ...")

    # — training curves ────────────────────────────────────────────────────────
    epochs_range = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Intent Classifier — Training Curves", fontsize=13, fontweight="bold")

    ax1.plot(epochs_range, tr_losses, label="Train",  color="#2196F3", linewidth=1.5)
    ax1.plot(epochs_range, te_losses, label="Test",   color="#F44336", linewidth=1.5, linestyle="--")
    ax1.set_title("Cross-Entropy Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs_range, [a * 100 for a in tr_accs], label="Train", color="#2196F3", linewidth=1.5)
    ax2.plot(epochs_range, [a * 100 for a in te_accs], label="Test",  color="#F44336", linewidth=1.5, linestyle="--")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 105); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=130, bbox_inches="tight")
    plt.close()
    print("       [OK] training_curves.png")

    # — confusion matrix ───────────────────────────────────────────────────────
    y_pred = _predict_all(model, te_loader)
    cm     = confusion_matrix(y_te, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, linecolor="white", ax=ax,
    )
    ax.set_title("Confusion Matrix — Test Set", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=10); ax.set_ylabel("True", fontsize=10)
    plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=130, bbox_inches="tight")
    plt.close()
    print("       [OK] confusion_matrix.png")

    # — classification report ─────────────────────────────────────────────────
    report = classification_report(y_te, y_pred, target_names=classes, digits=3)
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("Intent Classification Report\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"Dataset     : {len(df)} samples  |  {len(classes)} classes\n")
        fh.write(f"Train / Test: {len(X_tr)} / {len(X_te)}\n")
        fh.write(f"TF-IDF      : max_features={MAX_FEATURES}, ngram_range={NGRAM_RANGE}\n")
        fh.write(f"Architecture: {actual_features}->{HIDDEN1}->{HIDDEN2}->{len(classes)}\n")
        fh.write(f"Epochs      : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}\n")
        fh.write(f"Final test accuracy: {te_accs[-1]*100:.2f} %\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(report)
    print("       [OK] classification_report.txt")

    print("\n" + "=" * 60)
    print(f"  Classification Report (test set)\n")
    print(report)
    print("=" * 60)

    # Verify all 6 expected files exist
    artifacts = [
        os.path.join(MODEL_DIR, "intent_model.pt"),
        os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
        os.path.join(MODEL_DIR, "label_encoder.pkl"),
        os.path.join(RESULTS_DIR, "training_curves.png"),
        os.path.join(RESULTS_DIR, "confusion_matrix.png"),
        os.path.join(RESULTS_DIR, "classification_report.txt"),
    ]
    print("\n  Saved files:")
    for path in artifacts:
        size = os.path.getsize(path)
        status = "[OK]" if os.path.exists(path) else "[MISSING]"
        print(f"    {status}  {os.path.relpath(path, _HERE)}  ({size:,} bytes)")
    print()

    print("=" * 60)
    print(f"  Run 1 (125 samples): {PREV_ACCURACY:.1f}% | Run 2 (300 samples): {te_accs[-1]*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
