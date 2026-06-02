"""
SignBridge v2.0 — Training Pipeline
ViT-Tiny on MediaPipe landmark coordinates + LSTM temporal smoothing
Fixed for Windows: num_workers=0, correct DATA_DIR, added augmentation.
Run: python model/train_v2.py
"""

import os, sys, json, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import mediapipe as mp
import cv2

# ── PATHS (FIXED for your folder structure) ──────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CORRECT PATH (no "..", double nest because of asl_alphabet_train/asl_alphabet_train)
DATA_DIR = os.path.join(BASE_DIR, "data", "asl_alphabet", "asl_alphabet_train", "asl_alphabet_train")

MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "signbridge_v2.pth")
META_PATH  = os.path.join(MODEL_DIR, "class_names_v2.json")
LAND_CACHE = os.path.join(MODEL_DIR, "landmark_cache.npz")

# ── CONFIG ───────────────────────────────────────────────────
SEQ_LEN    = 15
LANDMARK_D = 63
BATCH_SIZE = 64
EPOCHS     = 30
LR         = 3e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MediaPipe
mp_hands = mp.solutions.hands

# ── LANDMARK EXTRACTION WITH AUGMENTATION ────────────────────

def augment_landmarks(lm_array: np.ndarray) -> np.ndarray:
    """
    Add small perturbations to landmarks to simulate real-world variation.
    This helps the model generalise to webcam lighting/position changes.
    """
    # Noise (std=0.01 ~ 1% of hand size)
    noise = np.random.randn(*lm_array.shape) * 0.01
    lm_aug = lm_array + noise

    # Random small rotation (max 5 degrees) around wrist (landmark 0)
    wrist = lm_aug[0:3].copy()
    theta = np.random.uniform(-0.087, 0.087)  # ±5 deg in radians
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    # Shift to wrist, rotate, shift back
    lm_rel = lm_aug.reshape(-1, 3) - wrist
    lm_rot = np.dot(lm_rel, rot_mat.T)
    lm_aug = (lm_rot + wrist).flatten()
    return lm_aug.astype(np.float32)

def extract_landmarks_from_image(img_path: str, hands) -> np.ndarray | None:
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0].landmark
    coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
    return coords

def build_landmark_cache(data_dir: str, cache_path: str, class_names: list):
    """Extract landmarks, apply augmentation, cache to disk."""
    print("  Extracting landmarks with augmentation (one-time, ~10 min)...")
    all_landmarks, all_labels = [], []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.5) as hands:
        for label_idx, cls in enumerate(class_names):
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            files = [f for f in os.listdir(cls_dir)
                     if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            count = 0
            for fname in files:
                lm = extract_landmarks_from_image(os.path.join(cls_dir, fname), hands)
                if lm is not None:
                    # Store original + 2 augmented versions per image
                    all_landmarks.append(lm)
                    all_labels.append(label_idx)
                    # Augmentations
                    for _ in range(2):
                        all_landmarks.append(augment_landmarks(lm))
                        all_labels.append(label_idx)
                    count += 1
            print(f"    {cls}: {count} images → {count*3} samples (augmented)")

    X = np.array(all_landmarks, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    np.savez(cache_path, X=X, y=y)
    print(f"  Cache saved: {cache_path}  ({len(X)} samples)\n")
    return X, y

# ── DATASET ──────────────────────────────────────────────────

class LandmarkSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        base = self.X[idx]
        noise = torch.randn(self.seq_len, LANDMARK_D) * 0.01
        seq = base.unsqueeze(0).expand(self.seq_len, -1) + noise
        return seq, self.y[idx]

# ── MODEL (ViT + LSTM) ───────────────────────────────────────

class LandmarkTransformer(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 128,
                 num_heads: int = 4, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_landmarks = 21
        self.landmark_dim = 3
        self.input_proj = nn.Linear(self.landmark_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_landmarks, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._attn_weights = None

    def forward(self, x, return_attn=False):
        if x.dim() == 3:
            x = x.mean(dim=1)
        B = x.shape[0]
        tokens = x.view(B, self.num_landmarks, self.landmark_dim)
        tokens = self.input_proj(tokens) + self.pos_embed
        out = self.transformer(tokens)
        out = self.norm(out)
        pooled = out.mean(dim=1)
        logits = self.classifier(pooled)
        if return_attn:
            attn = out.mean(dim=-1)
            attn = torch.softmax(attn, dim=-1)
            return logits, attn
        return logits

class TemporalSignBridge(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 128,
                 lstm_hidden: int = 256, lstm_layers: int = 2):
        super().__init__()
        self.vit = LandmarkTransformer(num_classes=embed_dim, embed_dim=embed_dim)
        self.vit.classifier = nn.Identity()
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=0.2, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_attn=False):
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        if return_attn:
            feats, attn = self.vit(x_flat, return_attn=True)
            attn = attn.view(B, T, -1).mean(dim=1)
        else:
            feats = self.vit(x_flat)
            attn = None
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        pooled = out[:, -1, :]
        logits = self.head(pooled)
        if return_attn:
            return logits, attn
        return logits

# ── TRAIN ────────────────────────────────────────────────────

def train():
    print(f"\n{'='*52}")
    print(f"  SignBridge v2 — Transformer + LSTM Training (Windows fixed)")
    print(f"  Device : {DEVICE}")
    print(f"  Data   : {DATA_DIR}")
    print(f"{'='*52}\n")

    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Dataset not found: {DATA_DIR}")
        sys.exit(1)

    # Class names
    class_names = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])
    num_classes = len(class_names)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(class_names, f)
    print(f"Classes: {class_names[:5]}... ({num_classes} total)")

    # Landmark cache
    if os.path.exists(LAND_CACHE):
        print("  Loading cached landmarks...")
        cache = np.load(LAND_CACHE)
        X, y = cache["X"], cache["y"]
        print(f"  Loaded {len(X)} samples\n")
    else:
        X, y = build_landmark_cache(DATA_DIR, LAND_CACHE, class_names)

    # Dataset + split
    dataset = LandmarkSequenceDataset(X, y, seq_len=SEQ_LEN)
    val_size = int(len(dataset) * 0.15)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # FIX for Windows: num_workers=0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)

    model = TemporalSignBridge(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.time()
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(seqs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * seqs.size(0)
            _, preds = logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
                logits = model(seqs)
                _, preds = logits.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100
        train_acc = correct / total * 100
        train_loss = running_loss / total
        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "class_names": class_names,
                "val_acc": val_acc,
                "config": {"seq_len": SEQ_LEN, "landmark_d": LANDMARK_D}
            }, MODEL_PATH)
            print(f"  ✓ Saved — {val_acc:.1f}%")

    print(f"\n  Done. Best val accuracy: {best_acc:.1f}%")
    print(f"  Model saved to: {MODEL_PATH}\n")

if __name__ == "__main__":
    train()