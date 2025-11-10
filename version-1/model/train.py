"""
SignBridge v1.0 — Training Pipeline
MobileNetV2 fine-tuned on ASL Alphabet dataset
Target: >90% validation accuracy
Run: python model/train.py
"""

import os, sys, time, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ── PATHS ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "asl_alphabet", "asl_alphabet_train", "asl_alphabet_train")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "signbridge_v1.pth")
META_PATH  = os.path.join(MODEL_DIR, "class_names.json")

# ── CONFIG ───────────────────────────────────────────────────
IMG_SIZE   = 128
BATCH_SIZE = 32
EPOCHS     = 15
LR         = 1e-4
VAL_SPLIT  = 0.15
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Unfreeze last 3 blocks
    for param in model.features[:-3].parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return model.to(DEVICE)


def train():
    print(f"\n{'='*50}")
    print(f"  SignBridge v1 — Training")
    print(f"  Device : {DEVICE}")
    print(f"  Data   : {DATA_DIR}")
    print(f"{'='*50}\n")

    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Dataset not found at:\n  {DATA_DIR}")
        print("Download from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        sys.exit(1)

    train_tf, val_tf = get_transforms()

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    class_names  = full_dataset.classes
    num_classes  = len(class_names)
    print(f"  Classes found : {num_classes} → {class_names}\n")

    # Save class names for inference
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(class_names, f)

    # Train/val split
    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset = datasets.ImageFolder(DATA_DIR, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    best_acc = 0.0
    scaler   = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    for epoch in range(1, EPOCHS + 1):
        # ── TRAIN
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct  += preds.eq(labels).sum().item()
            total    += labels.size(0)

        train_acc  = correct / total * 100
        train_loss = running_loss / total

        # ── VALIDATE
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total * 100
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.1f}% | "
              f"Val: {val_acc:.1f}% | "
              f"{elapsed:.1f}s")

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "class_names": class_names,
                "val_acc"    : val_acc,
            }, MODEL_PATH)
            print(f"  ✓ Saved best model — {val_acc:.1f}%")

    print(f"\n  Training complete. Best val accuracy: {best_acc:.1f}%")
    print(f"  Model saved to: {MODEL_PATH}\n")


if __name__ == "__main__":
    train()
