"""
SignBridge v1.0 — Inference Engine
MediaPipe hand detection + MobileNetV2 letter classification
"""

import os, json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "signbridge_v1.pth")
META_PATH  = os.path.join(BASE_DIR, "model", "class_names.json")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 128
CONF_THRESH = 0.75   # minimum confidence to accept prediction

mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils


class SignPredictor:

    def __init__(self):
        self.model       = None
        self.class_names = []
        self.transform   = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._load_model()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}\n"
                f"Run: python model/train.py"
            )
        checkpoint       = torch.load(MODEL_PATH, map_location=DEVICE)
        self.class_names = checkpoint["class_names"]
        num_classes      = len(self.class_names)

        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_channel, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        self.model = model.to(DEVICE)
        print(f"[SignBridge v1] Model loaded — {num_classes} classes | {DEVICE}")

    def predict_frame(self, frame: np.ndarray) -> dict:
        """
        Run inference on a single BGR frame.
        Returns prediction dict with letter, confidence, landmarks.
        """
        result = {
            "letter"    : None,
            "confidence": 0.0,
            "landmarks" : [],
            "hand_bbox" : None,
            "detected"  : False,
        }

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = self.hands.process(rgb)

        if not output.multi_hand_landmarks:
            return result

        result["detected"] = True
        h, w = frame.shape[:2]

        # Use first detected hand
        hand_lm = output.multi_hand_landmarks[0]

        # Extract landmark positions for overlay
        lm_points = []
        xs, ys = [], []
        for lm in hand_lm.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_points.append({"x": cx, "y": cy, "z": float(lm.z)})
            xs.append(cx); ys.append(cy)

        result["landmarks"] = lm_points

        # Bounding box with padding
        pad = 20
        x1  = max(0, min(xs) - pad)
        y1  = max(0, min(ys) - pad)
        x2  = min(w, max(xs) + pad)
        y2  = min(h, max(ys) + pad)
        result["hand_bbox"] = [x1, y1, x2, y2]

        # Crop hand region and classify
        hand_crop = frame[y1:y2, x1:x2]
        if hand_crop.size == 0:
            return result

        tensor = self.transform(hand_crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            conf, idx = probs.max(0)

        conf_val = conf.item()
        if conf_val >= CONF_THRESH:
            result["letter"]     = self.class_names[idx.item()]
            result["confidence"] = round(conf_val * 100, 1)

            # Top 3 predictions
            top3_vals, top3_idxs = probs.topk(3)
            result["top3"] = [
                {
                    "letter": self.class_names[i.item()],
                    "conf"  : round(v.item() * 100, 1),
                }
                for v, i in zip(top3_vals, top3_idxs)
            ]

        return result

    def draw_landmarks(self, frame: np.ndarray, landmarks: list,
                       bbox: list = None) -> np.ndarray:
        """Draw hand landmarks and bounding box on frame."""
        overlay = frame.copy()

        # Draw landmark points
        connections = [
            (0,1),(1,2),(2,3),(3,4),       # thumb
            (0,5),(5,6),(6,7),(7,8),       # index
            (0,9),(9,10),(10,11),(11,12),  # middle
            (0,13),(13,14),(14,15),(15,16),# ring
            (0,17),(17,18),(18,19),(19,20),# pinky
            (5,9),(9,13),(13,17),          # palm
        ]
        for a, b in connections:
            if a < len(landmarks) and b < len(landmarks):
                pt1 = (landmarks[a]["x"], landmarks[a]["y"])
                pt2 = (landmarks[b]["x"], landmarks[b]["y"])
                cv2.line(overlay, pt1, pt2, (0, 200, 255), 2)

        for lm in landmarks:
            cv2.circle(overlay, (lm["x"], lm["y"]), 4, (255, 255, 255), -1)
            cv2.circle(overlay, (lm["x"], lm["y"]), 4, (0, 150, 255),  1)

        # Draw bounding box
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 150), 2)

        return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    def close(self):
        self.hands.close()
