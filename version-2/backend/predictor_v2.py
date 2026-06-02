"""
SignBridge v2.0 — Inference Engine (Stable)
- Lower MediaPipe thresholds for better detection
- Mild CLAHE preprocessing
- Fallback when hand lost
- No aggressive preprocessing or smoothing
"""

import os, json, collections
import cv2
import numpy as np
import torch
import mediapipe as mp
import sys

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "model"))

MODEL_PATH = os.path.join(BASE_DIR, "model", "signbridge_v2.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN    = 15
CONF_THRESH = 0.90
HOLD_FRAMES = 25

mp_hands = mp.solutions.hands

def _load_model_class():
    import importlib.util
    train_path = os.path.join(BASE_DIR, "model", "train_v2.py")
    spec   = importlib.util.spec_from_file_location("train_v2", train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TemporalSignBridge

class SignPredictorV2:

    def __init__(self):
        self.model       = None
        self.class_names = []
        self.frame_buffer = collections.deque(maxlen=SEQ_LEN)
        self.hold_letter  = None
        self.hold_count   = 0
        self.confirmed    = False

        # Fallback state
        self.last_valid_landmarks = None
        self.last_known_lm_pts    = []
        self.last_known_bbox      = None
        self.no_hand_counter      = 0

        # MediaPipe with lower thresholds
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
        )
        self._load_model()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"v2 model not found: {MODEL_PATH}\n"
                f"Run: python model/train_v2.py"
            )
        TemporalSignBridge = _load_model_class()
        checkpoint         = torch.load(MODEL_PATH, map_location=DEVICE)
        self.class_names   = checkpoint["class_names"]
        num_classes        = len(self.class_names)

        model = TemporalSignBridge(num_classes=num_classes)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        self.model = model.to(DEVICE)
        print(f"[SignBridge v2] Model loaded — {num_classes} classes | {DEVICE}")

    def _preprocess_for_mediapipe(self, frame: np.ndarray) -> np.ndarray:
        """Mild CLAHE only – no sharpening, no gamma."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _extract_landmarks(self, frame: np.ndarray):
        frame = self._preprocess_for_mediapipe(frame)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None, [], None

        h, w    = frame.shape[:2]
        hand_lm = result.multi_hand_landmarks[0]
        coords  = []
        lm_pts  = []
        xs, ys  = [], []

        for lm in hand_lm.landmark:
            coords.extend([lm.x, lm.y, lm.z])
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_pts.append({"x": cx, "y": cy, "z": float(lm.z)})
            xs.append(cx); ys.append(cy)

        pad = 20
        bbox = [
            max(0, min(xs) - pad), max(0, min(ys) - pad),
            min(w, max(xs) + pad), min(h, max(ys) + pad),
        ]
        return np.array(coords, dtype=np.float32), lm_pts, bbox

    def predict_frame(self, frame: np.ndarray) -> dict:
        result = {
            "letter"      : None,
            "confidence"  : 0.0,
            "landmarks"   : [],
            "hand_bbox"   : None,
            "detected"    : False,
            "attn_weights": [],
            "hold_progress": 0.0,
            "confirmed"   : False,
            "top3"        : [],
        }

        coords, lm_pts, bbox = self._extract_landmarks(frame)

        if coords is None:
            self.no_hand_counter += 1
            if self.no_hand_counter < 5 and self.last_valid_landmarks is not None:
                coords = self.last_valid_landmarks
                lm_pts = self.last_known_lm_pts
                bbox   = self.last_known_bbox
                result["detected"] = True
                result["landmarks"] = lm_pts
                result["hand_bbox"] = bbox
            else:
                self.hold_count = 0
                self.hold_letter = None
                self.frame_buffer.clear()
                self.last_valid_landmarks = None
                return result
        else:
            self.no_hand_counter = 0
            self.last_valid_landmarks = coords
            self.last_known_lm_pts = lm_pts
            self.last_known_bbox = bbox
            result["detected"] = True
            result["landmarks"] = lm_pts
            result["hand_bbox"] = bbox

        self.frame_buffer.append(coords)
        if len(self.frame_buffer) < SEQ_LEN:
            while len(self.frame_buffer) < SEQ_LEN:
                self.frame_buffer.appendleft(coords)

        seq_tensor = torch.tensor(
            np.stack(list(self.frame_buffer)), dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits, attn = self.model(seq_tensor, return_attn=True)
            probs        = torch.softmax(logits, dim=1)[0]
            conf, idx    = probs.max(0)

        conf_val = conf.item()
        letter   = self.class_names[idx.item()]

        attn_np = attn[0].cpu().numpy().tolist()
        result["attn_weights"] = attn_np

        top3_vals, top3_idxs = probs.topk(3)
        result["top3"] = [
            {"letter": self.class_names[i.item()], "conf": round(v.item()*100,1)}
            for v, i in zip(top3_vals, top3_idxs)
        ]

        if conf_val < CONF_THRESH:
            return result

        result["letter"]     = letter
        result["confidence"] = round(conf_val * 100, 1)

        if letter == self.hold_letter:
            self.hold_count += 1
        else:
            self.hold_letter = letter
            self.hold_count  = 1
            self.confirmed   = False

        progress = min(self.hold_count / HOLD_FRAMES, 1.0)
        result["hold_progress"] = round(progress, 3)

        if self.hold_count >= HOLD_FRAMES and not self.confirmed:
            result["confirmed"] = True
            self.confirmed      = True
            self.hold_count     = 0

        return result

    def draw_overlay(self, frame: np.ndarray, pred: dict) -> np.ndarray:
        overlay    = frame.copy()
        landmarks  = pred.get("landmarks", [])
        attn       = pred.get("attn_weights", [])
        bbox       = pred.get("hand_bbox")

        if not landmarks:
            return frame

        if attn:
            attn_arr = np.array(attn)
            attn_arr = (attn_arr - attn_arr.min()) / (attn_arr.max() - attn_arr.min() + 1e-8)
        else:
            attn_arr = np.ones(len(landmarks))

        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]
        for a, b in connections:
            if a < len(landmarks) and b < len(landmarks):
                weight = (attn_arr[a] + attn_arr[b]) / 2
                color  = (int(50*(1-weight)), int(150+105*weight), int(255*weight))
                pt1 = (landmarks[a]["x"], landmarks[a]["y"])
                pt2 = (landmarks[b]["x"], landmarks[b]["y"])
                cv2.line(overlay, pt1, pt2, color, 2)

        for i, lm in enumerate(landmarks):
            w = attn_arr[i] if i < len(attn_arr) else 0.5
            color  = (int(50*(1-w)), int(150+105*w), int(255*w))
            radius = int(4 + 6*w)
            cv2.circle(overlay, (lm["x"], lm["y"]), radius, color, -1)

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,150), 2)

        return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    def close(self):
        self.hands.close()