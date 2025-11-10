"""
SignBridge v1.0 — Flask Backend
HTTP polling — webcam frames processed per request
Port: 5004
Run: python backend/app.py
"""

import os, sys, base64, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from predictor import SignPredictor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "frontend"),
    static_folder=os.path.join(BASE_DIR, "frontend"),
)

predictor    = SignPredictor()
word_buffer  = []          # simple letter buffer for v1
session_start = time.time()


def decode_frame(b64: str) -> np.ndarray:
    data  = base64.b64decode(b64.split(",")[-1])
    arr   = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """Receive base64 frame, return prediction."""
    data  = request.get_json()
    b64   = data.get("frame", "")
    if not b64:
        return jsonify({"error": "No frame"}), 400

    frame = decode_frame(b64)
    if frame is None:
        return jsonify({"error": "Bad frame"}), 400

    pred = predictor.predict_frame(frame)

    # Draw overlay and return annotated frame
    if pred["detected"] and pred["landmarks"]:
        frame = predictor.draw_landmarks(
            frame, pred["landmarks"], pred.get("hand_bbox")
        )

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    return jsonify({
        "letter"      : pred["letter"],
        "confidence"  : pred["confidence"],
        "detected"    : pred["detected"],
        "top3"        : pred.get("top3", []),
        "annotated"   : annotated_b64,
    })


@app.route("/api/word/push", methods=["POST"])
def word_push():
    """Add a confirmed letter to the word buffer."""
    data   = request.get_json()
    letter = data.get("letter", "")
    if letter and letter.isalpha():
        word_buffer.append(letter.upper())
    return jsonify({
        "word": "".join(word_buffer),
    })


@app.route("/api/word/clear", methods=["POST"])
def word_clear():
    word_buffer.clear()
    return jsonify({"word": ""})


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "model"  : "MobileNetV2",
        "version": "1.0",
        "device" : str(predictor.model.parameters().__next__().device)
                   if predictor.model else "cpu",
        "uptime" : round(time.time() - session_start),
    })


if __name__ == "__main__":
    print("📷 SignBridge v1.0 — http://localhost:5004")
    app.run(debug=False, host="0.0.0.0", port=5004, threaded=True)
