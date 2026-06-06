"""
SignBridge v2.0 — Flask + Socket.IO Backend
Real-time WebSocket streaming | REST API v1 | LLM correction | TTS | Export
Port: 5004
Run: python backend/app.py
"""

import os, sys, base64, json, time, datetime, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import requests
import pyttsx3
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from predictor_v2 import SignPredictorV2
from word_engine   import WordEngine

# ── PDF export (optional — skip if reportlab not installed) ──
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "frontend"),
    static_folder=os.path.join(BASE_DIR, "frontend"),
)
app.config["SECRET_KEY"] = "signbridge-v2-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

predictor     = SignPredictorV2()
word_engine   = WordEngine()
session_start = time.time()

# TTS engine (runs in background thread)
_tts_lock   = threading.Lock()
_tts_engine = None

def get_tts():
    global _tts_engine
    if _tts_engine is None:
        try:
            _tts_engine = pyttsx3.init()
            _tts_engine.setProperty("rate", 150)
        except Exception:
            pass
    return _tts_engine

# ── HELPERS ──────────────────────────────────────────────────

def decode_frame(b64: str) -> np.ndarray | None:
    try:
        data  = base64.b64decode(b64.split(",")[-1])
        arr   = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def llm_correct(sentence: str) -> str:
    """Send raw ASL sentence to Llama3 for natural English correction."""
    if not sentence.strip():
        return sentence
    try:
        prompt = (
            f"Convert this ASL fingerspelled text into natural English. "
            f"Fix grammar and spelling. Reply with ONLY the corrected sentence, nothing else.\n\n"
            f"Input: {sentence}\nOutput:"
        )
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model" : "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 100},
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("response", sentence).strip()
    except Exception:
        pass
    return sentence


# ── WEBSOCKET EVENTS ─────────────────────────────────────────

@socketio.on("frame")
def handle_frame(data):
    """Real-time frame processing over WebSocket."""
    b64   = data.get("frame", "")
    frame = decode_frame(b64)
    if frame is None:
        return

    pred = predictor.predict_frame(frame)

    # Draw attention heatmap overlay
    annotated_b64 = ""
    if pred["detected"]:
        frame = predictor.draw_overlay(frame, pred)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    # If a letter was confirmed, push to word engine
    word_state = None
    if pred.get("confirmed") and pred.get("letter") and pred.get("confidence", 0) >= 85:
        word_state = word_engine.push_letter(pred["letter"])

    emit("prediction", {
        "letter"        : pred["letter"],
        "confidence"    : pred["confidence"],
        "detected"      : pred["detected"],
        "attn_weights"  : pred["attn_weights"],
        "hold_progress" : pred["hold_progress"],
        "confirmed"     : pred["confirmed"],
        "top3"          : pred["top3"],
        "annotated"     : annotated_b64,
        "word_state"    : word_state,
    })


@socketio.on("push_letter")
def handle_push_letter(data):
    letter = data.get("letter", "")
    state  = word_engine.push_letter(letter)
    emit("word_update", state)


@socketio.on("accept_suggestion")
def handle_suggestion(data):
    word  = data.get("word", "")
    state = word_engine.accept_suggestion(word)
    emit("word_update", state)


@socketio.on("clear_word")
def handle_clear_word(_):
    emit("word_update", word_engine.clear_word())


@socketio.on("clear_sentence")
def handle_clear_sentence(_):
    emit("word_update", word_engine.clear_sentence())


# ── REST API v1 ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/v1/predict", methods=["POST"])
def api_predict():
    """
    REST endpoint for single-frame inference.
    POST { "frame": "<base64 jpeg>" }
    Returns prediction JSON.
    """
    data  = request.get_json()
    b64   = data.get("frame", "")
    frame = decode_frame(b64)
    if frame is None:
        return jsonify({"error": "Invalid frame"}), 400

    pred = predictor.predict_frame(frame)
    return jsonify({
        "letter"      : pred["letter"],
        "confidence"  : pred["confidence"],
        "detected"    : pred["detected"],
        "hold_progress": pred["hold_progress"],
        "confirmed"   : pred["confirmed"],
        "top3"        : pred["top3"],
        "attn_weights": pred["attn_weights"],
    })


@app.route("/api/v1/llm/correct", methods=["POST"])
def api_llm_correct():
    """LLM sentence correction endpoint."""
    data     = request.get_json()
    sentence = data.get("sentence", "")
    corrected = llm_correct(sentence)
    return jsonify({"original": sentence, "corrected": corrected})


@app.route("/api/v1/tts", methods=["POST"])
def api_tts():
    """Speak a sentence via pyttsx3."""
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    def speak():
        with _tts_lock:
            engine = get_tts()
            if engine:
                engine.say(text)
                engine.runAndWait()

    threading.Thread(target=speak, daemon=True).start()
    return jsonify({"status": "speaking", "text": text})


@app.route("/api/v1/export/json", methods=["GET"])
def export_json():
    """Export session log as JSON."""
    state = word_engine.get_state()
    log   = word_engine.get_session_log()
    payload = {
        "exported_at" : datetime.datetime.now().isoformat(),
        "sentence"    : state["sentence"],
        "word_count"  : state["word_count"],
        "letter_count": state["letter_count"],
        "session_log" : log,
        "duration_s"  : round(time.time() - session_start),
    }
    path = os.path.join(EXPORT_DIR, "session.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return send_file(path, as_attachment=True, download_name="signbridge_session.json")


@app.route("/api/v1/export/pdf", methods=["GET"])
def export_pdf():
    """Export session transcript as PDF."""
    if not PDF_AVAILABLE:
        return jsonify({"error": "reportlab not installed"}), 500

    state    = word_engine.get_state()
    sentence = state["sentence"]
    path     = os.path.join(EXPORT_DIR, "session.pdf")

    doc    = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story  = [
        Paragraph("SignBridge — Session Transcript", styles["Title"]),
        Spacer(1, 20),
        Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
        Paragraph(f"Duration: {round(time.time()-session_start)}s", styles["Normal"]),
        Spacer(1, 20),
        Paragraph("Recognized Text:", styles["Heading2"]),
        Paragraph(sentence or "(empty session)", styles["BodyText"]),
        Spacer(1, 20),
        Paragraph(f"Words: {state['word_count']} | Letters: {state['letter_count']}", styles["Normal"]),
    ]
    doc.build(story)
    return send_file(path, as_attachment=True, download_name="signbridge_transcript.pdf")


@app.route("/api/v1/status", methods=["GET"])
def status():
    return jsonify({
        "model"        : "ViT-Tiny + LSTM",
        "version"      : "2.0",
        "device"       : "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "uptime"       : round(time.time() - session_start),
        "llm"          : "llama3.2:3b",
        "tts"          : "pyttsx3",
        "pdf_export"   : PDF_AVAILABLE,
    })


@app.route("/api/v1/word/state", methods=["GET"])
def word_state():
    return jsonify(word_engine.get_state())


if __name__ == "__main__":
    print("🤟 SignBridge v2.0 — http://localhost:5004")
    print("   WebSocket + REST API | LLM via Ollama | TTS offline")
    socketio.run(app, debug=False, host="0.0.0.0", port=5004)
