"""
Voice Cloning Detector — Flask backend (SIH26104)

Run locally (in Antigravity / your own laptop):
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000 in your browser.

Needs model.pkl and scaler.pkl in the model/ folder — these come from
train_colab.ipynb (train it in Google Colab, then download both files here).
"""

import os
import io
import traceback

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

from features import load_audio, extract_features, waveform_preview

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

model = None
scaler = None
model_load_error = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        model_load_error = (
            "model.pkl / scaler.pkl nahi mile model/ folder mein. "
            "Pehle train_colab.ipynb Colab mein chalao, phir dono files "
            "is project ke model/ folder mein daalo."
        )
except Exception as e:
    model_load_error = f"Model load karne mein error: {e}"


ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


@app.route("/")
def index():
    return render_template("index.html", model_ready=(model is not None), model_error=model_load_error)


@app.route("/api/status")
def status():
    return jsonify({"model_ready": model is not None, "error": model_load_error})


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": model_load_error or "Model load nahi hua."}), 503

    if "audio" not in request.files:
        return jsonify({"error": "Koi audio file nahi mili upload mein."}), 400

    f = request.files["audio"]
    ext = os.path.splitext(f.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"File type '{ext}' supported nahi hai. WAV/MP3/M4A/OGG/FLAC try karo."}), 400

    try:
        audio_bytes = io.BytesIO(f.read())
        y, sr = load_audio(audio_bytes)

        if len(y) < sr * 0.5:
            return jsonify({"error": "Clip bahut chhoti hai (< 0.5 sec). Lambi clip try karo."}), 400

        feat = extract_features(y, sr)
        feat_scaled = scaler.transform(feat.reshape(1, -1))

        pred = model.predict(feat_scaled)[0]                 # 0 = REAL, 1 = FAKE
        proba = model.predict_proba(feat_scaled)[0]          # [P(real), P(fake)]

        label = "FAKE" if pred == 1 else "REAL"
        confidence = float(proba[1] if pred == 1 else proba[0]) * 100

        return jsonify({
            "label": label,
            "confidence": round(confidence, 1),
            "prob_real": round(float(proba[0]) * 100, 1),
            "prob_fake": round(float(proba[1]) * 100, 1),
            "waveform": waveform_preview(y),
            "duration_sec": round(len(y) / sr, 2),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Processing mein error aaya: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
