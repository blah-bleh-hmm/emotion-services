from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import av
import numpy as np
import io
import os
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# ------------------------
# Config
# ------------------------
MODEL_DIR = os.path.abspath("./mobile/finetuned_wav2vec2_model_mobile")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 16000

# ------------------------
# Load model and feature extractor
# ------------------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="Wav2Vec2 Audio Emotion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_audio(file_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """Decode any audio format to a float32 mono array at target_sr using PyAV."""
    container = av.open(io.BytesIO(file_bytes))
    resampler = av.audio.resampler.AudioResampler(
        format="fltp", layout="mono", rate=target_sr
    )
    samples = []
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            samples.append(resampled.to_ndarray()[0])
    for resampled in resampler.resample(None):  # flush remaining frames
        samples.append(resampled.to_ndarray()[0])
    return np.concatenate(samples).astype(np.float32)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    speech = decode_audio(audio_bytes, target_sr=TARGET_SR)

    inputs = feature_extractor(
        speech, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
    )
    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_values)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()

    predicted_emotion = model.config.id2label[pred_id]

    return {
        "predicted_emotion": predicted_emotion,
        "confidence": round(confidence, 4),
    }
