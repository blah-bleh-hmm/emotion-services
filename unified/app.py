from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import tensorflow as tf
import cv2
import av
import numpy as np
from PIL import Image
import io
import os
import json
import tempfile
from collections import Counter

# ----------------------------------------------------------------------
# Paths and Device Configurations
# ----------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIT_MODEL_DIR = os.path.join(BASE_DIR, "../video/Vit/model")
RESNET_MODEL_DIR = os.path.join(BASE_DIR, "../video/ResNet50/model/emotion_resnet50_savedmodel")
WAV2VEC_MODEL_DIR = os.path.join(BASE_DIR, "../audio/wav2vec/mobile/finetuned_wav2vec2_model_mobile")
HUBERT_MODEL_DIR = os.path.join(BASE_DIR, "../audio/hubert/hubert_base_pitch/hubert_base_pitch_audio")

AUDIO_TARGET_SR = 16000
RESNET_FRAME_SKIP = 5
resnet_labels = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

# ----------------------------------------------------------------------
# Lazy-loaded Model Singletons & Getters
# ----------------------------------------------------------------------
_vit_model = None
_vit_id2label = None
_vit_transforms = None

_resnet_model = None

_wav2vec_model = None
_wav2vec_feature_extractor = None

_hubert_model = None
_hubert_feature_extractor = None

def get_vit():
    global _vit_model, _vit_id2label, _vit_transforms
    if _vit_model is None:
        print("Loading ViT Model lazily...")
        from transformers import ViTForImageClassification
        _vit_model = ViTForImageClassification.from_pretrained(VIT_MODEL_DIR, torch_dtype=torch.float32)
        _vit_model.to(DEVICE)
        _vit_model.eval()

        with open(os.path.join(VIT_MODEL_DIR, "preprocessor_config.json"), "r") as f:
            preprocessor_config = json.load(f)

        with open(os.path.join(VIT_MODEL_DIR, "config.json"), "r") as f:
            config_json = json.load(f)
            _vit_id2label = config_json["id2label"]

        _vit_transforms = vit_get_transforms(preprocessor_config)
        print("ViT Model successfully loaded!")
    return _vit_model, _vit_id2label, _vit_transforms

def get_resnet():
    global _resnet_model
    if _resnet_model is None:
        print("Loading ResNet50 Model lazily...")
        _resnet_model = tf.keras.layers.TFSMLayer(
            RESNET_MODEL_DIR,
            call_endpoint="serving_default"
        )
        print("ResNet50 Model successfully loaded!")
    return _resnet_model

def get_wav2vec():
    global _wav2vec_model, _wav2vec_feature_extractor
    if _wav2vec_model is None:
        print("Loading Wav2Vec2 Model lazily...")
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        _wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_MODEL_DIR)
        _wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained(WAV2VEC_MODEL_DIR)
        _wav2vec_model.to(DEVICE)
        _wav2vec_model.eval()
        print("Wav2Vec2 Model successfully loaded!")
    return _wav2vec_model, _wav2vec_feature_extractor

def get_hubert():
    global _hubert_model, _hubert_feature_extractor
    if _hubert_model is None:
        print("Loading HuBERT Model lazily...")
        from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
        _hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_DIR)
        _hubert_model = HubertForSequenceClassification.from_pretrained(HUBERT_MODEL_DIR)
        _hubert_model.to(DEVICE)
        _hubert_model.eval()
        print("HuBERT Model successfully loaded!")
    return _hubert_model, _hubert_feature_extractor

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
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

# ViT helper transforms
def vit_get_transforms(preprocessor_config):
    size = preprocessor_config.get("size", {"height": 224, "width": 224})
    mean = preprocessor_config.get("image_mean", [0.5, 0.5, 0.5])
    std = preprocessor_config.get("image_std", [0.5, 0.5, 0.5])
    return Compose([
        Resize((size["height"], size["width"])),
        CenterCrop((size["height"], size["width"])),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

def vit_detect_faces(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    if isinstance(image, Image.Image):
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        cv_image = image
    faces = face_cascade.detectMultiScale(
        cv_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces

def vit_crop_face(image, bbox):
    x, y, w, h = bbox
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cropped = image_array[y:y+h, x:x+w]
    return Image.fromarray(cropped)

def vit_extract_frames(video_path, frame_rate=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_rate == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_count += 1
    cap.release()
    return frames

# ResNet50 helpers
def resnet_detect_faces(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces

def resnet_crop_face(image, bbox):
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

def resnet_predict_frame(frame, model):
    frame_resized = cv2.resize(frame, (48, 48))
    frame_normalized = frame_resized / 255.0
    frame_input = np.expand_dims(frame_normalized, axis=0)
    output = model(frame_input)
    preds = np.array(list(output.values())[0])[0]
    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])
    return resnet_labels[pred_idx], confidence

# ----------------------------------------------------------------------
# FastAPI Setup
# ----------------------------------------------------------------------
app = FastAPI(title="Unified Emotion Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "ok", "message": "Unified ML Service is running"}

# ----------------------------------------------------------------------
# ViT Service Endpoints
# ----------------------------------------------------------------------
@app.post("/vit/predict_image/")
async def vit_predict_image(file: UploadFile = File(...)):
    model, id2label, transforms = get_vit()
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    faces = vit_detect_faces(image)
    
    if len(faces) == 0:
        return {"detections": [], "message": "No faces detected in image"}
    
    detections = []
    for (x, y, w, h) in faces:
        face_image = vit_crop_face(image, (x, y, w, h))
        tensor = transforms(face_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(tensor)
            predicted_id = outputs.logits.argmax(-1).item()
            predicted_label = id2label[str(predicted_id)]
            confidence = torch.nn.functional.softmax(outputs.logits, dim=1)[0].max().item()
        
        detections.append({
            "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "emotion": predicted_label,
            "confidence": float(confidence)
        })
    return {"detections": detections}

@app.post("/vit/predict_video/")
async def vit_predict_video(file: UploadFile = File(...)):
    model, id2label, transforms = get_vit()
    temp_dir = os.environ.get("TEMP", "C:\\Windows\\Temp")
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, file.filename)

    with open(video_path, "wb") as f:
        f.write(await file.read())

    try:
        frames = vit_extract_frames(video_path, frame_rate=10)
        if len(frames) == 0:
            return {"error": "No frames extracted from video. Check the file."}

        all_detections = []
        face_emotion_counts = {}
        
        for frame_idx, frame in enumerate(frames):
            faces = vit_detect_faces(frame)
            frame_detections = []
            for (x, y, w, h) in faces:
                face_image = vit_crop_face(frame, (x, y, w, h))
                tensor = transforms(face_image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    outputs = model(tensor)
                    predicted_id = outputs.logits.argmax(-1).item()
                    predicted_label = id2label[str(predicted_id)]
                    confidence = torch.nn.functional.softmax(outputs.logits, dim=1)[0].max().item()
                
                face_emotion_counts[predicted_label] = face_emotion_counts.get(predicted_label, 0) + 1
                frame_detections.append({
                    "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "emotion": predicted_label,
                    "confidence": float(confidence)
                })
            
            if frame_detections:
                all_detections.append({
                    "frame_index": frame_idx,
                    "detections": frame_detections
                })
        
        overall_emotion = max(face_emotion_counts, key=face_emotion_counts.get) if face_emotion_counts else None
        
        return {
            "total_frames_processed": len(frames),
            "frames_with_detections": len(all_detections),
            "overall_dominant_emotion": overall_emotion,
            "frame_detections": all_detections
        }
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

# ----------------------------------------------------------------------
# ResNet50 Endpoints
# ----------------------------------------------------------------------
@app.post("/resnet/predict-image")
async def resnet_predict_image(file: UploadFile = File(...)):
    try:
        model = get_resnet()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        img = cv2.imread(tmp_path)
        if img is None:
            return JSONResponse({"error": "Cannot read image file"}, status_code=400)

        faces = resnet_detect_faces(img)
        if len(faces) == 0:
            return {"detections": [], "message": "No faces detected in image"}
        
        detections = []
        for (x, y, w, h) in faces:
            face_image = resnet_crop_face(img, (x, y, w, h))
            emotion, confidence = resnet_predict_frame(face_image, model)
            detections.append({
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "emotion": emotion,
                "confidence": float(confidence)
            })
        
        return {"detections": detections}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/resnet/predict-video")
async def resnet_predict_video(file: UploadFile = File(...)):
    try:
        model = get_resnet()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video_bytes = await file.read()
            tmp.write(video_bytes)
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return JSONResponse({"error": "OpenCV cannot open video file"}, status_code=400)

        frame_count = 0
        frame_detections = []
        emotion_counts = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % RESNET_FRAME_SKIP != 0:
                continue

            try:
                faces = resnet_detect_faces(frame)
                if len(faces) > 0:
                    frame_dets = []
                    for (x, y, w, h) in faces:
                        face_image = resnet_crop_face(frame, (x, y, w, h))
                        emotion, confidence = resnet_predict_frame(face_image, model)
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                        
                        frame_dets.append({
                            "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                            "emotion": emotion,
                            "confidence": float(confidence)
                        })
                    
                    if frame_dets:
                        frame_detections.append({
                            "frame_index": len(frame_detections),
                            "detections": frame_dets
                        })
            except Exception:
                continue

        cap.release()
        
        if len(frame_detections) == 0:
            return JSONResponse({"error": "No faces detected in video"}, status_code=400)

        overall_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else None

        return {
            "total_frames_processed": frame_count // RESNET_FRAME_SKIP,
            "frames_with_detections": len(frame_detections),
            "overall_dominant_emotion": overall_emotion,
            "frame_detections": frame_detections
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)

# ----------------------------------------------------------------------
# Wav2Vec2 Endpoints
# ----------------------------------------------------------------------
@app.post("/wav2vec/predict-audio")
async def wav2vec_predict_audio(file: UploadFile = File(...)):
    model, feature_extractor = get_wav2vec()
    audio_bytes = await file.read()
    speech = decode_audio(audio_bytes, target_sr=AUDIO_TARGET_SR)

    inputs = feature_extractor(
        speech, sampling_rate=AUDIO_TARGET_SR, return_tensors="pt", padding=True
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

# ----------------------------------------------------------------------
# HuBERT Endpoints
# ----------------------------------------------------------------------
@app.post("/hubert/predict-audio")
async def hubert_predict_audio(file: UploadFile = File(...)):
    model, feature_extractor = get_hubert()
    audio_bytes = await file.read()
    speech = decode_audio(audio_bytes, target_sr=AUDIO_TARGET_SR)

    inputs = feature_extractor(
        speech, sampling_rate=AUDIO_TARGET_SR, return_tensors="pt", padding=True
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
