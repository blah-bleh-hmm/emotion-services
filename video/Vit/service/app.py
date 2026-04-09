from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import json
from utils import extract_frames, get_transforms, prepare_video_tensor, detect_faces, crop_face
from PIL import Image
import io
import os

# ------------------------
# Config
# ------------------------
MODEL_DIR = "../model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_RATE = 10

# ------------------------
# Load model and preprocessor
# ------------------------
model = ViTForImageClassification.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
model.to(DEVICE)
model.eval()

with open(f"{MODEL_DIR}/preprocessor_config.json", "r") as f:
    preprocessor_config = json.load(f)

with open(f"{MODEL_DIR}/config.json", "r") as f:
    config_json = json.load(f)
    id2label = config_json["id2label"]

transforms = get_transforms(preprocessor_config)

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="ViT Emotion Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict_image/")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Detect faces in the image
    faces = detect_faces(image)
    
    if len(faces) == 0:
        return {"detections": [], "message": "No faces detected in image"}
    
    detections = []
    
    for (x, y, w, h) in faces:
        # Crop face region
        face_image = crop_face(image, (x, y, w, h))
        
        # Prepare tensor for model
        tensor = get_transforms(preprocessor_config)(face_image).unsqueeze(0).to(DEVICE)
        
        # Predict emotion
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

@app.post("/predict_video/")
async def predict_video(file: UploadFile = File(...)):
    # Save uploaded video to TEMP folder (Windows-safe)
    temp_dir = os.environ.get("TEMP", "C:\\Windows\\Temp")
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, file.filename)

    with open(video_path, "wb") as f:
        f.write(await file.read())

    try:
        # Extract frames
        frames = extract_frames(video_path, frame_rate=10)
        if len(frames) == 0:
            return {"error": "No frames extracted from video. Check the file."}

        # Process frames for face detection and emotion classification
        all_detections = []
        face_emotion_counts = {}
        
        for frame_idx, frame in enumerate(frames):
            faces = detect_faces(frame)
            
            frame_detections = []
            for (x, y, w, h) in faces:
                # Crop face region
                face_image = crop_face(frame, (x, y, w, h))
                
                # Prepare tensor for model
                tensor = get_transforms(preprocessor_config)(face_image).unsqueeze(0).to(DEVICE)
                
                # Predict emotion
                with torch.no_grad():
                    outputs = model(tensor)
                    predicted_id = outputs.logits.argmax(-1).item()
                    predicted_label = id2label[str(predicted_id)]
                    confidence = torch.nn.functional.softmax(outputs.logits, dim=1)[0].max().item()
                
                # Track emotion counts for overall aggregation
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
        
        # Aggregate overall emotion
        overall_emotion = max(face_emotion_counts, key=face_emotion_counts.get) if face_emotion_counts else None
        
        return {
            "total_frames_processed": len(frames),
            "frames_with_detections": len(all_detections),
            "overall_dominant_emotion": overall_emotion,
            "frame_detections": all_detections
        }
    
    finally:
        # Clean up temporary video file
        if os.path.exists(video_path):
            os.remove(video_path)

