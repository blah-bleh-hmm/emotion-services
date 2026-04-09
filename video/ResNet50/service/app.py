# app.py
import tempfile
from collections import Counter
import os

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# -------------------------
# CONSTANTS
# -------------------------
FRAME_SKIP = 5  # take every 5th frame
labels = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

# -------------------------
# INIT APP
# -------------------------
app = FastAPI(
    title="Emotion Detection API",
    version="0.1.0",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# LOAD MODEL
# -------------------------
# Using Keras 3 TFSMLayer for SavedModel
model = tf.keras.layers.TFSMLayer(
    "../model/emotion_resnet50_savedmodel",
    call_endpoint="serving_default"
)


# -------------------------
# HELPER: DETECT FACES
# -------------------------
def detect_faces(image):
    """
    Detect faces in an image using OpenCV Haar Cascade.
    Returns list of (x, y, w, h) bounding boxes.
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces


# -------------------------
# HELPER: CROP FACE
# -------------------------
def crop_face(image, bbox):
    """
    Crop a face region from an image given bounding box.
    Returns cropped image.
    """
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]


# -------------------------
# HELPER: PREDICT FRAME
# -------------------------
def predict_frame(frame):
    """
    Predict emotion for a single frame.
    Returns the emotion label and confidence.
    """
    frame_resized = cv2.resize(frame, (48, 48))
    frame_normalized = frame_resized / 255.0
    frame_input = np.expand_dims(frame_normalized, axis=0)

    output = model(frame_input)
    preds = np.array(list(output.values())[0])[0]
    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])
    
    return labels[pred_idx], confidence


# -------------------------
# IMAGE UPLOAD ENDPOINT
# -------------------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Read image with OpenCV
        img = cv2.imread(tmp_path)
        if img is None:
            return JSONResponse({"error": "Cannot read image file"}, status_code=400)

        # Detect faces in the image
        faces = detect_faces(img)
        
        if len(faces) == 0:
            return {"detections": [], "message": "No faces detected in image"}
        
        detections = []
        
        for (x, y, w, h) in faces:
            # Crop face region
            face_image = crop_face(img, (x, y, w, h))
            
            # Predict emotion
            emotion, confidence = predict_frame(face_image)
            
            detections.append({
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "emotion": emotion,
                "confidence": float(confidence)
            })
        
        return {"detections": detections}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# -------------------------
# VIDEO UPLOAD ENDPOINT
# -------------------------
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    try:
        # Save uploaded video temporarily
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
            if frame_count % FRAME_SKIP != 0:
                continue

            try:
                # Detect faces in the frame
                faces = detect_faces(frame)
                
                if len(faces) > 0:
                    frame_dets = []
                    for (x, y, w, h) in faces:
                        # Crop face region
                        face_image = crop_face(frame, (x, y, w, h))
                        
                        # Predict emotion
                        emotion, confidence = predict_frame(face_image)
                        
                        # Track emotion counts
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
                continue  # skip frames that fail

        cap.release()
        
        # Clean up temp video file
        if os.path.exists(video_path):
            os.remove(video_path)

        if len(frame_detections) == 0:
            return JSONResponse({"error": "No faces detected in video"}, status_code=400)

        # Aggregate overall emotion
        overall_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else None

        return {
            "total_frames_processed": frame_count // FRAME_SKIP,
            "frames_with_detections": len(frame_detections),
            "overall_dominant_emotion": overall_emotion,
            "frame_detections": frame_detections
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


