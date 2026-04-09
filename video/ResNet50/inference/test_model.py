import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter
model = load_model("../model/emotion_resnet50_v1.h5")
labels = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
video_path = r"C:\Users\UIET\Desktop\PUMAVE\Happy\S8-H-H-S3-R1-V.mpg"   # put video in this folder
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Cannot open video file")
frame_skip = 10   # process every 10th frame
frame_count = 0
predictions = []
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames
    if frame_count % frame_skip != 0:
        continue

    # Preprocess frame
    frame = cv2.resize(frame, (48, 48))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)

    # Predict
    pred = model.predict(frame, verbose=0)
    emotion_index = int(np.argmax(pred))
    predictions.append(labels[emotion_index])
cap.release()
if len(predictions) == 0:
    print("No frames processed")
else:
    final_emotion = Counter(predictions).most_common(1)[0][0]
    print("Video Emotion:", final_emotion)
    print("Frame-wise distribution:", Counter(predictions))
