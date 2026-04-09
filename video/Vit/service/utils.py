import cv2
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import json
import numpy as np

# ------------------------
# Video frame extraction
# ------------------------
def extract_frames(video_path, frame_rate=30):
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

# ------------------------
# Get transforms from preprocessor_config
# ------------------------
def get_transforms(preprocessor_config):
    size = preprocessor_config.get("size", {"height":224, "width":224})
    mean = preprocessor_config.get("image_mean", [0.5,0.5,0.5])
    std = preprocessor_config.get("image_std", [0.5,0.5,0.5])
    return Compose([
        Resize((size["height"], size["width"])),
        CenterCrop((size["height"], size["width"])),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

# ------------------------
# Prepare tensor for ViT inference
# ------------------------
def prepare_video_tensor(frames, transforms, device):
    frames_tensor = torch.stack([transforms(f) for f in frames]).to(device)
    video_tensor = frames_tensor.mean(dim=0).unsqueeze(0)  # [1,3,224,224]
    return video_tensor

# ------------------------
# Face detection using Haar Cascade
# ------------------------
def detect_faces(image):
    """
    Detect faces in an image using OpenCV Haar Cascade.
    Returns list of (x, y, w, h) bounding boxes.
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Convert PIL to OpenCV format
    if isinstance(image, Image.Image):
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        cv_image = image
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        cv_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces

# ------------------------
# Crop face from image
# ------------------------
def crop_face(image, bbox):
    """
    Crop a face region from an image given bounding box.
    Returns PIL Image of cropped face.
    """
    x, y, w, h = bbox
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    cropped = image_array[y:y+h, x:x+w]
    return Image.fromarray(cropped)
