Emotion_model_vit
│
├── model/                     
│   ├── config.json             
│   ├── model.safetensors       
│   ├── preprocessor_config.json 
│   └── training_args.bin       
│
├── service/                   
│   ├── app.py                
│   ├── utils.py               
│   └── requirements.txt       │
├── inference/                
│   └── run_inference.py
│
└── README.md                  Setup Instructions
1. Install Dependencies
Create a Python virtual environment 
pip install -r service/requirements.txt

2. Run the FastAPI Service
From the service/ folder, run:

code
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Service runs at: http://localhost:8000

Endpoints:

/predict_image/ → Predict emotion from an image

/predict_video/ → Predict emotion from a video

3. Test the API
Image Example:

python
Copy code
import requests

url = "http://localhost:8000/predict_image/"
files = {"file": open("sample_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
Video Example:

python
Copy code
url = "http://localhost:8000/predict_video/"
files = {"file": open("sample_video.mp4", "rb")}
response = requests.post(url, files=files)
print(response.json())
Sample Output:

json
Copy code
{"predicted_emotion": "Happy"}
4. Video Processing
Videos are processed by extracting frames at every 10th frame by default.

Frames are resized, center-cropped, normalized, and averaged before being passed to the model.

This matches the preprocessing pipeline used during training.

5. Preprocessing
The service uses the preprocessor_config.json from your saved model:

Resize to 224x224

Normalize using mean [0.5, 0.5, 0.5] and std [0.5, 0.5, 0.5]

Converts images to RGB

This ensures inference is consistent with training.

7. Adding New Videos or Images
Place your video/image file locally.

Send it via POST request to /predict_image/ or /predict_video/.

Receive JSON response with predicted emotion.

8. Supported Emotions
The model supports the following classes:

Anger

Disgust

Fear

Happy

Neutral

Sad