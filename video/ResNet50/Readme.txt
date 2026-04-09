Emotion_Model_Production/
│
├── config/
│   └── labels.json
│       # ["Anger","Disgust","Fear","Happy","Neutral","Sad"]
│
├── model/
│   ├── emotion_resnet50_v1.h5           
│   └── OR emotion_resnet50_savedmodel/ 
│       ├── assets/
│       ├── variables/
│       └── saved_model.pb
│
├── inference/
│   └── test_model.py                     
│
├── service/
│   ├── app.py                            
│   └── requirements.txt                  
│
└── README.md                            

File Details
1. config/labels.json
["Anger","Disgust","Fear","Happy","Neutral","Sad"]

2. model/

 A: .h5 → use tf.keras.models.load_model("../model/emotion_resnet50_v1.h5") in app.py.

 B: SavedModel → use tf.keras.layers.TFSMLayer("../model/emotion_resnet50_savedmodel") in app.py.


3. service/app.py
/predict-image → predicts emotion from an uploaded image.

/predict-video → predicts emotion from uploaded video, skipping frames, returns most common emotion + distribution.

4. service/requirements.txt
fastapi==0.111.1
uvicorn[standard]==0.23.2
tensorflow==2.14.0
numpy==1.26.0
opencv-python==4.9.0.73
python-multipart==0.0.6

5.  inference/test_model.py 

A standalone script to test model loading and prediction

7. Deployment Tips

cd Emotion_Model_Production/service

uvicorn app:app --reload


Test API: http://127.0.0.1:8000/docs

