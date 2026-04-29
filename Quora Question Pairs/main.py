import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
from result import predict

app = FastAPI(title="ML Model API")

# --- CONFIGURATION ---
MODEL_PATH = 'model.joblib'
VECTORIZER_PATH = 'vectorizer.joblib'
BUCKET_NAME = 'ml-model-bucket-test'

# --- MODELS & GLOBALS ---
model = None
vectorizer = None

# Define the schema for input data
class PredictionRequest(BaseModel):
    sentence1: str
    sentence2: str

def download_resource(blob_name, path):
    if not os.path.exists(path):
        print(f"Downloading {blob_name} from GCS...")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(path)
        print(f"Download complete: {path}")

# This replaces the 'main' logic for loading assets
@app.on_event("startup")
def load_assets():
    global model, vectorizer
    
    # Ensure local files exist
    download_resource('model.joblib', MODEL_PATH)
    download_resource('vectorizer.joblib', VECTORIZER_PATH)
    
    # Load into memory
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and Vectorizer loaded successfully.")

@app.post("/predict")
async def make_prediction(request: PredictionRequest):
    try:
        # FastAPI handles JSON parsing and validation automatically via Pydantic
        s1 = request.sentence1
        s2 = request.sentence2

        input_data = np.array([[s1, s2]])

        # Note: Ensure your 'predict' function is thread-safe or use 'await' if it's async
        prediction = predict(input_data) 
        result_value = int(prediction[0])
        
        # Determine the human-readable message
        if result_value == 1:
            message = "Sentences are similar"
        elif result_value == 0:
            message = "Sentences are not similar"
        else:
            message = "Invalid input"

        return {
            "prediction": result_value,
            "message": message,
            "status": "success"
        }

    except Exception as e:
        # FastAPI's HTTPException returns structured JSON errors automatically
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    # In production, you'd usually run this via command line: uvicorn main:app --host 0.0.0.0 --port 8080
    uvicorn.run(app, host='0.0.0.0', port=8080)
