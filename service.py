import pandas as pd
import numpy as np

from joblib import load
from pydantic import BaseModel
from fastapi import FastAPI
from src.features.feature_definitions import test_feature_build

app = FastAPI()

class PredictionInput(BaseModel):
    vendor_id : float
    vendor_id : float
    pickup_datetime : float
    dropoff_datetime : float
    passenger_count : float
    pickup_longitude : float
    pickup_latitude : float
    dropoff_longitude : float
    dropoff_latitude : float
    store_and_fwd_flag : float
    trip_duration : float
    pickup_date : float
    check_trip_duration : float
    pickup_pca0 : float
    pickup_pca1 : float
    dropoff_pca0 : float
    dropoff_pca1 : float
    distance_haversine : float
    distance_dummy_manhattan : float
    direction : float
    pca_manhattan : float

model_path  = 'model.joblib'
model = load(model_path)

@app.get('/')
def home():
    return 'Working Fine'

@app.post('/predict')
def predict(input_data : PredictionInput):
    features = {
         'vendor_id' : input_data.vendor_id,
        'pickup_datetime' : input_data.pickup_datetime,
        'dropoff_datetime' : input_data.dropoff_datetime,
        'passenger_count' : input_data.passenger_count,
        'pickup_longitude' : input_data.pickup_longitude,
        'pickup_latitude' : input_data.pickup_latitude,
        'dropoff_longitude' : input_data.dropoff_longitude,
        'dropoff_latitude' : input_data.dropoff_latitude,
        'store_and_fwd_flag' : input_data.store_and_fwd_flag,
        'trip_duration' : input_data.trip_duration,
        'pickup_date' : input_data.pickup_date,
        'check_trip_duration' : input_data.check_trip_duration,
        'pickup_pca0' : input_data.pickup_pca0,
        'pickup_pca1' : input_data.pickup_pca1,
        'dropoff_pca0' : input_data.dropoff_pca0,
        'dropoff_pca1' : input_data.dropoff_pca1,
        'distance_haversine' : input_data.distance_haversine,
        'distance_dummy_manhattan' : input_data.distance_dummy_manhattan,
        'direction' : input_data.direction,
        'pca_manhattan' : input_data.pca_manhattan,
            }
    features = pd.DataFrame(features , index=[0])
    features = test_feature_build(features)
    prediction = model.predict(features)[0].item()
    return {'prediction' : prediction}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)