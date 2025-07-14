import uvicorn
import pandas as pd 
from pydantic import BaseModel
from typing import Literal, Union
from fastapi import FastAPI
from joblib import load
import os

# Load the model and preprocessor
preprocessor = load('preprocessor.joblib')
model = load('model.joblib')

# API description
description = """
Welcome to the GetAround Price Prediction API !
Use this API to find the right daily rental price for your car depending on its characteristics.

Here are the available endpoints:

* `/predict`: You can use this endpoint to predict the daily rental price for your car.
It accepts a POST request with your car's characteristics as JSON input data. 
"""

tags_metadata = [
    {
        "name": "Prediction",
        "description": "Prediction of the daily rental price for your car"
    }
]

# App initialization
app = FastAPI(
    title="Rental Price Prediction API",
    description=description,
    version="1.0",
    contact={
    "name": "GetAround Price Prediction API - by Myriam Goyet",
    }, 
    openapi_tags=tags_metadata
)

# Input data model with example
class PredictionFeatures(BaseModel):
    car_type:Literal['convertible', 'coupe', 'estate', 'hatchback', 'sedan','subcompact', 'suv', 'van']
    model_key: Literal['Audi', 'BMW', 'Citroën', 'Ferrari', 'Mercedes', 'Mitsubishi', 'Nissan', 'Opel', 'PGO', 'Peugeot', 'Renault', 'SEAT', 'Toyota', 'Volkswagen', 'other']
    mileage: Union[int, float]
    engine_power: Union[int, float]
    fuel: Literal['diesel','petrol']
    has_gps: bool
    automatic_car: bool
    has_getaround_connect: bool
    private_parking_available: bool
    has_speed_regulator: bool
    has_air_conditioning: bool
    winter_tires: bool
    paint_color: Literal['black', 'white', 'red', 'silver', 'grey', 'blue', 'beige','brown']

    class Config:
        schema_extra = {
            "example": {
                "car_type": "suv",
                "model_key": "Toyota",
                "mileage": 40000,
                "engine_power": 110,
                "fuel": "petrol",
                "has_gps": True,
                "automatic_car": False,
                "has_getaround_connect": True,
                "private_parking_available": True,
                "has_speed_regulator": True,
                "has_air_conditioning": True,
                "winter_tires": False,
                "paint_color": "grey"
            }
        }

# Define enpoints 
# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "🚗 Welcome to the GetAround Price Prediction API!",
        "description": "Use this API to predict the daily rental price of a car based on its characteristics.",
        "version": "1.0",
        "author": "Myriam Goyet",
        "docs": "/docs",
        "alternative_docs": "/redoc",
        "predict_endpoint": {
            "method": "POST",
            "path": "/predict",
            "description": "Submit car features to get a daily rental price prediction."
        }
    }
# Prediction endpoint
@app.post("/predict", tags=["Prediction"])
async def predict(features: PredictionFeatures):
    data = pd.DataFrame(features.dict(), index=[0])
    X_processed = preprocessor.transform(data)
    prediction = model.predict(X_processed)
    return {"prediction": prediction.tolist()[0]}

# Run server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)