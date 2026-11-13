
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Create FastAPI app
app = FastAPI(title="House Price Prediction API")

# Load ML model
model = joblib.load("house_price_prediction_model.joblib")


# Input data structure
class House(BaseModel):
    location: str
    property_type: str
    size_sqft: int
    num_bedrooms: int
    num_bathrooms: int
    floor_number: int
    total_floors: int
    age_years: int
    furnishing: str
    distance_to_metro_km: float
    view_score: int
    year: int
    month: int
    day: int


# Prediction API
@app.post("/predict")
def predict(data: House):

    input_data = pd.DataFrame([{
        "location": data.location,
        "property_type": data.property_type,
        "size_sqft": data.size_sqft,
        "num_bedrooms": data.num_bedrooms,
        "num_bathrooms": data.num_bathrooms,
        "floor_number": data.floor_number,
        "total_floors": data.total_floors,
        "age_years": data.age_years,
        "furnishing": data.furnishing,
        "distance_to_metro_km": data.distance_to_metro_km,
        "view_score": data.view_score,
        "year": data.year,
        "month": data.month,
        "day": data.day
    }])

    prediction = float(model.predict(input_data)[0])

    return {
        "predicted_price": prediction,
        "result": f"Estimated House Price: ₹{prediction:,.2f}"
    }
