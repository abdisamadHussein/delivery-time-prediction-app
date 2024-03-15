from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_timetkaen
from app.model.model import __version__ as model_version

app = FastAPI()


class DeliveryData(BaseModel):
    Vehicle_condition: int
    distance: float
    multiple_deliveries: int
    Weatherconditions: str
    Road_traffic_density: str
    Festival: str
    City: str


class PredictionOut(BaseModel):
    estimated_delivery_time: float


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: DeliveryData):
    estimated_delivery_time = predict_timetkaen(payload)
    return  estimated_delivery_time
