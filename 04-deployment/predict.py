import pickle
from typing import Literal

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict


# Request
class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gender: Literal["male", "female"]
    seniorcitizen: Literal[0, 1]
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["no", "yes", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: Literal["no", "yes", "no_internet_service"]
    onlinebackup: Literal["no", "yes", "no_internet_service"]
    deviceprotection: Literal["no", "yes", "no_internet_service"]
    techsupport: Literal["no", "yes", "no_internet_service"]
    streamingtv: Literal["no", "yes", "no_internet_service"]
    streamingmovies: Literal["no", "yes", "no_internet_service"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ]
    tenure: int = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0.0)
    totalcharges: float = Field(..., ge=0.0)

# Response
class PredictResponse(BaseModel):
    churn_probability: float
    is_churn: bool


app = FastAPI(title="customer-churn-prediction")


with open("model_final.bin", "rb") as f_in:
    model_pipeline = pickle.load(f_in)

def predict_single(customer):
    result = model_pipeline.predict_proba(customer)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    proba = predict_single(customer.model_dump())

    return PredictResponse(
        churn_probability=proba,
        is_churn=(proba >= 0.5)
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7676)
