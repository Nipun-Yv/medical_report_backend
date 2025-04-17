from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
class MedicalReportData:
    blood_pressure:int
    specific_gravity:float
    albumin:float
    sugar:bool
    rbc_in_urine:bool
    blood_urea:int
    serum_creatinine:float
    sodium:float
    potassium:float
    hemoglobin:float
    wbc_count:int
    rbc_count:int
    hypertension:bool
    systolic_bp:int
    diastolic_bp:int
    cholestrol:int
    glucose:int
    smoke:bool
    alcohol:bool
    active:bool

class ConfidenceScoreRequestBody(BaseModel):
    pdf_url:str

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

@app.post("/confidence-score")
async def confidence_score_estimate(data:ConfidenceScoreRequestBody):
    print(data.pdf_url)
    return {
        "success":"true",
        "pdf":data.pdf_url
}