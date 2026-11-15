from datetime import datetime, timezone
import json
import os
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Load model ONCE at startup (never inside endpoint)
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="AI Resume Screening & Candidate Shortlisting API", version="1.0.0")


class CandidateFeatures(BaseModel):
    age: int = Field(..., ge=18, le=65, description="Candidate age (18-65)")
    education_level: Literal["Bachelors", "Masters", "PhD"]
    university_tier: Literal["Tier 1", "Tier 2", "Tier 3"]
    cgpa: float = Field(..., ge=0.0, le=10.0, description="CGPA on a 10-point scale")
    internships: int = Field(..., ge=0, le=15)
    projects: int = Field(..., ge=0, le=25)
    programming_languages: int = Field(..., ge=1, le=15)
    certifications: int = Field(..., ge=0, le=15)
    experience_years: float = Field(..., ge=0.0, le=40.0)
    hackathons: int = Field(..., ge=0, le=20)
    research_papers: int = Field(..., ge=0, le=20)
    skills_score: float = Field(..., ge=0.0, le=40.0)
    soft_skills_score: float = Field(..., ge=0.0, le=10.0)
    resume_length_words: int = Field(..., ge=50, le=2000)
    company_type: Literal["Startup", "Mid-size", "MNC"]


def generate_tips(data: CandidateFeatures) -> list[str]:
    tips = []
    if data.skills_score < 14.0:
        tips.append("Technical skills score is below average: add measurable tech stack achievements and certifications.")
    if data.internships == 0 and data.experience_years < 1.0:
        tips.append("Limited industry exposure: highlight 2-3 end-to-end capstone projects or open-source contributions.")
    if data.soft_skills_score < 6.0:
        tips.append("Strengthen leadership & communication signals by emphasizing team projects, hackathons, or mentoring.")
    if data.resume_length_words < 250:
        tips.append("Resume word count is brief: expand bullet points with quantified impact metrics.")
    if not tips:
        tips.append("Strong overall profile: tailor top keywords directly to the target company's job description.")
        tips.append("Highlight leadership outcomes and system design trade-offs during technical interviews.")
    return tips[:3]


def log_prediction(payload: dict, prediction: int, probability: float):
    os.makedirs("logs", exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": payload,
        "prediction": prediction,
        "shortlist_probability": round(probability, 4),
    }
    with open("logs/predictions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(candidate: CandidateFeatures):
    input_dict = candidate.model_dump()
    df = pd.DataFrame([input_dict])

    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])

    log_prediction(input_dict, pred, prob)

    return {
        "prediction": pred,
        "status": "Shortlisted (Suitable)" if pred == 1 else "Not Shortlisted",
        "shortlist_probability": round(prob, 4),
        "tips": generate_tips(candidate),
    }
