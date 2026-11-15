from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

STRONG_CANDIDATE = {
    "age": 25,
    "education_level": "Masters",
    "university_tier": "Tier 1",
    "cgpa": 9.3,
    "internships": 3,
    "projects": 6,
    "programming_languages": 5,
    "certifications": 3,
    "experience_years": 3.0,
    "hackathons": 3,
    "research_papers": 1,
    "skills_score": 29.0,
    "soft_skills_score": 9.0,
    "resume_length_words": 500,
    "company_type": "MNC",
}

WEAK_CANDIDATE = {
    "age": 22,
    "education_level": "Bachelors",
    "university_tier": "Tier 3",
    "cgpa": 5.1,
    "internships": 0,
    "projects": 0,
    "programming_languages": 1,
    "certifications": 0,
    "experience_years": 0.0,
    "hackathons": 0,
    "research_papers": 0,
    "skills_score": 4.0,
    "soft_skills_score": 3.5,
    "resume_length_words": 150,
    "company_type": "MNC",
}


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_ui_loads():
    res = client.get("/")
    assert res.status_code == 200
    assert "AI Resume Screening" in res.text


def test_domain_logic_strong_beats_weak():
    res_strong = client.post("/predict", json=STRONG_CANDIDATE)
    res_weak = client.post("/predict", json=WEAK_CANDIDATE)
    assert res_strong.status_code == 200
    assert res_weak.status_code == 200
    assert res_strong.json()["shortlist_probability"] > res_weak.json()["shortlist_probability"]


def test_out_of_range_returns_422():
    bad_payload = STRONG_CANDIDATE.copy()
    bad_payload["cgpa"] = 12.5  # Invalid: exceeds max 10.0
    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 422
