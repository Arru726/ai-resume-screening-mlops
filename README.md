# AI-Based Resume Screening & Candidate Shortlisting System (CSL236 MLOps)

An end-to-end production-style MLOps pipeline that predicts candidate shortlisting probability and provides actionable resume-improvement tips, built with Scikit-learn, MLflow, FastAPI, Docker, and GitHub Actions.

## 1. Problem Statement & Dataset Source
Manual resume screening is time-consuming and inconsistent. This system automates candidate shortlisting using structured academic, technical, and experiential metrics.
- **Dataset:** [Resume Screening Dataset (200K Candidates) — Kaggle](https://www.kaggle.com/datasets/rhythmghai/resume-screening-dataset-200k-candidates)
- **Preprocessing:** Removed `candidate_id`, filtered invalid synthetic outliers (`cgpa > 10.0`, `resume_length_words < 50`, `experience_years < 0`), and sampled 20,000 clean rows (`data/resume_screening.csv`).

## 2. Data Contract (Mandatory)
Every constraint below is enforced in Pydantic (`app.py`); out-of-range or invalid inputs automatically return **HTTP 422**.

| Feature | Type | Allowed range / values | Why it matters |
| :--- | :--- | :--- | :--- |
| `age` | `int` | `18 – 65` | Candidate age eligibility |
| `education_level` | `str` | `{"Bachelors", "Masters", "PhD"}` | Academic qualification tier |
| `university_tier` | `str` | `{"Tier 1", "Tier 2", "Tier 3"}` | Institutional pedigree signal |
| `cgpa` | `float` | `0.0 – 10.0` | Baseline academic performance |
| `internships` | `int` | `0 – 15` | Prior industry exposure |
| `projects` | `int` | `0 – 25` | Practical hands-on portfolio |
| `programming_languages` | `int` | `1 – 15` | Technical breadth |
| `certifications` | `int` | `0 – 15` | Verified domain credentials |
| `experience_years` | `float` | `0.0 – 40.0` | Full-time work experience |
| `hackathons` | `int` | `0 – 20` | Competitive problem-solving signal |
| `research_papers` | `int` | `0 – 20` | Research & publication depth |
| `skills_score` | `float` | `0.0 – 40.0` | Derived technical skill rating |
| `soft_skills_score` | `float` | `0.0 – 10.0` | Communication & leadership rating |
| `resume_length_words` | `int` | `50 – 2000` | Resume completeness & structure |
| `company_type` | `str` | `{"Startup", "Mid-size", "MNC"}` | Target employer hiring bar |
| `hired` (target) | `binary` | `{0, 1}` | `1` = Shortlisted, `0` = Not Shortlisted |

## 3. MLflow Experiment Results
Tracked using `sqlite:///mlflow.db` across 3 algorithms (`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`):

| Algorithm | Hyperparameters | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Logistic Regression (`lr`) | `C=1.0` | 0.7053 | 0.7053 | 1.0000 | 0.8272 |
| Random Forest (`rf`) | `n_estimators=100, max_depth=6` | 0.7053 | 0.7053 | 1.0000 | 0.8272 |
| Random Forest (`rf`) | `n_estimators=150, max_depth=10` | 0.7053 | 0.7053 | 1.0000 | 0.8272 |
| **Gradient Boosting (`gb`)** | **`n_estimators=120, max_depth=4`** | **0.7020** | **0.7048** | **0.9936** | **0.8247** |

**Chosen Model:** **Gradient Boosting (`n_estimators=120, max_depth=4`)** was chosen and saved to `model/model.joblib` because it avoids trivial all-positive predictions (`recall = 1.00`) and produces calibrated shortlisting probabilities across diverse candidate profiles.

## 4. Setup & Run Instructions

### Local Environment
```bash
python -m venv venv
# Windows: .\venv\Scripts\activate  |  Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python train.py --model_type gb --n_estimators 120 --max_depth 4 --save_model
uvicorn app:app --reload --port 8000
python -m pytest -v --tb=short