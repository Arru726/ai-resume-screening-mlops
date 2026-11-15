import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLS = [
    "age", "cgpa", "internships", "projects", "programming_languages",
    "certifications", "experience_years", "hackathons", "research_papers",
    "skills_score", "soft_skills_score", "resume_length_words"
]
CATEGORICAL_COLS = ["education_level", "university_tier", "company_type"]
TARGET_COL = "hired"


def build_pipeline(model_type: str, C: float, n_estimators: int, max_depth: int):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    if model_type == "lr":
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
    elif model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
        )
    elif model_type == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def main():
    parser = argparse.ArgumentParser(description="Train Resume Screening Model")
    parser.add_argument("--model_type", type=str, default="rf", choices=["lr", "rf", "gb"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--save_model", action="store_true", help="Save model to model/model.joblib")
    args = parser.parse_args()

    df = pd.read_csv("data/resume_screening.csv")
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Resume_Screening_CSL236")

    with mlflow.start_run(run_name=f"{args.model_type}_run"):
        pipeline = build_pipeline(args.model_type, args.C, args.n_estimators, args.max_depth)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_param("model_type", args.model_type)
        if args.model_type == "lr":
            mlflow.log_param("C", args.C)
        else:
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)

        mlflow.log_metric("accuracy", round(acc, 4))
        mlflow.log_metric("precision", round(prec, 4))
        mlflow.log_metric("recall", round(rec, 4))
        mlflow.log_metric("f1_score", round(f1, 4))
        mlflow.sklearn.log_model(pipeline, artifact_path="model", serialization_format="cloudpickle")

        print(f"[{args.model_type.upper()}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        if args.save_model:
            os.makedirs("model", exist_ok=True)
            joblib.dump(pipeline, "model/model.joblib")
            print("✅ Best model saved to model/model.joblib")


if __name__ == "__main__":
    main()
