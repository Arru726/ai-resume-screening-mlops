import json
import os
import pandas as pd

TRAIN_PATH = "data/resume_screening.csv"
LOG_PATH = "logs/predictions.jsonl"
FEATURES = ["cgpa", "internships", "projects", "experience_years", "skills_score", "soft_skills_score"]

if not os.path.exists(LOG_PATH):
    print("No prediction logs found yet. Make a few POST /predict requests first.")
    exit(0)

train_df = pd.read_csv(TRAIN_PATH)
with open(LOG_PATH, "r", encoding="utf-8") as f:
    logged_inputs = [json.loads(line)["input"] for line in f]

live_df = pd.DataFrame(logged_inputs)

print(f"Comparing {len(live_df)} live predictions against {len(train_df)} training rows:\n")
for col in FEATURES:
    t_mean, t_std = train_df[col].mean(), train_df[col].std()
    l_mean = live_df[col].mean()
    z_shift = abs(l_mean - t_mean) / (t_std + 1e-8)
    flag = "DRIFT DETECTED" if z_shift > 1.5 else "OK"
    print(f"{col:20s} | Train Mean: {t_mean:6.2f} | Live Mean: {l_mean:6.2f} | Shift: {z_shift:.2f} std [{flag}]")
