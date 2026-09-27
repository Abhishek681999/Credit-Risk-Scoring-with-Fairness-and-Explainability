"""Create separate Kaggle submissions for all three saved models."""
import json

import joblib
import numpy as np
import pandas as pd

from pipeline_utils import load_settings, prepare
from project_config import DATA, MODELS, OUTPUTS

model_dir = OUTPUTS / "models"
out = OUTPUTS / "submissions"
out.mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(DATA / "cs-test.csv")
template = pd.read_csv(DATA / "sampleEntry.csv")
if list(template.columns) != ["Id", "Probability"] or len(template) != len(raw):
    raise ValueError("Check the original sampleEntry.csv format and row count")
unnamed = [c for c in raw if c.startswith("Unnamed:")]
if len(unnamed) == 1 and not np.array_equal(raw[unnamed[0]].to_numpy(), template["Id"].to_numpy()):
    raise ValueError("Kaggle test IDs are not aligned with sampleEntry.csv")

features = json.loads((model_dir / "features.json").read_text())
test = prepare(raw, load_settings())
X_test = test[features]
if X_test.isna().any().any():
    raise ValueError("Missing model input in Kaggle test")

for name in MODELS:
    model = joblib.load(model_dir / f"{name}.joblib")
    scores = model.predict_proba(X_test)[:, 1]
    if not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError(f"Invalid {name} predictions")
    submission = template.copy()
    submission["Probability"] = scores
    filename = out / f"submission_{name}.csv"
    submission.to_csv(filename, index=False)
    print(f"{name}: {len(scores):,} rows; mean {scores.mean():.4f}; {filename}")
print("Submit one CSV at a time; Kaggle's leaderboard evaluates ROC-AUC.")
