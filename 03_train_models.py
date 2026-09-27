"""Train two baselines and three shortlisted credit-risk models."""

import json

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pipeline_utils import feature_columns
from project_config import MODELS, OUTPUTS, PROCESSED, SEED, TARGET

out = OUTPUTS / "models"
out.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(PROCESSED / "train.csv")
validation = pd.read_csv(PROCESSED / "validation.csv")

features = feature_columns(train)

X_train = train[features]
y_train = train[TARGET]
X_val = validation[features]
y_val = validation[TARGET]

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
positive_weight = negative_count / positive_count

# The first two are benchmarks. The last three receive the detailed
# calibration, cost, fairness, and mitigation analysis.
models = {
    "logistic_regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
        ),
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    ),
    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=positive_weight,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        scale_pos_weight=positive_weight,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    ),
    "ann": make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            alpha=0.01,
            early_stopping=True,
            validation_fraction=0.10,
            max_iter=150,
            n_iter_no_change=10,
            random_state=SEED,
        ),
    ),
}

predictions = pd.DataFrame({
    "y_true": y_val.to_numpy(),
    "age_group": validation["age_group"].to_numpy(),
})

results = []

for name, model in models.items():
    print(f"Training {name}...", flush=True)
    model.fit(X_train, y_train)

    scores = model.predict_proba(X_val)[:, 1]
    predictions[f"{name}_score"] = scores

    results.append({
        "model": name,
        "role": "shortlisted" if name in MODELS else "baseline",
        "roc_auc": roc_auc_score(y_val, scores),
        "average_precision": average_precision_score(y_val, scores),
        "brier": brier_score_loss(y_val, scores),
        "mean_score": scores.mean(),
    })

    joblib.dump(model, out / f"{name}.joblib")

predictions.to_csv(
    out / "validation_predictions.csv",
    index=False,
)
(out / "features.json").write_text(
    json.dumps(features, indent=2)
)

results_df = pd.DataFrame(results)
results_df.to_csv(
    out / "validation_metrics.csv",
    index=False,
)

print("\n=== FIVE-MODEL VALIDATION COMPARISON ===")
print(results_df.round(4).to_string(index=False))
print(f"\nObserved validation default rate: {y_val.mean():.4f}")
print(f"Models and predictions saved in: {out.resolve()}")