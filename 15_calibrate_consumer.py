from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/consumer_loans")

train = pd.read_csv(DATA_DIR / "consumer_train_raw.csv", low_memory=False)
validation = pd.read_csv(
    DATA_DIR / "consumer_validation_raw.csv",
    low_memory=False,
)
previous = pd.read_csv(OUTPUT_DIR / "validation_predictions.csv")

if len(previous) != len(validation):
    raise ValueError("Validation predictions and data have different row counts.")

if not previous["y_true"].reset_index(drop=True).equals(
    validation["target"].reset_index(drop=True)
):
    raise ValueError("Validation predictions are not aligned with the data.")

NUMERIC_FEATURES = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "collections_12_mths_ex_med",
    "pub_rec_bankruptcies",
    "credit_history_years",
]

CATEGORICAL_FEATURES = [
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_features(frame):
    result = frame.copy()

    earliest = pd.to_datetime(
        result["earliest_cr_line"],
        format="%b-%Y",
        errors="coerce",
    )

    result["credit_history_years"] = (
        result["issue_year"] - earliest.dt.year
    )

    result.loc[
        result["credit_history_years"] < 0,
        "credit_history_years",
    ] = float("nan")

    for column in NUMERIC_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    return result[FEATURES]


X_train = build_features(train)
X_validation = build_features(validation)
y_train = train["target"].astype(int)
y_validation = validation["target"].astype(int)

folds = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

comparison = []
calibrated_predictions = previous.copy()

for name in ["lightgbm", "xgboost"]:
    print(f"Calibrating {name} (this retrains the model several times)...", flush=True)

    # clone() takes the saved pipeline's settings without using its fitted
    # state. Calibration then fits on training-period rows only.
    fitted_pipeline = joblib.load(OUTPUT_DIR / f"{name}.joblib")
    unfitted_pipeline = clone(fitted_pipeline)

    calibrated_model = CalibratedClassifierCV(
        estimator=unfitted_pipeline,
        method="sigmoid",
        cv=folds,
        ensemble=False,
    )

    calibrated_model.fit(X_train, y_train)

    calibrated_score = calibrated_model.predict_proba(
        X_validation
    )[:, 1]

    raw_score = previous[f"{name}_probability"]
    calibrated_predictions[f"{name}_calibrated_probability"] = (
        calibrated_score
    )

    for version, score in [
        ("raw", raw_score),
        ("calibrated", calibrated_score),
    ]:
        comparison.append(
            {
                "model": name,
                "version": version,
                "roc_auc": roc_auc_score(y_validation, score),
                "average_precision": average_precision_score(
                    y_validation, score
                ),
                "brier_score": brier_score_loss(y_validation, score),
                "mean_predicted_probability": score.mean(),
                "observed_charge_off_rate": y_validation.mean(),
            }
        )

    joblib.dump(
        calibrated_model,
        OUTPUT_DIR / f"{name}_calibrated.joblib",
    )

table = pd.DataFrame(comparison)
table.to_csv(OUTPUT_DIR / "calibration_comparison.csv", index=False)
calibrated_predictions.to_csv(
    OUTPUT_DIR / "calibrated_validation_predictions.csv",
    index=False,
)

print("\n=== 2014 VALIDATION: BEFORE AND AFTER CALIBRATION ===")
print(table.round(4).to_string(index=False))
print(f"\nSaved calibrated models and results in: {OUTPUT_DIR}")