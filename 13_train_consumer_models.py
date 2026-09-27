from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/consumer_loans")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / "consumer_train_raw.csv", low_memory=False)
validation = pd.read_csv(
    DATA_DIR / "consumer_validation_raw.csv",
    low_memory=False,
)

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


def make_preprocessor():
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Missing"),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


models = {
    "logistic_regression": Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            (
                "model",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    ),
    "lightgbm": Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            (
                "model",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=100,
                    random_state=42,
                    verbosity=-1,
                ),
            ),
        ]
    ),
    "xgboost": Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.05,
                    min_child_weight=10,
                    subsample=0.85,
                    colsample_bytree=0.9,
                    reg_lambda=3,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=4,
                    random_state=42,
                ),
            ),
        ]
    ),
}

results = []
predictions = pd.DataFrame(
    {
        "issue_year": validation["issue_year"],
        "y_true": y_validation,
    }
)

fig, ax = plt.subplots(figsize=(8, 6))
baseline = y_validation.mean()

ax.axhline(
    baseline,
    linestyle="--",
    color="gray",
    label=f"Charge-off prevalence ({baseline:.3f})",
)

for name, pipeline in models.items():
    print(f"Training {name}...", flush=True)
    pipeline.fit(X_train, y_train)

    probability = pipeline.predict_proba(X_validation)[:, 1]
    predictions[f"{name}_probability"] = probability

    results.append(
        {
            "model": name,
            "roc_auc": roc_auc_score(y_validation, probability),
            "average_precision": average_precision_score(
                y_validation, probability
            ),
            "brier_score": brier_score_loss(
                y_validation, probability
            ),
            "mean_predicted_probability": probability.mean(),
        }
    )

    precision, recall, _ = precision_recall_curve(
        y_validation, probability
    )
    ax.plot(recall, precision, label=name)

    joblib.dump(pipeline, OUTPUT_DIR / f"{name}.joblib")

ax.set(
    xlabel="Recall",
    ylabel="Precision",
    title="Consumer loans: 2014 validation precision–recall",
    xlim=(0, 1),
    ylim=(0, 1),
)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "precision_recall.png", dpi=160)
plt.close(fig)

results_table = pd.DataFrame(results)
results_table.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
predictions.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)

print("\n=== 2014 VALIDATION RESULTS ===")
print(f"Training rows: {len(train):,}")
print(f"Validation rows: {len(validation):,}")
print(f"Validation charge-off prevalence: {baseline:.4f}")
print(results_table.round(4).to_string(index=False))
print(f"\nModels, predictions and chart saved in: {OUTPUT_DIR}")