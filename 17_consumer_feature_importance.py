from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split


DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/consumer_loans")

validation = pd.read_csv(
    DATA_DIR / "consumer_validation_raw.csv",
    low_memory=False,
)
model = joblib.load(OUTPUT_DIR / "lightgbm.joblib")

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


X = build_features(validation)
y = validation["target"].astype(int)

# Stratified sample: keeps approximately the same charge-off prevalence
# while avoiding repeated predictions over all 162,570 rows.
sample_indices, _ = train_test_split(
    range(len(validation)),
    train_size=10_000,
    stratify=y,
    random_state=42,
)

X_sample = X.iloc[sample_indices]
y_sample = y.iloc[sample_indices]

baseline_ap = average_precision_score(
    y_sample,
    model.predict_proba(X_sample)[:, 1],
)

print(f"Sample size: {len(X_sample):,}")
print(f"Sample charge-off rate: {y_sample.mean():.2%}")
print(f"Unshuffled average precision: {baseline_ap:.4f}")
print("\nShuffling each feature three times...", flush=True)

importance = permutation_importance(
    model,
    X_sample,
    y_sample,
    scoring="average_precision",
    n_repeats=3,
    random_state=42,
    n_jobs=1,
)

table = pd.DataFrame(
    {
        "feature": X_sample.columns,
        "ap_drop_mean": importance.importances_mean,
        "ap_drop_std": importance.importances_std,
    }
).sort_values("ap_drop_mean", ascending=False)

table.to_csv(
    OUTPUT_DIR / "consumer_feature_importance.csv",
    index=False,
)

print("\n=== FEATURES MOST USED FOR RANKING ===")
print(table.round(4).to_string(index=False))

top = table.head(12).iloc[::-1]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top["feature"], top["ap_drop_mean"])
ax.set(
    xlabel="Mean drop in average precision after shuffling",
    title="LightGBM feature importance: 2014 validation sample",
)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "consumer_feature_importance.png",
    dpi=160,
)
plt.close(fig)

print(f"\nSaved table and chart in: {OUTPUT_DIR}")