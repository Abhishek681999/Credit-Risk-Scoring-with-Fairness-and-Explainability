"""Compare equalized-odds mitigation for all three models on held-out rows."""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from project_config import (
    AGE_GROUPS,
    FN_COST,
    FP_COST,
    MODELS,
    OUTPUTS,
    POLICY_THRESHOLD,
    PROCESSED,
    TARGET,
)


class Float64ProbabilityAdapter(ClassifierMixin, BaseEstimator):
    """Return a fitted model's probabilities as float64 for Fairlearn."""

    def __init__(self, estimator):
        self.estimator = estimator

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y):
        raise RuntimeError("The wrapped model is already fitted")

    def predict_proba(self, X):
        probabilities = self.estimator.predict_proba(X)
        return np.asarray(probabilities, dtype=np.float64)

    def predict(self, X):
        return self.estimator.predict(X)


out = OUTPUTS / "mitigation"
out.mkdir(parents=True, exist_ok=True)

val = pd.read_csv(PROCESSED / "validation.csv")
oof = pd.read_csv(
    OUTPUTS / "evaluation/calibrated_validation_predictions.csv"
)
features = json.loads((OUTPUTS / "models/features.json").read_text())

if len(val) != len(oof):
    raise ValueError("Validation files have different row counts")

if not np.array_equal(
    val[TARGET].to_numpy(),
    oof["y_true"].to_numpy(),
):
    raise ValueError("Predictions are not aligned with validation.csv")

if not np.array_equal(
    val["age_group"].to_numpy(),
    oof["age_group"].to_numpy(),
):
    raise ValueError("Age groups are not aligned")

y = val[TARGET].to_numpy()
groups = val["age_group"].to_numpy()

# Use one half to fit the fairness postprocessor and the other to assess it.
fit_idx, eval_idx = train_test_split(
    np.arange(len(y)),
    test_size=0.50,
    stratify=y,
    random_state=123,
)

X_fit = val.iloc[fit_idx][features]
X_eval = val.iloc[eval_idx][features]
y_eval = y[eval_idx]
groups_eval = groups[eval_idx]


def count(y_true, y_pred):
    return len(y_true)


def observed_default_rate(y_true, y_pred):
    return np.mean(y_true)


def assess(model_name, method, predictions):
    """Return overall and age-group results for one decision policy."""
    _, fp, fn, tp = confusion_matrix(
        y_eval,
        predictions,
        labels=[0, 1],
    ).ravel()

    frame = MetricFrame(
        metrics={
            "applicants": count,
            "observed_default_rate": observed_default_rate,
            "default_flag_rate": selection_rate,
            "false_positive_rate": false_positive_rate,
            "true_positive_rate": true_positive_rate,
            "accuracy": accuracy_score,
        },
        y_true=y_eval,
        y_pred=predictions,
        sensitive_features=groups_eval,
    )

    overall = {
        "model": model_name,
        "method": method,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "default_flag_rate": predictions.mean(),
        "cost_per_1000": (
            1000 * (FP_COST * fp + FN_COST * fn) / len(y_eval)
        ),
        "dp_difference": demographic_parity_difference(
            y_eval,
            predictions,
            sensitive_features=groups_eval,
        ),
        "eo_difference": equalized_odds_difference(
            y_eval,
            predictions,
            sensitive_features=groups_eval,
        ),
    }

    by_group = frame.by_group.reindex(AGE_GROUPS).copy()
    by_group.insert(0, "model", model_name)
    by_group.insert(1, "method", method)
    by_group.insert(2, "age_group", by_group.index)

    return overall, by_group.reset_index(drop=True)


rows = []
group_tables = []

for name in MODELS:
    model = joblib.load(OUTPUTS / f"models/{name}.joblib")

    # Unchanged cost-based baseline on the held-out evaluation half.
    baseline_scores = oof.iloc[eval_idx][
        f"{name}_calibrated"
    ].to_numpy()
    baseline_predictions = (
        baseline_scores >= POLICY_THRESHOLD
    ).astype(int)

    row, group_table = assess(
        name,
        "baseline_cost_10_to_1",
        baseline_predictions,
    )
    rows.append(row)
    group_tables.append(group_table)

    print(
        f"Fitting equalized-odds postprocessor for {name}...",
        flush=True,
    )

    # The adapter changes only the probability array's dtype.
    # It avoids a float32/float64 assignment error in Fairlearn + pandas.
    float64_model = Float64ProbabilityAdapter(model)

    postprocessor = ThresholdOptimizer(
        estimator=float64_model,
        prefit=True,
        predict_method="predict_proba",
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
    )

    postprocessor.fit(
        X_fit,
        y[fit_idx],
        sensitive_features=groups[fit_idx],
    )

    # Randomized predictions can differ by seed.
    for seed in (42, 43, 44):
        mitigated_predictions = postprocessor.predict(
            X_eval,
            sensitive_features=groups_eval,
            random_state=seed,
        ).astype(int)

        row, group_table = assess(
            name,
            f"mitigated_seed_{seed}",
            mitigated_predictions,
        )
        rows.append(row)

        if seed == 42:
            group_tables.append(group_table)

summary = pd.DataFrame(rows)
by_group = pd.concat(group_tables, ignore_index=True)

summary.to_csv(out / "comparison.csv", index=False)
by_group.to_csv(out / "by_age.csv", index=False)

print(
    f"\nFairness optimizer fit rows: {len(fit_idx):,}; "
    f"evaluation rows: {len(eval_idx):,}"
)
print(
    "Baseline and mitigation use the same evaluation rows. "
    "Costs assume FN:FP = 10:1."
)

print("\n=== OVERALL COMPARISON ===")
print(summary.round(4).to_string(index=False))

print("\n=== AGE-GROUP METRICS: BASELINE AND MITIGATION SEED 42 ===")
display = by_group.copy()
rate_columns = [
    "observed_default_rate",
    "default_flag_rate",
    "false_positive_rate",
    "true_positive_rate",
    "accuracy",
]
display[rate_columns] *= 100
print(display.round(2).to_string(index=False))
print("Rate columns above are percentages; applicant counts are unchanged.")

# Compare equalized-odds gaps for each model.
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(MODELS))
width = 0.35

baseline = summary[
    summary["method"] == "baseline_cost_10_to_1"
].set_index("model")
mitigated = summary[
    summary["method"] == "mitigated_seed_42"
].set_index("model")

ax.bar(
    x - width / 2,
    baseline.loc[list(MODELS), "eo_difference"],
    width,
    label="Baseline",
)
ax.bar(
    x + width / 2,
    mitigated.loc[list(MODELS), "eo_difference"],
    width,
    label="Mitigated",
)
ax.set_xticks(x, MODELS)
ax.set_ylabel("Equalized-odds difference")
ax.set_title("Fairness comparison on the same evaluation rows")
ax.legend()

fig.tight_layout()
fig.savefig(out / "eo_before_after.png", dpi=160)
plt.close(fig)

print(f"\nTables and chart saved in {out}")