"""Calibrate and compare all five models on the same validation rows."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from pipeline_utils import logit_feature
from project_config import (
    AGE_GROUPS,
    FN_COST,
    FP_COST,
    MODELS,
    OUTPUTS,
    POLICY_THRESHOLD,
    SEED,
)


ALL_MODELS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "ann",
)

out = OUTPUTS / "evaluation"
out.mkdir(parents=True, exist_ok=True)

raw_df = pd.read_csv(OUTPUTS / "models" / "validation_predictions.csv")
y = raw_df["y_true"].to_numpy()
groups = raw_df["age_group"].to_numpy()

for name in ALL_MODELS:
    column = f"{name}_score"
    if column not in raw_df.columns:
        raise ValueError(
            f"Missing {column}. Run the updated five-model "
            "03_train_models.py first."
        )

# These are out-of-fold calibrated probabilities for evaluation.
oof = raw_df[["y_true", "age_group"]].copy()
folds = StratifiedKFold(
    n_splits=2,
    shuffle=True,
    random_state=SEED,
)


def group_count(y_true, y_pred):
    return len(y_true)


def observed_rate(y_true, y_pred):
    return np.mean(y_true)


summary_rows = []
cost_rows = []
group_tables = []

for name in ALL_MODELS:
    print(f"Calibrating and evaluating {name}...", flush=True)

    raw_score = raw_df[f"{name}_score"].to_numpy()
    z = logit_feature(raw_score)

    calibrated = np.empty(len(y), dtype=float)

    for fit_indices, assess_indices in folds.split(z, y):
        calibrator = LogisticRegression(max_iter=1000)
        calibrator.fit(z[fit_indices], y[fit_indices])

        calibrated[assess_indices] = (
            calibrator.predict_proba(z[assess_indices])[:, 1]
        )

    # This version is fitted on all labeled validation rows for
    # future unlabeled Kaggle test predictions. Evaluation above uses
    # the out-of-fold probabilities, not these fitted predictions.
    final_calibrator = LogisticRegression(max_iter=1000)
    final_calibrator.fit(z, y)
    joblib.dump(
        final_calibrator,
        out / f"{name}_calibrator.joblib",
    )

    oof[f"{name}_calibrated"] = calibrated

    decision = (calibrated >= POLICY_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y,
        decision,
        labels=[0, 1],
    ).ravel()

    frame = MetricFrame(
        metrics={
            "applicants": group_count,
            "observed_default_rate": observed_rate,
            "default_flag_rate": selection_rate,
            "false_positive_rate": false_positive_rate,
            "true_positive_rate": true_positive_rate,
            "accuracy": accuracy_score,
        },
        y_true=y,
        y_pred=decision,
        sensitive_features=groups,
    )

    group_table = frame.by_group.reindex(AGE_GROUPS).copy()
    group_table.insert(0, "model", name)
    group_table.insert(1, "age_group", group_table.index)
    group_tables.append(group_table.reset_index(drop=True))

    summary_rows.append(
        {
            "model": name,
            "roc_auc": roc_auc_score(y, raw_score),
            "average_precision": average_precision_score(
                y,
                raw_score,
            ),
            "raw_brier": brier_score_loss(y, raw_score),
            "calibrated_brier": brier_score_loss(y, calibrated),
            "calibrated_mean": calibrated.mean(),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "default_flag_rate": decision.mean(),
            "cost_per_1000": (
                1000 * (FP_COST * fp + FN_COST * fn) / len(y)
            ),
            "dp_difference": demographic_parity_difference(
                y,
                decision,
                sensitive_features=groups,
            ),
            "eo_difference": equalized_odds_difference(
                y,
                decision,
                sensitive_features=groups,
            ),
        }
    )

    # Show how the preferred model may change if the assumed
    # missed-default cost changes.
    for ratio in (5, 10, 20):
        threshold = 1 / (ratio + 1)
        scenario_decision = (calibrated >= threshold).astype(int)

        _, scenario_fp, scenario_fn, _ = confusion_matrix(
            y,
            scenario_decision,
            labels=[0, 1],
        ).ravel()

        cost_rows.append(
            {
                "model": name,
                "fn_to_fp_cost": ratio,
                "threshold": threshold,
                "false_positives": int(scenario_fp),
                "false_negatives": int(scenario_fn),
                "cost_per_1000": (
                    1000
                    * (scenario_fp + ratio * scenario_fn)
                    / len(y)
                ),
            }
        )


summary = pd.DataFrame(summary_rows)
by_age = pd.concat(group_tables, ignore_index=True)
costs = pd.DataFrame(cost_rows)

# New five-model outputs for our full comparison and dashboard.
summary.to_csv(out / "all5_summary.csv", index=False)
by_age.to_csv(out / "all5_by_age.csv", index=False)
costs.to_csv(out / "cost_scenarios.csv", index=False)
oof.to_csv(
    out / "calibrated_validation_predictions.csv",
    index=False,
)

# Keep the old three-model filenames current for any existing code
# that reads them. MODELS still contains XGBoost, LightGBM, and ANN.
summary[summary["model"].isin(MODELS)].to_csv(
    out / "top3_summary.csv",
    index=False,
)
by_age[by_age["model"].isin(MODELS)].to_csv(
    out / "top3_by_age.csv",
    index=False,
)


print(
    f"\nObserved default rate: {y.mean():.4f}; "
    f"illustrative FN:FP cost = {FN_COST}:{FP_COST}"
)
print(f"Common calibrated cutoff: {POLICY_THRESHOLD:.4f}")

print("\n=== FIVE-MODEL COMPARISON ===")
print(summary.round(4).to_string(index=False))

print("\n=== AGE-GROUP RESULTS ===")
age_view = by_age.copy()
rate_columns = [
    "observed_default_rate",
    "default_flag_rate",
    "false_positive_rate",
    "true_positive_rate",
    "accuracy",
]
age_view[rate_columns] *= 100
print(age_view.round(2).to_string(index=False))
print("Rate columns above are percentages.")

print("\n=== COST SENSITIVITY ===")
print(costs.round(3).to_string(index=False))


# Cost versus cutoff for every model.
fig, ax = plt.subplots(figsize=(10, 6))
thresholds = np.linspace(0.01, 0.40, 80)

for name in ALL_MODELS:
    scores = oof[f"{name}_calibrated"].to_numpy()
    curve = []

    for threshold in thresholds:
        decisions = scores >= threshold
        false_positives = np.sum((y == 0) & decisions)
        false_negatives = np.sum((y == 1) & ~decisions)

        curve.append(
            1000
            * (
                FP_COST * false_positives
                + FN_COST * false_negatives
            )
            / len(y)
        )

    ax.plot(thresholds, curve, label=name)

ax.axvline(
    POLICY_THRESHOLD,
    color="black",
    linestyle="--",
    label="10:1 theoretical cutoff",
)
ax.set(
    xlabel="Calibrated default-risk cutoff",
    ylabel="Illustrative cost units per 1,000 applicants",
    title="Cost versus decision cutoff",
)
ax.legend()
fig.tight_layout()
fig.savefig(out / "cost_vs_threshold.png", dpi=160)
plt.close(fig)


# Age-group flag rates at the common policy cutoff.
fig, ax = plt.subplots(figsize=(11, 6))
positions = np.arange(len(AGE_GROUPS))
width = 0.16

for position, name in enumerate(ALL_MODELS):
    subset = by_age[by_age["model"] == name].set_index("age_group")
    shift = position - (len(ALL_MODELS) - 1) / 2

    ax.bar(
        positions + shift * width,
        subset.loc[
            list(AGE_GROUPS),
            "default_flag_rate",
        ].to_numpy() * 100,
        width,
        label=name,
    )

ax.set_xticks(positions, AGE_GROUPS)
ax.set(
    ylabel="Applicants flagged (%)",
    title="Default-flag rate by age group at the 10:1 cutoff",
)
ax.legend()
fig.tight_layout()
fig.savefig(out / "age_flag_rates.png", dpi=160)
plt.close(fig)

print(f"\nFive-model tables and charts saved in: {out}")
print(
    "06_mitigation.py still uses the original three models "
    "from project_config.py."
)