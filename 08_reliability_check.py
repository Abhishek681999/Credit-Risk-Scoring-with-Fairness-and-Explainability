"""Check calibrated probability reliability overall and by age group."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import brier_score_loss

from project_config import OUTPUTS, PROCESSED, TARGET


MODELS = ("xgboost", "lightgbm", "ann")
OUT = OUTPUTS / "reliability"
OUT.mkdir(parents=True, exist_ok=True)

validation = pd.read_csv(PROCESSED / "validation.csv")
predictions = pd.read_csv(
    OUTPUTS / "evaluation" / "calibrated_validation_predictions.csv"
)

if len(validation) != len(predictions):
    raise ValueError("Validation and prediction row counts differ.")

if not np.array_equal(
    validation[TARGET].to_numpy(),
    predictions["y_true"].to_numpy(),
):
    raise ValueError("Validation labels are not aligned with predictions.")

if not np.array_equal(
    validation["age_group"].to_numpy(),
    predictions["age_group"].to_numpy(),
):
    raise ValueError("Age groups are not aligned with predictions.")

y = validation[TARGET].to_numpy()

overall_rows = []
age_rows = []
bin_tables = []

for name in MODELS:
    column = f"{name}_calibrated"

    if column not in predictions.columns:
        raise ValueError(f"Missing calibrated score column: {column}")

    score = predictions[column].to_numpy(dtype=float)

    if not np.isfinite(score).all():
        raise ValueError(f"{name} contains non-finite probabilities.")

    if ((score < 0) | (score > 1)).any():
        raise ValueError(f"{name} contains probabilities outside 0–1.")

    # Put applicants into 10 groups of roughly equal size,
    # ordered from lowest to highest predicted probability.
    frame = pd.DataFrame(
        {
            "actual": y,
            "predicted": score,
            "age_group": validation["age_group"].to_numpy(),
        }
    )
    frame["bin"] = pd.qcut(
        frame["predicted"],
        q=10,
        labels=False,
        duplicates="drop",
    )

    bins = (
        frame.groupby("bin", observed=True)
        .agg(
            applicants=("actual", "size"),
            mean_predicted=("predicted", "mean"),
            observed_default_rate=("actual", "mean"),
        )
        .reset_index()
    )
    bins.insert(0, "model", name)

    # Expected calibration error: weighted average absolute distance
    # between mean predicted and observed default rate in each bin.
    ece = (
        (
            bins["applicants"]
            * (
                bins["mean_predicted"]
                - bins["observed_default_rate"]
            ).abs()
        ).sum()
        / len(frame)
    )

    overall_rows.append(
        {
            "model": name,
            "applicants": len(frame),
            "observed_default_rate": frame["actual"].mean(),
            "mean_predicted": frame["predicted"].mean(),
            "brier": brier_score_loss(y, score),
            "ece_10_equal_count_bins": ece,
        }
    )
    bin_tables.append(bins)

    for age_group, group in frame.groupby("age_group", observed=True):
        age_rows.append(
            {
                "model": name,
                "age_group": age_group,
                "applicants": len(group),
                "observed_default_rate": group["actual"].mean(),
                "mean_predicted": group["predicted"].mean(),
                "brier": brier_score_loss(
                    group["actual"],
                    group["predicted"],
                ),
            }
        )

overall = pd.DataFrame(overall_rows)
by_age = pd.DataFrame(age_rows)
all_bins = pd.concat(bin_tables, ignore_index=True)

overall.to_csv(OUT / "overall_reliability.csv", index=False)
by_age.to_csv(OUT / "reliability_by_age.csv", index=False)
all_bins.to_csv(OUT / "reliability_bins.csv", index=False)

# Plot observed versus predicted default rates.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax in axes:
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Perfect agreement",
    )

    for name in MODELS:
        model_bins = all_bins[all_bins["model"] == name]
        ax.plot(
            model_bins["mean_predicted"],
            model_bins["observed_default_rate"],
            marker="o",
            label=name,
        )

    ax.set_xlabel("Mean predicted default probability")
    ax.set_ylabel("Actual default rate")
    ax.grid(alpha=0.25)

axes[0].set_title("Full probability range")
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)

axes[1].set_title("Zoom: probabilities up to 25%")
axes[1].set_xlim(0, 0.25)
axes[1].set_ylim(0, 0.25)
axes[1].legend()

fig.suptitle("Reliability of calibrated validation predictions")
fig.tight_layout()
fig.savefig(
    OUT / "reliability_curves.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close(fig)

# A focused check around the illustrative 9.09% policy threshold.
near_threshold_rows = []

for name in MODELS:
    score = predictions[f"{name}_calibrated"].to_numpy(dtype=float)
    mask = (score >= 0.07) & (score < 0.11)

    if mask.any():
        near_threshold_rows.append(
            {
                "model": name,
                "score_band": "7% to <11%",
                "applicants": int(mask.sum()),
                "mean_predicted": float(score[mask].mean()),
                "observed_default_rate": float(y[mask].mean()),
            }
        )

near_threshold = pd.DataFrame(near_threshold_rows)
near_threshold.to_csv(
    OUT / "near_policy_threshold.csv",
    index=False,
)

print("\n=== OVERALL RELIABILITY ===")
print(overall.round(4).to_string(index=False))

print("\n=== BY AGE GROUP ===")
print(by_age.round(4).to_string(index=False))

print("\n=== NEAR THE 9.09% POLICY THRESHOLD ===")
print(near_threshold.round(4).to_string(index=False))

print(f"\nChart and tables saved in: {OUT}")
print(
    "ECE depends on the chosen bins; use it alongside the chart "
    "and Brier score, not as the sole model-selection rule."
)