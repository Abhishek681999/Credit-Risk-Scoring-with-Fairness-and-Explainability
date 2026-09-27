"""Compare feature explanations across the trained credit-risk models."""

import json

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from project_config import OUTPUTS, POLICY_THRESHOLD, PROCESSED, TARGET


OUT = OUTPUTS / "explainability"
OUT.mkdir(parents=True, exist_ok=True)

ALL_MODELS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "ann",
)
SHAP_MODELS = ("xgboost", "lightgbm")


def positive_class_explanation(explanation):
    """Select the default class if SHAP returns separate class outputs."""
    values = np.asarray(explanation.values)

    if values.ndim == 3:
        if values.shape[2] != 2:
            raise ValueError(f"Unexpected SHAP shape: {values.shape}")
        return explanation[:, :, 1]

    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")

    return explanation


# Load the processed validation set and verify prediction alignment.
validation = pd.read_csv(PROCESSED / "validation.csv")
predictions = pd.read_csv(
    OUTPUTS / "evaluation" / "calibrated_validation_predictions.csv"
)
features = json.loads(
    (OUTPUTS / "models" / "features.json").read_text(encoding="utf-8")
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

X = validation[features]
y = validation[TARGET].to_numpy()

models = {
    name: joblib.load(OUTPUTS / "models" / f"{name}.joblib")
    for name in ALL_MODELS
}

print(f"Validation rows: {len(validation):,}")
print(f"Feature count: {len(features)}")


# ---------------------------------------------------------------------
# Part 1: SHAP for the two boosting models
# ---------------------------------------------------------------------

# Use the LightGBM calibrated policy to identify the same three cases
# for both sets of waterfall plots.
lightgbm_scores = predictions["lightgbm_calibrated"].to_numpy()
flagged = lightgbm_scores >= POLICY_THRESHOLD

case_masks = {
    "true_positive": (y == 1) & flagged,
    "false_positive": (y == 0) & flagged,
    "false_negative": (y == 1) & ~flagged,
}

case_indices = {}

for case_name, mask in case_masks.items():
    candidates = np.flatnonzero(mask)

    if len(candidates) == 0:
        print(f"No validation row found for {case_name}; skipping.")
        continue

    # Pick the highest LightGBM calibrated score in each category.
    case_indices[case_name] = int(
        candidates[np.argmax(lightgbm_scores[candidates])]
    )

# Same 1,500 validation rows for both SHAP beeswarm plots.
global_indices = (
    validation.sample(
        n=min(1500, len(validation)),
        random_state=42,
    )
    .index.to_numpy()
)
X_global = X.iloc[global_indices]

case_records = []
shap_importance_tables = []

for model_name in SHAP_MODELS:
    print(f"\nCalculating SHAP for {model_name}...", flush=True)
    model = models[model_name]

    # Explain the raw model score, not its calibrated probability.
    explainer = shap.TreeExplainer(model, model_output="raw")
    explanation = positive_class_explanation(explainer(X_global))

    shap.plots.beeswarm(
        explanation,
        max_display=len(features),
        show=False,
    )
    plt.savefig(
        OUT / f"{model_name}_shap_beeswarm.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close("all")

    importance = pd.DataFrame(
        {
            "model": model_name,
            "feature": features,
            "mean_absolute_shap": np.abs(
                np.asarray(explanation.values)
            ).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)

    importance.to_csv(
        OUT / f"{model_name}_shap_importance.csv",
        index=False,
    )
    shap_importance_tables.append(importance)

    for case_name, index in case_indices.items():
        row = X.iloc[[index]]
        local = positive_class_explanation(explainer(row))

        shap.plots.waterfall(
            local[0],
            max_display=len(features),
            show=False,
        )
        plt.savefig(
            OUT / f"{model_name}_{case_name}_waterfall.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close("all")

        calibrated_column = f"{model_name}_calibrated"

        case_records.append(
            {
                "model": model_name,
                "case_selected_using_lightgbm": case_name,
                "validation_row": index,
                "observed_default": int(y[index]),
                "age_group": validation.iloc[index]["age_group"],
                "raw_model_probability": float(
                    model.predict_proba(row)[0, 1]
                ),
                "calibrated_probability": float(
                    predictions.iloc[index][calibrated_column]
                ),
            }
        )

pd.concat(shap_importance_tables, ignore_index=True).to_csv(
    OUT / "boosting_shap_importance_comparison.csv",
    index=False,
)

cases = pd.DataFrame(case_records)
cases.to_csv(OUT / "boosting_same_cases.csv", index=False)


# ---------------------------------------------------------------------
# Part 2: Common permutation-importance comparison for all five models
# ---------------------------------------------------------------------

# A stratified sample keeps roughly the same default prevalence.
# Every model is assessed on precisely these same rows.
sample_size = min(5000, len(validation))

sample_indices, _ = train_test_split(
    np.arange(len(validation)),
    train_size=sample_size,
    stratify=y,
    random_state=42,
)

X_compare = X.iloc[sample_indices]
y_compare = y[sample_indices]

permutation_rows = []
model_ap_rows = []

for model_name in ALL_MODELS:
    print(
        f"Calculating permutation importance for {model_name}...",
        flush=True,
    )
    model = models[model_name]

    baseline_ap = average_precision_score(
        y_compare,
        model.predict_proba(X_compare)[:, 1],
    )

    result = permutation_importance(
        model,
        X_compare,
        y_compare,
        scoring="average_precision",
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )

    model_ap_rows.append(
        {
            "model": model_name,
            "sample_average_precision": baseline_ap,
        }
    )

    for position, feature in enumerate(features):
        permutation_rows.append(
            {
                "model": model_name,
                "feature": feature,
                "ap_drop_mean": float(result.importances_mean[position]),
                "ap_drop_std": float(result.importances_std[position]),
            }
        )

permutation_table = pd.DataFrame(permutation_rows)
permutation_table.to_csv(
    OUT / "five_model_permutation_importance.csv",
    index=False,
)

pd.DataFrame(model_ap_rows).to_csv(
    OUT / "five_model_sample_ap.csv",
    index=False,
)

# One chart showing feature importance on the same AP-drop scale.
heatmap = permutation_table.pivot(
    index="feature",
    columns="model",
    values="ap_drop_mean",
)
heatmap = heatmap[ list(ALL_MODELS) ]
heatmap = heatmap.loc[
    heatmap.mean(axis=1).sort_values(ascending=False).index
]

fig, ax = plt.subplots(figsize=(11, 8))
picture = ax.imshow(
    heatmap.to_numpy(),
    aspect="auto",
    cmap="Blues",
)
ax.set_xticks(
    np.arange(len(heatmap.columns)),
    heatmap.columns,
    rotation=30,
    ha="right",
)
ax.set_yticks(
    np.arange(len(heatmap.index)),
    heatmap.index,
)
ax.set_title(
    "Feature importance across five models\n"
    "Decrease in average precision after shuffling one feature"
)
fig.colorbar(picture, ax=ax, label="Average precision decrease")
fig.tight_layout()
fig.savefig(
    OUT / "five_model_permutation_heatmap.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close(fig)


print("\n=== TOP 5 FEATURES BY MODEL: PERMUTATION IMPORTANCE ===")

for model_name in ALL_MODELS:
    top = (
        permutation_table[
            permutation_table["model"] == model_name
        ]
        .sort_values("ap_drop_mean", ascending=False)
        .head(5)
    )
    print(f"\n{model_name}")
    print(
        top[["feature", "ap_drop_mean"]]
        .round(4)
        .to_string(index=False)
    )

print("\n=== SAME THREE APPLICANTS, TWO BOOSTING MODELS ===")
print(cases.round(4).to_string(index=False))

print(f"\nCharts and CSV files saved in: {OUT}")
print(
    "SHAP values explain each boosting model's raw score. "
    "Permutation importance compares AP drops across all five models."
)