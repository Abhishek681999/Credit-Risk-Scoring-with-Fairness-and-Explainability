from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


OUTPUT_DIR = Path("outputs/consumer_loans")
predictions = pd.read_csv(OUTPUT_DIR / "validation_predictions.csv")

MODELS = ["logistic_regression", "lightgbm", "xgboost"]
y_true = predictions["y_true"].astype(int)

if len(predictions) != 162_570:
    raise ValueError("Unexpected validation row count.")

rows = []

fig, ax = plt.subplots(figsize=(8, 7))

for model in MODELS:
    scores = predictions[f"{model}_probability"]

    if scores.isna().any() or not scores.between(0, 1).all():
        raise ValueError(f"Invalid probabilities for {model}")

    # Ten groups with approximately equal numbers of loans,
    # ordered from lowest to highest predicted risk.
    risk_decile = pd.qcut(
        scores.rank(method="first"),
        q=10,
        labels=False,
    ) + 1

    grouped = pd.DataFrame(
        {
            "risk_decile": risk_decile,
            "predicted": scores,
            "observed": y_true,
        }
    ).groupby("risk_decile", as_index=False).agg(
        loans=("observed", "size"),
        mean_predicted=("predicted", "mean"),
        observed_charge_off_rate=("observed", "mean"),
    )

    grouped.insert(0, "model", model)
    rows.append(grouped)

    ax.plot(
        grouped["mean_predicted"],
        grouped["observed_charge_off_rate"],
        marker="o",
        label=model,
    )

    print(f"\n=== {model} ===")
    print(
        f"Mean predicted: {scores.mean():.2%} | "
        f"Observed: {y_true.mean():.2%} | "
        f"Brier: {brier_score_loss(y_true, scores):.4f}"
    )
    print(
        grouped[
            [
                "risk_decile",
                "loans",
                "mean_predicted",
                "observed_charge_off_rate",
            ]
        ].round(4).to_string(index=False)
    )

table = pd.concat(rows, ignore_index=True)
table.to_csv(OUTPUT_DIR / "reliability_by_decile.csv", index=False)

limit = max(
    table["mean_predicted"].max(),
    table["observed_charge_off_rate"].max(),
) * 1.05

ax.plot(
    [0, limit],
    [0, limit],
    linestyle="--",
    color="black",
    label="Predicted = observed",
)
ax.set(
    xlabel="Mean predicted charge-off probability",
    ylabel="Observed charge-off rate",
    title="2014 consumer-loan reliability by predicted-risk decile",
    xlim=(0, limit),
    ylim=(0, limit),
)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "reliability_by_decile.png", dpi=160)
plt.close(fig)

constant_brier = brier_score_loss(
    y_true,
    np.repeat(y_true.mean(), len(y_true)),
)

print(f"\nConstant-rate Brier reference: {constant_brier:.4f}")
print(f"Saved table and chart in: {OUTPUT_DIR}")