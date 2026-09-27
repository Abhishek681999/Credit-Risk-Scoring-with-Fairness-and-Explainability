from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/consumer_loans")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / "consumer_train_raw.csv")
validation = pd.read_csv(DATA_DIR / "consumer_validation_raw.csv")
predictions = pd.read_csv(
    OUTPUT_DIR / "validation_predictions.csv"
)

if len(validation) != len(predictions):
    raise ValueError("Validation rows and predictions do not match.")

if not validation["target"].reset_index(drop=True).equals(
    predictions["y_true"].reset_index(drop=True)
):
    raise ValueError("Validation targets and predictions are misaligned.")

scores = predictions["lightgbm_probability"]
y = validation["target"].astype(int)

# Rank validation loans from lowest to highest predicted risk.
deciles = pd.qcut(
    scores.rank(method="first"),
    q=10,
    labels=False,
) + 1

ranked = pd.DataFrame(
    {
        "risk_decile": deciles,
        "score": scores,
        "charged_off": y,
    }
)

risk_table = ranked.groupby(
    "risk_decile",
    as_index=False,
).agg(
    loans=("charged_off", "size"),
    charged_off=("charged_off", "sum"),
    average_predicted_risk=("score", "mean"),
    observed_charge_off_rate=("charged_off", "mean"),
)

total_charge_offs = int(y.sum())
overall_rate = y.mean()

risk_table["share_of_all_charge_offs"] = (
    risk_table["charged_off"] / total_charge_offs
)
risk_table["lift_vs_overall"] = (
    risk_table["observed_charge_off_rate"] / overall_rate
)

risk_table.to_csv(
    OUTPUT_DIR / "risk_deciles.csv",
    index=False,
)

top = risk_table.loc[risk_table["risk_decile"] == 10].iloc[0]

print("\n=== LIGHTGBM: HIGHEST-RISK 10% OF 2014 LOANS ===")
print(f"Loans flagged for review: {int(top['loans']):,}")
print(f"Charge-offs in that group: {int(top['charged_off']):,}")
print(
    f"Observed charge-off rate: "
    f"{top['observed_charge_off_rate']:.2%}"
)
print(
    f"Share of all 2014 charge-offs captured: "
    f"{top['share_of_all_charge_offs']:.2%}"
)
print(f"Lift versus portfolio average: {top['lift_vs_overall']:.2f}x")

print("\n=== ALL RISK DECILES ===")
print(risk_table.round(4).to_string(index=False))

# Compare outcomes by loan purpose across the historical training period
# and the later 2014 validation period.
train_purpose = (
    train.groupby("purpose")
    .agg(
        train_loans=("target", "size"),
        train_charge_off_rate=("target", "mean"),
    )
)

validation_purpose = (
    validation.groupby("purpose")
    .agg(
        validation_loans=("target", "size"),
        validation_charge_off_rate=("target", "mean"),
    )
)

purpose_table = (
    train_purpose.join(validation_purpose, how="outer")
    .fillna(0)
    .reset_index()
    .sort_values("validation_loans", ascending=False)
)

purpose_table.to_csv(
    OUTPUT_DIR / "purpose_outcomes.csv",
    index=False,
)

print("\n=== LOAN PURPOSE: LARGEST 10 GROUPS ===")
print(
    purpose_table.head(10).round(4).to_string(index=False)
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(
    risk_table["risk_decile"],
    risk_table["observed_charge_off_rate"] * 100,
)
axes[0].axhline(
    overall_rate * 100,
    color="red",
    linestyle="--",
    label="2014 portfolio average",
)
axes[0].set(
    xlabel="Predicted-risk decile (10 = highest)",
    ylabel="Observed charge-off rate (%)",
    title="2014 outcomes by predicted risk",
)
axes[0].legend()

largest = purpose_table.head(8).iloc[::-1]
axes[1].barh(
    largest["purpose"],
    largest["validation_charge_off_rate"] * 100,
)
axes[1].set(
    xlabel="Observed charge-off rate (%)",
    title="2014 outcomes by loan purpose",
)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "consumer_portfolio_analysis.png",
    dpi=160,
)
plt.close(fig)

print(f"\nSaved tables and chart in: {OUTPUT_DIR}")