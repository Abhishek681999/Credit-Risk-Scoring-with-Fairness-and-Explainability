"""Inspect the raw labeled data; no model preprocessing occurs here."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pipeline_utils import remove_kaggle_index
from project_config import AGE_GROUPS, DATA, OUTPUTS, RATIOS, TARGET

out = OUTPUTS / "eda"
out.mkdir(parents=True, exist_ok=True)
df = remove_kaggle_index(pd.read_csv(DATA / "cs-training.csv"))

print(f"Rows: {len(df):,}; predictors: {len(df.columns) - 1}")
print("\nTarget (count and proportion):")
print(pd.DataFrame({"count": df[TARGET].value_counts().sort_index(),
                    "proportion": df[TARGET].value_counts(normalize=True).sort_index()}))
print("\nMissing values:")
print(pd.DataFrame({"count": df.isna().sum(), "percent": df.isna().mean() * 100}).query("count > 0").round(2))
print(f"\nExact duplicate rows: {df.duplicated().sum():,}")
print(f"Invalid age (<=0): {(df['age'] <= 0).sum():,}")
for col in RATIOS:
    print(f"\n{col} quantiles:\n{df[col].quantile([0, .5, .95, .99, .995, 1])}")

age = df.loc[df["age"] > 0].copy()
age["age_group"] = pd.cut(age["age"], bins=[0, 25, 60, float("inf")],
                          labels=list(AGE_GROUPS), right=False)
age_table = age.groupby("age_group", observed=True)[TARGET].agg(
    applicants="size", defaults="sum", default_rate="mean")
print(f"\nAge-group observed defaults:\n{age_table.round(4)}")
age_table.to_csv(out / "age_group_observed.csv")

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(6, 4))
df[TARGET].value_counts().sort_index().plot.bar(ax=ax)
ax.set_xticklabels(["No default", "Default"], rotation=0)
ax.set_title("Two-year default outcome")
fig.tight_layout(); fig.savefig(out / "class_balance.png", dpi=160); plt.close(fig)

cols = ["age", "MonthlyIncome", *RATIOS]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.flat, cols):
    upper = df[col].quantile(.99)
    for label in (0, 1):
        values = df.loc[(df[TARGET] == label) & df[col].between(0, upper), col].dropna()
        ax.hist(values, bins=40, density=True, alpha=.45, label=f"default={label}")
    ax.set_title(f"{col} (display up to 99th percentile)"); ax.legend()
fig.tight_layout(); fig.savefig(out / "feature_distributions.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(df.select_dtypes(include="number").corr(), cmap="coolwarm", center=0, ax=ax)
fig.tight_layout(); fig.savefig(out / "correlations.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
age_table["default_rate"].mul(100).plot.bar(ax=ax)
ax.set_ylabel("Observed default rate (%)"); ax.tick_params(axis="x", rotation=0)
fig.tight_layout(); fig.savefig(out / "observed_default_by_age.png", dpi=160); plt.close(fig)
print(f"\nCharts saved in {out}")
