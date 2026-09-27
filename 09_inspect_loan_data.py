from pathlib import Path
import pandas as pd

path = Path("data/loan.csv")

if not path.exists():
    raise FileNotFoundError(f"File not found: {path.resolve()}")

# Read only the first 20,000 rows for a quick inspection.
df = pd.read_csv(path, nrows=20_000, low_memory=False)

print(f"Rows inspected: {len(df):,}")
print(f"Number of columns: {len(df.columns)}")

print("\n=== ALL COLUMN NAMES ===")
for name in df.columns:
    print(name)

for column in ["loan_status", "purpose", "home_ownership", "issue_d"]:
    if column in df.columns:
        print(f"\n=== {column}: TOP VALUES ===")
        print(df[column].value_counts(dropna=False).head(15).to_string())

print("\n=== MOST MISSING COLUMNS IN THIS SAMPLE ===")
print(
    df.isna()
      .mean()
      .mul(100)
      .sort_values(ascending=False)
      .head(15)
      .round(1)
      .to_string()
)