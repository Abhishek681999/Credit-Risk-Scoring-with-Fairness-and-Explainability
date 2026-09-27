from collections import Counter
from pathlib import Path

import pandas as pd

path = Path("data/loan.csv")
columns = ["loan_status", "issue_d", "purpose"]

status_counts = Counter()
year_counts = Counter()
year_status_counts = Counter()
purpose_counts = Counter()
total_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(path, usecols=columns, chunksize=100_000, low_memory=False),
    start=1,
):
    total_rows += len(chunk)

    status_counts.update(chunk["loan_status"].fillna("(missing)").astype(str))
    purpose_counts.update(chunk["purpose"].fillna("(missing)").astype(str))

    dates = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
    years = dates.dt.year.fillna(-1).astype(int)
    statuses = chunk["loan_status"].fillna("(missing)").astype(str)

    year_counts.update(years)
    year_status_counts.update(zip(years, statuses))

    print(f"Processed {total_rows:,} rows...", end="\r", flush=True)

print(f"\n\n=== TOTAL ROWS: {total_rows:,} ===")

print("\n=== ALL LOAN STATUSES ===")
for status, count in status_counts.most_common():
    print(f"{status:30} {count:>10,}  ({count / total_rows:.2%})")

print("\n=== LOANS BY ISSUE YEAR ===")
for year, count in sorted(year_counts.items()):
    label = "Invalid/missing" if year == -1 else str(year)
    print(f"{label:16} {count:>10,}")

print("\n=== OUTCOMES BY ISSUE YEAR ===")
print(f"{'Year':<16} {'Fully Paid':>12} {'Charged Off':>12} {'Current':>12} {'Other':>12}")

for year in sorted(year_counts):
    paid = year_status_counts[year, "Fully Paid"]
    charged = year_status_counts[year, "Charged Off"]
    current = year_status_counts[year, "Current"]
    other = year_counts[year] - paid - charged - current
    label = "Invalid/missing" if year == -1 else str(year)

    print(f"{label:<16} {paid:>12,} {charged:>12,} {current:>12,} {other:>12,}")

print("\n=== TOP LOAN PURPOSES ===")
for purpose, count in purpose_counts.most_common(15):
    print(f"{purpose:25} {count:>10,}")