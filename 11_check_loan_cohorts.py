from collections import Counter
from pathlib import Path

import pandas as pd

path = Path("data/loan.csv")

counts = Counter()
total = 0

for chunk in pd.read_csv(
    path,
    usecols=["issue_d", "term", "loan_status"],
    chunksize=100_000,
    low_memory=False,
):
    dates = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
    years = dates.dt.year.fillna(-1).astype(int)

    terms = (
        chunk["term"]
        .fillna("Missing")
        .astype(str)
        .str.strip()
    )

    statuses = chunk["loan_status"].fillna("Missing").astype(str)

    counts.update(zip(years, terms, statuses))
    total += len(chunk)
    print(f"Processed {total:,} rows...", end="\r", flush=True)

print(f"\n\nTotal rows: {total:,}")
print("\n=== STATUS BY ISSUE YEAR AND LOAN TERM ===")
print(
    f"{'Year':<7} {'Term':<13} {'Fully Paid':>12} "
    f"{'Charged Off':>12} {'Current':>12} {'Other':>10} {'Total':>12}"
)

years_and_terms = sorted({(year, term) for year, term, _ in counts})

for year, term in years_and_terms:
    paid = counts[year, term, "Fully Paid"]
    charged = counts[year, term, "Charged Off"]
    current = counts[year, term, "Current"]

    cohort_total = sum(
        count
        for (row_year, row_term, _), count in counts.items()
        if row_year == year and row_term == term
    )
    other = cohort_total - paid - charged - current

    print(
        f"{year:<7} {term:<13} {paid:>12,} "
        f"{charged:>12,} {current:>12,} "
        f"{other:>10,} {cohort_total:>12,}"
    )