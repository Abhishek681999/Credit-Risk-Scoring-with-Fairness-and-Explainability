from pathlib import Path

import pandas as pd


SOURCE = Path("data/loan.csv")
OUTPUT_DIR = Path("data/processed")
TRAIN_PATH = OUTPUT_DIR / "consumer_train_raw.csv"
VALIDATION_PATH = OUTPUT_DIR / "consumer_validation_raw.csv"

# Candidate inputs available from the loan application or prior credit history.
# We will check definitions and missing values before using them in a model.
FEATURES = [
    "loan_amnt",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "collections_12_mths_ex_med",
    "mort_acc",
    "pub_rec_bankruptcies",
]

READ_COLUMNS = ["issue_d", "term", "loan_status"] + FEATURES
FINAL_STATUSES = {"Fully Paid", "Charged Off"}

if not SOURCE.exists():
    raise FileNotFoundError(f"Cannot find {SOURCE.resolve()}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Make repeat runs produce fresh files, rather than appending duplicate rows.
for path in (TRAIN_PATH, VALIDATION_PATH):
    if path.exists():
        path.unlink()

train_rows = 0
validation_rows = 0
train_defaults = 0
validation_defaults = 0
train_missing = pd.Series(0, index=FEATURES, dtype="int64")
validation_missing = pd.Series(0, index=FEATURES, dtype="int64")
rows_read = 0

for chunk in pd.read_csv(
    SOURCE,
    usecols=READ_COLUMNS,
    chunksize=100_000,
    low_memory=False,
):
    rows_read += len(chunk)

    issue_year = pd.to_datetime(
        chunk["issue_d"],
        format="%b-%Y",
        errors="coerce",
    ).dt.year

    term = chunk["term"].astype("string").str.strip()
    status = chunk["loan_status"].astype("string").str.strip()

    # All 36-month loans issued in 2012–2014 appeared resolved
    # in our earlier full-file cohort check. Verify that again here.
    cohort = issue_year.isin([2012, 2013, 2014]) & term.eq("36 months")
    unexpected = cohort & ~status.isin(FINAL_STATUSES)

    if unexpected.any():
        examples = status[unexpected].value_counts(dropna=False).to_dict()
        raise ValueError(
            f"Found {int(unexpected.sum())} unresolved or unexpected "
            f"statuses in the selected cohort: {examples}"
        )

    selected = cohort & status.isin(FINAL_STATUSES)

    for years, output_path, name in [
        ([2012, 2013], TRAIN_PATH, "train"),
        ([2014], VALIDATION_PATH, "validation"),
    ]:
        mask = selected & issue_year.isin(years)

        if not mask.any():
            continue

        result = chunk.loc[mask, FEATURES].copy()
        result.insert(0, "issue_year", issue_year.loc[mask].astype(int))
        result.insert(
            1,
            "target",
            status.loc[mask].eq("Charged Off").astype(int),
        )

        # Append each chunk; write the header only the first time.
        result.to_csv(
            output_path,
            mode="a",
            header=not output_path.exists(),
            index=False,
        )

        if name == "train":
            train_rows += len(result)
            train_defaults += int(result["target"].sum())
            train_missing += result[FEATURES].isna().sum()
        else:
            validation_rows += len(result)
            validation_defaults += int(result["target"].sum())
            validation_missing += result[FEATURES].isna().sum()

    print(f"Read {rows_read:,} source rows...", end="\r", flush=True)

if train_rows == 0 or validation_rows == 0:
    raise RuntimeError("A training or validation file is empty. Check the source data.")

print(f"\n\nTotal source rows read: {rows_read:,}")
print("\n=== SELECTED COHORT ===")
print("36-month loans with final status: Fully Paid or Charged Off")
print("Target: 1 = Charged Off; 0 = Fully Paid")

print(
    f"\nTrain (2012–2013): {train_rows:,} loans; "
    f"{train_defaults:,} charged off ({train_defaults / train_rows:.2%})"
)
print(
    f"Validation (2014): {validation_rows:,} loans; "
    f"{validation_defaults:,} charged off "
    f"({validation_defaults / validation_rows:.2%})"
)

print("\n=== TRAINING FEATURE MISSINGNESS (TOP 10) ===")
print(
    (100 * train_missing / train_rows)
    .sort_values(ascending=False)
    .head(10)
    .round(2)
    .to_string()
)

print("\n=== VALIDATION FEATURE MISSINGNESS (TOP 10) ===")
print(
    (100 * validation_missing / validation_rows)
    .sort_values(ascending=False)
    .head(10)
    .round(2)
    .to_string()
)

print(f"\nSaved: {TRAIN_PATH}")
print(f"Saved: {VALIDATION_PATH}")