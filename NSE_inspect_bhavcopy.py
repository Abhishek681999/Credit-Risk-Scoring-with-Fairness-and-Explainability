"""Inspect downloaded NSE daily bhavcopy ZIPs before building risk models."""

import csv
import io
import re
import zipfile
from collections import Counter
from pathlib import Path

SOURCE = Path("data/market/bhavcopy")
OUTPUT = Path("outputs/market")
PATTERN = re.compile(r"BhavCopy_NSE_CM_0_0_0_(\d{8})_F_0000\.csv\.zip$")

REQUIRED = {
    "TradDt",
    "TckrSymb",
    "SctySrs",
    "ClsPric",
    "PrvsClsgPric",
    "TtlTradgVol",
}


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def inspect_file(path):
    match = PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected filename: {path.name}")

    file_date = match.group(1)
    result = {
        "date": f"{file_date[:4]}-{file_date[4:6]}-{file_date[6:]}",
        "filename": path.name,
        "all_rows": 0,
        "equity_rows": 0,
        "bad_close": 0,
        "missing_previous_close": 0,
        "zero_or_missing_volume": 0,
        "duplicate_equity_symbols": 0,
        "date_mismatch_rows": 0,
    }
    seen_symbols = set()

    with zipfile.ZipFile(path) as archive:
        csv_files = [name for name in archive.namelist()
                     if name.lower().endswith(".csv")]
        if len(csv_files) != 1:
            raise ValueError(f"Expected one CSV inside {path.name}")

        with archive.open(csv_files[0]) as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text)

            missing = REQUIRED - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path.name} lacks columns: {sorted(missing)}")

            for row in reader:
                result["all_rows"] += 1

                if row["TradDt"] != result["date"]:
                    result["date_mismatch_rows"] += 1

                # EQ is the ordinary equity series we will examine first.
                if row["SctySrs"] != "EQ":
                    continue

                result["equity_rows"] += 1
                symbol = row["TckrSymb"]

                if symbol in seen_symbols:
                    result["duplicate_equity_symbols"] += 1
                seen_symbols.add(symbol)

                close = number(row["ClsPric"])
                previous = number(row["PrvsClsgPric"])
                volume = number(row["TtlTradgVol"])

                if close is None or close <= 0:
                    result["bad_close"] += 1
                if previous is None or previous <= 0:
                    result["missing_previous_close"] += 1
                if volume is None or volume <= 0:
                    result["zero_or_missing_volume"] += 1

    return result


def main():
    files = sorted(path for path in SOURCE.glob("*.zip")
                   if PATTERN.fullmatch(path.name))
    if not files:
        raise SystemExit(
            f"No NSE ZIP files found in {SOURCE.resolve()}\n"
            "Check that the downloader has saved files there."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    problems = []

    for index, path in enumerate(files, start=1):
        try:
            results.append(inspect_file(path))
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            problems.append((path.name, str(error)))

        if index % 50 == 0:
            print(f"Inspected {index:,} of {len(files):,} ZIPs...")

    if results:
        destination = OUTPUT / "nse_file_audit.csv"
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        years = Counter(row["date"][:4] for row in results)

        print("\n=== NSE DOWNLOAD AUDIT ===")
        print(f"Valid daily files: {len(results):,}")
        print(f"Date range: {results[0]['date']} to {results[-1]['date']}")
        print(f"Trading days by year: {dict(sorted(years.items()))}")
        print(f"Total security-day rows: "
              f"{sum(row['all_rows'] for row in results):,}")
        print(f"Ordinary equity-day rows (EQ): "
              f"{sum(row['equity_rows'] for row in results):,}")

        for field in (
            "bad_close",
            "missing_previous_close",
            "zero_or_missing_volume",
            "duplicate_equity_symbols",
            "date_mismatch_rows",
        ):
            print(f"{field}: {sum(row[field] for row in results):,}")

        print(f"Daily audit saved: {destination}")

    print(f"Unreadable or invalid files: {len(problems)}")
    for filename, error in problems[:10]:
        print(f"  {filename}: {error}")


if __name__ == "__main__":
    main()