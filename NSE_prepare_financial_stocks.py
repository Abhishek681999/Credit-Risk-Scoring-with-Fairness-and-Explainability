"""Extract a fixed financial-stock sample and flag price continuity issues."""

import csv
import io
import re
import zipfile
from collections import Counter
from pathlib import Path

SOURCE = Path("data/market/bhavcopy")
OUTPUT = Path("outputs/market")

# Illustrative portfolio universe, not an investment recommendation.
SYMBOLS = ("HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK")

FILE_PATTERN = re.compile(
    r"BhavCopy_NSE_CM_0_0_0_(\d{8})_F_0000\.csv\.zip$"
)

PRICE_FIELDS = [
    "date", "symbol", "close", "nse_previous_close",
    "volume", "daily_price_return",
]
FLAG_FIELDS = [
    "date", "symbol", "issue", "previous_file_close",
    "nse_previous_close", "close", "difference_percent",
]


def positive_float(value, field, filename, symbol):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{filename}: invalid {field} for {symbol}: {value!r}"
        ) from error

    if result <= 0:
        raise ValueError(
            f"{filename}: nonpositive {field} for {symbol}: {result}"
        )
    return result


def main():
    files = sorted(
        path for path in SOURCE.glob("*.zip")
        if FILE_PATTERN.fullmatch(path.name)
    )
    if not files:
        raise SystemExit(f"No bhavcopy ZIPs found in {SOURCE.resolve()}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    prices = []
    flags = []
    previous_file_close = {}
    counts = Counter()

    for index, path in enumerate(files, start=1):
        match = FILE_PATTERN.fullmatch(path.name)
        digits = match.group(1)
        expected_date = f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
        found_today = set()

        with zipfile.ZipFile(path) as archive:
            csv_names = [
                name for name in archive.namelist()
                if name.lower().endswith(".csv")
            ]
            if len(csv_names) != 1:
                raise ValueError(f"Expected one CSV in {path.name}")

            with archive.open(csv_names[0]) as binary:
                text = io.TextIOWrapper(
                    binary, encoding="utf-8-sig", newline=""
                )
                reader = csv.DictReader(text)

                for row in reader:
                    symbol = row["TckrSymb"]
                    if row["SctySrs"] != "EQ" or symbol not in SYMBOLS:
                        continue

                    if row["TradDt"] != expected_date:
                        raise ValueError(
                            f"Date mismatch for {symbol} in {path.name}"
                        )
                    if symbol in found_today:
                        raise ValueError(
                            f"Duplicate EQ row for {symbol} in {path.name}"
                        )
                    found_today.add(symbol)

                    close = positive_float(
                        row["ClsPric"], "close", path.name, symbol
                    )
                    nse_previous = positive_float(
                        row["PrvsClsgPric"],
                        "previous close",
                        path.name,
                        symbol,
                    )
                    volume = positive_float(
                        row["TtlTradgVol"], "volume", path.name, symbol
                    )

                    daily_return = close / nse_previous - 1

                    prices.append({
                        "date": expected_date,
                        "symbol": symbol,
                        "close": close,
                        "nse_previous_close": nse_previous,
                        "volume": volume,
                        "daily_price_return": daily_return,
                    })
                    counts[(symbol, expected_date[:4])] += 1

                    prior = previous_file_close.get(symbol)
                    if prior is not None:
                        gap = nse_previous / prior - 1
                        if abs(gap) > 0.005:
                            flags.append({
                                "date": expected_date,
                                "symbol": symbol,
                                "issue": "previous_close_discontinuity",
                                "previous_file_close": prior,
                                "nse_previous_close": nse_previous,
                                "close": close,
                                "difference_percent": round(100 * gap, 4),
                            })

                    if abs(daily_return) > 0.20:
                        flags.append({
                            "date": expected_date,
                            "symbol": symbol,
                            "issue": "daily_move_over_20_percent",
                            "previous_file_close": (
                                prior if prior is not None else ""
                            ),
                            "nse_previous_close": nse_previous,
                            "close": close,
                            "difference_percent": round(
                                100 * daily_return, 4
                            ),
                        })

                    previous_file_close[symbol] = close

        if index % 100 == 0:
            print(f"Processed {index:,} of {len(files):,} daily files...")

    price_path = OUTPUT / "financial_stocks_daily.csv"
    with price_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRICE_FIELDS)
        writer.writeheader()
        writer.writerows(prices)

    flag_path = OUTPUT / "price_continuity_flags.csv"
    with flag_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FLAG_FIELDS)
        writer.writeheader()
        writer.writerows(flags)

    print("\n=== FINANCIAL STOCK DATA ===")
    print(f"Daily files examined: {len(files):,}")
    print(f"Selected stock-day rows: {len(prices):,}")
    print(f"Price flags requiring review: {len(flags):,}")

    print("\nRows by stock and year:")
    for symbol in SYMBOLS:
        print(
            f"{symbol:12} "
            f"2025: {counts[(symbol, '2025')]:3}  "
            f"2026: {counts[(symbol, '2026')]:3}"
        )

    print("\nFirst 10 price flags:")
    for flag in flags[:10]:
        print(
            flag["date"],
            flag["symbol"],
            flag["issue"],
            f"{flag['difference_percent']:+.2f}%",
        )

    print(f"\nSaved: {price_path}")
    print(f"Saved: {flag_path}")


if __name__ == "__main__":
    main()