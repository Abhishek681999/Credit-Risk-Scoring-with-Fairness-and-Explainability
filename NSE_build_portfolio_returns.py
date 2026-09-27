"""Correct documented corporate actions and build illustrative bank returns."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = Path("outputs/market/financial_stocks_daily.csv")
OUTPUT = Path("outputs/market")

SYMBOLS = ("HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK")
WEIGHT = 1 / len(SYMBOLS)

# Number of shares held after the action for each one held before it.
# HDFCBANK: 1:1 bonus. KOTAKBANK: Rs 5 share split into five Re 1 shares.
SHARE_MULTIPLIERS = {
    ("2025-08-26", "HDFCBANK"): 2,
    ("2026-01-14", "KOTAKBANK"): 5,
}


def period_for(day):
    if day < "2026-01-01":
        return "development_2025"
    if day < "2026-07-01":
        return "validation_2026_jan_jun"
    return "final_test_2026_jul_sep"


def main():
    if not SOURCE.exists():
        raise SystemExit(f"Missing {SOURCE}. Run NSE_prepare_financial_stocks.py.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    by_date = defaultdict(dict)
    adjustments = []
    observed_actions = set()

    with SOURCE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            day = row["date"]
            symbol = row["symbol"]
            key = (day, symbol)

            if symbol not in SYMBOLS:
                raise ValueError(f"Unexpected symbol: {symbol}")
            if symbol in by_date[day]:
                raise ValueError(f"Duplicate row: {day} {symbol}")

            close = float(row["close"])
            previous = float(row["nse_previous_close"])
            raw_return = float(row["daily_price_return"])
            multiplier = SHARE_MULTIPLIERS.get(key, 1)

            if close <= 0 or previous <= 0:
                raise ValueError(f"Invalid price: {day} {symbol}")

            # Check the extraction script's return against the raw prices.
            calculated_raw = close / previous - 1
            if abs(calculated_raw - raw_return) > 1e-9:
                raise ValueError(f"Raw return mismatch: {day} {symbol}")

            # On a 1:1 bonus, two new shares replace one old share.
            # On a 5-for-1 split, five new shares replace one old share.
            adjusted_return = multiplier * close / previous - 1

            if multiplier != 1:
                observed_actions.add(key)

                # The raw drop should be near the mechanical price change.
                expected_raw = 1 / multiplier - 1
                if abs(raw_return - expected_raw) > 0.10:
                    raise ValueError(
                        f"Corporate action price needs manual review: "
                        f"{day} {symbol}, raw return {raw_return:.2%}"
                    )

                adjustments.append({
                    "date": day,
                    "symbol": symbol,
                    "share_multiplier": multiplier,
                    "raw_return": raw_return,
                    "adjusted_return": adjusted_return,
                })

            if abs(adjusted_return) > 0.20:
                raise ValueError(
                    f"Unreviewed return over 20%: "
                    f"{day} {symbol} {adjusted_return:.2%}"
                )

            by_date[day][symbol] = {
                "close": close,
                "nse_previous_close": previous,
                "share_multiplier": multiplier,
                "raw_return": raw_return,
                "adjusted_return": adjusted_return,
            }

    missing_actions = set(SHARE_MULTIPLIERS) - observed_actions
    if missing_actions:
        raise ValueError(
            f"Expected corporate action dates missing: {sorted(missing_actions)}"
        )

    stock_rows = []
    portfolio_rows = []

    for day in sorted(by_date):
        missing_symbols = set(SYMBOLS) - set(by_date[day])
        if missing_symbols:
            raise ValueError(
                f"{day}: missing stocks {sorted(missing_symbols)}"
            )

        period = period_for(day)
        portfolio_return = 0.0

        for symbol in SYMBOLS:
            item = by_date[day][symbol]
            portfolio_return += WEIGHT * item["adjusted_return"]

            stock_rows.append({
                "date": day,
                "period": period,
                "symbol": symbol,
                "weight": WEIGHT,
                **item,
            })

        portfolio_rows.append({
            "date": day,
            "period": period,
            "portfolio_return": portfolio_return,
            "portfolio_loss": -portfolio_return,
        })

    stock_path = OUTPUT / "adjusted_stock_returns.csv"
    with stock_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(stock_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(stock_rows)

    portfolio_path = OUTPUT / "bank_portfolio_daily_returns.csv"
    with portfolio_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(portfolio_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(portfolio_rows)

    adjustment_path = OUTPUT / "corporate_action_adjustments.csv"
    with adjustment_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(adjustments[0].keys())
        )
        writer.writeheader()
        writer.writerows(adjustments)

    counts = Counter(row["period"] for row in portfolio_rows)

    print("\n=== BANK PORTFOLIO RETURNS ===")
    print(f"Trading days: {len(portfolio_rows)}")
    print(f"Stock-day rows: {len(stock_rows)}")
    print("Portfolio: five bank stocks, 20% each, rebalanced daily")

    print("\nDays by period:")
    for period, count in counts.items():
        print(f"  {period}: {count}")

    print("\nCorporate action corrections:")
    for item in adjustments:
        print(
            f"  {item['date']} {item['symbol']}: "
            f"raw {item['raw_return']:.2%} -> "
            f"corrected {item['adjusted_return']:.2%}"
        )

    worst = min(portfolio_rows, key=lambda row: row["portfolio_return"])
    best = max(portfolio_rows, key=lambda row: row["portfolio_return"])
    print(
        f"\nWorst portfolio day: {worst['date']} "
        f"{worst['portfolio_return']:.2%}"
    )
    print(
        f"Best portfolio day:  {best['date']} "
        f"{best['portfolio_return']:.2%}"
    )

    print(f"\nSaved: {stock_path}")
    print(f"Saved: {portfolio_path}")
    print(f"Saved: {adjustment_path}")


if __name__ == "__main__":
    main()