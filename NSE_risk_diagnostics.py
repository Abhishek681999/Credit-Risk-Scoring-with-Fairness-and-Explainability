"""Explain when portfolio risk limits were crossed and what drove losses."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("outputs/market")
COMPARISON_FILE = ROOT / "var_model_daily_comparison.csv"
STOCK_FILE = ROOT / "adjusted_stock_returns.csv"


def longest_streak(values):
    """Consecutive rows represent consecutive observed trading sessions."""
    current = 0
    longest = 0

    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def main():
    for path in (COMPARISON_FILE, STOCK_FILE):
        if not path.exists():
            raise SystemExit(f"Missing {path}")

    comparison = pd.read_csv(
        COMPARISON_FILE, parse_dates=["date"]
    ).sort_values("date").reset_index(drop=True)

    stocks = pd.read_csv(STOCK_FILE, parse_dates=["date"])

    comparison["ewma_breach"] = (
        comparison["actual_loss"] > comparison["ewma_var_95"]
    ).astype(int)

    comparison["historical_breach"] = (
        comparison["actual_loss"] > comparison["historical_var_95"]
    ).astype(int)

    comparison["month"] = comparison["date"].dt.to_period("M").astype(str)

    monthly = (
        comparison.groupby(["period", "month"], sort=True)
        .agg(
            trading_days=("date", "size"),
            ewma_breaches=("ewma_breach", "sum"),
            historical_breaches=("historical_breach", "sum"),
            average_ewma_limit=("ewma_var_95", "mean"),
            average_historical_limit=("historical_var_95", "mean"),
        )
        .reset_index()
    )

    monthly["expected_breaches"] = monthly["trading_days"] * 0.05

    monthly["ewma_breach_rate_percent"] = (
        100 * monthly["ewma_breaches"] / monthly["trading_days"]
    )

    monthly["historical_breach_rate_percent"] = (
        100 * monthly["historical_breaches"] / monthly["trading_days"]
    )

    # Sort by calendar month rather than by the period label.
    monthly = monthly.sort_values("month").reset_index(drop=True)

    monthly_path = ROOT / "monthly_risk_monitor.csv"
    monthly.to_csv(monthly_path, index=False)

    stocks_by_date = {
        day: group for day, group in stocks.groupby("date")
    }
    breaches = []

    for row in comparison.itertuples(index=False):
        if not (row.ewma_breach or row.historical_breach):
            continue

        day_stocks = stocks_by_date.get(row.date)

        if day_stocks is None or len(day_stocks) != 5:
            raise ValueError(
                f"Expected five bank returns for {row.date.date()}"
            )

        day_stocks = day_stocks.copy()
        day_stocks["loss_contribution"] = (
            -day_stocks["weight"]
            * day_stocks["adjusted_return"]
        )

        if not np.isclose(
            day_stocks["loss_contribution"].sum(),
            row.actual_loss,
            atol=1e-10,
        ):
            raise ValueError(
                f"Bank contributions do not match portfolio loss "
                f"on {row.date.date()}"
            )

        largest = day_stocks.loc[
            day_stocks["loss_contribution"].idxmax()
        ]

        breaches.append({
            "date": row.date.strftime("%Y-%m-%d"),
            "period": row.period,
            "actual_loss_percent": 100 * row.actual_loss,
            "ewma_limit_percent": 100 * row.ewma_var_95,
            "historical_limit_percent": (
                100 * row.historical_var_95
            ),
            "ewma_breach": row.ewma_breach,
            "historical_breach": row.historical_breach,
            "largest_loss_contributor": largest["symbol"],
            "contribution_percentage_points": (
                100 * largest["loss_contribution"]
            ),
        })

    breach_log = pd.DataFrame(breaches)
    breach_path = ROOT / "risk_breach_log.csv"
    breach_log.to_csv(breach_path, index=False)

    print("\n=== MONTHLY RISK MONITOR ===")
    print(
        monthly[
            [
                "month",
                "trading_days",
                "expected_breaches",
                "historical_breaches",
                "ewma_breaches",
            ]
        ].round(2).to_string(index=False)
    )

    print("\n=== BREACH PATTERNS ===")
    for model in ("historical", "ewma"):
        column = f"{model}_breach"
        print(
            f"{model}: {int(comparison[column].sum())} breaches; "
            f"longest consecutive-trading-day run: "
            f"{longest_streak(comparison[column])}"
        )

    print("\n=== LARGEST LOSS DAYS THAT BREACHED A LIMIT ===")
    view = breach_log.nlargest(
        10, "actual_loss_percent"
    ).copy()
    print(view.round(2).to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(monthly))
    width = 0.28

    ax.bar(
        x - width,
        monthly["historical_breaches"],
        width,
        label="Historical crossings",
    )
    ax.bar(
        x,
        monthly["ewma_breaches"],
        width,
        label="EWMA crossings",
    )
    ax.bar(
        x + width,
        monthly["expected_breaches"],
        width,
        label="About 5% expected",
        color="#9e9e9e",
    )

    ax.set_xticks(x, monthly["month"], rotation=45)
    ax.set(
        title="Days each risk limit was crossed, by month",
        xlabel="Month",
        ylabel="Number of trading days",
    )
    ax.legend()
    fig.tight_layout()

    chart_path = ROOT / "monthly_risk_monitor.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)

    print(f"\nSaved: {monthly_path}")
    print(f"Saved: {breach_path}")
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()