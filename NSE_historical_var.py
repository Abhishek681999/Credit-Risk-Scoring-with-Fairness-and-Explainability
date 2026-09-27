"""Backtest one-day historical 95% VaR without using future returns."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE = Path("outputs/market/bank_portfolio_daily_returns.csv")
OUTPUT = Path("outputs/market")

CONFIDENCE = 0.95
WINDOW = 250
MIN_HISTORY = 200

PERIODS = (
    "validation_2026_jan_jun",
    "final_test_2026_jul_sep",
)


def main():
    if not SOURCE.exists():
        raise SystemExit(
            f"Missing {SOURCE}. Run NSE_build_portfolio_returns.py first."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE, parse_dates=["date"])
    data = data.sort_values("date").reset_index(drop=True)

    if data["date"].duplicated().any():
        raise ValueError("Duplicate portfolio dates")
    if not np.allclose(
        data["portfolio_loss"],
        -data["portfolio_return"],
        atol=1e-12,
    ):
        raise ValueError("Portfolio loss must equal negative return")

    results = []

    for index, current in data.iterrows():
        if current["period"] not in PERIODS:
            continue

        # Exclude today's loss. Nothing from today or later enters the forecast.
        earlier_losses = data.iloc[
            max(0, index - WINDOW):index
        ]["portfolio_loss"].to_numpy(dtype=float)

        if len(earlier_losses) < MIN_HISTORY:
            raise ValueError(
                f"Too little prior data for {current['date'].date()}: "
                f"{len(earlier_losses)} days"
            )

        var = float(np.quantile(earlier_losses, CONFIDENCE))
        tail_losses = earlier_losses[earlier_losses >= var]
        expected_shortfall = float(tail_losses.mean())
        actual_loss = float(current["portfolio_loss"])

        results.append({
            "date": current["date"],
            "period": current["period"],
            "history_days": len(earlier_losses),
            "portfolio_return": float(current["portfolio_return"]),
            "actual_loss": actual_loss,
            "var_95": var,
            "historical_tail_mean": expected_shortfall,
            "exceedance": int(actual_loss > var),
        })

    forecast = pd.DataFrame(results)
    destination = OUTPUT / "historical_var_backtest.csv"
    forecast.to_csv(destination, index=False)

    print("\n=== HISTORICAL 95% VaR BACKTEST ===")
    print(f"History window: up to {WINDOW} earlier trading days")
    print("Today's return is never used to estimate today's risk.")

    for period in PERIODS:
        part = forecast.loc[forecast["period"] == period]
        days = len(part)
        breaches = int(part["exceedance"].sum())

        print(f"\n{period}")
        print(f"  Days: {days}")
        print(f"  Expected exceedances at 5%: {days * 0.05:.1f}")
        print(f"  Actual exceedances: {breaches}")
        print(f"  Exceedance rate: {100 * breaches / days:.2f}%")
        print(f"  Mean estimated VaR: {100 * part['var_95'].mean():.2f}%")
        print(
            f"  Worst observed daily loss: "
            f"{100 * part['actual_loss'].max():.2f}%"
        )

    largest = forecast.nlargest(5, "actual_loss")[
        ["date", "period", "actual_loss", "var_95", "exceedance"]
    ].copy()
    largest["date"] = largest["date"].dt.strftime("%Y-%m-%d")
    largest[["actual_loss", "var_95"]] *= 100

    print("\nFive largest losses across validation and final test:")
    print(largest.round(2).to_string(index=False))
    print("Loss and VaR values in this table are percentages.")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        forecast["date"],
        100 * forecast["actual_loss"],
        label="Actual daily loss",
        color="#546e7a",
        linewidth=1,
    )
    ax.plot(
        forecast["date"],
        100 * forecast["var_95"],
        label="Prior-day historical 95% VaR",
        color="#1565c0",
        linewidth=1.5,
    )

    breaches = forecast.loc[forecast["exceedance"] == 1]
    ax.scatter(
        breaches["date"],
        100 * breaches["actual_loss"],
        color="#c62828",
        s=32,
        label="VaR exceedance",
        zorder=3,
    )

    ax.axvline(
        pd.Timestamp("2026-07-01"),
        color="black",
        linestyle="--",
        linewidth=1,
        label="Final test begins",
    )
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set(
        title="Five-bank portfolio: historical 95% VaR backtest",
        ylabel="One-day loss (%); gains appear below zero",
        xlabel="Date",
    )
    ax.legend()
    fig.tight_layout()

    chart = OUTPUT / "historical_var_backtest.png"
    fig.savefig(chart, dpi=160)
    plt.close(fig)

    print(f"\nSaved: {destination}")
    print(f"Saved: {chart}")


if __name__ == "__main__":
    main()