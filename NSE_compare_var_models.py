"""Compare rolling historical VaR with EWMA-normal VaR on the same dates."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("outputs/market")
PORTFOLIO_FILE = ROOT / "bank_portfolio_daily_returns.csv"
HISTORICAL_FILE = ROOT / "historical_var_backtest.csv"

CONFIDENCE = 0.95
NORMAL_95_QUANTILE = 1.6448536269514722
DECAY_CANDIDATES = (0.90, 0.94, 0.97, 0.99)
INITIAL_DAYS = 20

VALIDATION = "validation_2026_jan_jun"
FINAL_TEST = "final_test_2026_jul_sep"


def quantile_loss(actual_loss, var_limit):
    """Lower is better; penalizes breaches more than overestimation."""
    actual_loss = np.asarray(actual_loss, dtype=float)
    var_limit = np.asarray(var_limit, dtype=float)

    return np.where(
        actual_loss >= var_limit,
        CONFIDENCE * (actual_loss - var_limit),
        (1 - CONFIDENCE) * (var_limit - actual_loss),
    )


def ewma_forecasts(data, decay):
    returns = data["portfolio_return"].to_numpy(dtype=float)

    # Initialize using the first 20 days of 2025. Forecasts begin much later.
    variance = float(np.mean(returns[:INITIAL_DAYS] ** 2))
    rows = []

    for index in range(INITIAL_DAYS, len(data)):
        if index > INITIAL_DAYS:
            # Yesterday's return updates today's estimate.
            variance = (
                decay * variance
                + (1 - decay) * returns[index - 1] ** 2
            )

        current = data.iloc[index]
        if current["period"] not in (VALIDATION, FINAL_TEST):
            continue

        var_limit = NORMAL_95_QUANTILE * np.sqrt(variance)
        actual_loss = float(current["portfolio_loss"])

        rows.append({
            "date": current["date"],
            "period": current["period"],
            "actual_loss": actual_loss,
            "ewma_var_95": float(var_limit),
            "ewma_exceedance": int(actual_loss > var_limit),
        })

    return pd.DataFrame(rows)


def summary_row(model, period, actual, limits):
    breaches = int(np.sum(actual > limits))
    return {
        "model": model,
        "period": period,
        "days": len(actual),
        "expected_exceedances": round(0.05 * len(actual), 2),
        "actual_exceedances": breaches,
        "exceedance_rate_percent": round(
            100 * breaches / len(actual), 2
        ),
        "mean_var_percent": round(100 * np.mean(limits), 3),
        "mean_quantile_loss": round(
            float(np.mean(quantile_loss(actual, limits))), 6
        ),
    }


def main():
    for path in (PORTFOLIO_FILE, HISTORICAL_FILE):
        if not path.exists():
            raise SystemExit(f"Missing {path}")

    data = pd.read_csv(PORTFOLIO_FILE, parse_dates=["date"])
    data = data.sort_values("date").reset_index(drop=True)

    if len(data) < INITIAL_DAYS + 1:
        raise ValueError("Not enough returns to initialize EWMA")
    if data["date"].duplicated().any():
        raise ValueError("Duplicate portfolio dates")

    candidates = {}

    for decay in DECAY_CANDIDATES:
        forecast = ewma_forecasts(data, decay)
        validation = forecast.loc[forecast["period"] == VALIDATION]

        score = float(np.mean(quantile_loss(
            validation["actual_loss"],
            validation["ewma_var_95"],
        )))
        candidates[decay] = (score, forecast)

    # Select using validation only. Final-test outcomes are not used here.
    chosen_decay = min(
        DECAY_CANDIDATES,
        key=lambda decay: candidates[decay][0],
    )
    selected = candidates[chosen_decay][1]

    historical = pd.read_csv(HISTORICAL_FILE, parse_dates=["date"])
    historical = historical[
        ["date", "period", "actual_loss", "var_95"]
    ].rename(columns={"var_95": "historical_var_95"})

    comparison = selected.merge(
        historical,
        on=["date", "period"],
        how="inner",
        suffixes=("", "_historical"),
        validate="one_to_one",
    )

    if len(comparison) != len(selected):
        raise ValueError("EWMA and historical dates do not align")
    if not np.allclose(
        comparison["actual_loss"],
        comparison["actual_loss_historical"],
    ):
        raise ValueError("Actual losses differ between model files")

    comparison = comparison.drop(columns=["actual_loss_historical"])
    comparison = comparison.sort_values("date").reset_index(drop=True)
    comparison["historical_exceedance"] = (
        comparison["actual_loss"] > comparison["historical_var_95"]
    ).astype(int)

    forecast_path = ROOT / "var_model_daily_comparison.csv"
    comparison.to_csv(forecast_path, index=False)

    rows = []
    for period in (VALIDATION, FINAL_TEST):
        part = comparison.loc[comparison["period"] == period]
        actual = part["actual_loss"].to_numpy()

        for model, column in (
            ("historical_250", "historical_var_95"),
            (f"ewma_normal_{chosen_decay:.2f}", "ewma_var_95"),
        ):
            rows.append(summary_row(
                model,
                period,
                actual,
                part[column].to_numpy(),
            ))

    summary = pd.DataFrame(rows)
    summary_path = ROOT / "var_model_comparison.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== EWMA VALIDATION SETTINGS ===")
    for decay in DECAY_CANDIDATES:
        print(
            f"Decay {decay:.2f}: "
            f"validation quantile loss {candidates[decay][0]:.6f}"
        )
    print(f"Chosen decay using validation only: {chosen_decay:.2f}")

    print("\n=== SAME-DAY VaR COMPARISON ===")
    print(summary.to_string(index=False))
    print(
        "\nLower quantile loss is better. An exceedance is a daily loss "
        "greater than that day's forecast VaR."
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        comparison["date"],
        100 * comparison["actual_loss"],
        color="#607d8b",
        linewidth=0.9,
        label="Actual daily loss",
    )
    ax.plot(
        comparison["date"],
        100 * comparison["historical_var_95"],
        linewidth=1.4,
        label="Historical 95% VaR",
    )
    ax.plot(
        comparison["date"],
        100 * comparison["ewma_var_95"],
        linewidth=1.4,
        label=f"EWMA 95% VaR (decay {chosen_decay:.2f})",
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
        title="Daily bank-portfolio losses versus two prior-day risk limits",
        xlabel="Date",
        ylabel="Daily loss or VaR (%)",
    )
    ax.legend()
    fig.tight_layout()

    chart_path = ROOT / "var_model_comparison.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)

    print(f"\nSaved: {forecast_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()