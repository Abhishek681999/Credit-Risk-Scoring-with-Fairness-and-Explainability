"""Shared preprocessing so validation and Kaggle test use identical rules."""
import json

import pandas as pd

from project_config import AGE_GROUPS, PROCESSED, RATIOS, TARGET


def remove_kaggle_index(df):
    return df.drop(columns=[c for c in df if c.startswith("Unnamed:")]).copy()


def training_settings(train):
    return {
        "income_median": float(train["MonthlyIncome"].median()),
        "dependents_mode": float(train["NumberOfDependents"].mode().iloc[0]),
        "upper_caps_99_5_percentile": {
            col: float(train[col].quantile(0.995)) for col in RATIOS
        },
    }


def prepare(df, settings):
    """Transform a split without learning anything from that split."""
    out = remove_kaggle_index(df)
    out = out.drop(columns=[TARGET], errors="ignore") if TARGET in out and out[TARGET].isna().all() else out

    for col in ("MonthlyIncome", "NumberOfDependents"):
        out[f"{col}_missing"] = out[col].isna().astype(int)
    out["MonthlyIncome"] = out["MonthlyIncome"].fillna(settings["income_median"])
    out["NumberOfDependents"] = out["NumberOfDependents"].fillna(settings["dependents_mode"])

    for col, cap in settings["upper_caps_99_5_percentile"].items():
        out[f"{col}_above_cap"] = (out[col] > cap).astype(int)
        out[col] = out[col].clip(upper=cap)

    out["age_group"] = pd.cut(
        out["age"], bins=[0, 25, 60, float("inf")],
        labels=list(AGE_GROUPS), right=False,
    ).astype(str)
    out.loc[out["age"] <= 0, "age_group"] = "Unknown"
    return out


def load_settings():
    return json.loads((PROCESSED / "preprocessing_settings.json").read_text())


def feature_columns(train):
    # No direct use of age; age_group is kept only for the audit.
    return [c for c in train if c not in (TARGET, "age", "age_group")]


def logit_feature(scores):
    import numpy as np
    p = np.clip(scores, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)
