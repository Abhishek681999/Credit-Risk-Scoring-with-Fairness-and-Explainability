from pathlib import Path

DATA = Path("data")
PROCESSED = DATA / "processed"
OUTPUTS = Path("outputs")
TARGET = "SeriousDlqin2yrs"
RATIOS = ("RevolvingUtilizationOfUnsecuredLines", "DebtRatio")
MODELS = ("xgboost", "lightgbm", "ann")
AGE_GROUPS = ("Under 25", "25–59", "60+")
SEED = 42

# Illustrative policy only: missed default costs 10 units; false alarm costs 1.
FN_COST = 10
FP_COST = 1
POLICY_THRESHOLD = FP_COST / (FP_COST + FN_COST)
