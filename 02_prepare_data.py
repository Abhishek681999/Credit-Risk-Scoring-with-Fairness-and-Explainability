"""80/20 stratified split and train-only cleaning values."""
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline_utils import feature_columns, prepare, remove_kaggle_index, training_settings
from project_config import DATA, PROCESSED, SEED, TARGET

PROCESSED.mkdir(parents=True, exist_ok=True)
raw = remove_kaggle_index(pd.read_csv(DATA / "cs-training.csv"))
print(f"Original rows: {len(raw):,}; exact duplicates retained: {raw.duplicated().sum():,}")
print(f"Invalid ages removed: {(raw['age'] <= 0).sum():,}")
raw = raw.loc[raw["age"] > 0].copy()
train, validation = train_test_split(raw, test_size=.20, random_state=SEED,
                                     stratify=raw[TARGET])
settings = training_settings(train)
(PROCESSED / "preprocessing_settings.json").write_text(json.dumps(settings, indent=2))

for name, frame in (("train", train), ("validation", validation)):
    prepared = prepare(frame, settings)
    if prepared.isna().any().any():
        raise ValueError(f"Missing values remain in {name}")
    prepared.to_csv(PROCESSED / f"{name}.csv", index=False)
    print(f"{name}: {len(prepared):,} rows; default rate {prepared[TARGET].mean():.2%}")

print("Training-derived settings:")
print(json.dumps(settings, indent=2))
print("Age is excluded from model features but retained for the fairness audit.")
