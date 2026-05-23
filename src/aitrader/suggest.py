from __future__ import annotations

import argparse

from sklearn.ensemble import RandomForestClassifier

from .backtest import risk_plan
from .features import FEATURE_COLUMNS, add_features, load_candles_many
from .train import train_model


def latest_suggestion(data_path: str, config_path: str) -> None:
    _, labelled, config = train_model(data_path, config_path)

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=10,
        random_state=int(config["model"]["random_state"]),
        class_weight="balanced_subsample",
    )
    model.fit(labelled[FEATURE_COLUMNS], labelled["label"])

    threshold = float(config["strategy"]["buy_probability_threshold"])
    latest_by_symbol = add_features(load_candles_many(data_path)).groupby("symbol", sort=True).tail(1).copy()
    latest_by_symbol["buy_probability"] = model.predict_proba(latest_by_symbol[FEATURE_COLUMNS])[:, 1]
    latest_by_symbol = latest_by_symbol.sort_values("buy_probability", ascending=False)

    print("\nLatest intraday paper suggestions")
    for _, row in latest_by_symbol.iterrows():
        probability = float(row["buy_probability"])
        decision = "BUY" if probability >= threshold else "AVOID"
        plan = risk_plan(row, config)
        print("")
        print(f"symbol: {row['symbol']}")
        print(f"timestamp: {row['timestamp']}")
        print(f"decision: {decision}")
        print(f"buy_probability: {probability:.3f}")
        for key, value in plan.items():
            print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    latest_suggestion(args.data, args.config)


if __name__ == "__main__":
    main()
