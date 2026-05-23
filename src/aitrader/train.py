from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .backtest import summarize_predictions, threshold_trades
from .features import FEATURE_COLUMNS, add_features, add_labels, load_candles_many


def train_model(data_path: str, config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw = load_candles_many(data_path)
    featured = add_features(raw)
    labelled = add_labels(
        featured,
        holding_bars=int(config["strategy"]["holding_bars"]),
        threshold_bps=float(config["strategy"]["min_expected_return_bps"]),
    )

    min_rows = 200
    if len(labelled) < min_rows:
        raise ValueError(
            f"Need at least {min_rows} labelled candles after indicators; got {len(labelled)}. "
            "Use multiple days of 5-minute or 15-minute candles for each symbol."
        )

    split_idx = int(len(labelled) * (1 - float(config["model"]["test_size_pct"]) / 100))
    train_df = labelled.iloc[:split_idx]
    test_df = labelled.iloc[split_idx:]
    if train_df["label"].nunique() < 2:
        raise ValueError("Training split has only one class. Add more history or lower the return threshold.")
    if test_df.empty:
        raise ValueError("Test split is empty. Add more history or reduce model.test_size_pct.")

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=10,
        random_state=int(config["model"]["random_state"]),
        class_weight="balanced_subsample",
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"])

    test_df = test_df.copy()
    test_df["buy_probability"] = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    test_df["prediction"] = (test_df["buy_probability"] >= 0.5).astype(int)

    print("\nClassification report")
    print(classification_report(test_df["label"], test_df["prediction"], zero_division=0))
    print("Confusion matrix")
    print(confusion_matrix(test_df["label"], test_df["prediction"]))

    if test_df["label"].nunique() > 1:
        auc = roc_auc_score(test_df["label"], test_df["buy_probability"])
        print(f"ROC AUC: {auc:.3f}")

    summary = summarize_predictions(
        test_df,
        probability_column="buy_probability",
        threshold=float(config["strategy"]["buy_probability_threshold"]),
        cost_bps=float(config["strategy"]["estimated_round_trip_cost_bps"]),
    )
    print("\nBacktest summary for thresholded BUY signals")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print_backtest_interpretation(summary)

    report_path = save_trade_report(test_df, config)
    if report_path:
        print(f"\nSaved thresholded trade log: {report_path}")

    return model, labelled, config


def save_trade_report(test_df, config: dict) -> Path | None:
    reports_dir = config.get("model", {}).get("reports_dir")
    if not reports_dir:
        return None

    trades = threshold_trades(
        test_df,
        probability_column="buy_probability",
        threshold=float(config["strategy"]["buy_probability_threshold"]),
        cost_bps=float(config["strategy"]["estimated_round_trip_cost_bps"]),
    )
    if trades.empty:
        return None

    path = Path(reports_dir) / "threshold_trades.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "timestamp",
        "symbol",
        "close",
        "buy_probability",
        "future_return",
        "gross_return_pct",
        "net_return_pct",
        "win",
        "equity_curve_pct",
        "drawdown_pct",
    ]
    trades[columns].to_csv(path, index=False)
    return path


def print_backtest_interpretation(summary: dict[str, float]) -> None:
    if summary["trades"] == 0:
        print("interpretation: no trades passed the probability threshold.")
    elif summary["profit_factor"] < 1 or summary["avg_return_pct"] <= 0:
        print("interpretation: research only; thresholded signals are not profitable after estimated costs.")
    else:
        print("interpretation: promising in this backtest, but still paper trade before live orders.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    train_model(args.data, args.config)


if __name__ == "__main__":
    main()
