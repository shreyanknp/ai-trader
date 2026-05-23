from __future__ import annotations

import pandas as pd


def threshold_trades(
    df: pd.DataFrame,
    probability_column: str,
    threshold: float,
    cost_bps: float,
) -> pd.DataFrame:
    trades = df[df[probability_column] >= threshold].copy()
    if trades.empty:
        return trades

    cost = cost_bps / 10_000
    trades["net_return"] = trades["future_return"] - cost
    trades["gross_return_pct"] = trades["future_return"] * 100
    trades["net_return_pct"] = trades["net_return"] * 100
    trades["win"] = trades["net_return"] > 0
    trades["equity_curve_pct"] = trades["net_return_pct"].cumsum()
    trades["drawdown_pct"] = trades["equity_curve_pct"] - trades["equity_curve_pct"].cummax()
    return trades


def summarize_predictions(
    df: pd.DataFrame,
    probability_column: str,
    threshold: float,
    cost_bps: float,
) -> dict[str, float]:
    trades = threshold_trades(df, probability_column, threshold, cost_bps)
    if trades.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "payoff_ratio": 0.0,
            "profit_factor": 0.0,
            "avg_return_pct": 0.0,
            "median_return_pct": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
        }

    winners = trades[trades["win"]]
    losers = trades[~trades["win"]]
    avg_win = winners["net_return_pct"].mean() if not winners.empty else 0.0
    avg_loss = losers["net_return_pct"].mean() if not losers.empty else 0.0
    gross_profit = winners["net_return_pct"].sum()
    gross_loss = abs(losers["net_return_pct"].sum())
    payoff_ratio = avg_win / abs(avg_loss) if avg_loss else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss else 0.0

    return {
        "trades": int(len(trades)),
        "win_rate_pct": round(float(trades["win"].mean() * 100), 2),
        "avg_win_pct": round(float(avg_win), 3),
        "avg_loss_pct": round(float(avg_loss), 3),
        "payoff_ratio": round(float(payoff_ratio), 3),
        "profit_factor": round(float(profit_factor), 3),
        "avg_return_pct": round(float(trades["net_return_pct"].mean()), 3),
        "median_return_pct": round(float(trades["net_return_pct"].median()), 3),
        "total_return_pct": round(float(trades["net_return_pct"].sum()), 3),
        "max_drawdown_pct": round(float(trades["drawdown_pct"].min()), 3),
        "best_trade_pct": round(float(trades["net_return_pct"].max()), 3),
        "worst_trade_pct": round(float(trades["net_return_pct"].min()), 3),
    }


def risk_plan(row: pd.Series, config: dict) -> dict[str, float]:
    close = float(row["close"])
    atr = float(row["atr"])
    risk_cfg = config["risk"]
    strategy = config["strategy"]

    stop = close - strategy["stop_loss_atr_multiple"] * atr
    target = close + strategy["target_atr_multiple"] * atr
    risk_per_share = max(close - stop, 0.01)

    account_size = float(risk_cfg["account_size_inr"])
    risk_budget = account_size * float(risk_cfg["risk_per_trade_pct"]) / 100
    max_position_value = account_size * float(risk_cfg["max_position_pct"]) / 100

    shares_by_risk = int(risk_budget // risk_per_share)
    shares_by_cap = int(max_position_value // close)
    quantity = max(min(shares_by_risk, shares_by_cap), 0)

    return {
        "entry_reference": round(close, 2),
        "stop_loss": round(stop, 2),
        "target": round(target, 2),
        "quantity": quantity,
        "capital_used": round(quantity * close, 2),
        "max_loss": round(quantity * risk_per_share, 2),
    }
