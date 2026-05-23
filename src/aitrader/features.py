from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "ma_fast_dist",
    "ma_slow_dist",
    "rsi_14",
    "atr_pct",
    "vwap_dist",
    "volume_ratio",
    "minute_of_day",
]


def load_candles(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def load_candles_many(path_or_paths: str) -> pd.DataFrame:
    paths = resolve_data_paths(path_or_paths)
    frames = [load_candles(str(path)) for path in paths]
    if not frames:
        raise ValueError(f"No candle CSV files found for {path_or_paths}")
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def resolve_data_paths(path_or_paths: str) -> list[Path]:
    values = [value.strip() for value in path_or_paths.split(",") if value.strip()]
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.csv")))
        else:
            matches = [Path(match) for match in sorted(glob.glob(value))] if any(char in value for char in "*?[]") else [path]
            paths.extend(matches)
    return [path for path in paths if path.exists()]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, symbol_df in df.groupby("symbol", sort=False):
        x = symbol_df.copy()
        close = x["close"]
        high = x["high"]
        low = x["low"]
        volume = x["volume"]

        x["return_1"] = close.pct_change(1)
        x["return_3"] = close.pct_change(3)
        x["return_6"] = close.pct_change(6)

        ma_fast = close.rolling(5).mean()
        ma_slow = close.rolling(20).mean()
        x["ma_fast_dist"] = close / ma_fast - 1
        x["ma_slow_dist"] = close / ma_slow - 1

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, pd.NA)
        x["rsi_14"] = 100 - (100 / (1 + rs))

        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14).mean()
        x["atr"] = atr
        x["atr_pct"] = atr / close

        typical_price = (high + low + close) / 3
        session = x["timestamp"].dt.date
        cumulative_pv = (typical_price * volume).groupby(session).cumsum()
        cumulative_volume = volume.groupby(session).cumsum()
        vwap = cumulative_pv / cumulative_volume
        x["vwap_dist"] = close / vwap - 1

        x["volume_ratio"] = volume / volume.rolling(20).mean()
        x["minute_of_day"] = x["timestamp"].dt.hour * 60 + x["timestamp"].dt.minute
        parts.append(x)

    featured = pd.concat(parts, ignore_index=True)
    return featured.dropna(subset=FEATURE_COLUMNS + ["atr"]).reset_index(drop=True)


def add_labels(df: pd.DataFrame, holding_bars: int, threshold_bps: float) -> pd.DataFrame:
    parts = []
    threshold = threshold_bps / 10_000
    for _, symbol_df in df.groupby("symbol", sort=False):
        x = symbol_df.copy()
        future_close = x["close"].shift(-holding_bars)
        x["future_return"] = future_close / x["close"] - 1
        x["label"] = (x["future_return"] > threshold).astype(int)
        parts.append(x)
    labelled = pd.concat(parts, ignore_index=True)
    return labelled.dropna(subset=["future_return", "label"]).reset_index(drop=True)
